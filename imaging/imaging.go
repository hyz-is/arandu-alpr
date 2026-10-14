// Package imaging turns decoded frames into the exact numbers the two models expect.
//
// Everything here exists because a preprocessing mistake does not look like a bug. It looks like
// a model that reads badly. The detector fed a tensor in the wrong plane order still returns
// boxes; the OCR fed a crop divided by 255 still returns characters. Nothing errors, nothing
// logs, and the only symptom is accuracy that is quietly worse than the model's.
//
// So the invariants are stated once, here, and the tests hold them:
//
//   - detector input: NCHW, RGB, /255, letterbox with grey 114 padding
//   - OCR input:      NHWC, RGB, [0,255] and NOT divided by 255 -- the rescale is baked into
//     the graph, and dividing here destroys the reading
//   - crops come from the ORIGINAL frame, never from the letterbox, because cropping the
//     letterbox throws away resolution exactly on the distant plates
//
// The numbers match the sibling C++ engine (alpr-c/crates/alpr-detect/src/preprocess.cpp) so the
// two can be compared reading by reading during the migration. Where a constant is named there,
// it keeps its name here.
//
// # Allocation
//
// The per-frame path allocates nothing. Scaler owns every buffer it needs and sizes them on the
// first frame of a given geometry; a fixed camera therefore pays that cost once in its life.
// A Scaler is not safe for concurrent use -- one per camera goroutine, which is also how the
// buffers stay reusable at all.
package imaging

import (
	"errors"
	"math"

	"github.com/hyz-is/arandu-alpr/contract"
	"github.com/hyz-is/arandu-alpr/engine"
)

// DetectorInputSize is the square side of the detector tensor.
// DetectorInputSize is the engine's, aliased so the two cannot drift apart.
const DetectorInputSize = engine.DetectorInputSize

// OCR input geometry. The OCR tensor is [1,64,128,3]: height 64, width 128, channels last.
const (
	OCRInputWidth  = 128
	OCRInputHeight = 64
)

// PadValue is the grey the letterbox pads with, on the 0-255 scale.
//
// 114 is not decorative. It is the value the detector was trained against, and padding with
// black instead puts a hard edge around the picture that the model reads as structure.
const PadValue = 114

// Crop expansion, as fractions of the box's own dimensions.
//
// A box drawn tight around a plate cuts the glyph strokes, and the CTC head then confuses the
// characters that differ only at their extremities: D/O, 5/6, 8/B. The expansion is asymmetric
// because the mounting frame and the bolts sit above the characters while the bottom edge is
// usually already clean.
//
// The names are the ones the C++ engine uses (CROP_EXPAND_X, CROP_EXPAND_TOP_Y,
// CROP_EXPAND_BOTTOM_Y) so a divergence between the engines can be traced to a number.
const (
	CropExpandX       = 0.15
	CropExpandTopY    = 0.22
	CropExpandBottomY = 0.05
)

// ErrFrameSize is returned when a frame's pixel buffer does not match its declared geometry.
//
// Packed BGR24 is the only format this package accepts, so len(Pix) must be exactly W*H*3. A
// frame that is one row short would otherwise be read past its end or, worse, be silently
// reinterpreted at the wrong stride -- which produces a skewed picture that still detects.
var ErrFrameSize = errors.New("imaging: frame is not packed BGR24 of the declared size")

// ErrDstSize is returned when the destination tensor slice is not the exact expected length.
var ErrDstSize = errors.New("imaging: destination tensor has the wrong length")

// The letterbox transform lives in the engine package, not here.
//
// It used to be declared twice -- Ratio/DW/DH here and Scale/PadX/PadY there -- computing the
// same three numbers for the forward map and the inverse. They agreed, and that is the point:
// two correct copies of one geometry stay correct only until somebody edits one. A box that
// comes back through a drifted inverse has plausible coordinates pointing at the wrong part of
// the picture, which reads downstream as an ordinary weak detection.
//
// So this package produces engine.Letterbox and the engine's post-processing consumes it.

// Scaler holds every buffer the per-frame path needs.
//
// It is stateful on purpose. The bilinear weight tables depend only on (source size -> target
// size), and a camera's source size does not change, so the tables are computed on the first
// frame and reused for the rest of the process's life. Recomputing them per frame would be two
// allocations and a few thousand divisions on a path that runs at frame rate.
//
// Not safe for concurrent use.
type Scaler struct {
	target int

	// Letterbox weight tables, valid for the (srcW, srcH) that built them.
	lbSrcW, lbSrcH int
	lbInfo         engine.Letterbox
	lbNewW, lbNewH int
	lbPadX, lbPadY int
	lbXOff0        []int32 // byte offset of the left tap within a source row
	lbXOff1        []int32
	lbXW           []float32
	lbYOff0        []int32 // byte offset of the top tap's row start
	lbYOff1        []int32
	lbYW           []float32

	// Crop resize tables. The crop's source size changes with every detection, so these are
	// rewritten per call -- but they are OCRInputWidth and OCRInputHeight long, which is 192
	// entries, not the source's dimensions.
	crXOff0 [OCRInputWidth]int32
	crXOff1 [OCRInputWidth]int32
	crXW    [OCRInputWidth]float32
	crYOff0 [OCRInputHeight]int32
	crYOff1 [OCRInputHeight]int32
	crYW    [OCRInputHeight]float32

	// gray is the crop's luma, reused across calls and grown to the largest crop seen. A plate
	// crop is tens of kilobytes, so holding the high-water mark costs nothing worth measuring.
	gray []byte
}

// NewScaler returns a Scaler producing a target x target detector tensor.
//
// target is a parameter rather than the constant so the geometry can be tested at sizes small
// enough to write out by hand; production passes DetectorInputSize.
func NewScaler(target int) (*Scaler, error) {
	if target < 2 {
		return nil, errors.New("imaging: letterbox target must be at least 2")
	}
	return &Scaler{target: target}, nil
}

// TensorLen is the exact length Letterbox requires of dst.
func (s *Scaler) TensorLen() int { return 3 * s.target * s.target }

// Letterbox writes f into dst as the detector's input tensor and reports how to undo it.
//
// dst is written in place and must be exactly TensorLen() long: it is the tensor's own backing
// array, so the resize, the colour swap, the normalisation and the NCHW scatter all happen in
// one pass with no intermediate image anywhere. The obvious implementation -- resize to a byte
// canvas, then convert -- costs a second full traversal and a megabyte of garbage per frame.
//
// Only the padding bands are filled with grey. Painting the whole tensor first would write 1.2
// million floats that the content pass immediately overwrites.
func (s *Scaler) Letterbox(f engine.Frame, dst []float32) (engine.Letterbox, error) {
	if f.Width <= 0 || f.Height <= 0 || len(f.Pix) != f.Width*f.Height*3 {
		return engine.Letterbox{}, ErrFrameSize
	}
	if len(dst) != s.TensorLen() {
		return engine.Letterbox{}, ErrDstSize
	}

	s.prepareLetterbox(f.Width, f.Height)

	t := s.target
	plane := t * t
	const inv255 = float32(1) / 255

	s.padLetterbox(dst)

	src := f.Pix
	for oy := 0; oy < s.lbNewH; oy++ {
		row0 := s.lbYOff0[oy]
		row1 := s.lbYOff1[oy]
		wy := s.lbYW[oy]
		base := (s.lbPadY+oy)*t + s.lbPadX

		for ox := 0; ox < s.lbNewW; ox++ {
			x0 := s.lbXOff0[ox]
			x1 := s.lbXOff1[ox]
			wx := s.lbXW[ox]

			// Four taps, three channels. Indices are byte offsets into packed BGR24, so the
			// blue channel is at +0 and red at +2.
			a := row0 + x0
			b := row0 + x1
			c := row1 + x0
			d := row1 + x1

			top0 := float32(src[a])
			top1 := float32(src[b])
			bot0 := float32(src[c])
			bot1 := float32(src[d])
			blue := (top0+(top1-top0)*wx)*(1-wy) + (bot0+(bot1-bot0)*wx)*wy

			top0 = float32(src[a+1])
			top1 = float32(src[b+1])
			bot0 = float32(src[c+1])
			bot1 = float32(src[d+1])
			green := (top0+(top1-top0)*wx)*(1-wy) + (bot0+(bot1-bot0)*wx)*wy

			top0 = float32(src[a+2])
			top1 = float32(src[b+2])
			bot0 = float32(src[c+2])
			bot1 = float32(src[d+2])
			red := (top0+(top1-top0)*wx)*(1-wy) + (bot0+(bot1-bot0)*wx)*wy

			p := base + ox
			dst[p] = red * inv255
			dst[plane+p] = green * inv255
			dst[2*plane+p] = blue * inv255
		}
	}

	return s.lbInfo, nil
}

// prepareLetterbox rebuilds the weight tables, and only when the source geometry moved.
//
// The early return is the whole point: a fixed camera reaches this function on every frame and
// leaves it immediately, which is what keeps Letterbox at zero allocations in steady state.
func (s *Scaler) prepareLetterbox(srcW, srcH int) {
	if s.lbSrcW == srcW && s.lbSrcH == srcH {
		return
	}

	t := s.target
	ratio := math.Min(float64(t)/float64(srcW), float64(t)/float64(srcH))
	newW := clampInt(int(math.Round(float64(srcW)*ratio)), 1, t)
	newH := clampInt(int(math.Round(float64(srcH)*ratio)), 1, t)

	// DW/DH stay fractional; the placement offset truncates. That asymmetry is deliberate and
	// matches the C++ engine, which places at int(dw) and inverts with the float.
	s.lbInfo = engine.Letterbox{
		Scale:  ratio,
		PadX:   float64(t-newW) / 2,
		PadY:   float64(t-newH) / 2,
		FrameW: srcW,
		FrameH: srcH,
	}
	s.lbNewW, s.lbNewH = newW, newH
	s.lbPadX, s.lbPadY = int(s.lbInfo.PadX), int(s.lbInfo.PadY)

	s.lbXOff0 = growInt32(s.lbXOff0, newW)
	s.lbXOff1 = growInt32(s.lbXOff1, newW)
	s.lbXW = growFloat32(s.lbXW, newW)
	s.lbYOff0 = growInt32(s.lbYOff0, newH)
	s.lbYOff1 = growInt32(s.lbYOff1, newH)
	s.lbYW = growFloat32(s.lbYW, newH)

	buildAxis(srcW, newW, 3, s.lbXOff0, s.lbXOff1, s.lbXW)
	buildAxis(srcH, newH, srcW*3, s.lbYOff0, s.lbYOff1, s.lbYW)

	s.lbSrcW, s.lbSrcH = srcW, srcH
}

// padLetterbox fills the four padding bands, and nothing else.
//
// The bands are the same on every frame of a given geometry, but they are rewritten each call
// because dst belongs to the caller: assuming the previous contents survived would make this
// function's correctness depend on what someone else did with their buffer.
func (s *Scaler) padLetterbox(dst []float32) {
	const grey = float32(PadValue) / 255
	t := s.target
	plane := t * t

	top := s.lbPadY * t
	bottomStart := (s.lbPadY + s.lbNewH) * t

	for p := 0; p < 3; p++ {
		base := p * plane
		fillFloat32(dst[base:base+top], grey)
		fillFloat32(dst[base+bottomStart:base+plane], grey)

		if s.lbPadX == 0 && s.lbPadX+s.lbNewW == t {
			continue
		}
		rightStart := s.lbPadX + s.lbNewW
		for y := s.lbPadY; y < s.lbPadY+s.lbNewH; y++ {
			row := base + y*t
			fillFloat32(dst[row:row+s.lbPadX], grey)
			fillFloat32(dst[row+rightStart:row+t], grey)
		}
	}
}

// CropOCRTensorLen is the exact length Crop requires of dst.
const CropOCRTensorLen = OCRInputWidth * OCRInputHeight * 3

// Crop writes the OCR's input tensor for one box, taken from the ORIGINAL frame.
//
// dst is NHWC (row-major, channels last, RGB) and on the **[0,255] scale -- it is not divided by
// 255**. The rescale is a node inside the OCR graph, so dividing here feeds the model values a
// second factor of 255 too small and the reading collapses without any error surfacing. This is
// the single easiest thing in this package to "fix" into a broken state; see the test that pins
// a white pixel at 255.0.
//
// The box is expanded by the CropExpand* fractions before clamping, and the clamp is what keeps
// the expansion inside the frame buffer.
func (s *Scaler) Crop(f engine.Frame, box contract.BBox, dst []float32) error {
	if f.Width <= 0 || f.Height <= 0 || len(f.Pix) != f.Width*f.Height*3 {
		return ErrFrameSize
	}
	if len(dst) != CropOCRTensorLen {
		return ErrDstSize
	}

	x0, y0, cw, ch := clampedRegion(ExpandBox(box), f.Width, f.Height)

	buildAxis(cw, OCRInputWidth, 3, s.crXOff0[:], s.crXOff1[:], s.crXW[:])
	buildAxis(ch, OCRInputHeight, f.Width*3, s.crYOff0[:], s.crYOff1[:], s.crYW[:])

	src := f.Pix
	origin := int32((y0*f.Width + x0) * 3)

	for oy := 0; oy < OCRInputHeight; oy++ {
		row0 := origin + s.crYOff0[oy]
		row1 := origin + s.crYOff1[oy]
		wy := s.crYW[oy]
		out := oy * OCRInputWidth * 3

		for ox := 0; ox < OCRInputWidth; ox++ {
			xa := s.crXOff0[ox]
			xb := s.crXOff1[ox]
			wx := s.crXW[ox]

			a := row0 + xa
			b := row0 + xb
			c := row1 + xa
			d := row1 + xb

			t0 := float32(src[a])
			t1 := float32(src[b])
			b0 := float32(src[c])
			b1 := float32(src[d])
			blue := (t0+(t1-t0)*wx)*(1-wy) + (b0+(b1-b0)*wx)*wy

			t0 = float32(src[a+1])
			t1 = float32(src[b+1])
			b0 = float32(src[c+1])
			b1 = float32(src[d+1])
			green := (t0+(t1-t0)*wx)*(1-wy) + (b0+(b1-b0)*wx)*wy

			t0 = float32(src[a+2])
			t1 = float32(src[b+2])
			b0 = float32(src[c+2])
			b1 = float32(src[d+2])
			red := (t0+(t1-t0)*wx)*(1-wy) + (b0+(b1-b0)*wx)*wy

			p := out + ox*3
			dst[p] = red
			dst[p+1] = green
			dst[p+2] = blue
		}
	}
	return nil
}

// ExpandBox applies the asymmetric crop expansion. It does not clamp: the caller's clamp is
// what knows the frame.
func ExpandBox(b contract.BBox) contract.BBox {
	dx := CropExpandX * b.Width
	top := CropExpandTopY * b.Height
	bottom := CropExpandBottomY * b.Height
	return contract.BBox{
		X:      b.X - dx,
		Y:      b.Y - top,
		Width:  b.Width + 2*dx,
		Height: b.Height + top + bottom,
	}
}

// FocusVariance is the variance of the Laplacian of the box's grey pixels.
//
// It feeds GateConfig.AcceptFocus, which runs before the OCR: this number is what decides
// whether a crop is worth a GPU pass, so it is computed on the raw box, unexpanded and before
// any resize. Both details matter. The gate normalises its threshold by the box's own area, so
// measuring an expanded region would compare a variance against a threshold calibrated for a
// different rectangle; and measuring after the resize to 128x64 would measure the resampler's
// smoothing rather than the lens's.
//
// The kernel is OpenCV's default ksize=1 Laplacian, [[0,1,0],[1,-4,1],[0,1,0]], so the numbers
// stay comparable with the Python service's cv2.Laplacian(gray, CV_64F).var() that the
// threshold of 80 was calibrated against. Border pixels are excluded rather than reflected; on a
// plate-sized crop that is under 3% of the pixels and well inside the threshold's resolution.
func (s *Scaler) FocusVariance(f engine.Frame, box contract.BBox) float64 {
	if f.Width <= 0 || f.Height <= 0 || len(f.Pix) != f.Width*f.Height*3 {
		return 0
	}
	x0, y0, cw, ch := clampedRegion(box, f.Width, f.Height)
	if cw < 3 || ch < 3 {
		return 0
	}

	s.gray = growByte(s.gray, cw*ch)
	gray := s.gray[:cw*ch]
	src := f.Pix
	for y := 0; y < ch; y++ {
		row := ((y0+y)*f.Width + x0) * 3
		out := y * cw
		for x := 0; x < cw; x++ {
			i := row + x*3
			gray[out+x] = luma(src[i], src[i+1], src[i+2])
		}
	}

	// One pass, integer accumulators. The Laplacian of uint8 input is an exact integer in
	// [-1020,1020], so sum and sum of squares are exact and the variance carries no drift from
	// accumulating millions of float32 additions.
	var sum, sumSq int64
	n := int64((cw - 2) * (ch - 2))
	for y := 1; y < ch-1; y++ {
		row := y * cw
		up := row - cw
		down := row + cw
		for x := 1; x < cw-1; x++ {
			lap := int64(gray[up+x]) + int64(gray[down+x]) +
				int64(gray[row+x-1]) + int64(gray[row+x+1]) -
				4*int64(gray[row+x])
			sum += lap
			sumSq += lap * lap
		}
	}

	mean := float64(sum) / float64(n)
	variance := float64(sumSq)/float64(n) - mean*mean
	if variance < 0 {
		return 0
	}
	return variance
}

// luma is BT.601, rounded and clamped -- the same expression as the C++ bgr_to_gray.
//
// The obvious integer fixed-point form is a bit faster and differs by one LSB on some inputs.
// One LSB of grey is one unit of Laplacian, and this number is compared against a threshold the
// other engine also compares against, so the arithmetic is kept identical instead.
func luma(b, g, r byte) byte {
	v := math.Round(float64(0.299*float32(r) + 0.587*float32(g) + 0.114*float32(b)))
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return byte(v)
}

// AdjustBrightnessContrast rewrites pix in place, and does nothing at all when both are 100.
//
// The early return is the feature. The Python service ran this as a float32 round trip over the
// whole frame -- `astype(float32)/255`, arithmetic, `clip`, `*255`, `astype(uint8)` -- which is
// four full-frame passes and two temporary arrays per frame. Nearly every camera in the fleet
// runs at 100/100, so nearly every camera was paying for an identity transform.
//
// The transform itself is the Python one, expressed as a 256-entry table because it is a
// per-channel scalar function: contrast pivots around mid-grey, then brightness shifts. The
// final conversion truncates rather than rounds, matching numpy's astype(uint8), so a frame
// adjusted by either service is byte-identical.
func AdjustBrightnessContrast(pix []byte, brightness, contrast int) {
	if brightness == 100 && contrast == 100 {
		return
	}

	var lut [256]byte
	contrastFactor := float64(contrast) / 100
	brightnessShift := float64(brightness-100) / 100
	for i := range lut {
		v := (float64(i)/255-0.5)*contrastFactor + 0.5 + brightnessShift
		v = clampFloat(v, 0, 1) * 255
		lut[i] = byte(v)
	}

	for i, v := range pix {
		pix[i] = lut[v]
	}
}

// clampedRegion turns a float box into an integer region guaranteed to be inside a W x H frame.
//
// It always returns at least one pixel: a degenerate region would make every caller check, and
// the callers that forgot would index out of range on a box that arrived inverted.
func clampedRegion(b contract.BBox, frameW, frameH int) (x0, y0, w, h int) {
	x0 = clampInt(int(math.Floor(b.X)), 0, frameW-1)
	y0 = clampInt(int(math.Floor(b.Y)), 0, frameH-1)
	x1 := clampInt(int(math.Ceil(b.X2())), x0+1, frameW)
	y1 := clampInt(int(math.Ceil(b.Y2())), y0+1, frameH)
	return x0, y0, x1 - x0, y1 - y0
}

// buildAxis fills the bilinear tap offsets and weights for one axis.
//
// stride is the distance in bytes between two neighbouring samples along that axis, so the same
// function serves columns (stride 3, packed BGR24) and rows (stride srcW*3).
//
// The source coordinate is (dst + 0.5) * scale - 0.5, which aligns pixel centres. The naive
// dst * scale aligns pixel corners instead and shifts the whole picture by half a destination
// pixel -- invisible on a photograph, and a systematic bias in a box the model regresses.
func buildAxis(srcLen, dstLen, stride int, off0, off1 []int32, weights []float32) {
	scale := float64(srcLen) / float64(dstLen)
	for i := 0; i < dstLen; i++ {
		f := (float64(i)+0.5)*scale - 0.5
		lo := int(math.Floor(f))
		w := f - float64(lo)
		if lo < 0 {
			lo, w = 0, 0
		}
		hi := lo + 1
		if hi > srcLen-1 {
			hi = srcLen - 1
		}
		if lo > srcLen-1 {
			lo, w = srcLen-1, 0
		}
		off0[i] = int32(lo * stride)
		off1[i] = int32(hi * stride)
		weights[i] = float32(w)
	}
}

// fillFloat32 writes v across dst.
//
// The doubling copy is not micro-optimisation for its own sake: the padding bands of a 16:9
// frame are around 40% of the tensor, and copy() on a float32 slice compiles to a memmove while
// a plain range loop does not.
func fillFloat32(dst []float32, v float32) {
	if len(dst) == 0 {
		return
	}
	dst[0] = v
	for filled := 1; filled < len(dst); filled *= 2 {
		copy(dst[filled:], dst[:filled])
	}
}

// grow* return a slice of at least n capacity, reusing the existing backing array when it fits.
// They are the reason a steady-state frame allocates nothing.
func growInt32(s []int32, n int) []int32 {
	if cap(s) >= n {
		return s[:n]
	}
	return make([]int32, n)
}

func growFloat32(s []float32, n int) []float32 {
	if cap(s) >= n {
		return s[:n]
	}
	return make([]float32, n)
}

func growByte(s []byte, n int) []byte {
	if cap(s) >= n {
		return s[:n]
	}
	return make([]byte, n)
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func clampFloat(v, lo, hi float64) float64 {
	if math.IsNaN(v) {
		return lo
	}
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
