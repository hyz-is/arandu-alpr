package imaging

import (
	"bytes"
	"image/jpeg"
	"math"
	"testing"

	"github.com/hyz-is/arandu-alpr/contract"
	"github.com/hyz-is/arandu-alpr/engine"
)

// solidFrame builds a packed BGR24 frame of one colour.
func solidFrame(w, h int, b, g, r byte) engine.Frame {
	pix := make([]byte, w*h*3)
	for i := 0; i < len(pix); i += 3 {
		pix[i], pix[i+1], pix[i+2] = b, g, r
	}
	return engine.Frame{Pix: pix, Width: w, Height: h}
}

// frameFunc builds a frame whose pixels come from a function of (x, y).
func frameFunc(w, h int, fn func(x, y int) (b, g, r byte)) engine.Frame {
	pix := make([]byte, w*h*3)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := (y*w + x) * 3
			pix[i], pix[i+1], pix[i+2] = fn(x, y)
		}
	}
	return engine.Frame{Pix: pix, Width: w, Height: h}
}

func newScaler(t *testing.T, target int) *Scaler {
	t.Helper()
	s, err := NewScaler(target)
	if err != nil {
		t.Fatalf("NewScaler: %v", err)
	}
	return s
}

// TestLetterboxRoundTrip is the test that protects every detection's coordinates.
//
// A box that survives letterbox -> unletterbox unchanged is the only evidence that Ratio, DW and
// DH describe the transform that actually happened. When they do not, boxes land somewhere
// plausible in the frame and the failure reads as a bad model rather than bad arithmetic.
func TestLetterboxRoundTrip(t *testing.T) {
	const frameW, frameH = 1920, 1080
	s := newScaler(t, DetectorInputSize)
	dst := make([]float32, s.TensorLen())

	info, err := s.Letterbox(solidFrame(frameW, frameH, 10, 20, 30), dst)
	if err != nil {
		t.Fatalf("Letterbox: %v", err)
	}

	for _, want := range []contract.BBox{
		{X: 100, Y: 200, Width: 300, Height: 80},
		{X: 0, Y: 0, Width: 1920, Height: 1080},
		{X: 1500, Y: 900, Width: 120, Height: 40},
	} {
		// Forward and inverse both come from engine.Letterbox, so the round trip proves the
		// pair agrees rather than proving one copy agrees with another copy.
		got := info.ToOriginal(info.ToLetterbox(want))

		const tol = 1e-6
		if math.Abs(got.X-want.X) > tol || math.Abs(got.Y-want.Y) > tol ||
			math.Abs(got.Width-want.Width) > tol || math.Abs(got.Height-want.Height) > tol {
			t.Errorf("round trip of %+v gave %+v", want, got)
		}
	}
}

// TestUnletterboxClampsToFrame: the detector regularly proposes boxes over the padding, and an
// unclamped result is a negative origin that indexes out of the frame buffer downstream.
func TestUnletterboxClampsToFrame(t *testing.T) {
	info := engine.Letterbox{Scale: 0.5, PadX: 0, PadY: 100, FrameW: 640, FrameH: 480}
	got := info.ToOriginal(contract.BBox{X: -40, Y: 0, Width: 2000, Height: 4000})

	if got.X != 0 || got.Y != 0 {
		t.Errorf("origin not clamped: %+v", got)
	}
	if got.X2() > 640 || got.Y2() > 480 {
		t.Errorf("extent not clamped: %+v", got)
	}
}

// TestLetterboxPaddingIsExactAndConfined pins two things at once: the padding value is exactly
// 114/255, and it lands only in the bands. A pad that leaks one row into the content is a
// one-row grey stripe the model reads as an edge.
func TestLetterboxPaddingIsExactAndConfined(t *testing.T) {
	const target = 640
	const srcW, srcH = 300, 100
	const cb, cg, cr = 50, 100, 200

	s := newScaler(t, target)
	dst := make([]float32, s.TensorLen())
	if _, err := s.Letterbox(solidFrame(srcW, srcH, cb, cg, cr), dst); err != nil {
		t.Fatalf("Letterbox: %v", err)
	}

	// Geometry recomputed independently of the implementation.
	ratio := math.Min(float64(target)/srcW, float64(target)/srcH)
	newW := int(math.Round(srcW * ratio))
	newH := int(math.Round(srcH * ratio))
	padX := (target - newW) / 2
	padY := (target - newH) / 2

	const grey = float32(PadValue) / 255
	plane := target * target
	want := [3]float32{float32(cr) / 255, float32(cg) / 255, float32(cb) / 255}

	for p := 0; p < 3; p++ {
		for y := 0; y < target; y++ {
			for x := 0; x < target; x++ {
				got := dst[p*plane+y*target+x]
				inContent := y >= padY && y < padY+newH && x >= padX && x < padX+newW
				if !inContent {
					if got != grey {
						t.Fatalf("plane %d (%d,%d): padding is %v, want exactly %v", p, x, y, got, grey)
					}
					continue
				}
				if math.Abs(float64(got-want[p])) > 1e-4 {
					t.Fatalf("plane %d (%d,%d): content is %v, want %v", p, x, y, got, want[p])
				}
			}
		}
	}
}

// TestLetterboxIsNCHW: three distinct constant channels prove each plane carries its own
// channel and that the order is RGB, not the BGR that arrives from ffmpeg.
//
// Swapping R and B here costs accuracy and nothing else: the detector still returns boxes.
func TestLetterboxIsNCHW(t *testing.T) {
	const target = 64
	s := newScaler(t, target)
	dst := make([]float32, s.TensorLen())

	// Deliberately distinct and far apart, so a swapped plane cannot pass by rounding.
	const b, g, r = 20, 130, 240
	if _, err := s.Letterbox(solidFrame(target, target, b, g, r), dst); err != nil {
		t.Fatalf("Letterbox: %v", err)
	}

	plane := target * target
	for i, want := range [3]float32{float32(r) / 255, float32(g) / 255, float32(b) / 255} {
		for j := 0; j < plane; j++ {
			if math.Abs(float64(dst[i*plane+j]-want)) > 1e-4 {
				t.Fatalf("plane %d index %d = %v, want %v (planes must be R, G, B in that order)",
					i, j, dst[i*plane+j], want)
			}
		}
	}
}

// TestCropIsNHWCInZeroTo255 is the test that keeps the OCR readable.
//
// The OCR graph rescales internally. A crop divided by 255 here arrives 255x too small, the
// model still returns ten characters, and nothing anywhere reports a problem. So: a white pixel
// must be 255.0, never 1.0.
func TestCropIsNHWCInZeroTo255(t *testing.T) {
	s := newScaler(t, DetectorInputSize)
	dst := make([]float32, CropOCRTensorLen)

	f := solidFrame(400, 200, 255, 255, 255)
	if err := s.Crop(f, contract.BBox{X: 50, Y: 50, Width: 200, Height: 60}, dst); err != nil {
		t.Fatalf("Crop: %v", err)
	}
	for i, v := range dst {
		if v != 255 {
			t.Fatalf("index %d = %v; white must stay 255.0 -- a 1.0 means someone divided by 255", i, v)
		}
	}

	// Channel order and NHWC interleaving, from a frame that is one colour throughout.
	const b, g, r = 10, 90, 200
	if err := s.Crop(solidFrame(400, 200, b, g, r), contract.BBox{X: 50, Y: 50, Width: 200, Height: 60}, dst); err != nil {
		t.Fatalf("Crop: %v", err)
	}
	for i := 0; i < len(dst); i += 3 {
		if math.Abs(float64(dst[i]-r)) > 0.01 || math.Abs(float64(dst[i+1]-g)) > 0.01 || math.Abs(float64(dst[i+2]-b)) > 0.01 {
			t.Fatalf("pixel %d = (%v,%v,%v), want RGB (%d,%d,%d) interleaved", i/3, dst[i], dst[i+1], dst[i+2], r, g, b)
		}
	}
}

// TestExpandBoxUsesTheEngineConstants keeps the two engines comparable number by number.
func TestExpandBoxUsesTheEngineConstants(t *testing.T) {
	got := ExpandBox(contract.BBox{X: 100, Y: 100, Width: 200, Height: 100})

	want := contract.BBox{
		X:      100 - CropExpandX*200,
		Y:      100 - CropExpandTopY*100,
		Width:  200 + 2*CropExpandX*200,
		Height: 100 + CropExpandTopY*100 + CropExpandBottomY*100,
	}
	if got != want {
		t.Fatalf("ExpandBox = %+v, want %+v", got, want)
	}
	if CropExpandX != 0.15 || CropExpandTopY != 0.22 || CropExpandBottomY != 0.05 {
		t.Fatal("the expansion constants moved; the C++ engine's crops no longer match")
	}
}

// TestCropStaysInsideTheFrame: the expansion pushes a plate at the frame edge outside the
// buffer, and the clamp is the only thing between that and an out-of-range read.
func TestCropStaysInsideTheFrame(t *testing.T) {
	s := newScaler(t, DetectorInputSize)
	dst := make([]float32, CropOCRTensorLen)
	f := solidFrame(64, 48, 30, 60, 90)

	for _, box := range []contract.BBox{
		{X: 0, Y: 0, Width: 20, Height: 10},     // top-left corner
		{X: 44, Y: 38, Width: 20, Height: 10},   // bottom-right corner
		{X: -50, Y: -50, Width: 20, Height: 10}, // entirely off the frame
		{X: 60, Y: 44, Width: 200, Height: 200}, // far past the far edge
		{X: 0, Y: 0, Width: 64, Height: 48},     // the whole frame
		{X: 10, Y: 10, Width: 0.4, Height: 0.4}, // sub-pixel
	} {
		if err := s.Crop(f, box, dst); err != nil {
			t.Fatalf("Crop(%+v): %v", box, err)
		}
		for i, v := range dst {
			if math.IsNaN(float64(v)) {
				t.Fatalf("Crop(%+v) produced NaN at %d", box, i)
			}
		}
	}
}

// TestFocusVarianceSeparatesSharpFromBlurred.
//
// The absolute number is calibration, not correctness; what this pins is the ordering the gate
// depends on. A Laplacian written with the wrong sign or the wrong neighbour set can still
// produce large numbers, just not larger ones for the sharp picture.
func TestFocusVarianceSeparatesSharpFromBlurred(t *testing.T) {
	const w, h = 64, 64

	sharp := frameFunc(w, h, func(x, y int) (byte, byte, byte) {
		if (x/2+y/2)%2 == 0 {
			return 0, 0, 0
		}
		return 255, 255, 255
	})
	blurred := frameFunc(w, h, func(x, y int) (byte, byte, byte) {
		v := byte(128 + 60*math.Sin(float64(x)/20))
		return v, v, v
	})
	flat := solidFrame(w, h, 128, 128, 128)

	s := newScaler(t, DetectorInputSize)
	box := contract.BBox{X: 0, Y: 0, Width: w, Height: h}

	vSharp := s.FocusVariance(sharp, box)
	vBlurred := s.FocusVariance(blurred, box)
	vFlat := s.FocusVariance(flat, box)

	if !(vSharp > vBlurred) {
		t.Errorf("sharp variance %v is not greater than blurred %v", vSharp, vBlurred)
	}
	if !(vBlurred > vFlat) {
		t.Errorf("blurred variance %v is not greater than flat %v", vBlurred, vFlat)
	}
	if vFlat != 0 {
		t.Errorf("a flat image has variance %v, want 0", vFlat)
	}
}

// TestFocusVarianceReusesItsGrayBuffer: the buffer grows to the high-water mark and is never
// reallocated afterwards, which is what keeps the gate off the allocator on the hot path.
func TestFocusVarianceReusesItsGrayBuffer(t *testing.T) {
	s := newScaler(t, DetectorInputSize)
	f := frameFunc(320, 240, func(x, y int) (byte, byte, byte) {
		v := byte((x * y) % 251)
		return v, v, v
	})
	box := contract.BBox{X: 0, Y: 0, Width: 320, Height: 240}

	s.FocusVariance(f, box) // grow once
	if n := testing.AllocsPerRun(50, func() { s.FocusVariance(f, box) }); n != 0 {
		t.Errorf("FocusVariance allocates %v times per call in steady state, want 0", n)
	}
}

// TestBrightnessContrastAtDefaultsWritesNothing.
//
// This is not a micro-optimisation with a test attached. The transform is a float round trip,
// and at 100/100 it is only *algebraically* the identity: (i/255 - 0.5) + 0.5 then *255 does not
// return every i exactly, so running it would silently shift some pixels by one. The guard is
// what makes 100/100 mean untouched, which is what nearly every camera in the fleet runs.
func TestBrightnessContrastAtDefaultsWritesNothing(t *testing.T) {
	pix := make([]byte, 256*3)
	for i := range pix {
		pix[i] = byte(i % 256)
	}
	before := append([]byte(nil), pix...)

	AdjustBrightnessContrast(pix, 100, 100)

	if !bytes.Equal(pix, before) {
		for i := range pix {
			if pix[i] != before[i] {
				t.Fatalf("byte %d changed from %d to %d at 100/100", i, before[i], pix[i])
			}
		}
	}
	if n := testing.AllocsPerRun(50, func() { AdjustBrightnessContrast(pix, 100, 100) }); n != 0 {
		t.Errorf("the 100/100 path allocates %v times, want 0", n)
	}
}

// TestBrightnessContrastActuallyAdjusts, and clamps rather than wrapping. A wrapped byte turns a
// bright sky into a dark one, which looks like a broken camera.
func TestBrightnessContrastActuallyAdjusts(t *testing.T) {
	pix := []byte{0, 64, 128, 192, 255}

	brighter := append([]byte(nil), pix...)
	AdjustBrightnessContrast(brighter, 150, 100)
	for i := range pix {
		if brighter[i] < pix[i] {
			t.Fatalf("brightness 150 darkened byte %d: %d -> %d", i, pix[i], brighter[i])
		}
	}
	if brighter[len(brighter)-1] != 255 {
		t.Fatalf("white clamped to %d, want 255", brighter[len(brighter)-1])
	}

	flatter := append([]byte(nil), pix...)
	AdjustBrightnessContrast(flatter, 100, 0)
	for i, v := range flatter {
		if v != 127 { // contrast 0 collapses everything onto mid-grey
			t.Fatalf("contrast 0 gave %d at %d, want mid-grey", v, i)
		}
	}
}

// TestLetterboxAllocatesNothingInSteadyState.
//
// The weight tables are built on the first frame of a geometry and reused for the rest of the
// camera's life. This is the assertion that the reuse is real: a fixed camera at 30fps that
// allocates one table per frame is a garbage collector running at 30Hz for no reason.
func TestLetterboxAllocatesNothingInSteadyState(t *testing.T) {
	s := newScaler(t, DetectorInputSize)
	f := solidFrame(1280, 720, 40, 80, 120)
	dst := make([]float32, s.TensorLen())

	if _, err := s.Letterbox(f, dst); err != nil { // first frame builds the tables
		t.Fatalf("Letterbox: %v", err)
	}

	if n := testing.AllocsPerRun(20, func() {
		if _, err := s.Letterbox(f, dst); err != nil {
			t.Fatal(err)
		}
	}); n != 0 {
		t.Errorf("Letterbox allocates %v times per frame in steady state, want 0", n)
	}
}

// TestCropAllocatesNothing: the crop tables live in fixed-size arrays on the Scaler precisely
// because the crop's source size changes with every detection.
func TestCropAllocatesNothing(t *testing.T) {
	s := newScaler(t, DetectorInputSize)
	f := solidFrame(1280, 720, 40, 80, 120)
	dst := make([]float32, CropOCRTensorLen)
	box := contract.BBox{X: 300, Y: 200, Width: 180, Height: 60}

	if err := s.Crop(f, box, dst); err != nil {
		t.Fatalf("Crop: %v", err)
	}
	if n := testing.AllocsPerRun(20, func() {
		if err := s.Crop(f, box, dst); err != nil {
			t.Fatal(err)
		}
	}); n != 0 {
		t.Errorf("Crop allocates %v times per call, want 0", n)
	}
}

// TestLetterboxRejectsMismatchedGeometry: a frame whose buffer disagrees with its declared size
// would otherwise be read at the wrong stride, producing a skewed picture that still detects.
func TestLetterboxRejectsMismatchedGeometry(t *testing.T) {
	s := newScaler(t, 64)
	dst := make([]float32, s.TensorLen())

	f := solidFrame(32, 32, 1, 2, 3)
	f.Pix = f.Pix[:len(f.Pix)-3]
	if _, err := s.Letterbox(f, dst); err != ErrFrameSize {
		t.Errorf("short frame gave %v, want ErrFrameSize", err)
	}

	if _, err := s.Letterbox(solidFrame(32, 32, 1, 2, 3), dst[:len(dst)-1]); err != ErrDstSize {
		t.Errorf("short tensor gave %v, want ErrDstSize", err)
	}
}

// TestScalerRebuildsTablesWhenTheSourceSizeChanges: one Scaler must not carry a stale geometry
// into a stream that reconnected at a different resolution.
func TestScalerRebuildsTablesWhenTheSourceSizeChanges(t *testing.T) {
	const target = 64
	s := newScaler(t, target)
	dst := make([]float32, s.TensorLen())

	wide, err := s.Letterbox(solidFrame(128, 32, 1, 2, 3), dst)
	if err != nil {
		t.Fatalf("Letterbox: %v", err)
	}
	tall, err := s.Letterbox(solidFrame(32, 128, 1, 2, 3), dst)
	if err != nil {
		t.Fatalf("Letterbox: %v", err)
	}

	if wide.PadY == tall.PadY || wide.PadX == tall.PadX {
		t.Fatalf("geometry did not change: wide=%+v tall=%+v", wide, tall)
	}
}

func TestEncodeJPEG(t *testing.T) {
	e, err := NewEncoder(85)
	if err != nil {
		t.Fatalf("NewEncoder: %v", err)
	}

	f := frameFunc(320, 240, func(x, y int) (byte, byte, byte) {
		return byte(x), byte(y), 200
	})

	full, err := e.EncodeFrame(f)
	if err != nil {
		t.Fatalf("EncodeFrame: %v", err)
	}
	img, err := jpeg.Decode(bytes.NewReader(full))
	if err != nil {
		t.Fatalf("decoding the frame we just encoded: %v", err)
	}
	if got := img.Bounds().Size(); got.X != 320 || got.Y != 240 {
		t.Errorf("frame jpeg is %v, want 320x240", got)
	}

	box := contract.BBox{X: 100, Y: 100, Width: 40, Height: 20}
	crop, err := e.EncodeCrop(f, box)
	if err != nil {
		t.Fatalf("EncodeCrop: %v", err)
	}
	cimg, err := jpeg.Decode(bytes.NewReader(crop))
	if err != nil {
		t.Fatalf("decoding the crop we just encoded: %v", err)
	}
	_, _, w, h := clampedRegion(ExpandBox(box), f.Width, f.Height)
	if got := cimg.Bounds().Size(); got.X != w || got.Y != h {
		t.Errorf("crop jpeg is %v, want %dx%d (the expanded box)", got, w, h)
	}
	if len(crop) >= len(full) {
		t.Errorf("a 40x20 crop encoded to %d bytes, no smaller than the full frame's %d", len(crop), len(full))
	}
}

// TestEncoderReturnsOwnedBytes: contract.Detection promises the receiver may keep FullJPEG
// forever. Handing back the pooled buffer would corrupt an image rather than fail.
func TestEncoderReturnsOwnedBytes(t *testing.T) {
	e, err := NewEncoder(60)
	if err != nil {
		t.Fatalf("NewEncoder: %v", err)
	}
	f := frameFunc(64, 64, func(x, y int) (byte, byte, byte) { return byte(x * 4), byte(y * 4), 128 })

	first, err := e.EncodeFrame(f)
	if err != nil {
		t.Fatalf("EncodeFrame: %v", err)
	}
	keep := append([]byte(nil), first...)

	for i := 0; i < 5; i++ {
		if _, err := e.EncodeFrame(solidFrame(64, 64, byte(i), byte(i), byte(i))); err != nil {
			t.Fatalf("EncodeFrame: %v", err)
		}
	}
	if !bytes.Equal(first, keep) {
		t.Fatal("a later encode overwrote bytes an earlier caller owns")
	}
}

func TestNewEncoderRejectsImpossibleQuality(t *testing.T) {
	for _, q := range []int{0, -1, 101} {
		if _, err := NewEncoder(q); err == nil {
			t.Errorf("NewEncoder(%d) accepted an out-of-range quality", q)
		}
	}
}

func TestTheCropHasItsOwnQuality(t *testing.T) {
	// A noisy picture, so quality moves the size: the crop of the same pixels is larger at 95 than
	// at 40, and the whole frame is not affected by the crop's setting.
	f := frameFunc(320, 240, func(x, y int) (byte, byte, byte) {
		return byte(x*7 + y*13), byte(x*3 ^ y*5), byte(x * y)
	})
	box := contract.BBox{X: 100, Y: 100, Width: 80, Height: 40}

	sharp, err := NewEncoderQualities(40, 95)
	if err != nil {
		t.Fatal(err)
	}
	soft, err := NewEncoderQualities(40, 40)
	if err != nil {
		t.Fatal(err)
	}
	cropSharp, _ := sharp.EncodeCrop(f, box)
	cropSoft, _ := soft.EncodeCrop(f, box)
	if len(cropSharp) <= len(cropSoft) {
		t.Errorf("crop at quality 95 is %d bytes and at 40 is %d: the crop quality is not applied", len(cropSharp), len(cropSoft))
	}
	frameA, _ := sharp.EncodeFrame(f)
	frameB, _ := soft.EncodeFrame(f)
	if !bytes.Equal(frameA, frameB) {
		t.Error("the crop's quality changed the whole frame's encoding")
	}
	if _, err := NewEncoderQualities(70, 0); err == nil {
		t.Error("a crop quality of 0 was accepted")
	}
}
