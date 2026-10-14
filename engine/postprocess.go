package engine

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/hyz-is/arandu-alpr/contract"
)

// The box tensor is [1, 4, N]: channel-major, not [1, N, 4].
//
// This is the most dangerous fact in this file. Reading it as N rows of four compiles, runs, and
// produces boxes with entirely plausible numbers -- positive centres, sane sizes, scores that pass
// the gate. They describe a different part of the picture, so the crop is of nothing and the OCR
// returns what looks like an ordinary bad read. Nothing fails and nothing is logged; the fleet just
// stops detecting.
//
// TestBoxTensorIsChannelMajor pins it against a synthetic tensor whose right answer is known.
const (
	boxChannels = 4
	chCX        = 0
	chCY        = 1
	chW         = 2
	chH         = 3
)

// ErrTensorShape means the tensor handed to Decode is not the one this Postprocessor was built
// for.
//
// It is a hard error rather than a best-effort decode on purpose. A 512 or 960 model produces a
// different anchor count, and decoding it against the 640 layout yields boxes -- wrong ones, with
// no symptom other than plates that never arrive.
var ErrTensorShape = errors.New("engine: detector tensor has the wrong shape")

// DetectorInputSize is the square the detector takes, and the single place it is written.
//
// Everything derived from it -- the anchor count, the letterbox geometry, the tensor shape -- is
// computed, never restated. A second copy is how a re-export at another size ends up half applied:
// one place updated, one place not, and boxes that are the right shape in the wrong place.
const DetectorInputSize = 640

// AnchorCount returns how many anchors a stride-8/16/32 detector emits for a square input.
//
// 8400 is derived, never written down: it is 80² + 40² + 20² for a 640 input. Writing the literal
// is what lets a 512 model (5376 anchors) or a 960 one (18900) be decoded as if it were a 640, and
// the arithmetic below is the only thing standing between a swapped model file and silent garbage.
//
// A size the strides do not divide has no anchor grid at all, and returns 0.
func AnchorCount(inputSize int) int {
	if inputSize <= 0 || inputSize%32 != 0 {
		return 0
	}
	total := 0
	for _, stride := range [...]int{8, 16, 32} {
		cells := inputSize / stride
		total += cells * cells
	}
	return total
}

// PostprocessConfig carries the knobs. They are values rather than environment reads so the
// decoding stays pure and testable.
type PostprocessConfig struct {
	// InputSize is the side of the detector's square input. It fixes the anchor count, and
	// therefore the only tensor shape this Postprocessor will accept.
	InputSize int

	// ScoreThreshold drops weak anchors during the single scan, before anything is sorted.
	ScoreThreshold float64

	// IoUThreshold is the overlap above which the weaker of two boxes is suppressed.
	IoUThreshold float64

	// MaxDetections caps how many boxes leave the detector for one frame.
	//
	// It is a cost ceiling, not a quality knob: every box that survives costs a crop and a full
	// OCR pass, so an unbounded frame is an unbounded GPU bill.
	MaxDetections int
}

// DefaultPostprocess mirrors what the Python service runs with.
//
// ScoreThreshold matches GateConfig.MinDetectionScore: this pass must never accept less than the
// gate would, or the gate is doing work the scan could have avoided. MaxDetections is the Python
// service's MAX_DETECTIONS_PER_FRAME.
func DefaultPostprocess() PostprocessConfig {
	return PostprocessConfig{
		InputSize:      640,
		ScoreThreshold: 0.24,
		IoUThreshold:   0.45,
		MaxDetections:  10,
	}
}

// Postprocessor turns the detector's two output tensors into candidates.
//
// It owns every buffer the four passes need, so decoding a frame allocates nothing. One instance
// belongs to one inference session: it is not safe for concurrent use, and it does not have to be
// -- SessionPool already guarantees a session is used by one goroutine at a time.
type Postprocessor struct {
	cfg     PostprocessConfig
	anchors int

	// Compared in the hot loop, so it is kept in the tensor's own type. Converting 8400 float32s
	// to float64 to compare against a float64 threshold is 8400 conversions to reach the same
	// answer.
	scoreThreshold float32

	// survivors holds everything above the threshold, in letterbox coordinates, and is sorted in
	// place. Its capacity is the full anchor count: typically twenty of 8400 survive, but a
	// degenerate model puts every anchor above the threshold, and that is precisely the frame on
	// which growing a slice is worst.
	survivors []Candidate

	// kept holds the NMS result, still in letterbox coordinates, so the greedy pass can measure
	// overlap against boxes that have not yet been transformed.
	kept []Candidate
}

// NewPostprocessor validates the configuration and pre-allocates the buffers.
func NewPostprocessor(cfg PostprocessConfig) (*Postprocessor, error) {
	anchors := AnchorCount(cfg.InputSize)
	if anchors == 0 {
		return nil, fmt.Errorf("engine: input size %d has no stride-8/16/32 anchor grid", cfg.InputSize)
	}
	if cfg.MaxDetections < 1 {
		return nil, fmt.Errorf("engine: max detections must be at least 1, got %d", cfg.MaxDetections)
	}
	if !(cfg.IoUThreshold >= 0 && cfg.IoUThreshold <= 1) {
		return nil, fmt.Errorf("engine: iou threshold must be in [0,1], got %v", cfg.IoUThreshold)
	}
	return &Postprocessor{
		cfg:            cfg,
		anchors:        anchors,
		scoreThreshold: float32(cfg.ScoreThreshold),
		survivors:      make([]Candidate, 0, anchors),
		kept:           make([]Candidate, 0, cfg.MaxDetections),
	}, nil
}

// Anchors is the tensor length this Postprocessor accepts.
func (p *Postprocessor) Anchors() int { return p.anchors }

// Config returns the configuration this Postprocessor was built with.
func (p *Postprocessor) Config() PostprocessConfig { return p.cfg }

// Decode turns one frame's detector output into candidates in ORIGINAL frame coordinates.
//
// boxes is [1, 4, anchors] and scores is [1, 1, anchors]. dst is appended to and returned, in the
// stdlib's AppendFormat shape, so a caller reusing a slice between frames allocates nothing.
//
// Four passes:
//
//  1. one scan of the anchors, filtering by score and converting centre-form to corner-form in
//     the same pass -- there is no reason to walk 8400 anchors twice
//  2. sort by score, descending
//  3. greedy suppression by IoU, each candidate compared only against what is already kept, so
//     the cost is bounded by MaxDetections rather than by the survivor count squared
//  4. unletterbox, LAST
//
// The order of 3 and 4 is the only interesting one. IoU is a ratio of areas, and a letterbox is a
// uniform scale plus a translation, so every area in that ratio scales by the same factor and the
// ratio is unchanged: suppressing before or after the transform gives an identical result. Doing
// it after means transforming at most MaxDetections boxes instead of every survivor.
func (p *Postprocessor) Decode(boxes, scores []float32, lb Letterbox, dst []Candidate) ([]Candidate, error) {
	if len(scores) != p.anchors || len(boxes) != p.anchors*boxChannels {
		return dst, fmt.Errorf("%w: input size %d expects boxes[%d] and scores[%d], got boxes[%d] and scores[%d]",
			ErrTensorShape, p.cfg.InputSize, p.anchors*boxChannels, p.anchors, len(boxes), len(scores))
	}

	// Pass 1: filter and convert. The channel bases are hoisted out of the loop; they are what
	// makes this channel-major rather than row-major.
	cxBase := chCX * p.anchors
	cyBase := chCY * p.anchors
	wBase := chW * p.anchors
	hBase := chH * p.anchors

	p.survivors = p.survivors[:0]
	for i, score := range scores {
		// Negated rather than `score < threshold`: a NaN score fails every ordered comparison, so
		// `<` would let it through and send a garbage crop to the OCR.
		if !(score >= p.scoreThreshold) {
			continue
		}
		cx := float64(boxes[cxBase+i])
		cy := float64(boxes[cyBase+i])
		w := float64(boxes[wBase+i])
		h := float64(boxes[hBase+i])
		p.survivors = append(p.survivors, Candidate{
			Box:   contract.BBox{X: cx - w/2, Y: cy - h/2, Width: w, Height: h},
			Score: float64(score),
		})
	}

	// Pass 2: slices.SortFunc, not sort.Slice. sort.Slice takes a closure over the slice, which
	// escapes to the heap on every call -- an allocation per frame, per camera, forever.
	slices.SortFunc(p.survivors, byScoreDesc)

	// Pass 3: greedy suppression.
	p.kept = p.kept[:0]
	for _, cand := range p.survivors {
		if len(p.kept) >= p.cfg.MaxDetections {
			break
		}
		if p.overlapsKept(cand.Box) {
			continue
		}
		p.kept = append(p.kept, cand)
	}

	// Pass 4: unletterbox, on the handful that survived.
	for _, cand := range p.kept {
		cand.Box = lb.ToOriginal(cand.Box)
		dst = append(dst, cand)
	}
	return dst, nil
}

// overlapsKept reports whether b is suppressed by something already accepted.
//
// Because the scan runs in descending score order, anything already in kept is stronger, so the
// first sufficient overlap is decisive and the rest need not be measured.
func (p *Postprocessor) overlapsKept(b contract.BBox) bool {
	for _, k := range p.kept {
		if iou(k.Box, b) > p.cfg.IoUThreshold {
			return true
		}
	}
	return false
}

// byScoreDesc is a package-level function rather than a literal at the call site so that no
// closure is created and nothing escapes.
func byScoreDesc(a, b Candidate) int {
	switch {
	case a.Score > b.Score:
		return -1
	case a.Score < b.Score:
		return 1
	default:
		return 0
	}
}

// iou is intersection over union of two boxes.
//
// Degenerate boxes -- zero or negative area, which a model can emit -- return 0 rather than a
// division result, so a collapsed box suppresses nothing instead of suppressing everything.
func iou(a, b contract.BBox) float64 {
	x1 := math.Max(a.X, b.X)
	y1 := math.Max(a.Y, b.Y)
	x2 := math.Min(a.X2(), b.X2())
	y2 := math.Min(a.Y2(), b.Y2())

	w := x2 - x1
	h := y2 - y1
	if !(w > 0) || !(h > 0) {
		return 0
	}
	inter := w * h
	union := a.Width*a.Height + b.Width*b.Height - inter
	if !(union > 0) {
		return 0
	}
	return inter / union
}

// Letterbox is the affine map between the original frame and the detector's square input.
//
// The detector sees a square; a camera does not produce one. The frame is scaled by one factor on
// both axes -- one factor, so the plate is not stretched -- and centred in the square with bars on
// two sides. Every coordinate the detector emits is in that square.
//
// This type is the inverse of that map and the only way back. A box that skips it has plausible
// numbers pointing at the wrong part of the picture, which is worse than no box at all: the crop
// is of nothing and the reading that comes back looks like any other weak read.
type Letterbox struct {
	// Scale is the single factor applied to both axes: min(inputSize/frameW, inputSize/frameH).
	Scale float64
	// PadX and PadY are the bars, on one side. The frame is centred, so the other side has the
	// same.
	PadX, PadY float64

	// FrameW and FrameH bound the result. Zero disables clamping.
	FrameW, FrameH int
}

// ToLetterbox maps a box from the frame into the detector's input.
//
// The service never needs this at run time -- frames go in as pixels, not boxes. It exists because
// it is the forward map, and a test that builds a synthetic tensor has to be able to write a box
// the way the detector would have emitted it. Deriving the inverse from the same two numbers is
// what keeps the two from drifting apart.
func (l Letterbox) ToLetterbox(b contract.BBox) contract.BBox {
	return contract.BBox{
		X:      b.X*l.Scale + l.PadX,
		Y:      b.Y*l.Scale + l.PadY,
		Width:  b.Width * l.Scale,
		Height: b.Height * l.Scale,
	}
}

// ToOriginal maps a box from the detector's input back to the frame.
//
// The result is clamped to the frame. A box the model pushed into the padding maps to a negative
// coordinate, and a negative offset into a pixel buffer is either a panic or somebody else's
// memory. Clamping does not weaken the gate's edge filter: a clamped box lands exactly on the
// border, and the filter rejects anything touching it.
func (l Letterbox) ToOriginal(b contract.BBox) contract.BBox {
	if l.Scale == 0 {
		return b
	}
	x1 := (b.X - l.PadX) / l.Scale
	y1 := (b.Y - l.PadY) / l.Scale
	x2 := (b.X2() - l.PadX) / l.Scale
	y2 := (b.Y2() - l.PadY) / l.Scale

	if l.FrameW > 0 && l.FrameH > 0 {
		w, h := float64(l.FrameW), float64(l.FrameH)
		x1 = clamp(x1, 0, w)
		x2 = clamp(x2, 0, w)
		y1 = clamp(y1, 0, h)
		y2 = clamp(y2, 0, h)
	}
	return contract.BBox{X: x1, Y: y1, Width: x2 - x1, Height: y2 - y1}
}

func clamp(v, lo, hi float64) float64 {
	if !(v > lo) { // also catches NaN, which must not become a crop offset
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
