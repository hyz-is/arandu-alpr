package engine

import (
	"errors"
	"math"
	"testing"

	"github.com/hyz-is/arandu-alpr/contract"
)

// newTensors builds the two output tensors of a detector with the given anchor count, filled the
// way a real one fills them: every anchor carries a plausible box, and only a few carry a score.
//
// The boxes are non-zero everywhere on purpose. A tensor that is zero except where the test writes
// would let a row-major reader produce an obviously empty box and look merely "wrong"; filling it
// makes the row-major answer plausible, which is the actual failure being guarded against.
func newTensors(anchors int) (boxes, scores []float32) {
	boxes = make([]float32, anchors*boxChannels)
	scores = make([]float32, anchors)
	for a := 0; a < anchors; a++ {
		boxes[chCX*anchors+a] = float32(a%600) + 20
		boxes[chCY*anchors+a] = float32((a*7)%600) + 20
		boxes[chW*anchors+a] = float32(a%40) + 10
		boxes[chH*anchors+a] = float32(a%20) + 5
	}
	return boxes, scores
}

// writeAnchor places one box, in corner form, at anchor a -- channel-major, the way the model
// writes it.
func writeAnchor(boxes, scores []float32, anchors, a int, b contract.BBox, score float32) {
	boxes[chCX*anchors+a] = float32(b.X + b.Width/2)
	boxes[chCY*anchors+a] = float32(b.Y + b.Height/2)
	boxes[chW*anchors+a] = float32(b.Width)
	boxes[chH*anchors+a] = float32(b.Height)
	scores[a] = score
}

func newTestPostprocessor(t *testing.T, mutate func(*PostprocessConfig)) *Postprocessor {
	t.Helper()
	cfg := DefaultPostprocess()
	if mutate != nil {
		mutate(&cfg)
	}
	p, err := NewPostprocessor(cfg)
	if err != nil {
		t.Fatalf("NewPostprocessor: %v", err)
	}
	return p
}

func closeEnough(a, b contract.BBox) bool {
	const eps = 1e-3
	return math.Abs(a.X-b.X) < eps && math.Abs(a.Y-b.Y) < eps &&
		math.Abs(a.Width-b.Width) < eps && math.Abs(a.Height-b.Height) < eps
}

// TestBoxTensorIsChannelMajor.
//
// The tensor is [1, 4, N]. Reading it as [1, N, 4] compiles, runs, and returns boxes that look
// entirely reasonable -- they just describe a different part of the picture, so the crop is of
// nothing and the OCR result reads like an ordinary weak plate. Nothing fails and nothing is
// logged.
//
// This is the test that would fail, so it also computes what the row-major reading would have
// given and asserts the decoder did not return that.
func TestBoxTensorIsChannelMajor(t *testing.T) {
	p := newTestPostprocessor(t, nil)
	anchors := p.Anchors()

	boxes, scores := newTensors(anchors)
	const target = 1234
	want := contract.BBox{X: 80, Y: 190, Width: 40, Height: 20}
	writeAnchor(boxes, scores, anchors, target, want, 0.9)

	got, err := p.Decode(boxes, scores, NewLetterbox(640, 640, 640), nil)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("got %d candidates, want 1", len(got))
	}
	if !closeEnough(got[0].Box, want) {
		t.Errorf("box = %+v, want %+v\n\n"+
			"The box tensor is channel-major: cx is boxes[0*N+i], cy is boxes[1*N+i], and so on. "+
			"Reading it as boxes[i*4+k] gives four numbers belonging to four different anchors.",
			got[0].Box, want)
	}

	// What a row-major reader would have produced. It is a perfectly plausible box, which is the
	// entire problem.
	rowMajor := contract.BBox{
		X:      float64(boxes[target*4+0]) - float64(boxes[target*4+2])/2,
		Y:      float64(boxes[target*4+1]) - float64(boxes[target*4+3])/2,
		Width:  float64(boxes[target*4+2]),
		Height: float64(boxes[target*4+3]),
	}
	if closeEnough(got[0].Box, rowMajor) {
		t.Fatalf("the synthetic tensor does not distinguish the two layouts; the test proves nothing")
	}
}

// TestAnchorCountIsDerivedFromInputSize.
//
// 8400 is (640/8)² + (640/16)² + (640/32)². Writing the literal instead is what lets a 512 or 960
// model be decoded against the 640 layout, which produces boxes rather than an error.
func TestAnchorCountIsDerivedFromInputSize(t *testing.T) {
	for _, c := range []struct {
		size int
		want int
	}{
		{640, 8400},  // 80² + 40² + 20²
		{512, 5376},  // 64² + 32² + 16²
		{960, 18900}, // 120² + 60² + 30²
		{320, 2100},
		{0, 0},
		{-640, 0},
		{100, 0}, // no stride-32 grid
		{641, 0},
	} {
		if got := AnchorCount(c.size); got != c.want {
			t.Errorf("AnchorCount(%d) = %d, want %d", c.size, got, c.want)
		}
	}

	if _, err := NewPostprocessor(PostprocessConfig{InputSize: 100, MaxDetections: 10}); err == nil {
		t.Error("NewPostprocessor accepted an input size with no anchor grid")
	}

	// A model swap is the real scenario: same code, a tensor of another size, and it must fail
	// loudly instead of decoding garbage.
	p := newTestPostprocessor(t, nil)
	wrong := AnchorCount(512)
	boxes, scores := newTensors(wrong)
	if _, err := p.Decode(boxes, scores, NewLetterbox(640, 640, 640), nil); !errors.Is(err, ErrTensorShape) {
		t.Errorf("Decode of a 512-sized tensor: err = %v, want ErrTensorShape", err)
	}

	// A truncated tensor is the other half: the right anchor count with the wrong channel count.
	boxes, scores = newTensors(p.Anchors())
	if _, err := p.Decode(boxes[:len(boxes)-1], scores, Letterbox{Scale: 1}, nil); !errors.Is(err, ErrTensorShape) {
		t.Errorf("Decode of a truncated box tensor: err = %v, want ErrTensorShape", err)
	}
}

func TestNMSSuppressesOverlapping(t *testing.T) {
	p := newTestPostprocessor(t, nil)
	anchors := p.Anchors()
	boxes, scores := newTensors(anchors)

	strong := contract.BBox{X: 100, Y: 100, Width: 100, Height: 40}
	// Shifted by 10px: IoU is well above 0.45, so this is the same plate seen by a neighbouring
	// anchor rather than a second plate.
	weak := contract.BBox{X: 110, Y: 105, Width: 100, Height: 40}
	writeAnchor(boxes, scores, anchors, 10, weak, 0.70)
	writeAnchor(boxes, scores, anchors, 11, strong, 0.95)

	got, err := p.Decode(boxes, scores, Letterbox{Scale: 1}, nil)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("got %d candidates, want 1: %+v", len(got), got)
	}
	if !closeEnough(got[0].Box, strong) {
		t.Errorf("kept %+v, want the higher-scoring %+v", got[0].Box, strong)
	}
	// The tensor is float32, so the score comes back with float32 precision widened; comparing it
	// for exact equality against a float64 literal would fail on a correct decoder.
	if math.Abs(got[0].Score-0.95) > 1e-6 {
		t.Errorf("score = %v, want 0.95", got[0].Score)
	}
}

func TestNMSKeepsDistinct(t *testing.T) {
	p := newTestPostprocessor(t, nil)
	anchors := p.Anchors()
	boxes, scores := newTensors(anchors)

	// Two plates on two cars: no overlap at all.
	left := contract.BBox{X: 20, Y: 100, Width: 90, Height: 30}
	right := contract.BBox{X: 400, Y: 300, Width: 90, Height: 30}
	writeAnchor(boxes, scores, anchors, 500, left, 0.60)
	writeAnchor(boxes, scores, anchors, 900, right, 0.80)

	got, err := p.Decode(boxes, scores, Letterbox{Scale: 1}, nil)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("got %d candidates, want 2: %+v", len(got), got)
	}
	// Descending score is part of the contract: the caller spends its OCR budget top-down.
	if !closeEnough(got[0].Box, right) || !closeEnough(got[1].Box, left) {
		t.Errorf("got %+v, want the right box first (higher score) then the left", got)
	}
}

// TestNMSRespectsMaxDetections.
//
// Every surviving box costs a crop and a full OCR pass. Without the cap, one frame of a degenerate
// model spends the GPU budget of a whole camera.
func TestNMSRespectsMaxDetections(t *testing.T) {
	const max = 10
	p := newTestPostprocessor(t, func(c *PostprocessConfig) { c.MaxDetections = max })
	anchors := p.Anchors()
	boxes, scores := newTensors(anchors)

	// 30 plates, none overlapping, with scores rising so the expected survivors are known.
	const planted = 30
	for i := 0; i < planted; i++ {
		b := contract.BBox{X: float64(i%6) * 100, Y: float64(i/6) * 100, Width: 60, Height: 25}
		writeAnchor(boxes, scores, anchors, 100+i*13, b, float32(0.30+0.02*float64(i)))
	}

	got, err := p.Decode(boxes, scores, Letterbox{Scale: 1}, nil)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if len(got) != max {
		t.Fatalf("got %d candidates, want %d", len(got), max)
	}
	// The cap must keep the strongest, not the first ones the scan happened to meet.
	weakestKept := got[len(got)-1].Score
	wantWeakest := 0.30 + 0.02*float64(planted-max)
	if math.Abs(weakestKept-wantWeakest) > 1e-6 {
		t.Errorf("weakest kept score = %v, want %v: the cap dropped strong boxes instead of weak ones",
			weakestKept, wantWeakest)
	}
}

// TestUnletterboxRunsAfterNMSAndGivesOriginalCoordinates.
//
// Two claims in one test, because they are the same design decision.
//
// The coordinates that leave the decoder are the frame's, not the letterbox's -- a box in
// letterbox space has plausible numbers pointing at the wrong part of the picture.
//
// And the transform runs after suppression, which is safe because IoU is a ratio of areas and the
// letterbox scales every area by the same factor: the ratio, and therefore every suppression
// decision, is identical either way. The proof is that the buffer NMS worked on still holds
// letterbox coordinates once Decode has returned frame coordinates.
func TestUnletterboxRunsAfterNMSAndGivesOriginalCoordinates(t *testing.T) {
	const frameW, frameH = 1920, 1080
	p := newTestPostprocessor(t, nil)
	anchors := p.Anchors()
	lb := NewLetterbox(frameW, frameH, p.Config().InputSize)

	// 640/1920 = 1/3, and the 1080-high frame becomes 360 tall, so the bars are 140 top and
	// bottom. Pinning them here means a change to the letterbox arithmetic fails a test rather
	// than moving every box in production.
	if math.Abs(lb.Scale-1.0/3) > 1e-9 || math.Abs(lb.PadX) > 1e-9 || math.Abs(lb.PadY-140) > 1e-9 {
		t.Fatalf("letterbox = %+v, want scale 1/3, padX 0, padY 140", lb)
	}

	want := contract.BBox{X: 600, Y: 450, Width: 300, Height: 90}
	overlapping := contract.BBox{X: 630, Y: 465, Width: 300, Height: 90}

	b, s := newTensors(anchors)
	writeAnchor(b, s, anchors, 42, lb.ToLetterbox(want), 0.91)
	writeAnchor(b, s, anchors, 43, lb.ToLetterbox(overlapping), 0.55)

	got, err := p.Decode(b, s, lb, nil)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("got %d candidates, want 1 (the overlapping pair is one plate): %+v", len(got), got)
	}
	if !closeEnough(got[0].Box, want) {
		t.Errorf("box = %+v, want the original-frame %+v", got[0].Box, want)
	}

	// The NMS buffer is untransformed: suppression happened in letterbox space.
	if len(p.kept) != 1 {
		t.Fatalf("kept %d boxes internally, want 1", len(p.kept))
	}
	if !closeEnough(p.kept[0].Box, lb.ToLetterbox(want)) {
		t.Errorf("the NMS buffer holds %+v, want letterbox coordinates %+v: the transform ran too early",
			p.kept[0].Box, lb.ToLetterbox(want))
	}
}

// TestUnletterboxClampsToTheFrame.
//
// A box the model pushed into the padding maps to a negative coordinate, and a negative offset
// into a pixel buffer is a panic or somebody else's memory.
func TestUnletterboxClampsToTheFrame(t *testing.T) {
	lb := NewLetterbox(1920, 1080, 640)
	got := lb.ToOriginal(contract.BBox{X: -50, Y: 0, Width: 100, Height: 100})
	if got.X < 0 || got.Y < 0 {
		t.Errorf("ToOriginal = %+v, want no negative coordinate", got)
	}
	if got.X2() > 1920 || got.Y2() > 1080 {
		t.Errorf("ToOriginal = %+v, want it inside 1920x1080", got)
	}
}

// TestLetterboxRoundTrips pins the forward and inverse maps against each other, which is what
// keeps the crop and the detector input describing the same rectangle.
func TestLetterboxRoundTrips(t *testing.T) {
	for _, dims := range [][2]int{{1920, 1080}, {640, 480}, {1080, 1920}, {640, 640}} {
		lb := NewLetterbox(dims[0], dims[1], 640)
		in := contract.BBox{X: 100, Y: 80, Width: 200, Height: 60}
		if out := lb.ToOriginal(lb.ToLetterbox(in)); !closeEnough(out, in) {
			t.Errorf("%dx%d: round trip gave %+v, want %+v", dims[0], dims[1], out, in)
		}
	}
}

// TestNMSAllocatesNothingInSteadyState.
//
// Detection runs on every frame of every camera. An allocation here is one per frame per camera,
// forever, and it is the kind that never shows up as a leak -- only as a GC that never rests.
func TestNMSAllocatesNothingInSteadyState(t *testing.T) {
	p := newTestPostprocessor(t, nil)
	anchors := p.Anchors()
	b, s := newTensors(anchors)

	// A busy but realistic frame: twenty anchors above the threshold, half of them duplicates of
	// each other, which is roughly what a lorry at a gate produces.
	for i := 0; i < 20; i++ {
		box := contract.BBox{X: float64(i%5) * 120, Y: float64(i/5) * 120, Width: 80, Height: 30}
		writeAnchor(b, s, anchors, 200+i*7, box, float32(0.3+0.03*float64(i)))
	}

	dst := make([]Candidate, 0, p.Config().MaxDetections)
	got := testing.AllocsPerRun(100, func() {
		out, err := p.Decode(b, s, NewLetterbox(1920, 1080, 640), dst[:0])
		if err != nil || len(out) == 0 {
			t.Fatalf("Decode: %v, %d candidates", err, len(out))
		}
	})
	if got != 0 {
		t.Errorf("Decode allocates %v times per frame, want 0\n\n"+
			"Every buffer the four passes need belongs to the Postprocessor, and dst is the "+
			"caller's. A non-zero here means one of them is being rebuilt per frame -- check that "+
			"the sort is slices.SortFunc and not sort.Slice, whose closure escapes.", got)
	}
}

// BenchmarkDecode is the per-frame cost of the four passes, and the number the allocation budget
// is argued from.
func BenchmarkDecode(b *testing.B) {
	cfg := DefaultPostprocess()
	p, err := NewPostprocessor(cfg)
	if err != nil {
		b.Fatal(err)
	}
	boxes, scores := newTensors(p.Anchors())
	for i := 0; i < 20; i++ {
		box := contract.BBox{X: float64(i%5) * 120, Y: float64(i/5) * 120, Width: 80, Height: 30}
		writeAnchor(boxes, scores, p.Anchors(), 200+i*7, box, float32(0.3+0.03*float64(i)))
	}
	lb := NewLetterbox(1920, 1080, cfg.InputSize)
	dst := make([]Candidate, 0, cfg.MaxDetections)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := p.Decode(boxes, scores, lb, dst[:0]); err != nil {
			b.Fatal(err)
		}
	}
}

// TestDecodeRejectsNaNScores.
//
// A degenerate model emits NaN, and `score < threshold` is false for NaN: the box would sail
// through every ordered comparison and reach the OCR as a crop of noise.
func TestDecodeRejectsNaNScores(t *testing.T) {
	p := newTestPostprocessor(t, nil)
	anchors := p.Anchors()
	b, s := newTensors(anchors)
	writeAnchor(b, s, anchors, 7, contract.BBox{X: 10, Y: 10, Width: 50, Height: 20},
		float32(math.NaN()))

	got, err := p.Decode(b, s, Letterbox{Scale: 1}, nil)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("a NaN score produced %d candidates, want 0", len(got))
	}
}

// TestDecodeAppendsToDst pins the AppendFormat shape: whatever the caller already had is still
// there afterwards.
func TestDecodeAppendsToDst(t *testing.T) {
	p := newTestPostprocessor(t, nil)
	anchors := p.Anchors()
	b, s := newTensors(anchors)
	writeAnchor(b, s, anchors, 3, contract.BBox{X: 10, Y: 10, Width: 50, Height: 20}, 0.8)

	dst := []Candidate{{Score: -1}}
	got, err := p.Decode(b, s, Letterbox{Scale: 1}, dst)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if len(got) != 2 || got[0].Score != -1 {
		t.Errorf("Decode did not append to dst: %+v", got)
	}
}

// NewLetterbox computes the transform for a frame going into a square input, the way the
// imaging scaler does in production; the tests build their expectations with it.
func NewLetterbox(frameW, frameH, inputSize int) Letterbox {
	if frameW <= 0 || frameH <= 0 || inputSize <= 0 {
		// An identity map is the only safe answer: it leaves coordinates as they came rather than
		// scaling them by a number derived from a zero.
		return Letterbox{Scale: 1, FrameW: frameW, FrameH: frameH}
	}
	side := float64(inputSize)
	scale := math.Min(side/float64(frameW), side/float64(frameH))
	return Letterbox{
		Scale:  scale,
		PadX:   (side - float64(frameW)*scale) / 2,
		PadY:   (side - float64(frameH)*scale) / 2,
		FrameW: frameW,
		FrameH: frameH,
	}
}
