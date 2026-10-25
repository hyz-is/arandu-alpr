package camera

import (
	"bytes"
	"context"
	"image/jpeg"
	"strings"
	"testing"
	"time"

	"github.com/hyz-is/arandu-alpr"
	"github.com/hyz-is/arandu-alpr/contract"
	"github.com/hyz-is/arandu-alpr/engine"
	"github.com/hyz-is/arandu-alpr/imaging"
)

// probs is the OCR head's answer for a text, with one slot misread when wrong >= 0: the wrong
// character leads there and the right one is the runner-up, which is how a soft frame fails.
func probs(text string, conf float32, wrong int, wrongCh byte) []float32 {
	a := len(alpr.Alphabet)
	out := make([]float32, alpr.MaxPlateSlots*a)
	for slot := 0; slot < alpr.MaxPlateSlots; slot++ {
		ch := byte(alpr.PadChar)
		if slot < len(text) {
			ch = text[slot]
		}
		rest := (1 - conf) / float32(a-1)
		for k := 0; k < a; k++ {
			out[slot*a+k] = rest
		}
		out[slot*a+strings.IndexByte(alpr.Alphabet, ch)] = conf
		if slot == wrong {
			out[slot*a+strings.IndexByte(alpr.Alphabet, ch)] = conf * 0.45
			out[slot*a+strings.IndexByte(alpr.Alphabet, wrongCh)] = conf * 0.55
		}
	}
	return out
}

func readingOf(p []float32) engine.Reading {
	r := alpr.DecodePlate(p)
	return engine.Reading{Plate: r.Plate, CharConfidences: r.CharConfidences, Confidence: r.Confidence(), Probabilities: p}
}

func consensusRig(t *testing.T) *testRig {
	t.Helper()
	cfg := alpr.DefaultConsensus()
	return newRig(t, func(c *Config) {
		c.Consensus = &cfg
		c.Focus = fakeFocus{variance: 300}
		c.Dedup = alpr.NewDeduper(time.Minute, 1)
	})
}

// drive reads one frame per entry, 125 ms apart, with the plate each reading says.
func drive(t *testing.T, r *testRig, readings ...engine.Reading) {
	t.Helper()
	r.engine.setCandidates([]engine.Candidate{
		{Box: contract.BBox{X: 800, Y: 400, Width: 300, Height: 90}, Score: 0.9},
	})
	for _, rd := range readings {
		r.engine.reading = rd
		r.cam.step(context.Background(), r.freshFrame())
		r.clock.Advance(125 * time.Millisecond)
	}
}

func TestConsensusReportsOneCarOnceWithTheRightPlate(t *testing.T) {
	// The reading that began this: one car, read frame after frame with a different slip each time.
	// Without the consensus the sink would have been called for each spelling.
	r := consensusRig(t)
	drive(t, r,
		readingOf(probs("ABC1D23", 0.85, 2, 'G')),
		readingOf(probs("ABC1D23", 0.85, 4, 'O')),
		readingOf(probs("ABC1D23", 0.85, 6, '8')),
		readingOf(probs("ABC1D23", 0.85, -1, 0)),
		readingOf(probs("ABC1D23", 0.85, 0, '8')),
		readingOf(probs("ABC1D23", 0.85, -1, 0)),
	)
	r.clock.Advance(5 * time.Second)
	r.cam.step(context.Background(), r.freshFrame()) // the clock the track is judged by is the frames'

	if got := r.sink.callCount(); got != 1 {
		t.Fatalf("sink calls = %d for one car, want 1", got)
	}
	if got := r.sink.lastDetection().Plate; got != "ABC1D23" {
		t.Errorf("plate %q, want ABC1D23", got)
	}
}

func TestConsensusHoldsAReadingThatNoOtherFrameBacks(t *testing.T) {
	// Without the consensus a single frame at 0.62 passes the gate and is delivered. With it, one
	// frame that nothing else supports is not a plate.
	r := consensusRig(t)
	drive(t, r, readingOf(probs("ABC1D23", 0.62, -1, 0)))
	r.clock.Advance(5 * time.Second)
	r.engine.setCandidates(nil)
	r.cam.step(context.Background(), r.freshFrame())

	if got := r.sink.callCount(); got != 0 {
		t.Errorf("sink calls = %d for one 0.62 frame, want 0", got)
	}
}

func TestConsensusDecidesEarlyAndNotAgain(t *testing.T) {
	r := consensusRig(t)
	rd := readingOf(probs("ABC1D23", 0.9, -1, 0))
	drive(t, r, rd, rd, rd, rd)
	if got := r.sink.callCount(); got != 1 {
		t.Fatalf("sink calls after four agreeing frames = %d, want 1: the car is still in view", got)
	}
	drive(t, r, rd, rd, rd, rd)
	r.clock.Advance(5 * time.Second)
	r.engine.setCandidates(nil)
	r.cam.step(context.Background(), r.freshFrame())
	if got := r.sink.callCount(); got != 1 {
		t.Errorf("sink calls = %d after the car stayed and left, want 1", got)
	}
}

func TestConsensusDeliversThePicturesOfTheBestFrame(t *testing.T) {
	// The frame a reading came from is borrowed and gone when the track decides, so the picture in
	// the Detection has to be a copy taken then.
	cfg := alpr.DefaultConsensus()
	enc, err := imaging.NewEncoder(80)
	if err != nil {
		t.Fatal(err)
	}
	r := newRig(t, func(c *Config) {
		c.Consensus = &cfg
		c.Encoder = enc
		c.IncludeCropImage = true
		c.Focus = fakeFocus{variance: 300}
	})
	r.engine.setCandidates([]engine.Candidate{
		{Box: contract.BBox{X: 800, Y: 400, Width: 300, Height: 90}, Score: 0.9},
	})
	rd := readingOf(probs("ABC1D23", 0.9, -1, 0))
	r.engine.reading = rd

	// A real frame, painted a colour no other test frame has, then reused and overwritten the way
	// the capture ring does it.
	pix := make([]byte, frameW*frameH*3)
	for i := range pix {
		pix[i] = 200
	}
	for i := 0; i < 4; i++ {
		f := engine.Frame{Pix: pix, Width: frameW, Height: frameH, CapturedAt: r.clock.Now()}
		r.cam.step(context.Background(), f)
		for j := range pix {
			pix[j] = 0 // the ring hands the buffer to the next frame
		}
		r.clock.Advance(125 * time.Millisecond)
	}
	if r.sink.callCount() != 1 {
		t.Fatalf("sink calls = %d", r.sink.callCount())
	}
	jpegBytes := r.sink.lastDetection().CropJPEG
	if len(jpegBytes) == 0 {
		t.Fatal("the verdict carries no crop")
	}
	// The ring zeroed the buffer after every frame. A crop that is the colour the first frame had
	// is a crop of the copy; a black one is a crop of the borrowed buffer after it was reused.
	img, err := jpeg.Decode(bytes.NewReader(jpegBytes))
	if err != nil {
		t.Fatalf("the crop is not a JPEG: %v", err)
	}
	if rr, _, _, _ := img.At(img.Bounds().Dx()/2, img.Bounds().Dy()/2).RGBA(); rr>>8 < 180 {
		t.Errorf("the crop's centre is %d, want about 200: the picture was not copied", rr>>8)
	}
	if len(r.cam.kept) != 0 {
		t.Errorf("%d frames still held after the verdict", len(r.cam.kept))
	}
}

func TestConsensusLetsGoOfFramesOfTracksThatNeverDecide(t *testing.T) {
	cfg := alpr.DefaultConsensus()
	enc, _ := imaging.NewEncoder(80)
	r := newRig(t, func(c *Config) {
		c.Consensus = &cfg
		c.Encoder = enc
		c.IncludeFullImage = true
		c.Focus = fakeFocus{variance: 300}
	})
	r.engine.setCandidates([]engine.Candidate{
		{Box: contract.BBox{X: 800, Y: 400, Width: 300, Height: 90}, Score: 0.9},
	})
	r.engine.reading = readingOf(probs("QZK4", 0.45, -1, 0)) // evidence, not a plate
	pix := make([]byte, 64)
	f := engine.Frame{Pix: pix, Width: frameW, Height: frameH, CapturedAt: r.clock.Now()}
	// The frame's size does not match its geometry; the fakes never read it, and keeping it must
	// not either.
	r.cam.step(context.Background(), f)
	if len(r.cam.kept) != 1 {
		t.Fatalf("kept = %d, want the track's best frame", len(r.cam.kept))
	}
	r.clock.Advance(10 * time.Second)
	r.engine.setCandidates(nil)
	r.cam.step(context.Background(), r.freshFrame())
	if len(r.cam.kept) != 0 {
		t.Errorf("kept = %d after the track ended, want 0", len(r.cam.kept))
	}
	if r.sink.callCount() != 0 {
		t.Errorf("a track below the bar reached the sink")
	}
}

func TestWithoutConsensusEveryReadingStillGoesThroughTheGate(t *testing.T) {
	// The default is what it was: no behaviour changes for a camera that does not ask.
	r := newRig(t, func(c *Config) { c.Focus = fakeFocus{variance: 300} })
	r.engine.setCandidates([]engine.Candidate{
		{Box: contract.BBox{X: 800, Y: 400, Width: 300, Height: 90}, Score: 0.9},
	})
	r.engine.reading = readingOf(probs("ABC1D23", 0.9, -1, 0))
	r.cam.step(context.Background(), r.freshFrame())
	if r.sink.callCount() != 1 {
		t.Errorf("sink calls = %d, want 1", r.sink.callCount())
	}
}

func TestWhileAPlateIsFollowedEveryFrameIsRead(t *testing.T) {
	// The camera idles at one look a quarter of a second, and the moment it has a plate to follow it
	// reads every frame it is given: the frames of a vehicle that crosses in two seconds are what
	// make its verdict right, and an empty road does not need them.
	cfg := alpr.DefaultConsensus()
	active := time.Duration(0)
	r := newRig(t, func(c *Config) {
		c.Consensus = &cfg
		c.ActiveDetectInterval = &active
		c.Spec.MinDetectInterval = 250 * time.Millisecond
		c.Focus = fakeFocus{variance: 300}
	})
	ctx := context.Background()
	step := func() { r.cam.step(ctx, r.freshFrame()); r.clock.Advance(50 * time.Millisecond) }

	// Nothing in view: one pass in five frames at 50 ms.
	r.engine.setCandidates(nil)
	for i := 0; i < 20; i++ {
		step()
	}
	if got := r.engine.detectCallCount(); got != 4 {
		t.Fatalf("idle detector calls over 20 frames = %d, want 4 (one per 250 ms)", got)
	}

	// A plate appears and is followed: from then on every frame is read.
	r.engine.setCandidates([]engine.Candidate{
		{Box: contract.BBox{X: 800, Y: 400, Width: 300, Height: 90}, Score: 0.9},
	})
	r.engine.reading = readingOf(probs("ABC1D23", 0.9, -1, 0))
	for r.engine.detectCallCount() < 5 { // until the pass that sees it
		step()
	}
	before := r.engine.detectCallCount()
	for i := 0; i < 10; i++ {
		step()
	}
	if got := r.engine.detectCallCount() - before; got != 10 {
		t.Errorf("detector calls over 10 frames of a followed plate = %d, want 10", got)
	}

	// And back to idling once the plate is gone and its track has ended.
	r.engine.setCandidates(nil)
	r.clock.Advance(5 * time.Second)
	step() // flushes the track
	before = r.engine.detectCallCount()
	for i := 0; i < 20; i++ {
		step()
	}
	if got := r.engine.detectCallCount() - before; got > 4 {
		t.Errorf("detector calls over 20 idle frames after the track ended = %d, want 4", got)
	}
}

func TestBackpressureStillWinsOverAnActiveTrack(t *testing.T) {
	cfg := alpr.DefaultConsensus()
	active := time.Duration(0)
	r := newRig(t, func(c *Config) {
		c.Consensus = &cfg
		c.ActiveDetectInterval = &active
		c.Spec.MinDetectInterval = 100 * time.Millisecond
		c.MaxDetectInterval = time.Second
		c.Focus = fakeFocus{variance: 300}
	})
	r.cam.consensus.Observe(alpr.Observation{
		At: r.clock.Now(), Box: alpr.BBox{X: 800, Y: 400, Width: 300, Height: 90}, DetectorScore: 0.9,
		Plate: "ABC1D23", Confidence: 0.9, Probabilities: probs("ABC1D23", 0.9, -1, 0),
	})
	if got := r.cam.detectInterval(r.clock.Now()); got != 0 {
		t.Fatalf("interval with a followed plate = %v, want 0", got)
	}
	r.cam.applyBackpressure(r.clock.Now())
	if got := r.cam.detectInterval(r.clock.Now()); got == 0 {
		t.Errorf("a track kept the budget at 0 while the sink asked for relief")
	}
}

func TestConsensusWorksWithAnEngineThatKeptOnlyTheText(t *testing.T) {
	r := consensusRig(t)
	rd := engine.Reading{Plate: "ABC1D23", CharConfidences: []float64{0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9}, Confidence: 0.9}
	drive(t, r, rd, rd, rd, rd)
	if got := r.sink.callCount(); got != 1 || r.sink.lastDetection().Plate != "ABC1D23" {
		t.Errorf("sink calls = %d, plate %q", got, r.sink.lastDetection().Plate)
	}
}

func TestAnyNumberOfPlatesHoldsAtMostAFewPictures(t *testing.T) {
	// A car park shows a dozen plates at once, and each of them is a frame of 11 MB at 1440p.
	cfg := alpr.DefaultConsensus()
	enc, _ := imaging.NewEncoder(80)
	r := newRig(t, func(c *Config) {
		c.Consensus = &cfg
		c.Encoder = enc
		c.IncludeCropImage = true
	})
	for i := 0; i < 12; i++ {
		f := engine.Frame{Pix: make([]byte, 64), Width: frameW, Height: frameH, CapturedAt: r.clock.Now()}
		r.cam.keep(uint64(i+1), f)
		r.clock.Advance(time.Millisecond)
	}
	if len(r.cam.kept) > maxKeptFrames {
		t.Errorf("%d pictures held for 12 undecided plates, the bound is %d", len(r.cam.kept), maxKeptFrames)
	}
}
