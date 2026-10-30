package camera

import (
	"bytes"
	"context"
	"image/jpeg"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/hyz-is/arandu-alpr"
	"github.com/hyz-is/arandu-alpr/contract"
	"github.com/hyz-is/arandu-alpr/engine"
	"github.com/hyz-is/arandu-alpr/imaging"
)

// platesByBox answers each box with the plate that sits there, which the one-reading fakeEngine
// cannot: several vehicles in view at once is the case this file is about.
type platesByBox struct {
	*fakeEngine
	plates map[float64]engine.Reading
}

func (e platesByBox) Recognize(_ context.Context, _ engine.Frame, box contract.BBox) (engine.Reading, error) {
	return e.plates[box.X*10000+box.Y], nil
}

// allDetections keeps every delivery; the fakeSink keeps only the last.
type allDetections struct {
	mu  sync.Mutex
	got []contract.Detection
}

func (s *allDetections) Detect(_ context.Context, d contract.Detection) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.got = append(s.got, d)
	return nil
}

func (s *allDetections) all() []contract.Detection {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]contract.Detection(nil), s.got...)
}

// crowd is six vehicles in view at once, far enough apart to be six tracks, one more than two
// rows of maxKeptFrames hold.
func crowd() (boxes []contract.BBox, plates map[float64]engine.Reading, texts []string) {
	texts = []string{"ABC1D23", "DEF2G34", "HIJ3K45", "LMN4P56", "QRS5T67", "UVW6X78"}
	plates = map[float64]engine.Reading{}
	i := 0
	for _, y := range []float64{100, 800} {
		for _, x := range []float64{100, 800, 1500} {
			b := contract.BBox{X: x, Y: y, Width: 100, Height: 30}
			boxes = append(boxes, b)
			plates[x*10000+y] = readingOf(probs(texts[i], 0.95, -1, 0))
			i++
		}
	}
	return boxes, plates, texts
}

// gatedEncoder holds every encode made by the camera's encoder goroutine until the test lets it
// go, which is how a test makes "the pictures are not ready yet" a fact and not a race. An encode
// made by the camera's own goroutine, for a verdict whose frame never left the loop, is not held.
type gatedEncoder struct {
	Encoder
	gate chan struct{}
}

func (g gatedEncoder) EncodeFrame(f engine.Frame) ([]byte, error) {
	g.hold()
	return g.Encoder.EncodeFrame(f)
}

func (g gatedEncoder) EncodeCrop(f engine.Frame, box contract.BBox) ([]byte, error) {
	g.hold()
	return g.Encoder.EncodeCrop(f, box)
}

func (g gatedEncoder) hold() {
	pcs := make([]uintptr, 16)
	frames := runtime.CallersFrames(pcs[:runtime.Callers(3, pcs)])
	for {
		fr, more := frames.Next()
		if strings.HasSuffix(fr.Function, "(*Camera).encode") {
			<-g.gate
			return
		}
		if !more {
			return
		}
	}
}

// startEncoder starts the rig's encoder goroutine for the test's length.
func startEncoder(t *testing.T, r *testRig) {
	t.Helper()
	stop := r.cam.startEncoder(context.Background())
	t.Cleanup(stop)
}

// settle steps empty frames until every verdict waiting for its pictures has been delivered.
func settle(t *testing.T, r *testRig) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for len(r.cam.parked) > 0 || len(r.cam.flight) > 0 {
		if time.Now().After(deadline) {
			t.Fatalf("%d verdicts and %d frames still with the encoder after 5s", len(r.cam.parked), len(r.cam.flight))
		}
		time.Sleep(time.Millisecond)
		r.cam.step(context.Background(), r.freshFrame())
	}
}

// crowdRig is a camera with a consensus and both pictures asked for, looking at six vehicles.
func crowdRig(t *testing.T, enc Encoder) (*testRig, *allDetections, []string) {
	t.Helper()
	cfg := alpr.DefaultConsensus()
	boxes, plates, texts := crowd()
	sink := &allDetections{}
	r := newRig(t, func(c *Config) {
		c.Consensus = &cfg
		c.Encoder = enc
		c.IncludeFullImage = true
		c.IncludeCropImage = true
		c.Focus = fakeFocus{variance: 300}
		c.Sink = sink
	})
	r.cam.cfg.Engine = platesByBox{fakeEngine: r.engine, plates: plates}
	var cands []engine.Candidate
	for _, b := range boxes {
		cands = append(cands, engine.Candidate{Box: b, Score: 0.9})
	}
	r.engine.setCandidates(cands)
	return r, sink, texts
}

// watchCrowd steps six frames of the crowd and then the clock past the gap, so every track is
// decided.
func watchCrowd(r *testRig) {
	pix := make([]byte, frameW*frameH*3)
	for i := range pix {
		pix[i] = 200
	}
	for i := 0; i < 6; i++ {
		r.cam.step(context.Background(), engine.Frame{Pix: pix, Width: frameW, Height: frameH, CapturedAt: r.clock.Now()})
		r.clock.Advance(125 * time.Millisecond)
	}
	r.clock.Advance(5 * time.Second)
	r.engine.setCandidates(nil)
	r.cam.step(context.Background(), r.freshFrame()) // the clock the tracks are judged by is the frames'
}

func TestEveryPlateOfACrowdReachesTheSinkWithItsPictures(t *testing.T) {
	// More vehicles undecided at once than maxKeptFrames hold raw. The one whose frame is let go
	// first still decides later, and went to the Sink with no picture at all: 275 of 1461 reads of
	// the busiest camera in an hour.
	enc, err := imaging.NewEncoder(80)
	if err != nil {
		t.Fatal(err)
	}
	r, sink, texts := crowdRig(t, enc)
	startEncoder(t, r)
	watchCrowd(r)
	settle(t, r)

	got := sink.all()
	if len(got) != len(texts) {
		t.Fatalf("sink calls = %d for %d vehicles, want one each", len(got), len(texts))
	}
	for _, d := range got {
		if len(d.FullJPEG) == 0 {
			t.Errorf("%s reached the sink with no full picture", d.Plate)
		}
		if len(d.CropJPEG) == 0 {
			t.Errorf("%s reached the sink with no crop", d.Plate)
			continue
		}
		img, err := jpeg.Decode(bytes.NewReader(d.CropJPEG))
		if err != nil {
			t.Errorf("%s: the crop is not a JPEG: %v", d.Plate, err)
			continue
		}
		// The crop is cut from the best reading's box, handed over or not.
		if w := img.Bounds().Dx(); w < int(d.Box.Width) || w > 2*int(d.Box.Width) {
			t.Errorf("%s: crop is %d px wide for a %v px plate", d.Plate, w, d.Box.Width)
		}
	}
	if n := r.errorCount(stageImages); n != 0 {
		t.Errorf("%v verdicts went out without pictures", n)
	}
	if len(r.cam.kept) != 0 || r.cam.raw != 0 {
		t.Errorf("%d frames (%d raw) still held after every verdict", len(r.cam.kept), r.cam.raw)
	}
}

func TestAVerdictDoesNotHoldTheLoopForItsPictures(t *testing.T) {
	// The encoder is held. Four of the six vehicles decide on the frames they have; the two whose
	// frames went to the encoder are decided too, and the loop goes on without them. Their
	// verdicts go out on the first frame after the encoder lets go.
	enc, _ := imaging.NewEncoder(80)
	gated := gatedEncoder{Encoder: enc, gate: make(chan struct{})}
	r, sink, texts := crowdRig(t, gated)
	startEncoder(t, r)

	watchCrowd(r) // would never return if a verdict waited for the encoder
	if got := len(sink.all()); got != len(texts)-2 {
		t.Fatalf("sink calls = %d with the encoder held, want the %d whose frames never left the loop", got, len(texts)-2)
	}
	if len(r.cam.parked) != 2 {
		t.Fatalf("%d verdicts parked, want 2", len(r.cam.parked))
	}

	close(gated.gate)
	settle(t, r)
	got := sink.all()
	if len(got) != len(texts) {
		t.Fatalf("sink calls = %d after the encoder finished, want %d", len(got), len(texts))
	}
	for _, d := range got {
		if len(d.FullJPEG) == 0 || len(d.CropJPEG) == 0 {
			t.Errorf("%s reached the sink with %d and %d bytes of pictures", d.Plate, len(d.FullJPEG), len(d.CropJPEG))
		}
	}
}

func TestAnEncoderThatIsBehindCostsPicturesAndNotTheLoop(t *testing.T) {
	// The encoder is held and more tracks are evicted than it can have in flight: the camera does
	// not wait, the oldest tracks lose their pictures, and each verdict that goes out without is
	// counted.
	enc, _ := imaging.NewEncoder(80)
	gated := gatedEncoder{Encoder: enc, gate: make(chan struct{})}
	cfg := alpr.DefaultConsensus()
	r := newRig(t, func(c *Config) {
		c.Consensus = &cfg
		c.Encoder = gated
		c.IncludeFullImage = true
	})
	startEncoder(t, r)
	defer close(gated.gate)

	pix := make([]byte, 320*240*3)
	box := contract.BBox{X: 100, Y: 100, Width: 100, Height: 30}
	tracks := maxKeptFrames + maxInFlight + 2
	for i := 0; i < tracks; i++ {
		r.cam.keep(uint64(i+1), engine.Frame{Pix: pix, Width: 320, Height: 240, CapturedAt: r.clock.Now()}, box)
		r.clock.Advance(time.Millisecond)
	}
	if r.cam.raw != maxKeptFrames {
		t.Errorf("%d raw frames held, want %d", r.cam.raw, maxKeptFrames)
	}
	if len(r.cam.flight) != maxInFlight {
		t.Errorf("%d frames with the encoder, want %d", len(r.cam.flight), maxInFlight)
	}
	lost := 0
	for i := 0; i < tracks; i++ {
		if r.cam.kept[uint64(i+1)] == nil {
			lost++
			r.cam.emitVerdict(context.Background(), alpr.Verdict{Track: uint64(i + 1), Plate: "ABC1D23"}, r.clock.Now())
		}
	}
	if lost != tracks-maxKeptFrames-maxInFlight {
		t.Errorf("%d tracks lost their pictures, want %d", lost, tracks-maxKeptFrames-maxInFlight)
	}
	if n := r.errorCount(stageImages); int(n) != lost {
		t.Errorf("%v verdicts counted without pictures, want %d", n, lost)
	}
}

func TestATrackTakesARawFrameBackOnlyWhenOneIsFree(t *testing.T) {
	// A track whose frame went to the encoder and then reads better takes a raw frame back only when
	// one is free: taking one from another track would send that one to the encoder, and a crowd
	// would encode a frame per reading. When it does take it, the pictures made from the old frame
	// must not travel with the new one.
	enc, _ := imaging.NewEncoder(80)
	cfg := alpr.DefaultConsensus()
	r := newRig(t, func(c *Config) {
		c.Consensus = &cfg
		c.Encoder = enc
		c.IncludeFullImage = true
	})
	startEncoder(t, r)
	pix := make([]byte, 320*240*3)
	box := contract.BBox{X: 100, Y: 100, Width: 100, Height: 30}
	frameAt := func() engine.Frame {
		return engine.Frame{Pix: pix, Width: 320, Height: 240, CapturedAt: r.clock.Now()}
	}
	for i := 0; i < maxKeptFrames+1; i++ {
		r.cam.keep(uint64(i+1), frameAt(), box)
		r.clock.Advance(time.Millisecond)
	}
	old := r.cam.kept[1]
	if len(old.pix) != 0 || old.job == nil {
		t.Fatalf("track 1 holds %d raw bytes and job %v, want its frame with the encoder", len(old.pix), old.job)
	}

	r.cam.keep(1, frameAt(), box)
	if k := r.cam.kept[1]; len(k.pix) != 0 || k.job == nil {
		t.Errorf("track 1 took a raw frame with none free")
	}
	if r.cam.raw != maxKeptFrames {
		t.Errorf("%d raw frames held, want %d", r.cam.raw, maxKeptFrames)
	}

	r.cam.drop(2)
	r.cam.keep(1, frameAt(), box)
	if k := r.cam.kept[1]; len(k.pix) == 0 || k.job != nil {
		t.Errorf("track 1 holds %d raw bytes and job %v, want the new raw frame alone", len(k.pix), k.job)
	}
	if r.cam.raw != maxKeptFrames {
		t.Errorf("%d raw frames held, want %d", r.cam.raw, maxKeptFrames)
	}
	// The buffer the encoder had is back with the spares once it is done, and counts nowhere.
	deadline := time.Now().Add(5 * time.Second)
	for len(r.cam.flight) > 0 {
		if time.Now().After(deadline) {
			t.Fatal("the encoder never finished")
		}
		time.Sleep(time.Millisecond)
		r.cam.reap()
	}
}

func TestTheEncodedPicturesOfUndecidedTracksAreBounded(t *testing.T) {
	enc, _ := imaging.NewEncoder(80)
	cfg := alpr.DefaultConsensus()
	r := newRig(t, func(c *Config) {
		c.Consensus = &cfg
		c.Encoder = enc
		c.IncludeFullImage = true
	})
	startEncoder(t, r)
	pix := make([]byte, 320*240*3)
	box := contract.BBox{X: 100, Y: 100, Width: 100, Height: 30}
	for i := 0; i < maxKeptFrames+maxEncodedFrames+10; i++ {
		r.cam.keep(uint64(i+1), engine.Frame{Pix: pix, Width: 320, Height: 240, CapturedAt: r.clock.Now()}, box)
		r.clock.Advance(time.Millisecond)
		// Let the encoder keep up, as a camera does between two tracks appearing.
		for deadline := time.Now().Add(5 * time.Second); len(r.cam.flight) > 0; {
			if time.Now().After(deadline) {
				t.Fatal("the encoder never finished")
			}
			time.Sleep(100 * time.Microsecond)
			r.cam.reap()
		}
	}
	if r.cam.raw != maxKeptFrames {
		t.Errorf("%d raw frames held, want %d", r.cam.raw, maxKeptFrames)
	}
	if got := len(r.cam.kept); got != maxKeptFrames+maxEncodedFrames {
		t.Errorf("%d tracks held, want the raw %d and the encoded %d", got, maxKeptFrames, maxEncodedFrames)
	}
	if r.cam.kept[1] != nil {
		t.Errorf("the oldest track still holds pictures past the bound")
	}
}

func TestTheEncoderGoroutineEndsWithTheCameraContext(t *testing.T) {
	enc, _ := imaging.NewEncoder(80)
	cfg := alpr.DefaultConsensus()
	r := newRig(t, func(c *Config) {
		c.Consensus = &cfg
		c.Encoder = enc
		c.IncludeFullImage = true
		c.Clock = time.Now
	})
	before := runtime.NumGoroutine()

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- r.cam.Run(ctx) }()
	<-r.dialer.opened
	// Running: the camera's own goroutine and the encoder's.
	deadline := time.Now().Add(5 * time.Second)
	for runtime.NumGoroutine() < before+2 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if got := runtime.NumGoroutine(); got < before+2 {
		t.Fatalf("%d goroutines while running, want at least %d: the encoder did not start", got, before+2)
	}
	cancel()
	<-done
	for runtime.NumGoroutine() > before && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if got := runtime.NumGoroutine(); got > before {
		t.Errorf("%d goroutines after Run returned, %d before it started", got, before)
	}
	if r.cam.encoding {
		t.Errorf("the camera still says its encoder runs")
	}
}

func TestAJobStillQueuedWhenTheCameraStopsIsFinishedAsFailed(t *testing.T) {
	// Nothing may wait on a frame the encoder will never reach.
	enc, _ := imaging.NewEncoder(80)
	gated := gatedEncoder{Encoder: enc, gate: make(chan struct{})}
	cfg := alpr.DefaultConsensus()
	r := newRig(t, func(c *Config) {
		c.Consensus = &cfg
		c.Encoder = gated
		c.IncludeFullImage = true
	})
	stop := r.cam.startEncoder(context.Background())
	pix := make([]byte, 320*240*3)
	box := contract.BBox{X: 100, Y: 100, Width: 100, Height: 30}
	for i := 0; i < maxKeptFrames+3; i++ {
		r.cam.keep(uint64(i+1), engine.Frame{Pix: pix, Width: 320, Height: 240, CapturedAt: r.clock.Now()}, box)
		r.clock.Advance(time.Millisecond)
	}
	go func() { time.Sleep(20 * time.Millisecond); close(gated.gate) }()
	stop()
	for _, j := range r.cam.flight {
		if !j.finished() {
			t.Errorf("a job is left that nothing will ever finish")
		}
	}
}
