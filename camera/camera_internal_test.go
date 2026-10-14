package camera

import (
	"context"
	"testing"
	"time"

	"github.com/hyz-is/arandu-alpr"
	"github.com/hyz-is/arandu-alpr/contract"
	"github.com/hyz-is/arandu-alpr/engine"
	"github.com/hyz-is/arandu-alpr/imaging"
	"github.com/hyz-is/arandu-alpr/metrics"
)

const (
	frameW = 1920
	frameH = 1080
)

// The optional helpers are declared as interfaces so this package does not depend on imaging, but
// the shapes were chosen to fit imaging exactly. These assertions are what keeps that true: a
// signature drifting apart would otherwise only be discovered by whoever wires the two together,
// as a compile error in a third package that changed nothing.
var (
	_ Focus   = (*imaging.Scaler)(nil)
	_ Encoder = (*imaging.Encoder)(nil)
	_ Adjust  = imaging.AdjustBrightnessContrast
)

// fakeFocus reports one sharpness for every crop. Real focus needs pixels; the gate only needs
// the number, which is exactly why alpr.AcceptFocus takes a variance rather than a frame.
type fakeFocus struct{ variance float64 }

func (f fakeFocus) FocusVariance(_ engine.Frame, _ contract.BBox) float64 { return f.variance }

// countingAdjust counts how often brightness/contrast was actually applied.
type countingAdjust struct{ calls int }

func (a *countingAdjust) adjust(_ []byte, _, _ int) { a.calls++ }

type testRig struct {
	cam    *Camera
	clock  *testClock
	engine *fakeEngine
	sink   *fakeSink
	reg    *metrics.SnapshotRegistry
	dialer *countingDialer
}

// newRig builds a camera wired to fakes, with the knobs a test usually wants already set.
func newRig(t *testing.T, mutate func(*Config)) *testRig {
	t.Helper()

	clock := newTestClock()
	eng := &fakeEngine{}
	sink := &fakeSink{}
	reg := metrics.NewSnapshotRegistry()
	dialer := newCountingDialer(64)

	cfg := Config{
		Spec: contract.CameraSpec{
			ID:         "cam-1",
			Revision:   1,
			Enabled:    true,
			FrameRate:  30,
			Brightness: 100,
			Contrast:   100,
		},
		Dialer:  dialer,
		Engine:  eng,
		Gate:    alpr.DefaultGate(),
		Sink:    sink,
		Metrics: reg,
		Clock:   clock.Now,
		// No deadline: context.WithTimeout allocates, and the allocation budget test needs the
		// detection path to be measurable on its own.
		DetectTimeout:    0,
		RecognizeTimeout: 0,
	}
	if mutate != nil {
		mutate(&cfg)
	}

	cam, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return &testRig{cam: cam, clock: clock, engine: eng, sink: sink, reg: reg, dialer: dialer}
}

// frame builds a frame captured at the given instant. The pixels are never read by the fakes;
// they exist so the buffer is the size a real one would be.
func frame(capturedAt time.Time) engine.Frame {
	return engine.Frame{
		Pix:        make([]byte, 64),
		Width:      frameW,
		Height:     frameH,
		CapturedAt: capturedAt,
	}
}

func (r *testRig) staleFrame() engine.Frame {
	return frame(r.clock.Now().Add(-20 * time.Second))
}

func (r *testRig) freshFrame() engine.Frame {
	return frame(r.clock.Now())
}

// dropCount reads one reason of alpr_frames_dropped_total.
func (r *testRig) dropCount(reason string) float64 {
	for _, s := range r.reg.Snapshot() {
		if s.Name == metrics.FramesDropped && s.Labels[LabelReason] == reason {
			return s.Value
		}
	}
	return 0
}

// errorCount reads one stage of alpr_inference_errors_total.
func (r *testRig) errorCount(stage string) float64 {
	for _, s := range r.reg.Snapshot() {
		if s.Name == metrics.InferenceErrors && s.Labels[LabelStage] == stage {
			return s.Value
		}
	}
	return 0
}

func (r *testRig) counter(name string) float64 {
	for _, s := range r.reg.Snapshot() {
		if s.Name == name {
			return s.Value
		}
	}
	return 0
}

// TestStaleFrameDropsButDoesNotReconnectBeforeThreshold.
//
// A late frame is always thrown away -- it cannot notify anyone in time. Reconnecting is a
// different decision, and it is expensive: reopening a stream costs seconds of blindness. Doing
// it on the first late frame is what turned a saturated GPU into a feedback loop in the Python
// service (late -> reconnect -> later), and the fleet's throughput collapsed while every camera
// reported healthy.
func TestStaleFrameDropsButDoesNotReconnectBeforeThreshold(t *testing.T) {
	r := newRig(t, func(c *Config) {
		c.MaxIngestLag = 8 * time.Second
		c.StaleFramesBeforeReconnect = 30
	})
	ctx := context.Background()

	for i := 1; i <= 29; i++ {
		if got := r.cam.step(ctx, r.staleFrame()); got != stepContinue {
			t.Fatalf("frame %d asked for a reconnect before the threshold", i)
		}
	}
	if got := r.dropCount(metrics.DropStale); got != 29 {
		t.Errorf("stale drops = %v, want 29", got)
	}

	if got := r.cam.step(ctx, r.staleFrame()); got != stepReconnect {
		t.Fatalf("frame 30 did not ask for a reconnect")
	}
	if got := r.dropCount(metrics.DropStale); got != 30 {
		t.Errorf("stale drops = %v, want 30 (the frame that triggers the reconnect is still a drop)", got)
	}
	if r.cam.staleRun != 0 {
		t.Errorf("staleRun = %d after the reconnect, want 0", r.cam.staleRun)
	}
}

// TestOneFreshFrameResetsStaleRun.
//
// **This is the real bug.** A run counter that only ever grows turns an occasional late frame --
// which every camera has -- into a permanent reconnect cycle: reach 30 once, and from then on
// every 30th frame reopens the stream whether or not anything is wrong.
//
// That is the shape of the incident where detections fell from 490 a minute to 12. The reset is
// one line and nothing at runtime looks different without it, which is why it has a test of its
// own rather than a comment.
func TestOneFreshFrameResetsStaleRun(t *testing.T) {
	r := newRig(t, func(c *Config) {
		c.MaxIngestLag = 8 * time.Second
		c.StaleFramesBeforeReconnect = 30
	})
	ctx := context.Background()

	reconnects := 0
	feed := func(f engine.Frame) {
		if r.cam.step(ctx, f) == stepReconnect {
			reconnects++
		}
	}

	for i := 0; i < 29; i++ {
		feed(r.staleFrame())
	}
	feed(r.freshFrame())
	for i := 0; i < 29; i++ {
		feed(r.staleFrame())
	}

	if reconnects != 0 {
		t.Fatalf("reconnects = %d, want 0: 58 late frames with a good one between them are two "+
			"runs of 29, not one run of 58", reconnects)
	}
	if r.cam.staleRun != 29 {
		t.Errorf("staleRun = %d, want 29", r.cam.staleRun)
	}
}

// TestOCRIsNotCalledForDetectionsTheGateRejects.
//
// The gate's order is the reason detection and reading are two calls instead of one. Every check
// before Recognize is arithmetic; Recognize is a GPU pass. The Python service ran fast_alpr, which
// does both at once, so a blurred plate at the edge of the frame cost the same as a perfect one.
func TestOCRIsNotCalledForDetectionsTheGateRejects(t *testing.T) {
	r := newRig(t, func(c *Config) {
		c.Focus = fakeFocus{variance: 0} // every crop is out of focus
	})
	ctx := context.Background()

	r.engine.setCandidates([]engine.Candidate{
		{Box: contract.BBox{X: 800, Y: 400, Width: 300, Height: 90}, Score: 0.05}, // below MinDetectionScore
		{Box: contract.BBox{X: 0, Y: 400, Width: 300, Height: 90}, Score: 0.9},    // touching the border
		{Box: contract.BBox{X: 800, Y: 400, Width: 300, Height: 90}, Score: 0.9},  // sharp enough? no
	})

	r.cam.step(ctx, r.freshFrame())

	if got := r.engine.detectCallCount(); got != 1 {
		t.Fatalf("detector calls = %d, want 1", got)
	}
	if got := r.engine.ocrCallCount(); got != 0 {
		t.Fatalf("OCR ran %d times for detections the gate rejected; the whole point of the "+
			"order is that it does not", got)
	}

	// And the same candidate does reach the OCR once it is sharp, so the test above is measuring
	// the gate and not a loop that never calls Recognize at all.
	sharp := newRig(t, func(c *Config) { c.Focus = fakeFocus{variance: 1e6} })
	sharp.engine.setCandidates([]engine.Candidate{
		{Box: contract.BBox{X: 800, Y: 400, Width: 300, Height: 90}, Score: 0.9},
	})
	sharp.engine.reading = engine.Reading{Plate: "ABC1D23", Confidence: 0.95}
	sharp.cam.step(ctx, sharp.freshFrame())
	if got := sharp.engine.ocrCallCount(); got != 1 {
		t.Fatalf("OCR calls for an accepted detection = %d, want 1", got)
	}
}

// TestSinkErrorDoesNotKillTheCamera.
//
// The sink is the application's boundary: a database, a queue, a webhook. When it has a bad
// minute the camera must keep reading, because the alternative is a working camera taken off the
// air by something that is not the camera's fault -- and it comes back with no frames of the
// minutes it missed.
func TestSinkErrorDoesNotKillTheCamera(t *testing.T) {
	r := newRig(t, func(c *Config) {
		c.Focus = fakeFocus{variance: 1e6}
	})
	ctx := context.Background()

	r.engine.setCandidates([]engine.Candidate{
		{Box: contract.BBox{X: 800, Y: 400, Width: 300, Height: 90}, Score: 0.9},
	})
	r.engine.reading = engine.Reading{Plate: "ABC1D23", Confidence: 0.95}
	r.sink.setErr(errSinkDown)

	for i := 0; i < 3; i++ {
		if got := r.cam.step(ctx, r.freshFrame()); got != stepContinue {
			t.Fatalf("frame %d: a sink error stopped the camera", i)
		}
		r.clock.Advance(time.Second)
	}

	if got := r.sink.callCount(); got != 3 {
		t.Errorf("sink calls = %d, want 3: every frame after a failure must still be offered", got)
	}
	if got := r.counter(metrics.DetectedPlates); got != 3 {
		t.Errorf("detected plates = %v, want 3: the reading happened whatever the sink did", got)
	}

	// Counted, not just logged. A sink failing quietly and a camera that stopped seeing plates
	// look identical from a dashboard, and they need opposite responses.
	if got := r.errorCount(stageSink); got != 3 {
		t.Errorf("sink errors counted = %v, want 3", got)
	}
}

// TestNoDetectionIsReportedOnceByMaintenance.
//
// A camera knocked out of alignment keeps streaming perfectly and reads nothing. Nothing else in
// the pipeline can tell that apart from a quiet car park, so it is the periodic pass that says so
// -- once, not every minute until someone mutes it.
func TestNoDetectionIsReportedOnceByMaintenance(t *testing.T) {
	events := &recordingObserver{}
	r := newRig(t, func(c *Config) {
		c.Observer = events
		c.NoDetectionAfter = 10 * time.Minute
	})
	ctx := context.Background()

	r.cam.lastDetectionAt = r.clock.Now()

	r.clock.Advance(9 * time.Minute)
	r.cam.maintain(ctx)
	if got := events.count(contract.EventCameraStale); got != 0 {
		t.Errorf("reported silence after 9 minutes of a 10 minute threshold (%d events)", got)
	}

	r.clock.Advance(2 * time.Minute)
	r.cam.maintain(ctx)
	r.cam.maintain(ctx)
	r.cam.maintain(ctx)
	if got := events.count(contract.EventCameraStale); got != 1 {
		t.Errorf("silence reported %d times, want 1", got)
	}
}

// TestBackpressureRaisesTheDetectInterval.
//
// contract.ErrBackpressure is the application saying "I cannot keep up". The only useful answer
// is to do less work, and the only place that can decide to do less work is here. Without it the
// pressure has nowhere to go but a queue, and a queue of frames is latency that never comes back.
func TestBackpressureRaisesTheDetectInterval(t *testing.T) {
	r := newRig(t, func(c *Config) {
		c.Spec.MinDetectInterval = 100 * time.Millisecond
		c.MaxDetectInterval = time.Second
		c.BackpressureRelax = 30 * time.Second
		c.Focus = fakeFocus{variance: 1e6}
	})
	ctx := context.Background()

	r.engine.setCandidates([]engine.Candidate{
		{Box: contract.BBox{X: 800, Y: 400, Width: 300, Height: 90}, Score: 0.9},
	})
	r.engine.reading = engine.Reading{Plate: "ABC1D23", Confidence: 0.95}
	r.sink.setErr(contract.ErrBackpressure)

	r.cam.step(ctx, r.freshFrame())
	if got := r.cam.interval; got != 200*time.Millisecond {
		t.Fatalf("detect interval = %s after backpressure, want 200ms", got)
	}

	// 150ms later the camera would have detected at the old budget. Under backpressure it drops.
	r.clock.Advance(150 * time.Millisecond)
	r.cam.step(ctx, r.freshFrame())
	if got := r.dropCount(metrics.DropSkip); got != 1 {
		t.Errorf("skip drops = %v, want 1: the raised budget must actually reject a frame", got)
	}
	if got := r.engine.detectCallCount(); got != 1 {
		t.Errorf("detector calls = %d, want 1", got)
	}

	// And it decays: backpressure is a pause, not a demotion. A camera that never returned to its
	// configured rate would quietly stay degraded long after the outage that caused it.
	r.clock.Advance(31 * time.Second)
	r.sink.setErr(nil)
	r.cam.step(ctx, r.freshFrame())
	if got := r.cam.interval; got != 100*time.Millisecond {
		t.Errorf("detect interval = %s after the relax window, want the configured 100ms", got)
	}
}

// TestCameraLoopDoesNotSleepForFrameRate.
//
// The Python loop ends with `await asyncio.sleep(max(0.001, 1/fps - elapsed))`. Ported literally,
// a 1fps camera would take a second per frame no matter how fast the frame was ready.
//
// That sleep is not throughput control, it is a workaround for two things Go does not have:
// OpenCV's read() draining a decoder buffer faster than the camera fills it, and asyncio needing
// an explicit yield for background work to run at all. Here the producer keeps one slot and
// overwrites it, so a sleep would only hold the newest frame in the channel while the loop waits.
//
// The camera is configured at 1fps and given 20 frames: a pacing loop needs ~20 seconds.
func TestCameraLoopDoesNotSleepForFrameRate(t *testing.T) {
	r := newRig(t, func(c *Config) {
		c.Spec.FrameRate = 1
		c.Spec.MinDetectInterval = 0
		c.Clock = time.Now // this test is about wall-clock time, so it uses the wall clock
	})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	done := make(chan error, 1)
	go func() { done <- r.cam.Run(ctx) }()

	stream := <-r.dialer.opened
	const frames = 20
	start := time.Now()
	for i := 0; i < frames; i++ {
		stream.ch <- frame(time.Now())
	}

	deadline := time.Now().Add(5 * time.Second)
	for r.engine.detectCallCount() < frames {
		if time.Now().After(deadline) {
			t.Fatalf("only %d of %d frames processed in 5s; the loop is pacing itself",
				r.engine.detectCallCount(), frames)
		}
		time.Sleep(time.Millisecond)
	}
	elapsed := time.Since(start)

	cancel()
	<-done

	// Twenty frames at 1fps would be ~20s with a pacer. Half a second is generous room for a
	// loaded CI machine and still three orders of magnitude away from sleeping per frame.
	if elapsed > 500*time.Millisecond {
		t.Errorf("20 frames took %s at FrameRate=1; the loop is sleeping for the frame rate", elapsed)
	}
}

// TestSteadyStateIterationAllocatesNothing.
//
// A fleet is thousands of frames a second. One allocation per frame is a garbage collector that
// runs during detection, and a GC pause lands on every camera at once -- so the cost of a
// forgotten `make` inside the loop is not measured in bytes, it is measured in the frames the
// whole worker missed while it ran.
func TestSteadyStateIterationAllocatesNothing(t *testing.T) {
	ctx := context.Background()

	t.Run("frame inside the detection budget", func(t *testing.T) {
		r := newRig(t, func(c *Config) {
			c.Spec.MinDetectInterval = time.Second
		})
		f := r.freshFrame()
		r.cam.step(ctx, f) // the first frame takes the budget

		if got := testing.AllocsPerRun(200, func() { r.cam.step(ctx, f) }); got != 0 {
			t.Errorf("dropping a frame inside the budget allocates %v times per frame", got)
		}
	})

	t.Run("frame that runs the detector and the gate", func(t *testing.T) {
		r := newRig(t, func(c *Config) {
			c.Spec.MinDetectInterval = 0
			c.Focus = fakeFocus{variance: 0}
		})
		r.engine.setCandidates([]engine.Candidate{
			{Box: contract.BBox{X: 800, Y: 400, Width: 300, Height: 90}, Score: 0.9},
		})
		f := r.freshFrame()

		if got := testing.AllocsPerRun(200, func() { r.cam.step(ctx, f) }); got != 0 {
			t.Errorf("a detection pass allocates %v times per frame; the candidate slice or the "+
				"active list is being rebuilt instead of reused", got)
		}
	})
}

// TestBrightnessIsNotAppliedAtOneHundred.
//
// 100/100 means untouched, and untouched has to mean "not called". A full pass over a 6MB frame
// to multiply every byte by one is the cheapest thing in the loop to get wrong and one of the
// most expensive to run.
func TestBrightnessIsNotAppliedAtOneHundred(t *testing.T) {
	adj := &countingAdjust{}
	r := newRig(t, func(c *Config) { c.Adjust = adj.adjust })
	r.cam.step(context.Background(), r.freshFrame())
	if adj.calls != 0 {
		t.Errorf("adjuster called %d times at 100/100", adj.calls)
	}

	adj2 := &countingAdjust{}
	r2 := newRig(t, func(c *Config) {
		c.Adjust = adj2.adjust
		c.Spec.Brightness = 120
	})
	r2.cam.step(context.Background(), r2.freshFrame())
	if adj2.calls != 1 {
		t.Errorf("adjuster called %d times at 120/100, want 1", adj2.calls)
	}
}

// TestRunReopensTheStreamAfterAStaleRun proves Run acts on what step decided: the reconnect is a
// real redial, and it is counted.
func TestRunReopensTheStreamAfterAStaleRun(t *testing.T) {
	r := newRig(t, func(c *Config) {
		c.StaleFramesBeforeReconnect = 30
		c.Clock = time.Now
	})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	done := make(chan error, 1)
	go func() { done <- r.cam.Run(ctx) }()

	stream := <-r.dialer.opened
	stale := frame(time.Now().Add(-20 * time.Second))
	for i := 0; i < 30; i++ {
		stream.ch <- stale
	}

	// The second dial can only happen after the thirtieth late frame, so receiving it is the
	// synchronisation: no sleep, no polling, no flake.
	select {
	case <-r.dialer.opened:
	case <-time.After(5 * time.Second):
		t.Fatal("the stream was never reopened after 30 consecutive late frames")
	}

	if got := r.dialer.dialCount(); got != 2 {
		t.Errorf("dials = %d, want 2 (the first open plus one reconnect)", got)
	}

	cancel()
	<-done
}

// TestBackoffIsJittered.
//
// Every camera's reconnect is triggered by the same event -- a switch reboot, a recorder
// restarting -- so an undithered `2^n` has two hundred cameras counting to the same second and
// reconnecting in the same millisecond. The Python service has no jitter; at its fleet size the
// recovery was its own second outage.
func TestBackoffIsJittered(t *testing.T) {
	r := newRig(t, nil)

	seen := map[time.Duration]int{}
	for i := 0; i < 200; i++ {
		d := r.cam.backoff(3)
		if d <= 0 || d > 8*time.Second {
			t.Fatalf("backoff(3) = %s, outside (0, 8s]", d)
		}
		seen[d]++
	}
	if len(seen) < 50 {
		t.Errorf("backoff(3) produced %d distinct values in 200 draws; the fleet would reconnect "+
			"in lockstep", len(seen))
	}

	if got := r.cam.backoff(40); got > r.cam.cfg.MaxReconnectBackoff {
		t.Errorf("backoff(40) = %s, above the %s ceiling", got, r.cam.cfg.MaxReconnectBackoff)
	}
}
