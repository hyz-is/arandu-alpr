package bench

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/hyz-is/arandu-alpr"
	"github.com/hyz-is/arandu-alpr/camera"
	"github.com/hyz-is/arandu-alpr/contract"
	"github.com/hyz-is/arandu-alpr/engine"
	"github.com/hyz-is/arandu-alpr/metrics"
)

// BenchmarkFleetThroughput is the camera loop's ceiling, with N cameras through the real
// capture path.
//
// The unit of work is one frame REACHING the loop, so ns/op is wall-clock time per frame across
// the whole fleet and halving it by doubling the cameras is the shape of a fleet that scales.
// The producer is unpaced, which is what makes this a ceiling rather than a capacity: a real
// camera delivers at its own rate and the answer to "how many fit" is Tier B's.
//
// Do NOT read -benchmem here as the allocation budget. This benchmark polls the registry to
// know when b.N frames have arrived, and Snapshot walks every series and allocates -- so the
// B/op reported is mostly the harness watching. The budget is BenchmarkCameraLoopSteadyState,
// which watches nothing.
//
//	go test -run '^$' -bench FleetThroughput -benchtime 20000x ./bench/
func BenchmarkFleetThroughput(b *testing.B) {
	for _, cameras := range []int{1, 4, 16} {
		b.Run(fmt.Sprintf("cameras=%d", cameras), func(b *testing.B) {
			reg := metrics.NewSnapshotRegistry()
			fake := readyFake(engine.FakeConfig{})
			sink := &countingSink{}

			f, err := startFleet(context.Background(), fleetConfig{
				cameras: cameras,
				engine:  fake,
				metrics: reg,
				sink:    sink,
				dir:     b.TempDir(),
			})
			if err != nil {
				b.Fatal(err)
			}

			// Let every camera open its stream before the clock starts. ffprobe and the ffmpeg
			// stand-in are two subprocesses each, and counting them as pipeline time would make
			// a sixteen-camera run look slower per frame than a one-camera run for a reason
			// that has nothing to do with the loop.
			warm := float64(cameras * 20)
			if _, err := waitForFrames(reg, warm, 30*time.Second); err != nil {
				f.stop()
				b.Fatal(err)
			}
			before := totals(reg.Snapshot())

			b.ResetTimer()
			start := time.Now()
			got, err := waitForFrames(reg, before.framesReceived+float64(b.N), 5*time.Minute)
			elapsed := time.Since(start)
			b.StopTimer()

			f.stop()
			if err != nil {
				b.Fatal(err)
			}

			after := totals(reg.Snapshot())
			calls := fake.Calls()
			frames := got - before.framesReceived
			seconds := elapsed.Seconds()

			b.ReportMetric(frames/seconds, "frames/s")
			b.ReportMetric(float64(calls.Detect)/seconds, "detect/s")
			b.ReportMetric(float64(calls.Recognize)/seconds, "ocr/s")
			b.ReportMetric((after.framesDropped-before.framesDropped)/frames, "dropped/frame")
		})
	}
}

// BenchmarkCameraLoopSteadyState is the per-frame allocation budget.
//
// It drives the loop through a plain channel rather than through ffmpeg, and that is the point:
// a frame arriving over a pipe carries a subprocess's worth of bookkeeping, and a regression in
// the loop's own allocations disappears inside it. Here every allocation reported belongs to
// step(), the gate, the fake and the metrics registry -- nothing else is running.
//
// Read the three cases together. The gated case is the one a real fleet spends its time in:
// most frames hold nothing worth an OCR pass, and the cheap checks are what keep it that way.
//
//	go test -run '^$' -bench CameraLoopSteadyState -benchmem ./bench/
func BenchmarkCameraLoopSteadyState(b *testing.B) {
	cases := []struct {
		name     string
		score    float64
		timeouts bool
	}{
		// The common frame: a weak box, discarded by arithmetic before any GPU work. 0.10 is
		// below DefaultGate's MinDetectionScore of 0.24.
		{name: "gate-rejects-before-ocr", score: 0.10},
		// A full pass: detect, gate, OCR, dedup.
		{name: "reading-accepted", score: 0.91},
		// The same pass with DetectTimeout and RecognizeTimeout set, which prices the context
		// and the timer each deadline costs.
		{name: "reading-accepted-with-deadlines", score: 0.91, timeouts: true},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			reg := metrics.NewSnapshotRegistry()
			fake := readyFake(engine.FakeConfig{
				Detections: []engine.Candidate{{Box: plateBox, Score: tc.score}},
			})
			stream := newChannelStream()

			cfg := camera.Config{
				Spec: contract.CameraSpec{
					ID:         "alloc",
					URL:        "chan://alloc",
					Enabled:    true,
					Brightness: 100,
					Contrast:   100,
					// Every frame, so the budget is the budget of a frame that is worked on
					// rather than of one the interval threw away.
					MinDetectInterval: 0,
				},
				Dialer:  &oneShotDialer{stream: stream},
				Engine:  fake,
				Gate:    alpr.DefaultGate(),
				Dedup:   alpr.NewDeduper(30*time.Second, 1),
				Metrics: reg,
				// One attempt: when the feed ends the loop must return, not sit in reconnect
				// backoff while the benchmark waits for it.
			}
			if tc.timeouts {
				cfg.DetectTimeout = 5 * time.Second
				cfg.RecognizeTimeout = 5 * time.Second
			}

			cam, err := camera.New(cfg)
			if err != nil {
				b.Fatal(err)
			}

			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			done := make(chan struct{})
			go func() {
				defer close(done)
				_ = cam.Run(ctx)
			}()

			frame := ownedFrame(time.Now())

			// Warm the loop's reusable slices. Their capacity IS the budget: the candidate
			// slice and the overlay list grow once and are appended into for the rest of the
			// camera's life, so measuring before they have settled prices a cost that is paid
			// once as if it were paid per frame.
			for i := 0; i < 128; i++ {
				frame.CapturedAt = time.Now()
				if !stream.send(frame) {
					b.Fatal("the loop stopped taking frames during warmup")
				}
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				frame.CapturedAt = time.Now()
				if !stream.send(frame) {
					b.Fatal("the loop stopped taking frames")
				}
			}
			// One more send returns only after the loop has finished the previous frame and
			// come back to its select, so every measured frame is accounted for before the
			// clock stops.
			frame.CapturedAt = time.Now()
			stream.send(frame)
			b.StopTimer()

			cancel()
			<-done

			if got := totals(reg.Snapshot()).framesReceived; got < float64(b.N) {
				b.Fatalf("the loop saw %.0f frames, fewer than the %d sent", got, b.N)
			}
		})
	}
}
