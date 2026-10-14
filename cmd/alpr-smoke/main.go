// Command alpr-smoke runs the whole camera pipeline against a real video file and prints what
// happened.
//
// It answers one question that `go test` does not: does the thing actually run end to end -- a
// real ffmpeg subprocess, real frames through the buffer ring, the real gate order, the real
// dedup -- on a machine with no GPU.
//
// The inference is engine.Fake, and that is the point rather than a limitation. Everything
// between the camera and the model is exercised for real; the model itself is the one part that
// needs CUDA, and it is isolated behind the engine interface precisely so the rest can be run
// and measured here.
//
//	go run ./cmd/alpr-smoke -input clip.mp4
//	ffmpeg -f lavfi -i testsrc=size=640x480:rate=10:duration=5 -pix_fmt yuv420p clip.mp4
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"syscall"
	"time"

	"github.com/hyz-is/arandu-alpr"
	"github.com/hyz-is/arandu-alpr/camera"
	"github.com/hyz-is/arandu-alpr/contract"
	"github.com/hyz-is/arandu-alpr/engine"
	"github.com/hyz-is/arandu-alpr/imaging"
	"github.com/hyz-is/arandu-alpr/metrics"
	"github.com/hyz-is/arandu-alpr/source"
	"github.com/hyz-is/arandu-alpr/streaming"
)

func main() {
	input := flag.String("input", "", "video file or RTSP URL to read")
	seconds := flag.Int("seconds", 10, "how long to run")
	detectEvery := flag.Duration("detect-every", 200*time.Millisecond, "minimum gap between detections")
	flag.Parse()

	if *input == "" {
		fmt.Fprintln(os.Stderr, "usage: alpr-smoke -input <file or rtsp url>")
		os.Exit(2)
	}

	if err := run(*input, *seconds, *detectEvery); err != nil {
		fmt.Fprintf(os.Stderr, "\nfailed: %v\n", err)
		os.Exit(1)
	}
}

// collector is the application half of the boundary: it does what the service refuses to do.
type collector struct {
	detections []contract.Detection
	events     []contract.Event
}

func (c *collector) Detect(_ context.Context, d contract.Detection) error {
	c.detections = append(c.detections, d)
	return nil
}

func (c *collector) Observe(_ context.Context, e contract.Event) { c.events = append(c.events, e) }

func run(input string, seconds int, detectEvery time.Duration) error {
	reg := metrics.NewSnapshotRegistry()
	sink := &collector{}

	// A plate in the middle of the frame, well inside the edge margin so the gate keeps it.
	// Fake returns it for every frame; the dedup is what decides how many reach the sink, and
	// watching that number is most of the point of this command.
	fake := engine.NewFake(engine.FakeConfig{
		Detections: []engine.Candidate{
			{Box: contract.BBox{X: 220, Y: 190, Width: 200, Height: 60}, Score: 0.91},
		},
		DefaultReading: engine.Reading{
			Plate:            "ABC1D23",
			CharConfidences:  []float64{.99, .98, .97, .99, .96, .98, .97},
			Confidence:       0.977,
			Region:           "Brazil",
			RegionConfidence: 0.88,
		},
		ReadyAtStart: true,
	})

	scaler, err := imaging.NewScaler(engine.DetectorInputSize)
	if err != nil {
		return err
	}
	encoder, err := imaging.NewEncoder(70)
	if err != nil {
		return err
	}

	dialer := source.NewFFmpegDialer(source.Config{
		DebugBufferOwnership: true,
		Runner:               realtimeRunner(input),
	})

	cam, err := camera.New(camera.Config{
		Spec: contract.CameraSpec{
			ID:                "smoke-1",
			URL:               input,
			Enabled:           true,
			MinDetectInterval: detectEvery,
			Brightness:        100,
			Contrast:          100,
		},
		Dialer:           dialerAdapter{dialer},
		Engine:           fake,
		Gate:             alpr.DefaultGate(),
		Dedup:            alpr.NewDeduper(30*time.Second, 1),
		Sink:             sink,
		Observer:         sink,
		Metrics:          reg,
		Adjust:           imaging.AdjustBrightnessContrast,
		Focus:            scaler,
		Encoder:          encoder,
		IncludeCropImage: true,
	})
	if err != nil {
		return err
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	ctx, timeout := context.WithTimeout(ctx, time.Duration(seconds)*time.Second)
	defer timeout()

	fmt.Printf("reading %s for %ds, detecting at most every %s\n\n", input, seconds, detectEvery)
	start := time.Now()
	runErr := cam.Run(ctx)
	elapsed := time.Since(start)

	report(reg, sink, fake, elapsed)

	// A run that ends because the clip ended or the deadline passed is a successful run.
	if runErr != nil && ctx.Err() == nil && !isEndOfStream(runErr) {
		return runErr
	}
	return nil
}

func isEndOfStream(err error) bool {
	return err != nil && (contains(err.Error(), "stream ended") || contains(err.Error(), "EOF"))
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

func report(reg *metrics.SnapshotRegistry, sink *collector, fake *engine.Fake, elapsed time.Duration) {
	counters := map[string]float64{}
	drops := map[string]float64{}
	var pipeline metrics.Sample

	for _, s := range reg.Snapshot() {
		switch s.Name {
		case metrics.FramesDropped:
			drops[s.Labels["reason"]] += s.Value
			counters[s.Name] += s.Value
		case metrics.PipelineDuration:
			pipeline = s
		default:
			counters[s.Name] += s.Value
		}
	}

	calls := fake.Calls()
	received := counters[metrics.FramesReceived]

	fmt.Printf("ran for %s\n\n", elapsed.Round(10*time.Millisecond))
	fmt.Println("frames")
	fmt.Printf("  received          %6.0f  (%.1f/s)\n", received, received/elapsed.Seconds())
	fmt.Printf("  dropped           %6.0f\n", counters[metrics.FramesDropped])
	for _, reason := range []string{metrics.DropStale, metrics.DropSkip, metrics.DropInferenceBusy, metrics.DropBlank} {
		if n := drops[reason]; n > 0 {
			fmt.Printf("    %-14s  %6.0f\n", reason, n)
		}
	}

	fmt.Println("\ninference")
	fmt.Printf("  detect calls      %6d\n", calls.Detect)
	fmt.Printf("  recognize calls   %6d\n", calls.Recognize)

	fmt.Println("\nresults")
	fmt.Printf("  plates to sink    %6d\n", len(sink.detections))
	fmt.Printf("  deduplicated      %6.0f\n", counters[metrics.Deduplicated])
	if pipeline.Count > 0 {
		fmt.Printf("  end-to-end p50    %6.1f ms\n", pipeline.Quantiles["p50"]*1000)
		fmt.Printf("  end-to-end p95    %6.1f ms\n", pipeline.Quantiles["p95"]*1000)
	}

	if len(sink.detections) > 0 {
		d := sink.detections[0]
		fmt.Printf("\nfirst detection\n")
		fmt.Printf("  plate             %s  (%.1f%% mean, region %s %.0f%%)\n",
			d.Plate, d.Confidence*100, d.Region, d.RegionConfidence*100)
		fmt.Printf("  box               %.0f,%.0f %.0fx%.0f\n", d.Box.X, d.Box.Y, d.Box.Width, d.Box.Height)
		fmt.Printf("  crop jpeg         %d bytes\n", len(d.CropJPEG))
		fmt.Printf("  captured -> read  %s\n", d.DetectedAt.Sub(d.CapturedAt).Round(time.Millisecond))
	}

	// The dedup number is the one worth explaining, because a reader who does not know the
	// cooldown will read "1 plate from 90 frames" as a failure rather than as the entire point.
	if received > 0 && len(sink.detections) > 0 {
		fmt.Printf("\n%.0f frames produced %d delivered plate(s): the deduper suppresses a repeat\n"+
			"of the same plate within its cooldown, so a car parked in view is one event, not one\n"+
			"per frame.\n", received, len(sink.detections))
	}
}

// realtimeRunner makes a video FILE behave like a camera.
//
// Without it the numbers this command prints are misleading rather than wrong. ffmpeg decodes a
// local file as fast as the CPU allows, so a six-second clip arrives as one burst of ninety
// frames in a fraction of a second and then ends. The loop then spends most of the wall clock in
// reconnect backoff, and the detection budget -- which is a rate, measured against the clock --
// has almost no clock to measure against. Detections come out looking capped by something other
// than the setting, because they are: by how little time the stream was actually open.
//
// `-re` paces the input at its native frame rate, and `-stream_loop -1` keeps it open, which is
// what a camera does. Neither flag is valid for a live source, so both are added only for a path
// that exists on disk.
func realtimeRunner(input string) streaming.Runner {
	if _, err := os.Stat(input); err != nil {
		return nil // a real URL: leave the command exactly as the capture builds it
	}
	return func(ctx context.Context, name string, args ...string) *exec.Cmd {
		if name != "ffmpeg" {
			return exec.CommandContext(ctx, name, args...)
		}
		out := make([]string, 0, len(args)+3)
		for _, a := range args {
			if a == "-i" {
				out = append(out, "-re", "-stream_loop", "-1")
			}
			out = append(out, a)
		}
		return exec.CommandContext(ctx, name, out...)
	}
}

// dialerAdapter satisfies camera.Dialer. The concrete return type of source.Dial is more useful
// to a caller than the interface, so the narrowing happens here rather than in the package.
type dialerAdapter struct{ inner *source.FFmpegDialer }

func (d dialerAdapter) Dial(ctx context.Context, spec contract.CameraSpec) (camera.Stream, error) {
	return d.inner.Dial(ctx, spec)
}
