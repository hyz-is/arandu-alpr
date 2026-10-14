package bench

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"sync"
	"time"

	"github.com/hyz-is/arandu-alpr"
	"github.com/hyz-is/arandu-alpr/camera"
	"github.com/hyz-is/arandu-alpr/contract"
	"github.com/hyz-is/arandu-alpr/engine"
	"github.com/hyz-is/arandu-alpr/metrics"
	"github.com/hyz-is/arandu-alpr/source"
	"github.com/hyz-is/arandu-alpr/streaming"
)

// The geometry every Tier A run uses.
//
// 640x480 rather than 1080p, and the reason is honesty rather than speed. A benchmark that
// spends most of its time in `cat` and `io.ReadFull` measures the pipe, not the pipeline. At
// this size a frame is 921 KB, which is enough for the blank check and the buffer ring to cost
// what they really cost, and small enough that the replay file fits in page cache.
const (
	frameWidth  = 640
	frameHeight = 480
	frameSize   = frameWidth * frameHeight * 3
)

// plateBox sits in the middle of the frame, well inside the edge margin, so DefaultGate keeps
// it. A box the gate rejects would make every OCR benchmark measure a branch that returns.
var plateBox = contract.BBox{X: 220, Y: 190, Width: 200, Height: 60}

// readyFake is the engine every Tier A run uses unless it is testing saturation.
//
// ReadyAtStart, because cold start is a different measurement and mixing it in is how a warmup
// artefact ends up quoted as a steady-state number.
func readyFake(cfg engine.FakeConfig) *engine.Fake {
	if cfg.Detections == nil {
		cfg.Detections = []engine.Candidate{{Box: plateBox, Score: 0.91}}
	}
	cfg.DefaultReading = engine.Reading{
		Plate:            "ABC1D23",
		CharConfidences:  []float64{.99, .98, .97, .99, .96, .98, .97},
		Confidence:       0.977,
		Region:           "Brazil",
		RegionConfidence: 0.88,
	}
	cfg.ReadyAtStart = true
	return engine.NewFake(cfg)
}

// rawClip writes a file of raw BGR24 frames for the fake ffmpeg to replay.
//
// A file rather than a generator, for the same reason streaming's own tests use one: the bytes
// are identical on every run, so two benchmark results differ because the code differed.
func rawClip(dir string, frames int) (string, error) {
	raw := make([]byte, frames*frameSize)
	for i := range raw {
		// Anything but a solid picture: streaming.isBlank samples every 997th byte and would
		// drop a uniform frame as signal loss before it ever reached the loop.
		raw[i] = byte(i % 251)
	}
	path := filepath.Join(dir, "clip.bgr24")
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		return "", err
	}
	return path, nil
}

// replayRunner stands in for ffprobe and ffmpeg: the probe answers with the geometry, and the
// capture replays the clip endlessly.
//
// Endlessly matters. A replay that ends makes the camera loop reconnect, and reconnect backoff
// is seconds -- so a throughput number taken over a clip that runs out is mostly a measurement
// of how long the loop spent waiting to try again.
//
// It is deliberately NOT paced. Tier A asks how fast the pipeline can go, so the producer runs
// flat out and the consumer is the thing being measured. Tier B, which asks how many real-time
// cameras fit, paces with `-re` and would be meaningless without it.
func replayRunner(clipPath string) streaming.Runner {
	probe := fmt.Sprintf(
		`{"streams":[{"width":%d,"height":%d,"r_frame_rate":"30/1","codec_name":"h264"}]}`,
		frameWidth, frameHeight)

	return func(ctx context.Context, name string, args ...string) *exec.Cmd {
		if name == "ffprobe" {
			return exec.CommandContext(ctx, "sh", "-c", "printf %s "+strconv.Quote(probe))
		}
		return exec.CommandContext(ctx, "sh", "-c",
			"while :; do cat "+strconv.Quote(clipPath)+" || exit 0; done")
	}
}

// dialerAdapter narrows source.FFmpegDialer to what the camera loop asks for. The concrete
// return type is more useful to a caller than the interface, so the narrowing happens here
// rather than in the package.
type dialerAdapter struct{ inner *source.FFmpegDialer }

func (d dialerAdapter) Dial(ctx context.Context, spec contract.CameraSpec) (camera.Stream, error) {
	return d.inner.Dial(ctx, spec)
}

// countingSink is the application half of the boundary, doing the least a Sink can do.
//
// It counts under a mutex rather than an atomic because it also has to be safe to read while
// cameras are writing, and a benchmark that races is a benchmark that is thrown away.
type countingSink struct {
	mu         sync.Mutex
	detections int
	lastPlate  string
}

func (s *countingSink) Detect(_ context.Context, d contract.Detection) error {
	s.mu.Lock()
	s.detections++
	s.lastPlate = d.Plate
	s.mu.Unlock()
	return nil
}

func (s *countingSink) count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.detections
}

// fleetConfig describes one Tier A run of N cameras against one shared engine.
type fleetConfig struct {
	cameras int
	engine  engine.Engine
	metrics metrics.Registry
	sink    contract.Sink

	// detectEvery is the per-camera detection budget. Zero means every frame, which is the
	// right setting for measuring the pipeline's ceiling and the wrong one for imitating
	// production.
	detectEvery time.Duration

	// clipFrames sizes the replay file.
	clipFrames int

	// timeouts, when true, puts a deadline on each inference call. It is a knob because the
	// deadline costs two allocations per frame (a context and a timer) and the budget test
	// exists to say so out loud.
	timeouts bool

	// disableDedup delivers every reading instead of one per cooldown.
	//
	// The saturation test needs it. With the deduper on, a fleet reading the same plate over
	// and over records the end-to-end duration once per camera, and a p95 over eight
	// observations cannot tell a bounded latency from a growing one -- which is the only
	// question that test is asking.
	disableDedup bool

	dir string
}

// dedupFor builds the deduper a run wants, or none.
//
// Thirty seconds and a distance of one are the production settings, so a run that keeps the
// deduper is measuring the pipeline an operator actually has.
func dedupFor(cfg fleetConfig) *alpr.Deduper {
	if cfg.disableDedup {
		return nil
	}
	return alpr.NewDeduper(30*time.Second, 1)
}

// fleet is a started set of cameras, and the handle that stops them.
type fleet struct {
	cancel  context.CancelFunc
	done    chan struct{}
	dialer  *source.FFmpegDialer
	started time.Time
}

// startFleet builds and runs cfg.cameras cameras against one dialer and one engine.
//
// One dialer for the fleet, not one each: the dialer holds the semaphore that bounds concurrent
// opens, and giving each camera its own would model a machine where that bound does not exist.
func startFleet(ctx context.Context, cfg fleetConfig) (*fleet, error) {
	if cfg.clipFrames <= 0 {
		cfg.clipFrames = 16
	}
	clip, err := rawClip(cfg.dir, cfg.clipFrames)
	if err != nil {
		return nil, err
	}

	// AllowPrivate: the bench replays a local clip, which the camera guard refuses by design.
	dialer := source.NewFFmpegDialer(source.Config{Runner: replayRunner(clip), AllowPrivate: true})

	runCtx, cancel := context.WithCancel(ctx)
	f := &fleet{cancel: cancel, done: make(chan struct{}), dialer: dialer}

	var wg sync.WaitGroup
	for i := 0; i < cfg.cameras; i++ {
		camCfg := camera.Config{
			Spec: contract.CameraSpec{
				ID:                fmt.Sprintf("bench-%03d", i),
				URL:               "file://" + clip,
				Enabled:           true,
				MinDetectInterval: cfg.detectEvery,
				Brightness:        100,
				Contrast:          100,
			},
			Dialer:  dialerAdapter{dialer},
			Engine:  cfg.engine,
			Gate:    alpr.DefaultGate(),
			Dedup:   dedupFor(cfg),
			Sink:    cfg.sink,
			Metrics: cfg.metrics,
			// No Focus: the blur gate needs a Scaler per camera, and its cost belongs to the
			// imaging benchmarks rather than to a measurement of the loop.
		}
		if cfg.timeouts {
			camCfg.DetectTimeout = 5 * time.Second
			camCfg.RecognizeTimeout = 5 * time.Second
		}

		cam, err := camera.New(camCfg)
		if err != nil {
			cancel()
			return nil, err
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			// A run that ends because the context ended is a run that finished. Everything
			// else would be a harness fault, and the fleet's caller has no way to see it, so
			// it is not silently swallowed: the tests assert on frame counts, which go to
			// zero the moment a camera stops early.
			_ = cam.Run(runCtx)
		}()
	}

	go func() {
		wg.Wait()
		close(f.done)
	}()

	f.started = time.Now()
	return f, nil
}

// stop cancels the fleet and waits for every camera to return.
func (f *fleet) stop() time.Duration {
	elapsed := time.Since(f.started)
	f.cancel()
	<-f.done
	return elapsed
}

// waitForFrames blocks until the registry has seen want frames, or the deadline passes.
//
// It polls rather than instruments the loop, because the loop must not grow a hook that exists
// only for a benchmark. The poll interval is coarse on purpose: Snapshot walks every series and
// allocates, and a tight poll would show up in the very allocation numbers being measured.
func waitForFrames(reg *metrics.SnapshotRegistry, want float64, timeout time.Duration) (float64, error) {
	deadline := time.Now().Add(timeout)
	for {
		got := totals(reg.Snapshot()).framesReceived
		if got >= want {
			return got, nil
		}
		if time.Now().After(deadline) {
			return got, fmt.Errorf("bench: only %.0f of %.0f frames after %s", got, want, timeout)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// fleetTotals is the handful of numbers a Tier A assertion is written against.
type fleetTotals struct {
	framesReceived float64
	framesDropped  float64
	dropsByReason  map[string]float64
	detected       float64
	deduplicated   float64
	inferenceErrs  float64

	pipeline metrics.Sample
	detector metrics.Sample
	ocr      metrics.Sample
}

// totals folds a snapshot into fleetTotals, summing across cameras.
func totals(samples []metrics.Sample) fleetTotals {
	t := fleetTotals{dropsByReason: map[string]float64{}}
	for _, s := range samples {
		switch s.Name {
		case metrics.FramesReceived:
			t.framesReceived += s.Value
		case metrics.FramesDropped:
			t.framesDropped += s.Value
			t.dropsByReason[s.Labels[camera.LabelReason]] += s.Value
		case metrics.DetectedPlates:
			t.detected += s.Value
		case metrics.Deduplicated:
			t.deduplicated += s.Value
		case metrics.InferenceErrors:
			t.inferenceErrs += s.Value
		case metrics.PipelineDuration:
			t.pipeline = mergeHistogram(t.pipeline, s)
		case metrics.DetectorDuration:
			t.detector = mergeHistogram(t.detector, s)
		case metrics.OCRDuration:
			t.ocr = mergeHistogram(t.ocr, s)
		}
	}
	return t
}

// mergeHistogram folds one camera's histogram into a running total.
//
// Counts and sums add exactly. Quantiles do not, so they are weighted by count -- which is an
// approximation, and is why every assertion on them here is a bound rather than an equality.
func mergeHistogram(into, s metrics.Sample) metrics.Sample {
	if s.Count == 0 {
		return into
	}
	if into.Count == 0 {
		out := s
		out.Quantiles = map[string]float64{
			"p50": s.Quantiles["p50"], "p95": s.Quantiles["p95"], "p99": s.Quantiles["p99"],
		}
		return out
	}
	total := float64(into.Count + s.Count)
	wInto, wS := float64(into.Count)/total, float64(s.Count)/total
	merged := metrics.Sample{
		Name:  into.Name,
		Count: into.Count + s.Count,
		Sum:   into.Sum + s.Sum,
		Quantiles: map[string]float64{
			"p50": into.Quantiles["p50"]*wInto + s.Quantiles["p50"]*wS,
			"p95": into.Quantiles["p95"]*wInto + s.Quantiles["p95"]*wS,
			"p99": into.Quantiles["p99"]*wInto + s.Quantiles["p99"]*wS,
		},
	}
	return merged
}

// ownedFrame builds a frame that owns its pixels, for the paths that measure the camera loop
// alone rather than the capture behind it.
func ownedFrame(at time.Time) engine.Frame {
	pix := make([]byte, frameSize)
	for i := range pix {
		pix[i] = byte(i % 251)
	}
	return engine.Frame{Pix: pix, Width: frameWidth, Height: frameHeight, CapturedAt: at}
}

// channelStream feeds a camera loop directly, with no subprocess and no ring.
//
// It exists so the allocation budget can be attributed. Driving the loop through ffmpeg
// measures the loop plus a pipe plus a decoder's worth of bookkeeping, and the number that
// comes out cannot be split back into its parts -- so a regression in the loop hides inside the
// noise of the capture.
type channelStream struct {
	frames chan engine.Frame
	closed chan struct{}
	once   sync.Once
}

func newChannelStream() *channelStream {
	return &channelStream{frames: make(chan engine.Frame), closed: make(chan struct{})}
}

func (s *channelStream) Frames() <-chan engine.Frame { return s.frames }

func (s *channelStream) Close() { s.once.Do(func() { close(s.closed) }) }

// send offers a frame and reports whether the loop took it.
func (s *channelStream) send(f engine.Frame) bool {
	select {
	case s.frames <- f:
		return true
	case <-s.closed:
		return false
	}
}

// finish ends the feed, which makes the camera's serve return.
func (s *channelStream) finish() { close(s.frames) }

// oneShotDialer hands out a prepared stream once and refuses to reconnect.
//
// Refusing is deliberate: a benchmark whose stream ends and silently reconnects is a benchmark
// measuring reconnect backoff, and the number looks like a slow pipeline.
type oneShotDialer struct {
	stream camera.Stream
	served bool
}

var errNoReconnect = errors.New("bench: the harness stream is not reconnectable")

func (d *oneShotDialer) Dial(context.Context, contract.CameraSpec) (camera.Stream, error) {
	if d.served {
		return nil, errNoReconnect
	}
	d.served = true
	return d.stream, nil
}
