// Package camera runs one camera.
//
// One goroutine per camera, and nothing per frame. Everything a frame touches -- the candidate
// slice, the active-detection list, the metric handles -- is owned by that goroutine and reused,
// so the steady state of a running fleet allocates nothing and the scheduler sees a fixed number
// of runnable goroutines rather than one per picture.
//
// The order of the eleven steps in step() is ported from the Python service's
// `_process_stream_loop`, which earned every one of them in production. Three are deliberate
// departures and each is marked where it lives:
//
//   - the detection budget is TIME, not a frame counter (see detectInterval)
//   - the loop never sleeps to pace itself (see serve)
//   - reconnect backoff carries jitter (see backoff)
package camera

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"time"

	"github.com/hyz-is/arandu-alpr"
	"github.com/hyz-is/arandu-alpr/contract"
	"github.com/hyz-is/arandu-alpr/engine"
	"github.com/hyz-is/arandu-alpr/metrics"
)

// Stream is a connected feed. It is the shape of streaming.Stream, restated here so this package
// does not depend on how frames are obtained: a test drives a channel, production drives ffmpeg,
// and the loop cannot tell the difference.
type Stream interface {
	// Frames yields decoded pictures, newest-wins. It closes when the feed ends.
	Frames() <-chan engine.Frame
	// Close stops the feed and releases whatever it holds. It must be safe to call twice.
	Close()
}

// Dialer opens a stream for a camera. The loop calls it again after every reconnect, so an
// implementation must not cache a dead connection.
type Dialer interface {
	Dial(ctx context.Context, spec contract.CameraSpec) (Stream, error)
}

// Adjust applies brightness and contrast to a frame's pixels, in place.
//
// It is a function rather than an interface because imaging.AdjustBrightnessContrast already has
// exactly this shape: an interface here would buy nothing and cost every caller a wrapper type.
//
// It takes the percentages rather than a prepared setting because the loop skips the call
// entirely at 100/100 -- an untouched frame must cost nothing, and that decision belongs next to
// the frame, not inside the adjustment.
type Adjust func(pix []byte, brightness, contrast int)

// Focus reports the Laplacian variance of a box's grayscale pixels, and is optional.
// *imaging.Scaler implements it.
//
// A nil Focus disables the blur gate rather than failing closed: a deployment without the imaging
// helper should read more plates than it should, never fewer, because a silently empty result set
// is the failure nobody notices.
type Focus interface {
	FocusVariance(f engine.Frame, box contract.BBox) float64
}

// Encoder produces the JPEGs a Detection may carry, and is optional. *imaging.Encoder implements
// it.
//
// The bytes it returns are owned by the caller and are never rewritten, which is what lets one
// encode serve both the sink and the live preview.
type Encoder interface {
	EncodeFrame(f engine.Frame) ([]byte, error)
	EncodeCrop(f engine.Frame, box contract.BBox) ([]byte, error)
}

// Config is everything one camera needs. Nothing here is optional except where it says so.
type Config struct {
	Spec   contract.CameraSpec
	Dialer Dialer
	Engine engine.Engine

	Gate  alpr.GateConfig
	Dedup *alpr.Deduper

	// Consensus, when set, makes the camera report a VEHICLE instead of a reading: the readings of
	// one plate across frames are followed as a track and decided together (alpr.Consensus), and
	// the Detection that reaches the Sink is the track's verdict, with the images of its best
	// frame. Nil keeps the older behaviour, one Detection per reading that passes the gate.
	//
	// It changes what the gate's MinOCRConfidence means: a single frame below it is still
	// evidence (ConsensusConfig.MinEvidence is its floor), and the bar is applied to the verdict.
	Consensus *alpr.ConsensusConfig

	Sink     contract.Sink     // optional
	Observer contract.Observer // optional
	Preview  contract.Preview  // optional
	Overlay  contract.Overlay  // optional; told what every detection pass found, while watched
	Stills   contract.Stills   // optional; told every frame arrives, and handed the pixels of the one it asks for
	Metrics  metrics.Registry  // optional; Discard when nil
	Logger   *slog.Logger      // optional; discards when nil

	Adjust  Adjust  // optional
	Focus   Focus   // optional
	Encoder Encoder // optional

	// IncludeFullImage and IncludeCropImage decide whether an accepted Detection carries its
	// pictures. They default to off because encoding is the most expensive thing that happens
	// after a plate is read, and an application that only wants the text should not pay for it.
	IncludeFullImage bool
	IncludeCropImage bool

	// Clock is the only source of time in the loop.
	//
	// Injectable because staleness, the detection budget and the dedup cooldown are all "has
	// enough time passed" questions. Against the wall clock every test of those becomes a
	// time.Sleep, which is slow when it passes and flaky when it does not.
	Clock func() time.Time

	// MaxIngestLag is how old a frame may be and still be worth a GPU pass.
	MaxIngestLag time.Duration
	// StaleFramesBeforeReconnect is how many consecutive late frames force a reconnect.
	StaleFramesBeforeReconnect int
	// DetectionTTL is how long an accepted box stays on the overlay.
	DetectionTTL time.Duration

	// DetectTimeout and RecognizeTimeout cap one inference call. Zero means no deadline.
	DetectTimeout    time.Duration
	RecognizeTimeout time.Duration

	// ActiveDetectInterval, when set, is the detection budget while a plate is being followed
	// (Consensus has a live track), in place of Spec.MinDetectInterval; nil keeps one budget.
	//
	// The two budgets want different things. An empty road needs one look in a quarter of a second
	// to notice a car, and every look is a GPU pass; a car crossing the picture is on screen for two
	// seconds and every frame of them is a reading that makes the verdict right. So the camera
	// idles cheaply and, the moment it sees a plate, reads every frame it is given until the plate
	// is gone. A camera that sees a vehicle one second in ten pays for one second in ten.
	//
	// Backpressure from the Sink still wins: a raised budget is not lowered for a track.
	ActiveDetectInterval *time.Duration

	// MaxDetectInterval caps how far backpressure may push the detection budget, and
	// BackpressureRelax is how long the raised budget survives without further complaint.
	MaxDetectInterval time.Duration
	BackpressureRelax time.Duration

	// MaintenanceInterval drives the periodic work.
	//
	// A ticker, never `frameCount % N`. With the counter the same code runs every 16 seconds on
	// a 2fps camera and every second on a 30fps one, so one number means two behaviours and the
	// fleet has both.
	MaintenanceInterval time.Duration
	// NoDetectionAfter is the silence that makes a camera worth mentioning. Zero disables it.
	NoDetectionAfter time.Duration

	// MaxReconnectBackoff caps the wait between reconnect attempts.
	//
	// There is no attempt limit, and that is the whole policy: a camera never gives up. It used
	// to stop after five failed dials, hand itself to a watchdog, and be marked OFFLINE -- and a
	// camera on a link that was down for an hour then needed something else to notice and bring
	// it back. A camera that simply keeps knocking, at a bounded interval, needs nobody.
	//
	// The backoff still grows, because a camera that is genuinely gone should not dial every
	// second forever; it grows to this ceiling and stays there.
	MaxReconnectBackoff time.Duration
}

// Defaults are the production values of the Python service this loop replaces.
const (
	DefaultMaxIngestLag               = 8 * time.Second
	DefaultStaleFramesBeforeReconnect = 30
	DefaultDetectionTTL               = 3 * time.Second
	DefaultDetectTimeout              = 5 * time.Second
	DefaultRecognizeTimeout           = 5 * time.Second
	DefaultMaxDetectInterval          = 2 * time.Second
	DefaultBackpressureRelax          = 30 * time.Second
	DefaultMaintenanceInterval        = 60 * time.Second
	DefaultMaxReconnectBackoff        = 30 * time.Second
)

// backpressureFloor is where the detection budget starts climbing when the camera is configured
// with no budget at all. Doubling zero is zero, so a camera set to "every frame" would ignore
// backpressure forever without it.
const backpressureFloor = 10 * time.Millisecond

func (c Config) withDefaults() Config {
	if c.Clock == nil {
		c.Clock = time.Now
	}
	if c.MaxIngestLag <= 0 {
		c.MaxIngestLag = DefaultMaxIngestLag
	}
	if c.StaleFramesBeforeReconnect <= 0 {
		c.StaleFramesBeforeReconnect = DefaultStaleFramesBeforeReconnect
	}
	if c.DetectionTTL <= 0 {
		c.DetectionTTL = DefaultDetectionTTL
	}
	if c.DetectTimeout < 0 {
		c.DetectTimeout = DefaultDetectTimeout
	}
	if c.RecognizeTimeout < 0 {
		c.RecognizeTimeout = DefaultRecognizeTimeout
	}
	if c.MaxDetectInterval <= 0 {
		c.MaxDetectInterval = DefaultMaxDetectInterval
	}
	if c.BackpressureRelax <= 0 {
		c.BackpressureRelax = DefaultBackpressureRelax
	}
	if c.MaintenanceInterval <= 0 {
		c.MaintenanceInterval = DefaultMaintenanceInterval
	}
	if c.MaxReconnectBackoff <= 0 {
		c.MaxReconnectBackoff = DefaultMaxReconnectBackoff
	}
	if c.Metrics == nil {
		c.Metrics = metrics.Discard()
	}
	if c.Logger == nil {
		c.Logger = slog.New(slog.DiscardHandler)
	}
	return c
}

// Camera is one camera's loop. It is owned by the goroutine that calls Run and is not safe for
// concurrent use -- that is the point of it, not a limitation.
type Camera struct {
	cfg Config

	// Reused across frames. Their capacity is the whole per-frame allocation budget.
	candidates []engine.Candidate
	// roiPix holds the ROI crop of the current frame (roi.go).
	roiPix []byte
	active []activeDetection
	boxes  []contract.BBox
	// items is what the current pass found, for the overlay.
	items []contract.PassItem

	// consensus follows this camera's plates when Config.Consensus is set; kept holds, per track,
	// a copy of the frame its best reading came from, because the borrowed frame is gone by the
	// time a track decides. spare is what a finished track gives back, so a camera that has
	// reached its working set copies frames without allocating.
	consensus *alpr.Consensus
	kept      map[uint64]*keptFrame
	spare     [][]byte

	staleRun     int
	lastDetectAt time.Time
	interval     time.Duration
	relaxAt      time.Time

	lastDetectionAt time.Time
	noDetectionSent bool

	m handles
}

// activeDetection is a box still worth drawing on the overlay.
type activeDetection struct {
	box contract.BBox
	at  time.Time
}

// maxActiveDetections bounds the overlay list. A camera pointed at a car park can hold plates in
// frame indefinitely, and an unbounded list is a leak that only shows up on the busiest site.
const maxActiveDetections = 32

// handles are the metric series this camera writes to, bound once.
//
// Binding here rather than per frame is the difference between a pointer dereference and a
// variadic call that escapes to the heap. Across a fleet at a few thousand frames a second that
// is the metrics layer costing nothing versus costing more than the gate it measures.
type handles struct {
	framesReceived metrics.Counter
	dropStale      metrics.Counter
	dropSkip       metrics.Counter
	dropBusy       metrics.Counter

	detectorDuration metrics.Histogram
	ocrDuration      metrics.Histogram
	pipelineDuration metrics.Histogram

	detected     metrics.Counter
	deduplicated metrics.Counter
	reconnects   metrics.Counter

	errDetect metrics.Counter
	errOCR    metrics.Counter
	errSink   metrics.Counter
}

// stageSink counts a reading that was read correctly and then lost on the way out.
//
// It rides on alpr_inference_errors_total rather than on a metric of its own because metrics.go
// owns every name this service emits, and a name only this package knows is a number that no
// dashboard and no alert will ever look at. The stage label already answers "where was it lost",
// and "at the boundary" is an answer worth being able to give: without it, a sink failing quietly
// looks exactly like a camera that stopped seeing plates.
const stageSink = "sink"

// Label names. They are constants because a metric bound with "camera" here and read with
// "camera_id" elsewhere is a query that returns nothing and an alert that never fires.
const (
	LabelCamera = "camera"
	LabelReason = "reason"
	LabelStage  = "stage"
)

func bind(r metrics.Registry, id string) handles {
	dropped := r.Counter(metrics.FramesDropped, LabelCamera, LabelReason)
	inferenceErrors := r.Counter(metrics.InferenceErrors, LabelCamera, LabelStage)
	return handles{
		framesReceived: r.Counter(metrics.FramesReceived, LabelCamera).With(id),
		dropStale:      dropped.With(id, metrics.DropStale),
		dropSkip:       dropped.With(id, metrics.DropSkip),
		dropBusy:       dropped.With(id, metrics.DropInferenceBusy),

		detectorDuration: r.Histogram(metrics.DetectorDuration, metrics.DefaultBuckets(), LabelCamera).With(id),
		ocrDuration:      r.Histogram(metrics.OCRDuration, metrics.DefaultBuckets(), LabelCamera).With(id),
		pipelineDuration: r.Histogram(metrics.PipelineDuration, metrics.PipelineBuckets(), LabelCamera).With(id),

		detected:     r.Counter(metrics.DetectedPlates, LabelCamera).With(id),
		deduplicated: r.Counter(metrics.Deduplicated, LabelCamera).With(id),
		reconnects:   r.Counter(metrics.StreamReconnects, LabelCamera).With(id),

		errDetect: inferenceErrors.With(id, metrics.StageDetect),
		errOCR:    inferenceErrors.With(id, metrics.StageOCR),
		errSink:   inferenceErrors.With(id, stageSink),
	}
}

// New builds a camera. It does not touch the network: Run does.
func New(cfg Config) (*Camera, error) {
	if cfg.Spec.ID == "" {
		return nil, errors.New("camera: spec has no ID")
	}
	if cfg.Dialer == nil {
		return nil, fmt.Errorf("camera %s: no dialer", cfg.Spec.ID)
	}
	if cfg.Engine == nil {
		return nil, fmt.Errorf("camera %s: no engine", cfg.Spec.ID)
	}
	cfg = cfg.withDefaults()

	c := &Camera{
		cfg:      cfg,
		interval: cfg.Spec.MinDetectInterval,
		m:        bind(cfg.Metrics, cfg.Spec.ID),
	}
	if cfg.Consensus != nil {
		c.consensus = alpr.NewConsensus(*cfg.Consensus)
		c.kept = map[uint64]*keptFrame{}
	}
	return c, nil
}

func (c *Camera) now() time.Time { return c.cfg.Clock() }

// following reports whether a plate is being followed right now.
func (c *Camera) following() bool {
	return c.consensus != nil && c.consensus.Tracks() > 0
}

// errStreamEnded and errStaleStream are internal: they say why serve returned.
var (
	errStreamEnded = errors.New("camera: stream ended")
	errStaleStream = errors.New("camera: stream is delivering only stale frames")
)

// Run reads the camera until the context is cancelled. Nothing else ends it.
//
// A camera never gives up. It used to stop after a handful of failed dials, hand itself to a
// watchdog, and be written OFFLINE -- which meant a camera behind a link that was down for an hour
// needed something else to notice it and bring it back, and that something else was a chain of
// timers, a database column and a retry window. A camera that simply keeps knocking replaces all
// of it, and cannot be forgotten by a part of the chain that failed.
//
// What survives is the backoff: the interval grows to MaxReconnectBackoff and stays there, so a
// camera that is genuinely gone costs one dial every thirty seconds rather than one per second,
// and the one that comes back is picked up within that.
func (c *Camera) Run(ctx context.Context) error {
	c.lastDetectionAt = c.now()

	// One ticker for the camera's whole life, not one per connection. A flapping camera
	// reconnecting every ten seconds would otherwise reset a sixty-second ticker forever and
	// never run its periodic work at all -- and a flapping camera is precisely the one whose
	// periodic work is worth reading.
	maintenance := time.NewTicker(c.cfg.MaintenanceInterval)
	defer maintenance.Stop()

	attempt := 0
	for {
		if err := ctx.Err(); err != nil {
			return err
		}

		stream, err := c.cfg.Dialer.Dial(ctx, c.cfg.Spec)
		if err != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			attempt++
			// The reason goes to the log: "unavailable" alone sends an operator to the camera when
			// the engine refused the address, or ffprobe found no video. Dial errors never carry
			// the stream's credentials (source.TestDialNeverPutsCredentialsInAnError).
			c.cfg.Logger.Warn("camera could not be opened", LabelCamera, c.cfg.Spec.ID, "attempt", attempt, "error", err)
			c.emit(ctx, contract.EventStreamReconnect, "stream unavailable, retrying")
			if !c.wait(ctx, c.backoff(attempt)) {
				return ctx.Err()
			}
			continue
		}

		attempt = 0
		c.emit(ctx, contract.EventCameraUp, "stream open")

		err = c.serve(ctx, stream, maintenance.C)
		stream.Close()

		if ctxErr := ctx.Err(); ctxErr != nil {
			return ctxErr
		}

		c.m.reconnects.Inc()
		if errors.Is(err, errStaleStream) {
			// A deliberate reconnect after a persistent lag, so it does not spend the backoff:
			// the stream is answering, it is just answering about the past. Waiting here would
			// add the outage the reconnect is meant to end.
			c.emit(ctx, contract.EventCameraStale, "frames persistently late, reopening stream")
			continue
		}

		attempt++
		c.emit(ctx, contract.EventStreamReconnect, "stream lost, reconnecting")
		if !c.wait(ctx, c.backoff(attempt)) {
			return ctx.Err()
		}
	}
}

// serve is the loop over one connected stream.
//
// **It never sleeps to pace itself, and that is a departure from the Python service**, whose last
// statement is `await asyncio.sleep(max(0.001, 1/fps - elapsed))`. That sleep exists for two
// reasons, and this loop has neither:
//
//  1. OpenCV's read() drains a decoder buffer at the speed of the buffer, not of the camera, so
//     without a pace the consumer races through queued stale pictures. Here the producer keeps a
//     single slot and overwrites it, so a frame arriving is already the newest one there is.
//  2. asyncio needs an explicit yield or the background tasks -- uploads, database writes, the
//     socket -- never run. Go preempts, and those tasks are other goroutines anyway.
//
// What is left of that sleep here is pure added latency: it would hold the newest frame in the
// channel while the loop waits out a rate it does not control. Throughput is governed by
// MinDetectInterval, which throws work away instead of delaying it.
func (c *Camera) serve(ctx context.Context, s Stream, maintenance <-chan time.Time) error {
	frames := s.Frames()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()

		case <-maintenance:
			c.maintain(ctx)

		case f, ok := <-frames:
			if !ok {
				return errStreamEnded
			}
			// The frame's pixels are BORROWED from the capture's fixed ring, and this is the
			// one place they are given back. It happens on every path out of step -- dropped
			// as stale, dropped inside the detection interval, dropped for a busy pool, or
			// fully processed -- because the ring is three buffers deep and a single missed
			// release stops the camera three frames later with no error anywhere.
			decision := c.step(ctx, f)
			engine.ReleaseFrame(f)
			if decision == stepReconnect {
				return errStaleStream
			}
		}
	}
}

// maintain is the periodic work: everything that is about elapsed time rather than about a frame.
func (c *Camera) maintain(ctx context.Context) {
	if c.cfg.NoDetectionAfter <= 0 || c.noDetectionSent {
		return
	}
	if c.now().Sub(c.lastDetectionAt) < c.cfg.NoDetectionAfter {
		return
	}
	c.noDetectionSent = true
	c.emit(ctx, contract.EventCameraStale, "no plate read recently; the camera may be out of frame")
}

// emit hands the application a lifecycle event.
//
// The message never carries the stream URL. That URL holds the camera's password and these
// messages land in operator-facing logs and panels, where a credential is not a detail -- it is a
// credential in a screenshot.
func (c *Camera) emit(ctx context.Context, kind contract.EventKind, message string) {
	if c.cfg.Observer == nil {
		return
	}
	c.cfg.Observer.Observe(ctx, contract.Event{
		CameraID: c.cfg.Spec.ID,
		Kind:     kind,
		Message:  message,
		At:       c.now(),
	})
}
