package engine

import (
	"context"
	"sync"
	"sync/atomic"
	"time"

	"github.com/hyz-is/arandu-alpr/contract"
)

// TB is the part of *testing.T that Fake needs.
//
// It is an interface rather than *testing.T because importing "testing" from a non-test file
// registers the testing flags in every binary that links this package -- including the service. A
// three-method interface costs nothing and *testing.T satisfies it as it is.
type TB interface {
	Helper()
	Errorf(format string, args ...any)
}

// FakeConfig describes the engine a test wants.
//
// The zero value is a usable engine that is not ready: nothing is detected, and Detect and
// Recognize return ErrNotReady until MarkReady is called. That default is deliberate -- a
// supervisor that never waits for readiness must fail against the fake, not in production.
type FakeConfig struct {
	// Detections is what every Detect returns, in ORIGINAL frame coordinates.
	//
	// The slice is shared, not copied, and Detect appends from it: with a caller-supplied dst of
	// sufficient capacity the whole call allocates nothing.
	Detections []Candidate

	// Readings maps a box to what the OCR "read" from it. contract.BBox is comparable, and the
	// camera loop hands Recognize exactly the box Detect produced, so equality is the right key.
	Readings map[contract.BBox]Reading

	// DefaultReading answers for a box that is not in Readings.
	DefaultReading Reading

	// ForbiddenBoxes are the boxes the gate was supposed to discard.
	//
	// Recognize being called with one of them is the failure this fake exists to catch: it means
	// the pipeline ran the expensive pass before the cheap checks, which on a GPU is the whole
	// difference between a fleet that fits and one that does not. It is reported through TB and
	// recorded in Calls, so a caller without a TB can still assert on it.
	ForbiddenBoxes []contract.BBox

	// TB, when set, receives the fake's own findings: a Recognize on a forbidden box, or a frame
	// whose pixel buffer does not match its dimensions.
	TB TB

	// DetectDelay and RecognizeDelay are synthetic latency. They are how saturation is tested:
	// without them every call returns instantly and no concurrency limit is ever reached.
	DetectDelay    time.Duration
	RecognizeDelay time.Duration

	// MaxConcurrent is how many calls may be in flight at once, across Detect and Recognize
	// together -- sessions are shared between the two, so counting them separately would model a
	// machine that does not exist. Above it, calls return ErrBusy. Zero means unlimited.
	MaxConcurrent int

	// DetectErr and RecognizeErr inject failures. Each is consulted with the 1-based call number
	// and returns the error to fail with, or nil to proceed. See FailAlways and FailAt.
	//
	// A function rather than a value so a test can fail the third call of ten, which is what the
	// inference-error path actually looks like: transient, not permanent.
	DetectErr    func(call int) error
	RecognizeErr func(call int, box contract.BBox) error

	// ReadyAtStart skips the cold start. Leave it false to exercise the warmup path.
	ReadyAtStart bool

	// Provider names the execution provider the fake reports. Empty means CPU, which is the
	// honest answer for a fake.
	Provider string

	// Model is what ModelInfo returns.
	Model contract.ModelInfo

	// RecordedBoxLimit bounds how many recognised boxes are kept for inspection. Zero means the
	// default; negative means none.
	//
	// It is bounded rather than unbounded so that recording never grows a slice on the hot path.
	// The call counters stay exact regardless -- only the sample of boxes is capped.
	RecordedBoxLimit int
}

// defaultRecordedBoxes is large enough for any assertion a test makes and small enough to
// pre-allocate without thinking about it.
const defaultRecordedBoxes = 1024

// FakeCalls is what the fake saw. It is a snapshot, owned by the caller.
type FakeCalls struct {
	Detect    int
	Recognize int

	// Busy and NotReady count the calls that were refused, which is what a backpressure test
	// asserts on.
	Busy     int
	NotReady int

	// Errors counts calls failed by an injected error.
	Errors int

	// Close counts Close calls; it must be exactly one in a clean shutdown.
	Close int

	// RecognizedBoxes is the boxes Recognize was called with, in order, up to RecordedBoxLimit.
	RecognizedBoxes []contract.BBox

	// ForbiddenRecognized is every call to Recognize on a box the gate should have dropped. A
	// non-empty slice here means the cheap-before-expensive ordering was violated.
	ForbiddenRecognized []contract.BBox
}

// Fake is a complete Engine with no GPU, no ONNX Runtime and no model file.
//
// It is what makes the rest of the service testable: the camera loop, the supervisor, the
// backpressure policy and the per-frame allocation budgets all run against it in an ordinary
// `go test` on a laptop. That is only worth anything if the fake is honest about the things that
// break in production -- cold start, saturation, transient errors, and the order in which the
// pipeline spends GPU time -- so it models all four.
//
// It is safe for concurrent use.
type Fake struct {
	cfg       FakeConfig
	forbidden map[contract.BBox]struct{}
	providers ProviderReport
	boxLimit  int

	ready  atomic.Bool
	closed atomic.Bool

	// inFlight is the concurrency gate. It is an atomic rather than a channel semaphore because
	// the refusal must be immediate: ErrBusy means "no session now", and a test that waits for
	// one is testing queueing, which is exactly the policy this service does not have.
	inFlight atomic.Int64

	mu    sync.Mutex
	calls FakeCalls
}

// Fake is the reference implementation of the whole interface; a method going missing must be a
// compile error here rather than a surprise at a call site.
var _ Engine = (*Fake)(nil)

// NewFake builds a Fake. It never fails, so a test can call it in a table entry.
func NewFake(cfg FakeConfig) *Fake {
	limit := cfg.RecordedBoxLimit
	switch {
	case limit == 0:
		limit = defaultRecordedBoxes
	case limit < 0:
		limit = 0
	}

	f := &Fake{
		cfg:       cfg,
		boxLimit:  limit,
		providers: fakeProviderReport(cfg),
	}
	if len(cfg.ForbiddenBoxes) > 0 {
		f.forbidden = make(map[contract.BBox]struct{}, len(cfg.ForbiddenBoxes))
		for _, b := range cfg.ForbiddenBoxes {
			f.forbidden[b] = struct{}{}
		}
	}
	f.calls.RecognizedBoxes = make([]contract.BBox, 0, limit)
	f.ready.Store(cfg.ReadyAtStart)
	return f
}

// MarkReady ends the synthetic cold start.
func (f *Fake) MarkReady() { f.ready.Store(true) }

// MarkNotReady puts the engine back into cold start, which is what a session rebuild looks like
// from outside.
func (f *Fake) MarkNotReady() { f.ready.Store(false) }

// InFlight is how many calls are executing right now. A saturation test uses it to know the fake
// is actually saturated before asserting that the next call is refused, rather than sleeping and
// hoping.
func (f *Fake) InFlight() int { return int(f.inFlight.Load()) }

// Ready reports whether inference can serve traffic.
func (f *Fake) Ready() error {
	if !f.ready.Load() {
		return ErrNotReady
	}
	return nil
}

// Detect returns the configured detections, appended to dst.
func (f *Fake) Detect(ctx context.Context, frame Frame, dst []Candidate) ([]Candidate, error) {
	f.checkFrame(frame)

	if err := f.enter(ctx); err != nil {
		return dst, err
	}
	defer f.leave()

	call := f.countDetect()
	if f.cfg.DetectErr != nil {
		if injected := f.cfg.DetectErr(call); injected != nil {
			f.countError()
			return dst, injected
		}
	}
	if err := f.stall(ctx, f.cfg.DetectDelay); err != nil {
		return dst, err
	}
	return append(dst, f.cfg.Detections...), nil
}

// Recognize returns the configured reading for a box.
func (f *Fake) Recognize(ctx context.Context, frame Frame, box contract.BBox) (Reading, error) {
	f.checkFrame(frame)

	if err := f.enter(ctx); err != nil {
		return Reading{}, err
	}
	defer f.leave()

	call := f.countRecognize(box)
	if f.cfg.RecognizeErr != nil {
		if injected := f.cfg.RecognizeErr(call, box); injected != nil {
			f.countError()
			return Reading{}, injected
		}
	}
	if err := f.stall(ctx, f.cfg.RecognizeDelay); err != nil {
		return Reading{}, err
	}
	if r, ok := f.cfg.Readings[box]; ok {
		return r, nil
	}
	return f.cfg.DefaultReading, nil
}

// Providers reports where the models ran. For a fake that is the CPU, and saying so is better
// than claiming a device that is not there.
func (f *Fake) Providers() ProviderReport { return f.providers }

// ModelInfo identifies the artefacts this engine would have loaded.
func (f *Fake) ModelInfo() contract.ModelInfo { return f.cfg.Model }

// Close releases nothing and says so, but still counts and still stops serving.
//
// A fake that kept answering after Close would hide the bug it exists to catch: a component that
// keeps using an engine past shutdown. Here that turns into ErrNotReady, which is the same answer
// a real engine gives once its sessions are gone.
func (f *Fake) Close(context.Context) error {
	f.closed.Store(true)
	f.ready.Store(false)
	f.mu.Lock()
	f.calls.Close++
	f.mu.Unlock()
	return nil
}

// Calls returns a snapshot of what the fake was asked to do.
//
// The slices are copies: an assertion that ran while a camera goroutine was still going would
// otherwise read a slice being appended to.
func (f *Fake) Calls() FakeCalls {
	f.mu.Lock()
	defer f.mu.Unlock()

	snapshot := f.calls
	snapshot.RecognizedBoxes = append([]contract.BBox(nil), f.calls.RecognizedBoxes...)
	snapshot.ForbiddenRecognized = append([]contract.BBox(nil), f.calls.ForbiddenRecognized...)
	return snapshot
}

// ResetCalls clears the counters, for a test that drives several phases through one engine.
func (f *Fake) ResetCalls() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls = FakeCalls{RecognizedBoxes: f.calls.RecognizedBoxes[:0]}
}

// enter applies the lifecycle and the concurrency limit, in that order. A nil error means the
// caller occupies a slot and must call leave.
//
// Readiness is checked before the limit because a cold engine is not busy, and reporting ErrBusy
// during warmup would send the camera loop into backpressure over a condition that resolves by
// itself.
//
// It returns an error rather than a release function because a returned closure captures the
// receiver and escapes to the heap -- one allocation per call, in the very fake the per-frame
// allocation budgets are measured against.
func (f *Fake) enter(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if !f.ready.Load() {
		f.mu.Lock()
		f.calls.NotReady++
		f.mu.Unlock()
		return ErrNotReady
	}

	if limit := f.cfg.MaxConcurrent; limit > 0 {
		if f.inFlight.Add(1) > int64(limit) {
			f.inFlight.Add(-1)
			f.mu.Lock()
			f.calls.Busy++
			f.mu.Unlock()
			return ErrBusy
		}
		return nil
	}
	f.inFlight.Add(1)
	return nil
}

func (f *Fake) leave() { f.inFlight.Add(-1) }

// stall is the synthetic latency, and it honours cancellation.
//
// A plain Sleep would make every saturation test unkillable and would let a camera whose context
// died keep occupying a slot -- the opposite of what the delay is there to model. The timer
// allocates, which is fine: a configured delay is never the hot path, and with no delay this
// returns before allocating anything.
func (f *Fake) stall(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return nil
	}
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (f *Fake) countDetect() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls.Detect++
	return f.calls.Detect
}

// countRecognize records the call and reports a box the gate should have dropped.
func (f *Fake) countRecognize(box contract.BBox) int {
	_, forbidden := f.forbidden[box]

	f.mu.Lock()
	f.calls.Recognize++
	call := f.calls.Recognize
	if len(f.calls.RecognizedBoxes) < f.boxLimit {
		f.calls.RecognizedBoxes = append(f.calls.RecognizedBoxes, box)
	}
	if forbidden {
		f.calls.ForbiddenRecognized = append(f.calls.ForbiddenRecognized, box)
	}
	f.mu.Unlock()

	if forbidden && f.cfg.TB != nil {
		f.cfg.TB.Helper()
		f.cfg.TB.Errorf("Recognize called for %+v, a box the gate should have discarded\n\n"+
			"The cheap checks -- score, edge, blur -- exist to run before the OCR, because the OCR "+
			"is a GPU pass. Reaching it means the pipeline is paying for every frame it should "+
			"have thrown away.", box)
	}
	return call
}

func (f *Fake) countError() {
	f.mu.Lock()
	f.calls.Errors++
	f.mu.Unlock()
}

// checkFrame reports a frame whose buffer does not match its dimensions.
//
// The pixel buffer is pooled and BGR24, so its length is fixed by the dimensions. A mismatch is a
// caller handing out a buffer from the wrong pool, and against a real engine it is a crop of
// somebody else's frame rather than an error.
func (f *Fake) checkFrame(frame Frame) {
	if f.cfg.TB == nil || frame.Width <= 0 || frame.Height <= 0 {
		return
	}
	if want := frame.Width * frame.Height * 3; len(frame.Pix) != want {
		f.cfg.TB.Helper()
		f.cfg.TB.Errorf("frame is %dx%d BGR24, which is %d bytes, but Pix holds %d",
			frame.Width, frame.Height, want, len(frame.Pix))
	}
}

// fakeProviderReport builds the report once, at construction, so Providers allocates nothing and
// two callers cannot see two different answers.
func fakeProviderReport(cfg FakeConfig) ProviderReport {
	provider := cfg.Provider
	if provider == "" {
		provider = "CPUExecutionProvider"
	}
	models := make([]ModelReport, 0, 2)
	for _, role := range [...]string{"detector", "ocr"} {
		sha := cfg.Model.Detector
		if role == "ocr" {
			sha = cfg.Model.OCR
		}
		models = append(models, ModelReport{
			Role:            role,
			Device:          cfg.Model.Device,
			SHA256:          sha,
			NodesByProvider: map[string]int{provider: 1},
		})
	}
	return ProviderReport{
		Models: models,
		// A fake runs on the CPU by definition. Claiming AllCUDA would make the one check that
		// catches a silent CPU fallback pass everywhere it is not true.
		AllCUDA:  false,
		CPUNodes: len(models),
	}
}

// FailAlways is the common injection: every call fails with err.
func FailAlways(err error) func(int) error {
	return func(int) error { return err }
}

// FailAt fails only the nth call (1-based), which is what a transient inference error looks like
// and the only shape that exercises recovery as well as the error path.
func FailAt(n int, err error) func(int) error {
	return func(call int) error {
		if call == n {
			return err
		}
		return nil
	}
}
