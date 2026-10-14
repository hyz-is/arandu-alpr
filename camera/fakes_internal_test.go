package camera

import (
	"context"
	"errors"
	"sync"
	"time"

	"github.com/hyz-is/arandu-alpr/contract"
	"github.com/hyz-is/arandu-alpr/engine"
)

// The fakes here are deliberately local to this package's tests.
//
// engine.Fake covers a whole engine; these cover exactly what one camera loop touches, and they
// count calls, which is what most of these tests actually assert. Sharing a general-purpose fake
// would mean asserting on someone else's counters and breaking this package when they are
// renamed.

// testClock is the injected clock. It moves only when a test moves it, so every assertion about
// staleness, the detection budget and backpressure is exact rather than "usually".
type testClock struct {
	mu  sync.Mutex
	now time.Time
}

func newTestClock() *testClock {
	return &testClock{now: time.Date(2026, 8, 26, 12, 0, 0, 0, time.UTC)}
}

func (c *testClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *testClock) Advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}

// fakeEngine answers with whatever the test set, and counts what was asked of it.
type fakeEngine struct {
	mu sync.Mutex

	candidates []engine.Candidate
	reading    engine.Reading
	detectErr  error
	ocrErr     error

	detectCalls int
	ocrCalls    int
}

func (f *fakeEngine) Detect(_ context.Context, _ engine.Frame, dst []engine.Candidate) ([]engine.Candidate, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.detectCalls++
	if f.detectErr != nil {
		return nil, f.detectErr
	}
	// Appending into dst is the contract: a caller reusing its slice must allocate nothing.
	return append(dst, f.candidates...), nil
}

func (f *fakeEngine) Recognize(_ context.Context, _ engine.Frame, _ contract.BBox) (engine.Reading, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.ocrCalls++
	if f.ocrErr != nil {
		return engine.Reading{}, f.ocrErr
	}
	return f.reading, nil
}

func (f *fakeEngine) Ready() error                     { return nil }
func (f *fakeEngine) Providers() engine.ProviderReport { return engine.ProviderReport{AllCUDA: true} }
func (f *fakeEngine) ModelInfo() contract.ModelInfo {
	return contract.ModelInfo{Detector: "det", OCR: "ocr"}
}
func (f *fakeEngine) Close(_ context.Context) error { return nil }
func (f *fakeEngine) ocrCallCount() int             { f.mu.Lock(); defer f.mu.Unlock(); return f.ocrCalls }
func (f *fakeEngine) detectCallCount() int          { f.mu.Lock(); defer f.mu.Unlock(); return f.detectCalls }
func (f *fakeEngine) setCandidates(c []engine.Candidate) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.candidates = c
}

// fakeSink records deliveries and can fail on demand.
type fakeSink struct {
	mu    sync.Mutex
	err   error
	calls int
	last  contract.Detection
}

func (s *fakeSink) Detect(_ context.Context, d contract.Detection) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.calls++
	s.last = d
	return s.err
}

func (s *fakeSink) callCount() int { s.mu.Lock(); defer s.mu.Unlock(); return s.calls }
func (s *fakeSink) setErr(e error) { s.mu.Lock(); defer s.mu.Unlock(); s.err = e }
func (s *fakeSink) lastDetection() contract.Detection {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.last
}

// fakeStream is a channel of frames the test drives.
type fakeStream struct {
	ch     chan engine.Frame
	closed chan struct{}
	once   sync.Once
}

func newFakeStream(buffer int) *fakeStream {
	return &fakeStream{ch: make(chan engine.Frame, buffer), closed: make(chan struct{})}
}

func (s *fakeStream) Frames() <-chan engine.Frame { return s.ch }
func (s *fakeStream) Close()                      { s.once.Do(func() { close(s.closed) }) }

// countingDialer hands out a fresh stream per dial and counts how often it was asked.
type countingDialer struct {
	mu      sync.Mutex
	buffer  int
	dials   int
	opened  chan *fakeStream
	failErr error
}

func newCountingDialer(buffer int) *countingDialer {
	return &countingDialer{buffer: buffer, opened: make(chan *fakeStream, 8)}
}

func (d *countingDialer) Dial(_ context.Context, _ contract.CameraSpec) (Stream, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.dials++
	if d.failErr != nil {
		return nil, d.failErr
	}
	s := newFakeStream(d.buffer)
	d.opened <- s
	return s, nil
}

func (d *countingDialer) dialCount() int { d.mu.Lock(); defer d.mu.Unlock(); return d.dials }

// errSinkDown stands in for whatever the application's downstream is having a bad minute about.
var errSinkDown = errors.New("sink: database unavailable")

// recordingObserver keeps the lifecycle events, by kind.
type recordingObserver struct {
	mu     sync.Mutex
	events []contract.Event
}

func (o *recordingObserver) Observe(_ context.Context, e contract.Event) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.events = append(o.events, e)
}

func (o *recordingObserver) count(kind contract.EventKind) int {
	o.mu.Lock()
	defer o.mu.Unlock()
	n := 0
	for _, e := range o.events {
		if e.Kind == kind {
			n++
		}
	}
	return n
}
