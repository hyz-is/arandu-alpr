package supervisor

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/hyz-is/arandu-alpr/contract"
	"github.com/hyz-is/arandu-alpr/metrics"
)

// testClock moves only when a test moves it, so the retry schedule can be asserted exactly
// instead of waited out.
type testClock struct {
	mu  sync.Mutex
	now time.Time
}

func newTestClock() *testClock {
	return &testClock{now: time.Date(2026, 8, 26, 12, 0, 0, 0, time.UTC)}
}

func (c *testClock) Now() time.Time { c.mu.Lock(); defer c.mu.Unlock(); return c.now }
func (c *testClock) Advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}

// fakeSource is the application's desired set, and counts how often it was asked for it.
type fakeSource struct {
	mu    sync.Mutex
	specs []contract.CameraSpec
	calls int
	err   error
}

func (s *fakeSource) Cameras(_ context.Context) ([]contract.CameraSpec, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.calls++
	if s.err != nil {
		return nil, s.err
	}
	return append([]contract.CameraSpec(nil), s.specs...), nil
}

func (s *fakeSource) callCount() int { s.mu.Lock(); defer s.mu.Unlock(); return s.calls }

func (s *fakeSource) set(specs ...contract.CameraSpec) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.specs = specs
}

// fakeHandle is a running camera the test can kill.
type fakeHandle struct {
	done chan struct{}
	err  error
	once sync.Once
}

func newFakeHandle() *fakeHandle { return &fakeHandle{done: make(chan struct{})} }

func (h *fakeHandle) Done() <-chan struct{} { return h.done }
func (h *fakeHandle) Err() error            { return h.err }
func (h *fakeHandle) Stop()                 { h.once.Do(func() { close(h.done) }) }

// die makes the camera stop by itself, as a lost stream would.
func (h *fakeHandle) die(err error) {
	h.once.Do(func() {
		h.err = err
		close(h.done)
	})
}

// fakeRunner records every start, by camera and by device.
type fakeRunner struct {
	mu sync.Mutex

	starts   map[string]int
	devices  map[string]int
	handles  map[string]*fakeHandle
	failWith map[string]error

	// hold, when non-nil, blocks every Start until it is closed. It is how the concurrency bound
	// is observed rather than assumed.
	hold    chan struct{}
	current int
	peak    int
}

func newFakeRunner() *fakeRunner {
	return &fakeRunner{
		starts:   map[string]int{},
		devices:  map[string]int{},
		handles:  map[string]*fakeHandle{},
		failWith: map[string]error{},
	}
}

func (r *fakeRunner) Start(_ context.Context, spec contract.CameraSpec, device int) (Handle, error) {
	r.mu.Lock()
	r.starts[spec.ID]++
	r.devices[spec.ID] = device
	r.current++
	if r.current > r.peak {
		r.peak = r.current
	}
	err := r.failWith[spec.ID]
	hold := r.hold
	r.mu.Unlock()

	if hold != nil {
		<-hold
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	r.current--
	if err != nil {
		return nil, err
	}
	h := newFakeHandle()
	r.handles[spec.ID] = h
	return h, nil
}

func (r *fakeRunner) startCount(id string) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.starts[id]
}

func (r *fakeRunner) handle(id string) *fakeHandle {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.handles[id]
}

func (r *fakeRunner) peakConcurrency() int { r.mu.Lock(); defer r.mu.Unlock(); return r.peak }

func (r *fakeRunner) fail(id string, err error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.failWith[id] = err
}

// succeed is fail's inverse, for the camera that comes back.
func (r *fakeRunner) succeed(id string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.failWith, id)
}

func spec(id string, revision uint64) contract.CameraSpec {
	return contract.CameraSpec{ID: id, Revision: revision, Enabled: true, Brightness: 100, Contrast: 100}
}

func newTestSupervisor(t *testing.T, src *fakeSource, run *fakeRunner, mutate func(*Config)) (*Supervisor, *testClock) {
	t.Helper()

	clock := newTestClock()
	cfg := Config{
		Source:  src,
		Runner:  run,
		Metrics: metrics.NewSnapshotRegistry(),
		Clock:   clock.Now,
		Devices: []int{0},
	}
	if mutate != nil {
		mutate(&cfg)
	}
	s, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return s, clock
}

var errStreamGone = errors.New("camera: stream lost")

// TestReconcileDoesNotRestartUnchangedCameras.
//
// The desired set arrives in full on every sync, which makes "restart everything I was told
// about" the easiest possible implementation and a fleet that never keeps a stream open for more
// than a sync interval. Revision is what separates "told again" from "changed".
func TestReconcileDoesNotRestartUnchangedCameras(t *testing.T) {
	src := &fakeSource{}
	run := newFakeRunner()
	src.set(spec("a", 1), spec("b", 1))

	s, _ := newTestSupervisor(t, src, run, nil)
	ctx := context.Background()

	for i := 0; i < 3; i++ {
		if err := s.reconcile(ctx); err != nil {
			t.Fatalf("reconcile %d: %v", i, err)
		}
	}

	if got := run.startCount("a"); got != 1 {
		t.Errorf("camera a started %d times across three identical syncs, want 1", got)
	}
	if got := run.startCount("b"); got != 1 {
		t.Errorf("camera b started %d times across three identical syncs, want 1", got)
	}

	// A moved revision is the one thing that does restart it, and only the camera that moved.
	src.set(spec("a", 2), spec("b", 1))
	if err := s.reconcile(ctx); err != nil {
		t.Fatalf("reconcile after revision bump: %v", err)
	}
	if got := run.startCount("a"); got != 2 {
		t.Errorf("camera a started %d times after its revision moved, want 2", got)
	}
	if got := run.startCount("b"); got != 1 {
		t.Errorf("camera b restarted when its neighbour changed (%d starts)", got)
	}
}

// TestCameraRemovedFromTheDesiredSetIsStopped.
//
// The other half of reconciliation: a snapshot that no longer names a camera is the only signal
// the service gets that it was deleted or disabled.
func TestCameraRemovedFromTheDesiredSetIsStopped(t *testing.T) {
	src := &fakeSource{}
	run := newFakeRunner()
	src.set(spec("a", 1), spec("b", 1))

	s, _ := newTestSupervisor(t, src, run, nil)
	ctx := context.Background()
	if err := s.reconcile(ctx); err != nil {
		t.Fatal(err)
	}

	handle := run.handle("b")
	disabled := spec("b", 1)
	disabled.Enabled = false
	src.set(spec("a", 1), disabled)
	if err := s.reconcile(ctx); err != nil {
		t.Fatal(err)
	}

	select {
	case <-handle.Done():
	default:
		t.Error("a disabled camera kept running")
	}
	if got := len(s.Snapshot()); got != 1 {
		t.Errorf("fleet has %d cameras, want 1", got)
	}
}

// TestWatcherHintsCoalesce.
//
// A single row update touching twenty cameras fires twenty notifications. Reconciling once per
// notification means nineteen passes with nothing to do, each of them reading the whole desired
// set -- and under a bulk edit that is a stampede against the application's database at exactly
// the moment it is busy.
func TestWatcherHintsCoalesce(t *testing.T) {
	src := &fakeSource{}
	run := newFakeRunner()
	src.set(spec("a", 1))

	hints := make(chan struct{}, 32)
	s, _ := newTestSupervisor(t, src, run, func(c *Config) {
		c.Watcher = staticWatcher{ch: hints}
		c.Debounce = 40 * time.Millisecond
		c.SyncInterval = time.Hour     // the periodic sync must not interfere
		c.WatchdogInterval = time.Hour // nor the watchdog
	})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- s.Run(ctx) }()

	waitFor(t, func() bool { return src.callCount() == 1 }, "the initial reconcile")

	for i := 0; i < 20; i++ {
		hints <- struct{}{}
	}

	waitFor(t, func() bool { return src.callCount() == 2 }, "one debounced reconcile")

	// And it stays at two: the other nineteen hints were folded into the same pass, not queued
	// behind it.
	time.Sleep(150 * time.Millisecond)
	if got := src.callCount(); got != 2 {
		t.Errorf("20 hints produced %d reconciles, want 2 (the initial one plus one debounced)", got)
	}

	cancel()
	<-done
}

// TestAFailedStartIsRetriedForever.
//
// A camera never gives up and is never declared offline. This used to stop after three failed
// starts, set an Offline flag and fall back to an hourly schedule -- and the flag was the
// beginning of a chain: a status column, a second query to find the camera again, and a
// forty-eight hour ceiling whose only job was to undo the giving-up.
//
// What is left is a backoff that grows to a ceiling and stays there. A camera unplugged over a
// weekend costs one attempt per ceiling interval and comes back on its own when it is plugged in,
// with nothing else in the system having to notice.
func TestAFailedStartIsRetriedForever(t *testing.T) {
	src := &fakeSource{}
	run := newFakeRunner()
	src.set(spec("a", 1))
	run.fail("a", errors.New("dial: no route to host"))

	s, clock := newTestSupervisor(t, src, run, func(c *Config) {
		c.RetryBackoff = 30 * time.Second
		c.RetryBackoffMax = 2 * time.Minute
	})
	ctx := context.Background()

	if err := s.reconcile(ctx); err != nil {
		t.Fatal(err)
	}

	// Too early: the watchdog respects the backoff, or a dead camera becomes a dial loop.
	s.check(ctx)
	if got := run.startCount("a"); got != 1 {
		t.Fatalf("start attempts = %d before the backoff expired, want 1", got)
	}

	// Keep failing well past where the old design would have given up, and past where the
	// backoff reaches its ceiling. The camera must still be being tried.
	for i := 0; i < 12; i++ {
		clock.Advance(3 * time.Minute)
		s.check(ctx)
	}

	st := status(t, s, "a")
	if st.Running {
		t.Fatal("a camera that never dials successfully is reported as running")
	}
	if got := run.startCount("a"); got < 10 {
		t.Errorf("start attempts = %d after twelve ceiling intervals; the camera gave up", got)
	}
	if st.Failures < 10 {
		t.Errorf("failures = %d, want them to keep counting", st.Failures)
	}

	// And when the camera comes back, so does it -- without anyone touching the panel.
	run.succeed("a")
	clock.Advance(3 * time.Minute)
	s.check(ctx)
	if !status(t, s, "a").Running {
		t.Error("the camera did not come back after it started answering")
	}
}

// TestWatchdogRestartsACameraThatDiedAndFreesItsDevice.
//
// A camera whose loop returned is not running, however healthy the record looks. Its device slot
// has to come back too -- a slot that outlives its camera is exactly the drift that the
// counter-based placement suffered from.
func TestWatchdogRestartsACameraThatDiedAndFreesItsDevice(t *testing.T) {
	src := &fakeSource{}
	run := newFakeRunner()
	src.set(spec("a", 1))

	s, _ := newTestSupervisor(t, src, run, func(c *Config) { c.Devices = []int{0, 1} })
	ctx := context.Background()
	if err := s.reconcile(ctx); err != nil {
		t.Fatal(err)
	}

	run.handle("a").die(errStreamGone)
	s.check(ctx)

	if got := run.startCount("a"); got != 2 {
		t.Errorf("start attempts = %d after the camera died, want 2", got)
	}
	if !status(t, s, "a").Running {
		t.Error("camera not running after the watchdog restarted it")
	}
	if total := s.Load()[0] + s.Load()[1]; total != 1 {
		t.Errorf("device load totals %d for one camera; a slot outlived its camera", total)
	}
}

// TestStartsRunInParallelWithinTheBound.
//
// Serial starts pay every camera's connect timeout in a line -- at 200 cameras that is half an
// hour before the last one sees a frame. The bound is a semaphore and not a lock: the Python
// service used one global lock here, and a single unreachable camera held every other camera's
// open behind its 30-second timeout.
func TestStartsRunInParallelWithinTheBound(t *testing.T) {
	src := &fakeSource{}
	run := newFakeRunner()
	specs := make([]contract.CameraSpec, 0, 30)
	for i := 0; i < 30; i++ {
		specs = append(specs, spec("cam-"+string(rune('a'+i%26))+string(rune('0'+i/26)), 1))
	}
	src.set(specs...)

	hold := make(chan struct{})
	run.hold = hold

	s, _ := newTestSupervisor(t, src, run, func(c *Config) { c.StartConcurrency = 10 })

	done := make(chan error, 1)
	go func() { done <- s.reconcile(context.Background()) }()

	// Ten starts get in; the eleventh waits. Releasing them all lets the pass finish.
	waitFor(t, func() bool { return run.peakConcurrency() == 10 }, "ten concurrent starts")
	time.Sleep(50 * time.Millisecond)
	if got := run.peakConcurrency(); got > 10 {
		t.Errorf("peak concurrent starts = %d, above the bound of 10", got)
	}
	close(hold)

	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if got := run.peakConcurrency(); got > 10 {
		t.Errorf("peak concurrent starts = %d, above the bound of 10", got)
	}
}

// staticWatcher hands back a channel the test writes to.
type staticWatcher struct{ ch chan struct{} }

func (w staticWatcher) Changed(_ context.Context) <-chan struct{} { return w.ch }

func status(t *testing.T, s *Supervisor, id string) Status {
	t.Helper()
	for _, st := range s.Snapshot() {
		if st.ID == id {
			return st
		}
	}
	t.Fatalf("camera %q is not in the fleet", id)
	return Status{}
}

// waitFor polls a condition with a deadline. Used only for things another goroutine does, never
// for anything the injected clock governs.
func waitFor(t *testing.T, cond func() bool, what string) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(time.Millisecond)
	}
}

// TestSpawnStopWaitsForTheCameraToFinish.
//
// Stop that returns early is the redeploy failure: the replacement camera asks for a GPU session
// while the one being torn down still holds it, and an ONNX session that is merely dropped keeps
// its VRAM until the process exits.
func TestSpawnStopWaitsForTheCameraToFinish(t *testing.T) {
	released := make(chan struct{})
	h := Spawn(context.Background(), func(ctx context.Context) error {
		<-ctx.Done()
		close(released) // stands in for "the session is actually gone"
		return ctx.Err()
	})

	h.Stop()

	select {
	case <-released:
	default:
		t.Fatal("Stop returned before the camera had finished")
	}
	select {
	case <-h.Done():
	default:
		t.Fatal("Done is not closed after Stop returned")
	}
	if !errors.Is(h.Err(), context.Canceled) {
		t.Errorf("Err = %v, want context.Canceled", h.Err())
	}

	h.Stop() // must be safe twice: the watchdog and a reconcile can both reach it
}
