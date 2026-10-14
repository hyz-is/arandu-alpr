// Package supervisor runs the fleet: it decides which cameras should exist, starts them, notices
// when one dies and puts it back.
//
// Two of its rules are the difference between a fleet that settles and one that never does:
//
//   - reconciliation compares (ID, Revision) and leaves everything else alone, so a sync can
//     never restart a healthy camera
//   - device placement is a live map of what is running, not a counter of what was ever started
//
// Both are ported from a Python service that had neither, and both failures were invisible: a
// fleet that restarts itself every minute and a fleet with one saturated GPU both look like
// "cameras are a bit flaky" from the outside.
package supervisor

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strconv"
	"sync"
	"time"

	"github.com/hyz-is/arandu-alpr/contract"
	"github.com/hyz-is/arandu-alpr/metrics"
)

// Handle is a camera that has been started.
type Handle interface {
	// Done is closed once the camera has stopped by itself.
	Done() <-chan struct{}
	// Err reports why it stopped. It is only meaningful after Done is closed.
	Err() error
	// Stop stops the camera and returns once it has. It must be safe to call twice.
	Stop()
}

// Runner starts one camera on one device.
//
// Start returns once the camera is running, or an error if it could not be started -- the
// supervisor's retry policy is built on that distinction, so an implementation
// that always succeeds and fails later moves every failure into the watchdog's slower path.
//
// Start calls are bounded by StartConcurrency. An implementation that returns before its stream
// is actually open moves the expensive part outside that bound and has to bound it itself, which
// is what streaming.Opener's semaphore is for.
type Runner interface {
	Start(ctx context.Context, spec contract.CameraSpec, device int) (Handle, error)
}

// Config is the fleet's policy.
type Config struct {
	Source contract.Source
	Runner Runner

	Watcher  contract.Watcher  // optional
	Observer contract.Observer // optional
	Metrics  metrics.Registry  // optional; Discard when nil
	Logger   *slog.Logger      // optional; discards when nil

	// Devices are the GPU ids this worker may use. Empty means a single device, id 0.
	Devices []int

	// Clock is the only source of time, so the retry schedule can be tested without waiting it out.
	Clock func() time.Time

	// SyncInterval is how often the desired set is read in full.
	//
	// The periodic sync is the truth and a Watcher only shortens the wait. A service that
	// reconciled only on notifications is one dropped notification away from running a camera
	// that was deleted, with nothing in either system showing a disagreement.
	SyncInterval time.Duration

	// Debounce is how long a hint waits for its neighbours.
	//
	// A row update touching twenty cameras fires twenty notifications. Without this, that is
	// twenty full reconciles, and the last nineteen have nothing to do.
	Debounce time.Duration

	// StartConcurrency bounds how many cameras are started at once.
	//
	// It exists so a fleet-wide restart does not open every stream in the same instant. It is a
	// semaphore and must never become a lock: the Python service serialised opening behind one
	// global lock, and a single unreachable camera then held every other camera's open behind its
	// 30-second connect timeout.
	StartConcurrency int

	// WatchdogInterval is how often the running set is checked for cameras that died.
	WatchdogInterval time.Duration
	// RetryBackoff is the first wait after a failed start; it doubles up to RetryBackoffMax.
	RetryBackoff    time.Duration
	RetryBackoffMax time.Duration
}

// Defaults mirror the Python service's production timings.
const (
	DefaultSyncInterval     = 60 * time.Second
	DefaultDebounce         = time.Second
	DefaultStartConcurrency = 10
	DefaultWatchdogInterval = 30 * time.Second
	DefaultRetryBackoff     = 30 * time.Second
	DefaultRetryBackoffMax  = 5 * time.Minute
)

func (c Config) withDefaults() Config {
	if c.Clock == nil {
		c.Clock = time.Now
	}
	if c.SyncInterval <= 0 {
		c.SyncInterval = DefaultSyncInterval
	}
	if c.Debounce <= 0 {
		c.Debounce = DefaultDebounce
	}
	if c.StartConcurrency <= 0 {
		c.StartConcurrency = DefaultStartConcurrency
	}
	if c.WatchdogInterval <= 0 {
		c.WatchdogInterval = DefaultWatchdogInterval
	}
	if c.RetryBackoff <= 0 {
		c.RetryBackoff = DefaultRetryBackoff
	}
	if c.RetryBackoffMax < c.RetryBackoff {
		c.RetryBackoffMax = DefaultRetryBackoffMax
	}
	if c.Metrics == nil {
		c.Metrics = metrics.Discard()
	}
	if c.Logger == nil {
		c.Logger = slog.New(slog.DiscardHandler)
	}
	return c
}

// record is one camera the supervisor is responsible for, running or not.
type record struct {
	spec   contract.CameraSpec
	device int

	handle Handle

	failures    int
	nextRetryAt time.Time
	lastErr     error
}

// Supervisor owns the fleet.
type Supervisor struct {
	cfg    Config
	placer *placer

	mu      sync.Mutex
	records map[string]*record

	gpuGauge metrics.GaugeVec
}

// LabelCamera is the label every fleet metric carries.
const LabelCamera = "camera"

// New builds a supervisor. It talks to nothing until Run is called.
func New(cfg Config) (*Supervisor, error) {
	if cfg.Source == nil {
		return nil, errors.New("supervisor: no source")
	}
	if cfg.Runner == nil {
		return nil, errors.New("supervisor: no runner")
	}
	cfg = cfg.withDefaults()

	return &Supervisor{
		cfg:      cfg,
		placer:   newPlacer(cfg.Devices),
		records:  map[string]*record{},
		gpuGauge: cfg.Metrics.Gauge(metrics.CameraGPU, LabelCamera),
	}, nil
}

func (s *Supervisor) now() time.Time { return s.cfg.Clock() }

// Run reconciles until the context ends, then stops every camera.
//
// Reconciles, watchdog passes and starts all happen on this one goroutine. That is deliberate:
// two reconciles overlapping would each see half of the other's work and fight over the same
// camera, and the bug that produces -- a camera started twice, or stopped by the pass that did
// not know it had just been started -- is the kind that only appears under load.
func (s *Supervisor) Run(ctx context.Context) error {
	defer s.stopAll()

	if err := s.reconcile(ctx); err != nil {
		s.cfg.Logger.Error("initial reconcile failed", "error", err)
	}

	sync := time.NewTicker(s.cfg.SyncInterval)
	defer sync.Stop()
	watchdog := time.NewTicker(s.cfg.WatchdogInterval)
	defer watchdog.Stop()

	var hints <-chan struct{}
	if s.cfg.Watcher != nil {
		hints = s.cfg.Watcher.Changed(ctx)
	}

	// The debounce timer is nil while idle. Arming it on the first hint and ignoring the rest is
	// what makes a notify storm one reconcile instead of one each -- and hints that arrive while
	// a reconcile is running wait in the channel and collapse into the same single pass after it.
	var debounce *time.Timer
	var debounced <-chan time.Time
	defer func() {
		if debounce != nil {
			debounce.Stop()
		}
	}()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()

		case <-sync.C:
			s.reconcileLogged(ctx)

		case _, ok := <-hints:
			if !ok {
				hints = nil // a closed watcher only costs us the shortcut, never the sync
				continue
			}
			drain(hints)
			if debounce == nil {
				debounce = time.NewTimer(s.cfg.Debounce)
				debounced = debounce.C
			}

		case <-debounced:
			debounce, debounced = nil, nil
			s.reconcileLogged(ctx)

		case <-watchdog.C:
			s.check(ctx)
		}
	}
}

// drain empties whatever is already queued on the hint channel.
func drain(ch <-chan struct{}) {
	for {
		select {
		case <-ch:
		default:
			return
		}
	}
}

func (s *Supervisor) reconcileLogged(ctx context.Context) {
	if err := s.reconcile(ctx); err != nil {
		// A failed sync is not a reason to tear the fleet down. The desired set is simply
		// unknown for now, and what is running keeps running -- which is right, because the most
		// likely cause is that the application's database is briefly unavailable, and stopping
		// every camera over that would turn a blip into an outage.
		s.cfg.Logger.Error("reconcile failed", "error", err)
	}
}

// reconcile brings the running set in line with the desired one.
//
// The diff is on (ID, Revision) and nothing else. Comparing whole specs would mean every field
// the comparison forgets is a camera silently running with stale settings -- and comparing
// nothing, as a "restart everything on sync" would, is a fleet that reboots itself every minute
// and never keeps a stream open long enough to read a plate.
func (s *Supervisor) reconcile(ctx context.Context) error {
	specs, err := s.cfg.Source.Cameras(ctx)
	if err != nil {
		return fmt.Errorf("supervisor: read desired cameras: %w", err)
	}

	desired := make(map[string]contract.CameraSpec, len(specs))
	for _, spec := range specs {
		if spec.ID == "" || !spec.Enabled {
			// A disabled camera is an absent camera. Keeping it as a stopped record would let it
			// come back through the watchdog's retry path, which is not what disabling means.
			continue
		}
		desired[spec.ID] = spec
	}

	var (
		toStop  []*record
		toStart []*record
	)

	s.mu.Lock()
	for id, rec := range s.records {
		want, ok := desired[id]
		switch {
		case !ok:
			toStop = append(toStop, rec)
			delete(s.records, id)
		case want.Revision != rec.spec.Revision:
			// Stop and start rather than reconfigure in place: the revision says something
			// changed, and the only thing that reliably applies every kind of change -- URL,
			// credentials, thresholds, placement -- is a fresh camera.
			toStop = append(toStop, rec)
			replacement := &record{spec: want}
			s.records[id] = replacement
			toStart = append(toStart, replacement)
		default:
			// Unchanged. Not touched, not restarted, not even looked at again.
		}
	}
	for id, want := range desired {
		if _, ok := s.records[id]; ok {
			continue
		}
		rec := &record{spec: want}
		s.records[id] = rec
		toStart = append(toStart, rec)
	}
	s.mu.Unlock()

	for _, rec := range toStop {
		s.stop(rec)
	}
	s.startAll(ctx, toStart)
	return nil
}

// startAll starts cameras in parallel, up to StartConcurrency at a time.
//
// Parallel because a fleet coming up otherwise pays every camera's connect timeout in a line:
// with a 10s timeout and 200 cameras, a serial start is over half an hour before the last camera
// sees a frame. Bounded because opening hundreds of streams in the same instant is a burst
// neither the network nor the decoder handles gracefully.
func (s *Supervisor) startAll(ctx context.Context, recs []*record) {
	if len(recs) == 0 {
		return
	}

	slots := make(chan struct{}, s.cfg.StartConcurrency)
	var wg sync.WaitGroup
	for _, rec := range recs {
		if ctx.Err() != nil {
			return
		}
		wg.Add(1)
		go func(rec *record) {
			defer wg.Done()
			slots <- struct{}{}
			defer func() { <-slots }()
			s.start(ctx, rec)
		}(rec)
	}
	wg.Wait()
}

// start puts one camera on a device and asks the runner for it.
func (s *Supervisor) start(ctx context.Context, rec *record) {
	s.mu.Lock()
	spec := rec.spec
	s.mu.Unlock()

	device := s.placer.Place(spec.ID, spec.Placement)
	handle, err := s.cfg.Runner.Start(ctx, spec, device)

	s.mu.Lock()
	defer s.mu.Unlock()

	if err != nil {
		s.placer.Release(spec.ID)
		rec.lastErr = err
		rec.failures++
		rec.handle = nil
		s.scheduleRetryLocked(ctx, rec, err)
		return
	}

	rec.device = device
	rec.handle = handle
	rec.failures = 0
	rec.lastErr = nil
	rec.nextRetryAt = time.Time{}
	s.gpuGauge.With(spec.ID).Set(float64(device))
	s.observe(ctx, spec.ID, contract.EventCameraUp, "camera started on device "+strconv.Itoa(device))
}

// scheduleRetryLocked decides when a failed camera is tried again. Callers hold the lock.
func (s *Supervisor) scheduleRetryLocked(ctx context.Context, rec *record, err error) {
	now := s.now()

	// The backoff grows to a ceiling and stays there. There is no attempt limit and no offline
	// state: a camera unplugged over a weekend comes back on its own when it is plugged in, and
	// spends the weekend costing one start attempt per ceiling interval rather than being
	// declared dead and needing something to resurrect it.
	//
	// Declaring it dead is what used to require a second retry schedule, a status column and a
	// 48-hour window -- three mechanisms whose only job was to undo the giving-up.
	backoff := s.cfg.RetryBackoff << uint(min(rec.failures-1, 16))
	if backoff > s.cfg.RetryBackoffMax || backoff <= 0 {
		backoff = s.cfg.RetryBackoffMax
	}
	rec.nextRetryAt = now.Add(backoff)
	s.cfg.Logger.Warn("camera failed to start, retrying",
		LabelCamera, rec.spec.ID, "attempt", rec.failures, "retry_in", backoff, "error", err)
}

// check is the watchdog pass: it finds cameras that stopped and the ones whose retry is due.
func (s *Supervisor) check(ctx context.Context) {
	now := s.now()

	var (
		toRelease []string
		toStart   []*record
	)

	s.mu.Lock()
	for id, rec := range s.records {
		if rec.handle == nil {
			if rec.nextRetryAt.IsZero() || !now.Before(rec.nextRetryAt) {
				toStart = append(toStart, rec)
			}
			continue
		}
		select {
		case <-rec.handle.Done():
			rec.lastErr = rec.handle.Err()
			rec.handle = nil
			rec.failures++
			toRelease = append(toRelease, id)
			s.scheduleRetryLocked(ctx, rec, rec.lastErr)
			// A camera that died gets its first retry on the next pass rather than after the
			// backoff: the common case is a stream that dropped and comes straight back, and
			// waiting 30 seconds to find that out is 30 seconds of missed plates.
			rec.nextRetryAt = time.Time{}
			toStart = append(toStart, rec)
		default:
			// Still running. Left alone.
		}
	}
	s.mu.Unlock()

	for _, id := range toRelease {
		// Released the moment the camera stops, not when it is replaced. A slot that outlives its
		// camera is exactly the drift the counter-based placement had.
		s.placer.Release(id)
	}
	s.startAll(ctx, toStart)
}

// stop stops one camera and frees its device.
func (s *Supervisor) stop(rec *record) {
	s.mu.Lock()
	handle := rec.handle
	rec.handle = nil
	id := rec.spec.ID
	s.mu.Unlock()

	if handle != nil {
		handle.Stop()
	}
	s.placer.Release(id)
}

// stopAll tears the fleet down, in parallel: shutdown is one place where a serial loop over
// hundreds of cameras turns a redeploy into a visible outage.
func (s *Supervisor) stopAll() {
	s.mu.Lock()
	recs := make([]*record, 0, len(s.records))
	for _, rec := range s.records {
		recs = append(recs, rec)
	}
	s.records = map[string]*record{}
	s.mu.Unlock()

	var wg sync.WaitGroup
	for _, rec := range recs {
		wg.Add(1)
		go func(rec *record) {
			defer wg.Done()
			s.stop(rec)
		}(rec)
	}
	wg.Wait()
}

func (s *Supervisor) observe(ctx context.Context, cameraID string, kind contract.EventKind, message string) {
	if s.cfg.Observer == nil {
		return
	}
	// The message never carries the stream URL: it holds the camera's password, and these lines
	// end up in operator-facing panels.
	s.cfg.Observer.Observe(ctx, contract.Event{
		CameraID: cameraID,
		Kind:     kind,
		Message:  message,
		At:       s.now(),
	})
}

// Status is what the supervisor knows about one camera.
type Status struct {
	ID       string
	Revision uint64
	Device   int
	Running  bool
	Failures int
	LastErr  error
}

// Snapshot reports the fleet, for diagnostics and tests.
func (s *Supervisor) Snapshot() []Status {
	s.mu.Lock()
	defer s.mu.Unlock()

	out := make([]Status, 0, len(s.records))
	for _, rec := range s.records {
		out = append(out, Status{
			ID:       rec.spec.ID,
			Revision: rec.spec.Revision,
			Device:   rec.device,
			Running:  rec.handle != nil,
			Failures: rec.failures,
			LastErr:  rec.lastErr,
		})
	}
	return out
}

// Load reports how many cameras each device is carrying.
func (s *Supervisor) Load() map[int]int { return s.placer.Load() }
