package alpr

// This file is the ONLY place in the service that knows the Arandu framework exists.
//
// Everything under app/Services/alpr -- the recognition pipeline, the camera loop, the
// supervisor, the ONNX layer -- is plain Go over the standard library, and a test
// (TestCoreDoesNotImportArandu) fails the build if that stops being true. The payoff is that the
// pipeline can be run, benchmarked and tested without booting an application, which is what makes
// the numbers in bench/ mean anything.
//
// The adapter is deliberately thin. What it mostly does is decide which work belongs in Boot and
// which belongs in Start, and that division is the one thing here worth getting right.

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os/exec"
	"sort"
	"strconv"
	"sync"
	"time"

	"github.com/hyz-is/arandu-alpr/contract"
	"github.com/hyz-is/arandu-alpr/engine"
	"github.com/hyz-is/arandu-alpr/metrics"
	"github.com/hyz-is/arandu-alpr/supervisor"
)

// ModuleConfig is what the application hands the module.
//
// Source and Sink are required: they are the two halves of the boundary. The service does not
// know where cameras come from and does not know what happens to a reading -- that is not
// modesty, it is what keeps billing, tenancy, persistence and notification out of a package that
// has no business making those decisions.
type ModuleConfig struct {
	Source contract.Source
	Sink   contract.Sink

	Watcher  contract.Watcher  // optional
	Observer contract.Observer // optional
	Preview  contract.Preview  // optional

	// Engine does the inference. In production this is the ONNX engine; in a test or a
	// benchmark it is engine.Fake, and neither the supervisor nor the camera loop can tell.
	Engine engine.Engine

	// Runner starts one camera. The application owns it because building a camera needs the
	// dialer, the gate and the encoder, and all three are deployment choices.
	Runner supervisor.Runner

	Supervisor supervisor.Config
	Metrics    metrics.Registry
	Logger     *slog.Logger

	// FFmpegPath and FFprobePath are checked at boot. Empty means "on PATH".
	FFmpegPath  string
	FFprobePath string

	// ExposeRoutes registers the operator endpoints. Off by default: a service does not get to
	// decide that an application has an unauthenticated metrics endpoint.
	ExposeRoutes bool
}

// Module is the Arandu module.
//
// It implements Name, Routes, Boot, Start, Close, Diagnose and Health. It deliberately does not
// implement Migratable -- the service owns no tables -- nor Schedulable, because the supervisor
// already owns its own timers and a scheduled task would be a second one racing the first.
type Module struct {
	cfg ModuleConfig
	sup *supervisor.Supervisor
	log *slog.Logger

	mu      sync.Mutex
	booted  bool
	started time.Time
}

// NewModule validates the configuration and takes a snapshot of it.
//
// Everything mutable is copied here rather than read later, so two Modules in one process cannot
// influence each other -- the class of bug that in the Python service showed up as a semaphore
// built at import time binding itself to the wrong event loop.
func NewModule(cfg ModuleConfig) (*Module, error) {
	if cfg.Source == nil {
		return nil, errors.New("alpr: ModuleConfig.Source is required; the service does not know where cameras come from")
	}
	if cfg.Sink == nil {
		return nil, errors.New("alpr: ModuleConfig.Sink is required; the service does not decide what happens to a reading")
	}
	if cfg.Engine == nil {
		return nil, errors.New("alpr: ModuleConfig.Engine is required")
	}
	if cfg.Runner == nil {
		return nil, errors.New("alpr: ModuleConfig.Runner is required; building a camera needs the dialer, gate and encoder the application owns")
	}
	if cfg.Metrics == nil {
		cfg.Metrics = metrics.Discard()
	}
	if cfg.Logger == nil {
		cfg.Logger = slog.New(slog.DiscardHandler)
	}

	sc := cfg.Supervisor
	sc.Source = cfg.Source
	sc.Runner = cfg.Runner
	sc.Watcher = cfg.Watcher
	sc.Observer = cfg.Observer
	sc.Metrics = cfg.Metrics
	sc.Logger = cfg.Logger

	sup, err := supervisor.New(sc)
	if err != nil {
		return nil, err
	}
	return &Module{cfg: cfg, sup: sup, log: cfg.Logger}, nil
}

// Name is the module's stable slug.
func (m *Module) Name() string { return "alpr" }

// Boot validates. It does not run.
//
// Nothing here opens a camera, starts a goroutine or reads a frame: Boot's job is to turn every
// failure that would otherwise appear later, as a puzzling symptom, into one sentence now.
//
// The model contract, the ONNX library and the CUDA devices are validated by the engine before
// it is handed to this module, which is why they are absent here -- the engine cannot be
// constructed without them.
func (m *Module) Boot(ctx context.Context) error {
	var problems []error

	// ffmpeg is a subprocess, so a missing binary is not a compile error and not a link error:
	// it is a camera that fails to open, once per camera, with an errno buried in a log.
	for name, path := range map[string]string{"ffmpeg": m.cfg.FFmpegPath, "ffprobe": m.cfg.FFprobePath} {
		if path == "" {
			path = name
		}
		if _, err := exec.LookPath(path); err != nil {
			problems = append(problems, fmt.Errorf(
				"alpr: %s not found (%q): every camera would fail to open, one at a time, "+
					"with an error that names a subprocess rather than a configuration", name, path))
		}
	}

	if err := m.cfg.Engine.Ready(); err != nil {
		problems = append(problems, fmt.Errorf("alpr: inference engine is not ready: %w", err))
	}

	// Reaching the camera source at boot turns "no cameras are running" -- which looks like a
	// service fault -- into "the source is unreachable", which is where the problem is.
	if _, err := m.cfg.Source.Cameras(ctx); err != nil {
		problems = append(problems, fmt.Errorf("alpr: camera source is unreachable: %w", err))
	}

	if err := errors.Join(problems...); err != nil {
		return err
	}

	m.mu.Lock()
	m.booted = true
	m.mu.Unlock()
	return nil
}

// Start runs the fleet, and returns only when ctx ends.
//
// It is separate from Boot for a specific reason: Start is called by the application's run path
// and never by a console command. Without that split, `aru migrate` -- or any other command that
// boots the container to reach a service -- would bring up every camera in the fleet as a side
// effect of asking one question.
func (m *Module) Start(ctx context.Context) error {
	m.mu.Lock()
	booted := m.booted
	m.started = time.Now()
	m.mu.Unlock()

	if !booted {
		return errors.New("alpr: Start called before Boot")
	}
	m.log.Info("alpr: starting camera fleet")
	return m.sup.Run(ctx)
}

// Close stops the fleet.
//
// It deliberately never calls runtime.GC. The Python service collected on every camera stop, to
// reclaim a session's memory, and it paused the whole process -- with a watchdog restarting
// processors every thirty seconds, that pause landed continuously, on the cameras that were
// still working. Releasing the resource explicitly is the fix; the collector is not.
func (m *Module) Close(ctx context.Context) error {
	m.log.Info("alpr: stopping camera fleet")
	return m.cfg.Engine.Close(ctx)
}

// Health is the yes-or-no answer, for `aru doctor` and the framework's health endpoint.
func (m *Module) Health(ctx context.Context) error {
	if err := m.cfg.Engine.Ready(); err != nil {
		return err
	}
	want, err := m.cfg.Source.Cameras(ctx)
	if err != nil {
		return fmt.Errorf("alpr: camera source unreachable: %w", err)
	}
	enabled := 0
	for _, c := range want {
		if c.Enabled {
			enabled++
		}
	}
	if enabled == 0 {
		return nil // nothing is meant to be running, so nothing being down is correct
	}
	running := 0
	for _, s := range m.sup.Snapshot() {
		if s.Running {
			running++
		}
	}
	if running == 0 {
		return fmt.Errorf("alpr: %d cameras are enabled and none is running", enabled)
	}
	return nil
}

// Diagnose reports what the module knows, in sentences a person can act on.
//
// It returns NOTHING when the fleet is healthy. A diagnostic that always says something trains
// its reader to skip it, and the one time it matters it is skipped too.
//
// The sentences here are chosen for a specific failure each -- every one of them is a thing that
// happened, in production, in the service this replaces, and took a code reading to explain.
func (m *Module) Diagnose(ctx context.Context) []string {
	var out []string

	if err := m.cfg.Engine.Ready(); err != nil {
		out = append(out, "The inference engine is not serving: "+err.Error()+
			". Sessions take eight to fifteen seconds each to build, so this is normal for the "+
			"first moments after a restart and a problem after that.")
	}

	// A model that fell back to the CPU does not fail. It gets slow, and slow is diagnosed as a
	// camera problem, a network problem, or a fleet that has outgrown its hardware.
	if p := m.cfg.Engine.Providers(); p.CPUNodes > 0 {
		out = append(out, fmt.Sprintf(
			"%d graph nodes are running on the CPU instead of the GPU. Throughput will be a "+
				"fraction of what the hardware can do, and nothing will report an error. "+
				"Check the node placement log from the first session of each model.", p.CPUNodes))
	}

	// Cameras that are not running and have failed before. Deliberately NOT called offline: no
	// camera in this service is ever declared offline, it keeps dialling at a bounded interval
	// until it answers. "Offline" would name a final state, and an operator who read it would go
	// restart something that is already trying.
	statuses := m.sup.Snapshot()
	var struggling []string
	for _, s := range statuses {
		if !s.Running && s.Failures > 0 {
			why := "no error recorded"
			if s.LastErr != nil {
				why = s.LastErr.Error()
			}
			struggling = append(struggling,
				fmt.Sprintf("%s (%d attempts, last: %s)", s.ID, s.Failures, why))
		}
	}
	if len(struggling) > 0 {
		sort.Strings(struggling)
		out = append(out, "Cameras not connected, still retrying: "+joinLimit(struggling, 5))
	}

	// The placement imbalance. The Python service picked a GPU with a counter that was never
	// decremented when a camera stopped, so after a few restarts one card carried the fleet and
	// the other idled -- with both reporting healthy.
	if load := m.sup.Load(); len(load) > 1 {
		lo, hi, loDev, hiDev := -1, -1, 0, 0
		for dev, n := range load {
			if lo < 0 || n < lo {
				lo, loDev = n, dev
			}
			if n > hi {
				hi, hiDev = n, dev
			}
		}
		if hi-lo > 1 && hi > 0 {
			out = append(out, fmt.Sprintf(
				"GPU %d is carrying %d cameras while GPU %d has %d. One card will be the "+
					"bottleneck and the other will idle, and both report healthy.",
				hiDev, hi, loDev, lo))
		}
	}

	return out
}

func joinLimit(items []string, max int) string {
	if len(items) <= max {
		return join(items, "; ")
	}
	return join(items[:max], "; ") + fmt.Sprintf("; and %d more", len(items)-max)
}

func join(items []string, sep string) string {
	out := ""
	for i, s := range items {
		if i > 0 {
			out += sep
		}
		out += s
	}
	return out
}

// FleetStatus is the operator view, for the optional status route.
type FleetStatus struct {
	Cameras  []supervisor.Status   `json:"cameras"`
	Load     map[string]int        `json:"load_by_device"`
	Provider engine.ProviderReport `json:"providers"`
	Model    contract.ModelInfo    `json:"model"`
	UptimeS  float64               `json:"uptime_seconds"`
}

// Status reports the fleet.
func (m *Module) Status() FleetStatus {
	m.mu.Lock()
	started := m.started
	m.mu.Unlock()

	load := map[string]int{}
	for dev, n := range m.sup.Load() {
		load[strconv.Itoa(dev)] = n
	}
	var uptime float64
	if !started.IsZero() {
		uptime = time.Since(started).Seconds()
	}
	return FleetStatus{
		Cameras:  m.sup.Snapshot(),
		Load:     load,
		Provider: m.cfg.Engine.Providers(),
		Model:    m.cfg.Engine.ModelInfo(),
		UptimeS:  uptime,
	}
}

// MetricsSnapshot reports the counters, if the registry can be read.
//
// It returns nil for a registry that cannot -- Discard, or an application's own backend that
// only forwards. That is not a failure: it means the numbers live somewhere else, and inventing
// an empty answer here would look like a service producing none.
func (m *Module) MetricsSnapshot() []metrics.Sample {
	if r, ok := m.cfg.Metrics.(interface{ Snapshot() []metrics.Sample }); ok {
		return r.Snapshot()
	}
	return nil
}
