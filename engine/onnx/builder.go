package onnx

import (
	"context"
	"errors"
	"fmt"
	"math/rand/v2"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/hyz-is/arandu-alpr/engine"
)

// DefaultLibraryPath is where the official ONNX Runtime tarball puts its shared object.
//
// The Go binding defaults to "onnxruntime.so", which is not what Microsoft ships -- their
// releases contain libonnxruntime.so. Without an explicit path the very first call fails at
// dlopen, before anything else has had a chance to run.
const DefaultLibraryPath = "/usr/local/lib/libonnxruntime.so"

// buildKind orders the two things the builder does.
type buildKind int

const (
	kindBuild buildKind = iota
	kindDestroy
)

// SessionSpec asks for one session.
type SessionSpec struct {
	Role     string // RoleDetector or RoleOCR
	Contract ModelContract
	Device   int // CUDA device id; negative means CPU

	// Verbose asks ONNX Runtime to log node placement while creating this session. Only the
	// first session of each (role, device) sets it -- see capturePlacement.
	Verbose bool
}

func (s SessionSpec) key() string { return s.Role + ":" + strconv.Itoa(s.Device) }

type buildRequest struct {
	kind  buildKind
	spec  SessionSpec
	built any // for kindDestroy
	reply chan buildResult
	ctx   context.Context
}

type buildResult struct {
	runner any
	report engine.ModelReport
	err    error
}

// Config configures the builder.
type Config struct {
	// LibraryPath overrides DefaultLibraryPath; ONNXRUNTIME_LIB_PATH overrides both.
	LibraryPath string

	// Devices are the CUDA device ids to use. Empty means CPU only.
	//
	// A list rather than a count, and never a hardcoded 0: a box with two GPUs where one is
	// reserved for something else is ordinary, and "device 0" is not a safe assumption.
	Devices []int

	WarmupRuns int

	// RequireAllCUDA aborts the boot when any node of any model lands on the CPU.
	//
	// A model that silently falls back does not fail; it gets slow, and slow is diagnosed as a
	// camera problem, a network problem, or a fleet that has simply outgrown its hardware.
	RequireAllCUDA bool

	// WarmupMaxSteadyMS fails a build whose warm inference is slower than this. Zero logs
	// instead of failing. A steady-state detector pass in the hundreds of milliseconds is the
	// signature of a CPU fallback, whatever the placement log said.
	WarmupMaxSteadyMS float64

	// ConvAlgoSearch is how hard cuDNN looks for a convolution algorithm: EXHAUSTIVE, HEURISTIC
	// or DEFAULT.
	//
	// EXHAUSTIVE is ONNX Runtime's default and gives the fastest steady state, but it benchmarks
	// every candidate algorithm at session creation -- which is a large share of the eight to
	// fifteen seconds a session costs to build. With several sessions across two cards that is
	// paid several times over, and it is paid again on every restart.
	//
	// Worth sweeping rather than assuming. HEURISTIC trades a little steady-state throughput for
	// a much shorter cold start, and cold start is what a fleet feels during a redeploy.
	ConvAlgoSearch string

	// ConvUseMaxWorkspace lets cuDNN allocate the workspace it actually wants.
	//
	// ONNX Runtime's documentation is unusually direct about this one: it is "an important flag
	// to use while using an fp16 model as this allows CuDNN to pick tensor core algorithms".
	// Clamped to 32MB, cuDNN falls back to an algorithm that does not use them.
	//
	// That matters here specifically. A T4 is Turing: its tensor cores are the difference
	// between FP32 and FP16 throughput, and FP16 is the largest lever this service has left.
	// The cost is peak VRAM, which on a 16GB card is not the binding constraint.
	ConvUseMaxWorkspace *bool

	// PreferNHWC asks the provider for NHWC kernels, transposing the graph as needed.
	//
	// NVIDIA's tensor cores prefer NHWC, so this can help -- or it can cost more in inserted
	// transposes than it saves, depending on how much of the graph has an NHWC kernel. ONNX
	// Runtime says as much. It is off by default and exists to be measured, not assumed.
	PreferNHWC bool

	// GPUMemLimit caps one session's arena, in bytes. Zero means unlimited.
	//
	// With several sessions per card, an unlimited arena means the first session to see a large
	// allocation can take space the others will need, and the failure surfaces later as a build
	// that fails on a card with apparently free memory.
	GPUMemLimit int64

	// TensorRT runs both models as INT8 on the tensor cores. Zero value: CUDA only.
	TensorRT TensorRTConfig
}

func (c Config) withDefaults() Config {
	if c.LibraryPath == "" {
		c.LibraryPath = DefaultLibraryPath
	}
	if c.WarmupRuns <= 0 {
		// More than one: the first inference pays cuDNN's exhaustive algorithm search and the
		// CUDA module load, so a single run measures the cold path and tells you nothing about
		// the one every frame will take.
		c.WarmupRuns = 3
	}
	return c
}

// Builder creates and destroys ONNX sessions, and it is the only thing in the service that may.
//
// It runs one loop, on the process's main thread, because of ONNX Runtime issue #26610: a CUDA
// session created on any other thread crashes with SIGSEGV. That constraint has a second payoff
// that would otherwise cost a global mutex -- a single loop is inherently serial, so concurrent
// session creation cannot happen, and concurrent creation is what produces
// `CUDA 900: operation not permitted when stream is capturing`.
//
// Everything else in the service talks to it over a channel and never touches ONNX Runtime.
type Builder struct {
	cfg Config

	requests chan buildRequest

	mu         sync.Mutex
	destroyers []registered         // what to destroy at shutdown, in creation order
	reports    []engine.ModelReport // one per built session
	failures   map[string]*failureState

	started   sync.Once
	serving   chan struct{}
	closeOnce sync.Once
}

// failureState backs off a (role, device) that keeps failing to build.
//
// Without it a GPU in a bad state turns every Borrow into a fresh twelve-second attempt on the
// single serial loop, and those attempts starve the healthy devices behind them. The pool has no
// way to know this: it hands the slot back as unbuilt and the next caller tries immediately.
// registered is one session the builder owns and must destroy.
type registered struct{ runner any }

type failureState struct {
	consecutive int
	nextAttempt time.Time
}

func NewBuilder(cfg Config) *Builder {
	return &Builder{
		cfg:      cfg.withDefaults(),
		requests: make(chan buildRequest),
		failures: map[string]*failureState{},
		serving:  make(chan struct{}),
	}
}

// Init loads the ONNX Runtime library and initialises the environment.
//
// It must run on the main thread, before Serve, and exactly once per process: the binding keeps
// the environment and the library handle in package globals, InitializeEnvironment errors if
// called twice, and DestroyEnvironment dlcloses the library.
func (b *Builder) Init() error {
	if !OnMainThread() {
		return errors.New(
			"alpr/onnx: Init must run on the process's main thread.\n" +
				"    Creating a CUDA session anywhere else crashes with SIGSEGV (ORT #26610).\n" +
				"    Import this package for its init() and call Init and Serve from main.main.")
	}
	ort.SetSharedLibraryPath(b.cfg.LibraryPath)
	if err := ort.InitializeEnvironment(ort.WithLogLevelVerbose()); err != nil {
		return fmt.Errorf("alpr/onnx: cannot initialise ONNX Runtime from %s: %w\n"+
			"    Set ONNXRUNTIME_LIB_PATH if the library is elsewhere", b.cfg.LibraryPath, err)
	}
	return nil
}

// Serve runs the builder loop. It blocks until ctx ends, then destroys everything it created.
//
// The application calls this from main.main and runs its own server in a goroutine beside it.
// That is an inversion of the usual shape, and it is not optional: see the package comment.
func (b *Builder) Serve(ctx context.Context) error {
	if !OnMainThread() {
		return errors.New("alpr/onnx: Serve must run on the process's main thread")
	}
	b.started.Do(func() { close(b.serving) })
	defer b.shutdown()

	// Destroy requests jump the queue. Destroying frees VRAM and building consumes it, so
	// serving them in arrival order is how a fleet that is churning cameras reaches OOM while
	// holding memory it has already finished with.
	var pending []buildRequest

	for {
		if len(pending) > 0 {
			req := takeHighestPriority(&pending)
			b.serve(req)
			continue
		}
		select {
		case <-ctx.Done():
			return nil
		case req := <-b.requests:
			pending = append(pending, req)
			// Drain whatever else is already waiting, so priority can be applied across the
			// whole set rather than to one request at a time.
			for draining := true; draining; {
				select {
				case more := <-b.requests:
					pending = append(pending, more)
				default:
					draining = false
				}
			}
		}
	}
}

func takeHighestPriority(pending *[]buildRequest) buildRequest {
	idx := 0
	for i, r := range *pending {
		if r.kind == kindDestroy {
			idx = i
			break
		}
	}
	req := (*pending)[idx]
	*pending = append((*pending)[:idx], (*pending)[idx+1:]...)
	return req
}

func (b *Builder) serve(req buildRequest) {
	// A caller that gave up must not hold the loop: its slot is already back in the pool.
	if req.ctx != nil && req.ctx.Err() != nil {
		req.reply <- buildResult{err: req.ctx.Err()}
		return
	}
	switch req.kind {
	case kindDestroy:
		destroyRunner(req.built)
		b.forget(req.built)
		req.reply <- buildResult{}
	default:
		runner, report, err := b.build(req.spec)
		req.reply <- buildResult{runner: runner, report: report, err: err}
	}
}

// Build asks for a session. It is safe to call from any goroutine.
//
// This is what a SessionPool's build closure calls. The calling goroutine only ever blocks on a
// channel -- it never touches ONNX Runtime, so which OS thread it happens to be on is
// irrelevant, and lazy construction inside Borrow stays safe.
func (b *Builder) Build(ctx context.Context, spec SessionSpec) (any, engine.ModelReport, error) {
	if err := b.waitServing(ctx); err != nil {
		return nil, engine.ModelReport{}, err
	}
	if wait, ok := b.backoffRemaining(spec); ok {
		return nil, engine.ModelReport{}, fmt.Errorf(
			"alpr/onnx: %s on device %d failed to build repeatedly; next attempt in %s",
			spec.Role, spec.Device, wait.Round(time.Second))
	}

	reply := make(chan buildResult, 1)
	select {
	case b.requests <- buildRequest{kind: kindBuild, spec: spec, reply: reply, ctx: ctx}:
	case <-ctx.Done():
		return nil, engine.ModelReport{}, ctx.Err()
	}
	select {
	case res := <-reply:
		b.recordOutcome(spec, res.err)
		return res.runner, res.report, res.err
	case <-ctx.Done():
		return nil, engine.ModelReport{}, ctx.Err()
	}
}

// Destroy releases a session, on the main thread, symmetrically with creation.
func (b *Builder) Destroy(ctx context.Context, runner any) error {
	reply := make(chan buildResult, 1)
	select {
	case b.requests <- buildRequest{kind: kindDestroy, built: runner, reply: reply, ctx: ctx}:
	case <-ctx.Done():
		return ctx.Err()
	}
	select {
	case <-reply:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (b *Builder) waitServing(ctx context.Context) error {
	select {
	case <-b.serving:
		return nil
	case <-ctx.Done():
		return fmt.Errorf("alpr/onnx: builder is not serving: %w", ctx.Err())
	}
}

func (b *Builder) backoffRemaining(spec SessionSpec) (time.Duration, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	f := b.failures[spec.key()]
	if f == nil || f.consecutive == 0 {
		return 0, false
	}
	if d := time.Until(f.nextAttempt); d > 0 {
		return d, true
	}
	return 0, false
}

func (b *Builder) recordOutcome(spec SessionSpec, err error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	key := spec.key()
	if err == nil {
		delete(b.failures, key)
		return
	}
	f := b.failures[key]
	if f == nil {
		f = &failureState{}
		b.failures[key] = f
	}
	f.consecutive++
	// Exponential with jitter, floored at 30s from the third failure: past that the cause is
	// almost never transient, and retrying faster only takes the serial loop away from devices
	// that still work.
	wait := time.Duration(1<<min(f.consecutive, 5)) * time.Second
	if f.consecutive >= 3 && wait < 30*time.Second {
		wait = 30 * time.Second
	}
	jitter := time.Duration(rand.Int64N(int64(wait / 4)))
	f.nextAttempt = time.Now().Add(wait + jitter)
}

func (b *Builder) build(spec SessionSpec) (any, engine.ModelReport, error) {
	report := engine.ModelReport{Role: spec.Role, Device: spec.Device, SHA256: short(spec.Contract.SHA256)}
	if w := checkFormat(spec.Contract.Path, spec.Device); w != "" {
		report.Warnings = append(report.Warnings, w)
	}

	opts, err := b.sessionOptions(spec)
	if err != nil {
		return nil, report, err
	}
	defer opts.Destroy()

	start := time.Now()
	var runner any
	var placed placement

	err = capturePlacement(spec.Verbose, &placed, func() error {
		var berr error
		switch spec.Role {
		case RoleDetector:
			runner, berr = newDetectorRunner(spec.Contract, opts, spec.Device)
		case RoleOCR:
			runner, berr = newOCRRunner(spec.Contract, opts, spec.Device)
		default:
			berr = fmt.Errorf("unknown model role %q", spec.Role)
		}
		return berr
	})
	if err != nil {
		return nil, report, err
	}
	report.BuildMS = msSince(start)
	report.NodesByProvider = placed.nodes
	// A fallback is a GPU session's node landing on the CPU. A CPU session runs every node there
	// by design, and reporting all of them as fallen back made a CPU-only deployment warn about
	// the GPU it does not have.
	if spec.Device >= 0 {
		report.CPUFallback = placed.fallback()
	}

	// TensorRT asked for and handed nothing: the session runs on CUDA in FP32, reads the same
	// plates, and loses the whole point of the configuration without a single error. The usual
	// cause is a TensorRT library the provider could not load.
	if b.cfg.TensorRT.Enabled && spec.Device >= 0 && len(placed.nodes) > 0 && placed.nodes[trtProvider] == 0 {
		msg := fmt.Sprintf("%s on device %d: ALPR_TENSORRT is on and no node went to TensorRT", spec.Role, spec.Device)
		if b.cfg.RequireAllCUDA {
			destroyRunner(runner)
			return nil, report, errors.New("alpr/onnx: " + msg + "; ALPR_REQUIRE_ALL_CUDA refuses it")
		}
		report.Warnings = append(report.Warnings, msg)
	}

	// Checked before the warmup: a session that will be refused should not cost the cuDNN
	// algorithm search first.
	if b.cfg.RequireAllCUDA && spec.Device >= 0 && spec.Verbose {
		if len(placed.nodes) == 0 {
			report.Warnings = append(report.Warnings, fmt.Sprintf(
				"node placement of %s on device %d was not captured, so ALPR_REQUIRE_ALL_CUDA "+
					"could not check it. ALPR_WARMUP_MAX_STEADY_MS is the remaining guard against a "+
					"CPU fallback.", spec.Role, spec.Device))
		} else if len(report.CPUFallback) > 0 {
			destroyRunner(runner)
			return nil, report, fmt.Errorf(
				"alpr/onnx: %s on device %d runs %s on the CPU, and ALPR_REQUIRE_ALL_CUDA refuses it.\n"+
					"    The CUDA provider has no kernel for these operators. QLinear*, QGemm and "+
					"Nhwc* come from a model quantized for the CPU -- the .ort files the mobile app "+
					"ships, or any QOperator export; a DequantizeLinear on the CPU comes from a static "+
					"INT8 export, whose int32 biases only TensorRT runs. %s is %s. On the GPU use "+
					"the weight-only INT8 .onnx or the FP16 .onnx (docs/MODELS.md)",
				spec.Role, spec.Device, describeOps(report.CPUFallback), roleVariable(spec.Role),
				spec.Contract.Path)
		}
	}

	first, steady, err := b.warmup(runner)
	if err != nil {
		destroyRunner(runner)
		return nil, report, err
	}
	report.FirstRunMS, report.SteadyRunMS = first, steady

	if b.cfg.WarmupMaxSteadyMS > 0 && steady > b.cfg.WarmupMaxSteadyMS {
		destroyRunner(runner)
		return nil, report, fmt.Errorf(
			"alpr/onnx: %s on device %d takes %.1fms warm, over the %.1fms limit.\n"+
				"    A steady-state pass this slow is the signature of a CPU fallback; check "+
				"the node placement report",
			spec.Role, spec.Device, steady, b.cfg.WarmupMaxSteadyMS)
	}

	b.mu.Lock()
	b.destroyers = append(b.destroyers, registered{runner: runner})
	b.reports = append(b.reports, report)
	b.mu.Unlock()

	return runner, report, nil
}

func (b *Builder) sessionOptions(spec SessionSpec) (*ort.SessionOptions, error) {
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("alpr/onnx: session options: %w", err)
	}

	// Every session is quiet except the first of each (role, device), which is asked to log
	// node placement. Verbose logging on every session would flood the log at fleet scale, and
	// the placement of the second session on a device is the same as the first.
	level := ort.LoggingLevel(ort.LoggingLevelWarning)
	if spec.Verbose {
		level = ort.LoggingLevelVerbose
	}
	if err := opts.SetLogSeverityLevel(level); err != nil {
		opts.Destroy()
		return nil, fmt.Errorf("alpr/onnx: log level: %w", err)
	}

	if spec.Device < 0 {
		return opts, nil // CPU
	}

	// TensorRT first: ONNX Runtime gives each node to the first provider that takes it, so the
	// CUDA provider appended below only gets what TensorRT leaves.
	if b.cfg.TensorRT.Enabled {
		if err := b.appendTensorRT(opts, spec); err != nil {
			opts.Destroy()
			return nil, err
		}
	}

	cuda, err := ort.NewCUDAProviderOptions()
	if err != nil {
		opts.Destroy()
		return nil, fmt.Errorf("alpr/onnx: cuda provider options: %w", err)
	}
	defer cuda.Destroy()

	algo := b.cfg.ConvAlgoSearch
	if algo == "" {
		algo = "EXHAUSTIVE"
	}
	// The default is already 1 on ONNX Runtime 1.14 and later; it is set explicitly because it
	// is the flag that decides whether an FP16 model reaches the tensor cores, and a default is
	// a poor place for a decision that load-bearing.
	maxWorkspace := "1"
	if b.cfg.ConvUseMaxWorkspace != nil && !*b.cfg.ConvUseMaxWorkspace {
		maxWorkspace = "0"
	}

	options := map[string]string{
		"device_id":                    strconv.Itoa(spec.Device),
		"cudnn_conv_algo_search":       algo,
		"cudnn_conv_use_max_workspace": maxWorkspace,
		// kSameAsRequested rather than the default kNextPowerOfTwo: with several sessions per
		// GPU the power-of-two arena inflates each one's reservation, and the sum is how a box
		// that fits on paper runs out of VRAM.
		"arena_extend_strategy":     "kSameAsRequested",
		"do_copy_in_default_stream": "1",
	}
	if b.cfg.PreferNHWC {
		options["prefer_nhwc"] = "1"
	}
	if b.cfg.GPUMemLimit > 0 {
		options["gpu_mem_limit"] = strconv.FormatInt(b.cfg.GPUMemLimit, 10)
	}

	if err := cuda.Update(options); err != nil {
		opts.Destroy()
		return nil, fmt.Errorf("alpr/onnx: cuda options: %w", err)
	}
	if err := opts.AppendExecutionProviderCUDA(cuda); err != nil {
		opts.Destroy()
		return nil, fmt.Errorf("alpr/onnx: append CUDA provider for device %d: %w", spec.Device, err)
	}
	return opts, nil
}

// warmup runs the session a few times before it is allowed to serve traffic.
//
// The input is pseudo-random rather than zeroed, and that is deliberate twice over: some kernels
// take a degenerate path on all-zero input, and a zero frame produces no detections at all, so a
// zero warmup never exercises the path that actually matters.
func (b *Builder) warmup(runner any) (firstMS, steadyMS float64, err error) {
	var fill []float32
	var run func() error

	switch r := runner.(type) {
	case *detectorRunner:
		fill = r.inputBuffer()
		run = func() error { _, _, e := r.run(); return e }
	case *ocrRunner:
		fill = r.inputBuffer()
		run = func() error { _, e := r.run(); return e }
	default:
		return 0, 0, fmt.Errorf("alpr/onnx: cannot warm up %T", runner)
	}

	// Deterministic, so two warmups on the same machine are comparable.
	rng := rand.New(rand.NewPCG(1, 2))
	for i := range fill {
		fill[i] = rng.Float32()
	}

	start := time.Now()
	if err := run(); err != nil {
		return 0, 0, fmt.Errorf("alpr/onnx: first inference failed: %w", err)
	}
	firstMS = msSince(start)

	var total float64
	runs := b.cfg.WarmupRuns - 1
	for i := 0; i < runs; i++ {
		s := time.Now()
		if err := run(); err != nil {
			return firstMS, 0, fmt.Errorf("alpr/onnx: warmup inference failed: %w", err)
		}
		total += msSince(s)
	}
	if runs > 0 {
		steadyMS = total / float64(runs)
	}
	return firstMS, steadyMS, nil
}

// Reports returns what was built and where it ran.
func (b *Builder) Reports() []engine.ModelReport {
	b.mu.Lock()
	defer b.mu.Unlock()
	return append([]engine.ModelReport(nil), b.reports...)
}

func (b *Builder) shutdown() {
	b.closeOnce.Do(func() {
		b.mu.Lock()
		pending := b.destroyers
		b.destroyers = nil
		b.mu.Unlock()

		// Reverse order, then the environment. Destroying the environment first would be a
		// use-after-free: it dlcloses the library, taking the CUDA context with it while
		// sessions still reference both.
		for i := len(pending) - 1; i >= 0; i-- {
			destroyRunner(pending[i].runner)
		}
		_ = ort.DestroyEnvironment()
	})
}

// forget drops a runner from the shutdown registry.
//
// Without it a session destroyed through Destroy is destroyed a second time when Serve returns
// and walks everything it built. That is survivable with onnxruntime_go v1.31.0 -- Destroy zeroes
// the handle and a released tensor falls through to ReleaseValue(nullptr), which is a no-op -- but
// it is survivable by accident, in someone else's library, and a version that tightens it turns a
// clean shutdown into a crash on the way out.
func (b *Builder) forget(runner any) {
	b.mu.Lock()
	defer b.mu.Unlock()
	for i, fn := range b.destroyers {
		if fn.runner == runner {
			b.destroyers = append(b.destroyers[:i], b.destroyers[i+1:]...)
			return
		}
	}
}

func destroyRunner(runner any) {
	switch r := runner.(type) {
	case *detectorRunner:
		r.destroy()
	case *ocrRunner:
		r.destroy()
	}
}

func msSince(t time.Time) float64 { return float64(time.Since(t).Microseconds()) / 1000.0 }

// describeOps lists operator counts in a stable order: "QGemm x2, QLinearConv x186".
func describeOps(ops map[string]int) string {
	names := make([]string, 0, len(ops))
	for op := range ops {
		names = append(names, op)
	}
	sort.Strings(names)
	parts := make([]string, 0, len(names))
	for _, op := range names {
		parts = append(parts, fmt.Sprintf("%s x%d", op, ops[op]))
	}
	return strings.Join(parts, ", ")
}

// roleVariable names the environment variable that chose the model of a role.
func roleVariable(role string) string {
	if role == RoleOCR {
		return "OCR_MODEL_PATH"
	}
	return "DETECTOR_MODEL_PATH"
}
