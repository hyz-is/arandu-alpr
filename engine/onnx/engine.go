package onnx

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/hyz-is/arandu-alpr"
	"github.com/hyz-is/arandu-alpr/contract"
	"github.com/hyz-is/arandu-alpr/engine"
	"github.com/hyz-is/arandu-alpr/imaging"
	"github.com/hyz-is/arandu-alpr/metrics"
)

// EngineConfig configures the ONNX-backed engine.
type EngineConfig struct {
	DetectorPath string
	OCRPath      string

	// SessionsPerDevice sizes each pool.
	//
	// It is sized by what saturates a GPU, NOT by the number of cameras -- two to four is
	// usually enough. The Python service built one session per camera, and with 39 cameras a
	// restart meant 39 serial eight-to-fifteen-second builds: roughly 470 seconds of a service
	// that was up and answering nothing, which is exactly the p90 measured in production.
	DetectorSessionsPerDevice int
	OCRSessionsPerDevice      int

	// BorrowTimeout caps how long a frame waits for a session before it is dropped.
	//
	// Dropping is the point. Under saturation the service loses frames, which a fleet recovers
	// from, instead of growing a queue, which it does not: every queued frame is older than the
	// one behind it and the answer arrives after it stopped being useful.
	BorrowTimeout time.Duration

	Postprocess engine.PostprocessConfig
	Metrics     metrics.Registry
}

func (c EngineConfig) withDefaults() EngineConfig {
	if c.DetectorSessionsPerDevice <= 0 {
		c.DetectorSessionsPerDevice = 2
	}
	if c.OCRSessionsPerDevice <= 0 {
		c.OCRSessionsPerDevice = 3
	}
	if c.BorrowTimeout <= 0 {
		c.BorrowTimeout = 250 * time.Millisecond
	}
	if c.Postprocess.InputSize == 0 {
		c.Postprocess = engine.DefaultPostprocess()
	}
	if c.Metrics == nil {
		c.Metrics = metrics.Discard()
	}
	return c
}

// detectorSession is a runner plus the per-frame scratch that only its holder may touch.
//
// The postprocessor travels WITH the session rather than being shared, because it holds reusable
// buffers and is not safe for concurrent use. Bundling them is what makes "hold the session" and
// "hold the buffers" the same statement. The letterbox is the exception, and on purpose: it is
// built before a session is borrowed, from the engine's own scratch pool (see Detect).
type detectorSession struct {
	runner *detectorRunner
	post   *engine.Postprocessor
}

type ocrSession struct {
	runner *ocrRunner
	scaler *imaging.Scaler
	crop   []float32
}

// Engine is the ONNX-backed engine.Engine.
type Engine struct {
	cfg     EngineConfig
	builder *Builder

	detectorContract ModelContract
	ocrContract      ModelContract

	devices []int
	next    atomic.Uint64

	detectors map[int]*alpr.SessionPool[*detectorSession]
	ocrs      map[int]*alpr.SessionPool[*ocrSession]

	ready atomic.Bool

	// scratch holds the letterbox buffers that are built before a session is borrowed. It is a
	// field and not package state, and it holds as many as there are frames in flight.
	scratch sync.Pool

	mu      sync.Mutex
	reports []engine.ModelReport

	// Metric handles, bound once. Formatting label values per frame allocates and escapes; a
	// bound handle is a pointer dereference, and this path runs at fleet frame rate.
	mDetect      metrics.Histogram
	mOCR         metrics.Histogram
	mWait        metrics.Histogram
	mInUse       metrics.Gauge
	mErrDetect   metrics.Counter
	mErrOCR      metrics.Counter
	mErrBuild    metrics.Counter
	modelVersion metrics.Gauge
}

var _ engine.Engine = (*Engine)(nil)

// New validates the models, builds the pools and warms one session of each role per device.
//
// It must be called after Builder.Init and while Builder.Serve is running, because every session
// it creates is created on the builder's thread.
func New(ctx context.Context, builder *Builder, cfg EngineConfig) (*Engine, error) {
	cfg = cfg.withDefaults()

	// Validate before building anything. A model that does not match must stop the boot, not
	// produce readings nobody can tell are wrong.
	det, derr := ValidateDetector(cfg.DetectorPath)
	ocr, oerr := ValidateOCR(cfg.OCRPath)
	if err := errors.Join(derr, oerr); err != nil {
		return nil, err
	}
	// The model says what square it takes. The letterbox and the decoder follow it, so a 608
	// export cannot be fed a 640 letterbox and decoded against the wrong anchor grid.
	cfg.Postprocess.InputSize = det.InputSize

	devices := builder.cfg.Devices
	if len(devices) == 0 {
		devices = []int{-1} // CPU
	}

	e := &Engine{
		cfg:              cfg,
		builder:          builder,
		detectorContract: det,
		ocrContract:      ocr,
		devices:          devices,
		detectors:        map[int]*alpr.SessionPool[*detectorSession]{},
		ocrs:             map[int]*alpr.SessionPool[*ocrSession]{},
	}
	e.bindMetrics()

	for _, dev := range devices {
		dp, err := e.newDetectorPool(dev)
		if err != nil {
			return nil, err
		}
		op, err := e.newOCRPool(dev)
		if err != nil {
			return nil, err
		}
		e.detectors[dev], e.ocrs[dev] = dp, op
	}

	if err := e.warmOnePerDevice(ctx); err != nil {
		return nil, err
	}
	e.ready.Store(true)
	return e, nil
}

func (e *Engine) bindMetrics() {
	r := e.cfg.Metrics
	e.mDetect = r.Histogram(metrics.DetectorDuration, metrics.DefaultBuckets(), "gpu").With("all")
	e.mOCR = r.Histogram(metrics.OCRDuration, metrics.DefaultBuckets(), "gpu").With("all")
	e.mWait = r.Histogram(metrics.SessionWait, metrics.DefaultBuckets(), "gpu").With("all")
	e.mInUse = r.Gauge(metrics.SessionPoolInUse, "gpu").With("all")
	errs := r.Counter(metrics.InferenceErrors, "gpu", "stage")
	e.mErrDetect = errs.With("all", metrics.StageDetect)
	e.mErrOCR = errs.With("all", metrics.StageOCR)
	e.mErrBuild = errs.With("all", metrics.StageSessionBuild)
	e.modelVersion = r.Gauge(metrics.ModelVersion, "detector", "ocr").
		With(short(e.detectorContract.SHA256), short(e.ocrContract.SHA256))
	e.modelVersion.Set(1)
}

func (e *Engine) newDetectorPool(device int) (*alpr.SessionPool[*detectorSession], error) {
	first := true
	var firstMu sync.Mutex

	return alpr.NewSessionPool(e.cfg.DetectorSessionsPerDevice, func() (*detectorSession, error) {
		// Only the first session of each (role, device) is built verbosely, because verbose is
		// how node placement gets reported and the second session on a device places the same
		// way as the first. Every session verbose would flood the log at fleet scale.
		firstMu.Lock()
		verbose := first
		first = false
		firstMu.Unlock()

		built, report, err := e.builder.Build(context.Background(), SessionSpec{
			Role: RoleDetector, Contract: e.detectorContract, Device: device, Verbose: verbose,
		})
		if err != nil {
			e.mErrBuild.Inc()
			return nil, err
		}
		e.record(report)

		post, err := engine.NewPostprocessor(e.cfg.Postprocess)
		if err != nil {
			return nil, err
		}
		return &detectorSession{runner: built.(*detectorRunner), post: post}, nil
	})
}

func (e *Engine) newOCRPool(device int) (*alpr.SessionPool[*ocrSession], error) {
	first := true
	var firstMu sync.Mutex

	return alpr.NewSessionPool(e.cfg.OCRSessionsPerDevice, func() (*ocrSession, error) {
		firstMu.Lock()
		verbose := first
		first = false
		firstMu.Unlock()

		built, report, err := e.builder.Build(context.Background(), SessionSpec{
			Role: RoleOCR, Contract: e.ocrContract, Device: device, Verbose: verbose,
		})
		if err != nil {
			e.mErrBuild.Inc()
			return nil, err
		}
		e.record(report)

		scaler, err := imaging.NewScaler(e.cfg.Postprocess.InputSize)
		if err != nil {
			return nil, err
		}
		return &ocrSession{runner: built.(*ocrRunner), scaler: scaler}, nil
	})
}

// warmOnePerDevice pays the first-inference cost before traffic arrives.
//
// The first pass through a session costs seconds -- cuDNN's algorithm search, the CUDA module
// load -- and paying it under load looks like a camera fault rather than a cold start.
func (e *Engine) warmOnePerDevice(ctx context.Context) error {
	for _, dev := range e.devices {
		d, release, err := e.detectors[dev].Borrow(ctx)
		if err != nil {
			return fmt.Errorf("alpr/onnx: warming detector on device %d: %w", dev, err)
		}
		_ = d
		release()

		o, releaseOCR, err := e.ocrs[dev].Borrow(ctx)
		if err != nil {
			return fmt.Errorf("alpr/onnx: warming ocr on device %d: %w", dev, err)
		}
		_ = o
		releaseOCR()
	}
	return nil
}

func (e *Engine) record(r engine.ModelReport) {
	e.mu.Lock()
	e.reports = append(e.reports, r)
	e.mu.Unlock()
}

// pick chooses a device for one call, round-robin.
//
// Detection and recognition of the same frame need not land on the same GPU: every tensor in
// this binding is CPU-backed, so the data returns to host memory between the two passes anyway.
func (e *Engine) pick() int {
	if len(e.devices) == 1 {
		return e.devices[0]
	}
	return e.devices[int(e.next.Add(1)-1)%len(e.devices)]
}

func (e *Engine) Ready() error {
	if !e.ready.Load() {
		return engine.ErrNotReady
	}
	return nil
}

// letterboxScratch is what one Detect call needs to letterbox a frame without holding a session:
// a scaler, which keeps the resize weights of the geometry it saw last, and the tensor it fills.
// Whoever holds the scratch is the only one using it.
type letterboxScratch struct {
	scaler *imaging.Scaler
	tensor []float32
}

// getScratch returns a scratch from the pool, or builds one the size of the detector's input.
func (e *Engine) getScratch() (*letterboxScratch, error) {
	if s, ok := e.scratch.Get().(*letterboxScratch); ok {
		return s, nil
	}
	scaler, err := imaging.NewScaler(e.cfg.Postprocess.InputSize)
	if err != nil {
		return nil, err
	}
	return &letterboxScratch{scaler: scaler, tensor: make([]float32, scaler.TensorLen())}, nil
}

// Detect runs the detector over one frame.
func (e *Engine) Detect(ctx context.Context, f engine.Frame, dst []engine.Candidate) ([]engine.Candidate, error) {
	if err := e.Ready(); err != nil {
		return dst, err
	}
	dev := e.pick()

	// The letterbox is the CPU half of detection, and it is done BEFORE a session is borrowed. A
	// session holds a GPU slot; while the host is short of CPU, a letterbox run after the borrow
	// kept the slot busy for as long as the scheduler took to give the thread a core, and the
	// GPU idled behind it. Done first, the slot is held for the copy and the inference only.
	scratch, err := e.getScratch()
	if err != nil {
		e.mErrDetect.Inc()
		return dst, err
	}
	defer e.scratch.Put(scratch)
	lb, err := scratch.scaler.Letterbox(f, scratch.tensor)
	if err != nil {
		e.mErrDetect.Inc()
		return dst, err
	}

	borrowCtx, cancel := context.WithTimeout(ctx, e.cfg.BorrowTimeout)
	defer cancel()

	waitStart := time.Now()
	sess, release, err := e.detectors[dev].Borrow(borrowCtx)
	e.mWait.ObserveSince(waitStart)
	if err != nil {
		// A deadline here is saturation, not a fault: report it as busy so the camera drops the
		// frame instead of treating it as an inference failure and reconnecting.
		if errors.Is(err, context.DeadlineExceeded) {
			return dst, engine.ErrBusy
		}
		e.mErrDetect.Inc()
		return dst, err
	}
	defer release()
	e.mInUse.Add(1)
	defer e.mInUse.Add(-1)

	// The session's input tensor is the memory ONNX Runtime reads, so the finished letterbox is
	// copied into it: one sequential copy of 4.9 MB at 640x640, against the resize it replaces.
	if n := copy(sess.runner.inputBuffer(), scratch.tensor); n != len(scratch.tensor) {
		e.mErrDetect.Inc()
		return dst, fmt.Errorf("alpr/onnx: detector input holds %d values, the letterbox has %d", n, len(scratch.tensor))
	}

	// The context is deliberately not wired to RunOptions.Terminate. Terminate is a sticky
	// switch that has to be unset, and aborting a CUDA Run mid-stream is exactly what leaves the
	// context in the state you cannot recover from without restarting the process. Cancellation
	// governs the wait for a session, not an inference already on the GPU.
	start := time.Now()
	boxes, scores, err := sess.runner.run()
	e.mDetect.ObserveSince(start)
	if err != nil {
		e.mErrDetect.Inc()
		return dst, fmt.Errorf("alpr/onnx: detector on device %d: %w", dev, err)
	}

	return sess.post.Decode(boxes, scores, lb, dst)
}

// Recognize reads one plate.
func (e *Engine) Recognize(ctx context.Context, f engine.Frame, box contract.BBox) (engine.Reading, error) {
	if err := e.Ready(); err != nil {
		return engine.Reading{}, err
	}
	dev := e.pick()

	borrowCtx, cancel := context.WithTimeout(ctx, e.cfg.BorrowTimeout)
	defer cancel()

	waitStart := time.Now()
	sess, release, err := e.ocrs[dev].Borrow(borrowCtx)
	e.mWait.ObserveSince(waitStart)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			return engine.Reading{}, engine.ErrBusy
		}
		e.mErrOCR.Inc()
		return engine.Reading{}, err
	}
	defer release()
	e.mInUse.Add(1)
	defer e.mInUse.Add(-1)

	// The crop is taken from the ORIGINAL frame, never from the detector's 640 letterbox.
	// Cropping the letterbox throws away resolution exactly on the small, distant plates, which
	// are the ones the OCR is already least sure about.
	if err := sess.scaler.Crop(f, box, sess.runner.inputBuffer()); err != nil {
		e.mErrOCR.Inc()
		return engine.Reading{}, err
	}

	start := time.Now()
	r, err := sess.runner.run()
	e.mOCR.ObserveSince(start)
	if err != nil {
		e.mErrOCR.Inc()
		return engine.Reading{}, fmt.Errorf("alpr/onnx: ocr on device %d: %w", dev, err)
	}

	return engine.Reading{
		Plate:            r.Plate,
		CharConfidences:  r.CharConfidences,
		Confidence:       r.Confidence(),
		Region:           r.Region,
		RegionConfidence: r.RegionConfidence,
		Probabilities:    r.Probabilities,
	}, nil
}

// Providers reports where the models actually ran.
func (e *Engine) Providers() engine.ProviderReport {
	e.mu.Lock()
	reports := append([]engine.ModelReport(nil), e.reports...)
	e.mu.Unlock()

	// Only the first session of each (role, device) is built verbosely, so the others carry no
	// placement. They place the same way as the first, and counting them as "unmeasured" made
	// all_cuda false on every fleet with more than one session per device -- that is, always.
	rep := engine.ProviderReport{Models: reports}
	measured, clean := 0, true
	for _, m := range reports {
		if len(m.NodesByProvider) == 0 {
			continue
		}
		measured++
		p := placement{nodes: m.NodesByProvider}
		for _, n := range m.CPUFallback {
			rep.CPUNodes += n
		}
		if len(m.CPUFallback) > 0 || !p.onlyCUDAAndCPU() {
			clean = false
		}
	}
	rep.AllCUDA = measured > 0 && clean
	return rep
}

func (e *Engine) ModelInfo() contract.ModelInfo {
	return contract.ModelInfo{
		Detector: short(e.detectorContract.SHA256),
		OCR:      short(e.ocrContract.SHA256),
	}
}

// Close drains the pools and destroys every session, on the builder's thread.
//
// Draining first is what stops a session being destroyed while a camera is still inside Run.
// Dropping sessions instead of destroying them would leave their VRAM held until the process
// exits, which in a redeploy means the container coming up finds the GPU still occupied.
func (e *Engine) Close(ctx context.Context) error {
	e.ready.Store(false)

	var errs []error
	for _, dev := range e.devices {
		sessions, err := e.detectors[dev].Drain(ctx)
		if err != nil {
			errs = append(errs, err)
		}
		for _, s := range sessions {
			errs = append(errs, e.builder.Destroy(ctx, s.runner))
		}

		ocrs, err := e.ocrs[dev].Drain(ctx)
		if err != nil {
			errs = append(errs, err)
		}
		for _, s := range ocrs {
			errs = append(errs, e.builder.Destroy(ctx, s.runner))
		}
	}
	return errors.Join(errs...)
}
