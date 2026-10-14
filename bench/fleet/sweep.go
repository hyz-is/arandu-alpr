//go:build cuda

package main

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"sync"
	"time"

	"github.com/hyz-is/arandu-alpr"
	"github.com/hyz-is/arandu-alpr/bench/measure"
	"github.com/hyz-is/arandu-alpr/bench/report"
	"github.com/hyz-is/arandu-alpr/camera"
	"github.com/hyz-is/arandu-alpr/contract"
	"github.com/hyz-is/arandu-alpr/engine"
	"github.com/hyz-is/arandu-alpr/engine/onnx"
	"github.com/hyz-is/arandu-alpr/imaging"
	"github.com/hyz-is/arandu-alpr/metrics"
	"github.com/hyz-is/arandu-alpr/source"
)

// sweep runs every (devices, sessions) combination, raising the camera count until the SLO
// breaks.
//
// The engine is built ONCE per (devices, sessions) and reused across every camera count in that
// combination. Rebuilding it per step would put an eight-to-fifteen second session build inside
// each warmup, and a warmup that is mostly cold start does not settle the fleet -- which is the
// only thing a warmup is for.
func sweep(ctx context.Context, builder *onnx.Builder, cfg sweepConfig) error {
	gitSHA, gitDirty := measure.GitSHA(ctx, cfg.Root)
	models, err := describeModels(cfg)
	if err != nil {
		return err
	}

	for _, devices := range cfg.DeviceSets {
		for _, sessions := range cfg.SessionCounts {
			if err := ctx.Err(); err != nil {
				return nil
			}

			eng, err := onnx.New(ctx, builder, onnx.EngineConfig{
				DetectorPath:              cfg.DetectorPath,
				OCRPath:                   cfg.OCRPath,
				DetectorSessionsPerDevice: sessions,
				OCRSessionsPerDevice:      sessions + cfg.OCRExtra,
				BorrowTimeout:             cfg.BorrowTimeout,
				Postprocess:               postprocessConfig(),
			})
			if err != nil {
				return fmt.Errorf("building the engine for devices %v with %d sessions: %w", devices, sessions, err)
			}

			// A silent CPU fallback is the failure this check exists for. It does not error, it
			// gets slow, and slow is then diagnosed as a network problem or as hardware that has
			// been outgrown -- so the sweep would report a capacity that is really a placement
			// bug.
			if providers := eng.Providers(); cfg.RequireAllCUDA && !providers.AllCUDA {
				_ = eng.Close(ctx)
				return fmt.Errorf("%d of the graph's nodes landed on the CPU; the sweep would "+
					"measure a placement bug and call it capacity", providers.CPUNodes)
			}

			capacity := 0
			for cameras := cfg.Start; cameras <= cfg.Max; cameras += cfg.Step {
				if ctx.Err() != nil {
					break
				}

				run, err := measureStep(ctx, cfg, eng, devices, sessions, cameras)
				if err != nil {
					_ = eng.Close(ctx)
					return err
				}
				run.GitSHA, run.GitDirty = gitSHA, gitDirty
				run.Models = models

				if err := report.AppendFile(cfg.Out, run); err != nil {
					_ = eng.Close(ctx)
					return err
				}
				summarise(run)

				if run.SLO.Crossed {
					break
				}
				capacity = cameras
			}

			// The headline, and the reason the sweep exists. It is printed to stderr so that
			// stdout stays a clean stream of JSON lines when -out is not given.
			fmt.Fprintf(os.Stderr,
				"\n==> devices %v, %d detector + %d ocr sessions each: %d cameras inside a p95 of %.2fs\n\n",
				devices, sessions, sessions+cfg.OCRExtra, capacity, cfg.SLOP95)

			if err := eng.Close(ctx); err != nil {
				return fmt.Errorf("closing the engine: %w", err)
			}
		}
	}
	return nil
}

// measureStep runs one camera count: publish, warm up, measure, tear down.
func measureStep(
	ctx context.Context,
	cfg sweepConfig,
	eng engine.Engine,
	devices []int,
	sessions, cameras int,
) (report.Run, error) {
	pubs, err := startPublishers(ctx, cfg, cameras)
	if err != nil {
		return report.Run{}, err
	}
	defer pubs.stop()

	phased := measure.NewPhased()
	sink := &countingSink{}

	encoder, err := imaging.NewEncoder(cfg.JPEGQuality)
	if err != nil {
		return report.Run{}, err
	}

	dialer := source.NewFFmpegDialer(source.Config{
		// Bounded, not serialised. One unreachable camera must fail beside the others rather
		// than in front of them: the Python service serialised opening behind a global lock and
		// a single dead camera held the whole fleet behind its 30-second connect timeout.
		MaxConcurrentOpens: 16,
		// Hardware decode, when the card has it. The sweep measures cameras-per-GPU, and with
		// CPU decode that number can be bounded by the host rather than by the card -- which
		// would make the answer a fact about this server, not about the hardware.
		HWAccel: cfg.HWAccel,
		// The sweep is an operator's measurement, run beside the cameras or against local
		// clips: both are addresses the production guard refuses on purpose.
		AllowPrivate: true,
	})

	runCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	var wg sync.WaitGroup
	for i, url := range pubs.urls() {
		// A Scaler per camera, because it holds the bilinear weight tables for that camera's
		// geometry and is not safe for concurrent use. Sharing one would be the cheapest
		// possible way to make every focus measurement wrong.
		scaler, err := imaging.NewScaler(engine.DetectorInputSize)
		if err != nil {
			return report.Run{}, err
		}

		cam, err := camera.New(camera.Config{
			Spec: contract.CameraSpec{
				ID:                fmt.Sprintf("bench%03d", i),
				URL:               url,
				Enabled:           true,
				MinDetectInterval: cfg.DetectEvery,
				Brightness:        100,
				Contrast:          100,
			},
			Dialer:           dialerAdapter{dialer},
			Engine:           eng,
			Gate:             alpr.DefaultGate(),
			Dedup:            alpr.NewDeduper(cfg.DedupCooldown, cfg.DedupDistance),
			Sink:             sink,
			Metrics:          phased,
			Adjust:           imaging.AdjustBrightnessContrast,
			Focus:            scaler,
			Encoder:          encoder,
			IncludeCropImage: cfg.IncludeCrop,
			IncludeFullImage: cfg.IncludeFull,
			DetectTimeout:    cfg.DetectTimeout,
			RecognizeTimeout: cfg.OCRTimeout,
		})
		if err != nil {
			return report.Run{}, err
		}

		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = cam.Run(runCtx)
		}()
	}

	fmt.Fprintf(os.Stderr, "%d cameras: warming up for %s...\n", cameras, cfg.Warmup)
	if !sleep(runCtx, cfg.Warmup) {
		return report.Run{}, ctx.Err()
	}

	// The samplers start with the window, not with the fleet: a VRAM peak recorded during
	// session build is a peak the running fleet never reaches, and reporting it would shrink
	// how many sessions somebody believes fit.
	gpu := measure.NewGPUSampler(runCtx, time.Second)
	proc := measure.NewProcSampler(runCtx, time.Second)
	defer gpu.Stop()
	defer proc.Stop()

	phased.Arm()
	started := time.Now()
	fmt.Fprintf(os.Stderr, "%d cameras: measuring for %s...\n", cameras, cfg.Window)
	if !sleep(runCtx, cfg.Window) {
		return report.Run{}, ctx.Err()
	}
	elapsed := time.Since(started)

	// Read everything BEFORE the cameras are stopped. Cancelling a camera cancels a context it
	// may be inside Detect with, and the engine then returns context.Canceled -- which the loop
	// counts as an inference error. That is teardown noise, and letting it into the window
	// would put a fault in every report.
	samples := phased.Window()
	gpu.Stop()
	proc.Stop()
	gpuStats := gpu.Stats()
	procStats := proc.Stats()

	cancel()
	wg.Wait()

	return assemble(cfg, devices, sessions, cameras, elapsed, samples, gpuStats, procStats, gpu.Note(), sink), nil
}

// assemble turns one window's snapshot into the line that gets written.
func assemble(
	cfg sweepConfig,
	devices []int,
	sessions, cameras int,
	elapsed time.Duration,
	samples []metrics.Sample,
	gpuStats []measure.GPUStats,
	procStats measure.ProcStats,
	gpuNote string,
	sink *countingSink,
) report.Run {
	seconds := elapsed.Seconds()
	frames := report.CounterTotal(samples, metrics.FramesReceived)
	detections := report.CounterTotal(samples, metrics.DetectedPlates)
	drops := report.DropsByReason(samples)

	var dropTotal float64
	for _, v := range drops {
		dropTotal += v
	}

	pipeline := report.FromSnapshot(samples, metrics.PipelineDuration)
	busyShare := 0.0
	if frames > 0 {
		busyShare = drops[metrics.DropInferenceBusy] / frames
	}

	hostname, _ := os.Hostname()

	run := report.Run{
		Tier:      "B",
		StartedAt: time.Now().Add(-elapsed),
		Elapsed:   seconds,
		Host: report.Host{
			Hostname: hostname,
			OS:       runtime.GOOS,
			Arch:     runtime.GOARCH,
			NumCPU:   runtime.NumCPU(),
			GoVer:    runtime.Version(),
		},
		Config: report.Config{
			Cameras:             cameras,
			Devices:             devices,
			SessionsPerDevice:   sessions,
			OCRSessions:         sessions + cfg.OCRExtra,
			BorrowTimeoutMS:     ms(cfg.BorrowTimeout),
			MinDetectIntervalMS: ms(cfg.DetectEvery),
			MaxIngestLagMS:      ms(camera.DefaultMaxIngestLag),
			DetectTimeoutMS:     ms(cfg.DetectTimeout),
			RecognizeTimeoutMS:  ms(cfg.OCRTimeout),
			HWAccel:             cfg.HWAccel,
			IncludeFullImage:    cfg.IncludeFull,
			IncludeCropImage:    cfg.IncludeCrop,
			JPEGQuality:         cfg.JPEGQuality,
			Gate:                alpr.DefaultGate(),
			Postprocess:         postprocessConfig(),
			Dedup: report.Dedup{
				CooldownSeconds: cfg.DedupCooldown.Seconds(),
				MaxDistance:     cfg.DedupDistance,
			},
			WarmupSeconds: cfg.Warmup.Seconds(),
			WindowSeconds: cfg.Window.Seconds(),
			Clips:         cfg.Clips,
			RTSPBase:      cfg.RTSPServer,
			// startPublishers always passes -re. It is recorded rather than assumed because a
			// line whose input was not paced describes a burst, not a fleet.
			Realtime: true,
		},
		Throughput: report.Throughput{
			FramesReceived:     frames,
			FramesPerSecond:    perSecond(frames, seconds),
			DetectionsTotal:    detections,
			DetectionsPerMin:   perSecond(detections, seconds) * 60,
			DeduplicatedTotal:  report.CounterTotal(samples, metrics.Deduplicated),
			FramesDroppedTotal: dropTotal,
		},
		Latency: map[string]report.Quantiles{
			"pipeline":     pipeline,
			"detector":     report.FromSnapshot(samples, metrics.DetectorDuration),
			"ocr":          report.FromSnapshot(samples, metrics.OCRDuration),
			"session_wait": report.FromSnapshot(samples, metrics.SessionWait),
		},
		Drops: drops,
		Process: report.Process{
			PeakRSSMiB:   procStats.PeakRSSMiB,
			SteadyRSSMiB: procStats.SteadyRSSMiB,
			CPUPercent:   procStats.CPUPercent,
			Goroutines:   procStats.Goroutines,
			HeapMiB:      procStats.HeapMiB,
		},
		SLO: report.SLO{
			Metric:                metrics.PipelineDuration,
			P95BudgetSeconds:      cfg.SLOP95,
			P95ObservedSeconds:    pipeline.P95,
			MaxBusyDropRatio:      cfg.SLOMaxBusyDrops,
			ObservedBusyDropRatio: busyShare,
			// Two conditions, and the second one is not decoration. A fleet can hold its p95 by
			// discarding most of what it sees, and a latency-only bar calls that a pass.
			Crossed: pipeline.P95 > cfg.SLOP95 || busyShare > cfg.SLOMaxBusyDrops,
		},
	}

	for _, g := range gpuStats {
		run.GPUs = append(run.GPUs, report.GPU{
			Index:       g.Index,
			Name:        g.Name,
			PeakMiB:     g.PeakMiB,
			SteadyMiB:   g.SteadyMiB,
			TotalMiB:    g.TotalMiB,
			Samples:     g.Samples,
			UtilPercent: g.UtilMean,
			UtilPeakPct: g.UtilPeak,
		})
	}
	if gpuNote != "" {
		run.Notes = append(run.Notes, gpuNote)
	}
	if procStats.Note != "" {
		run.Notes = append(run.Notes, procStats.Note)
	}
	if pipeline.Count < 100 {
		run.Notes = append(run.Notes, fmt.Sprintf(
			"the end-to-end p95 rests on %d observations, which is too few to read as a percentile; "+
				"lengthen -window or lower -detect-every", pipeline.Count))
	}
	if delivered := sink.count(); float64(delivered) != detections {
		run.Notes = append(run.Notes, fmt.Sprintf(
			"the sink received %d readings while the counter recorded %.0f; the difference is "+
				"readings delivered outside the measurement window", delivered, detections))
	}
	return run
}

// summarise prints the one line a human watching the sweep needs.
func summarise(run report.Run) {
	verdict := "within SLO"
	if run.SLO.Crossed {
		verdict = "OVER"
	}
	fmt.Fprintf(os.Stderr,
		"  %3d cameras  %6.1f fps  %6.0f det/min  p50 %5.0fms  p95 %5.0fms  p99 %5.0fms  busy-drop %4.1f%%  %s\n",
		run.Config.Cameras,
		run.Throughput.FramesPerSecond,
		run.Throughput.DetectionsPerMin,
		run.Latency["pipeline"].P50*1000,
		run.Latency["pipeline"].P95*1000,
		run.Latency["pipeline"].P99*1000,
		run.SLO.ObservedBusyDropRatio*100,
		verdict,
	)
}

// describeModels hashes every artefact the sweep will run.
//
// Done once, before anything is measured, so a missing or unreadable model fails in a second
// rather than after the first ten-minute window.
func describeModels(cfg sweepConfig) ([]report.Model, error) {
	out := make([]report.Model, 0, 2)
	for _, m := range []struct{ role, path string }{
		{onnx.RoleDetector, cfg.DetectorPath},
		{onnx.RoleOCR, cfg.OCRPath},
	} {
		sum, size, err := measure.FileSHA256(m.path)
		if err != nil {
			return nil, fmt.Errorf("hashing the %s model: %w", m.role, err)
		}
		out = append(out, report.Model{Role: m.role, Path: m.path, SHA256: sum, Bytes: size})
	}
	return out, nil
}

// countingSink is the application half of the boundary. It counts and returns, because anything
// slower would be measured as pipeline latency: Sink.Detect runs on the camera's own goroutine
// and blocks it.
type countingSink struct {
	mu         sync.Mutex
	detections int
}

func (s *countingSink) Detect(_ context.Context, _ contract.Detection) error {
	s.mu.Lock()
	s.detections++
	s.mu.Unlock()
	return nil
}

func (s *countingSink) count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.detections
}

// dialerAdapter narrows source.FFmpegDialer to the interface the camera loop asks for.
type dialerAdapter struct{ inner *source.FFmpegDialer }

func (d dialerAdapter) Dial(ctx context.Context, spec contract.CameraSpec) (camera.Stream, error) {
	return d.inner.Dial(ctx, spec)
}

// sleep waits, reporting false if the context ended first.
func sleep(ctx context.Context, d time.Duration) bool {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-timer.C:
		return true
	case <-ctx.Done():
		return false
	}
}

func ms(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }

func perSecond(total, seconds float64) float64 {
	if seconds <= 0 {
		return 0
	}
	return total / seconds
}
