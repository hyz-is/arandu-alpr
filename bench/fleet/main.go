//go:build cuda

// Command fleet answers one question: how many cameras does one GPU carry?
//
// It answers it by sweeping upward until a declared SLO breaks, and the crossing IS the answer.
// That is the whole design. "We ran forty and it looked fine" is not an answer -- it cannot be
// reproduced, cannot be compared to last month's, and cannot be argued with. A number produced
// by raising N until the p95 of alpr_pipeline_duration_seconds crosses a budget stated on the
// command line can be all three.
//
// # The protocol
//
// Every part of it exists because leaving it out produces a number that means something else.
//
//   - INPUT is clips replayed as RTSP with ffmpeg's -re. Without the pacing the clip arrives as
//     one burst and every rate is a rate of nothing. See publisher.go.
//   - WARMUP is discarded, and the measurement window is a separate registry rather than a
//     subtraction, because a quantile cannot be subtracted. See measure.Phased.
//   - The SWEEP raises cameras by a fixed step until the SLO breaks. The last N inside the SLO
//     is the reported capacity.
//   - The LINE carries the whole config, the git SHA and the SHA256 of every model. Two runs
//     without those cannot be compared and must not be.
//
// # Running it
//
//	mediamtx &                                  # an RTSP server for the clips
//	go build -tags cuda -o fleet ./bench/fleet
//	./fleet -clips /data/clips \
//	        -detector /models/detector.ort -ocr /models/ocr.ort \
//	        -devices 0 -start 8 -step 8 -max 96 \
//	        -slo-p95 1.0 -warmup 60s -window 10m \
//	        -out sweep.jsonl
//
// Then diff two sweeps:
//
//	go run ./bench/compare -baseline before.jsonl -current sweep.jsonl
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/hyz-is/arandu-alpr/engine"
	"github.com/hyz-is/arandu-alpr/engine/onnx"
)

// sweepConfig is everything the sweep was told to do. It is filled from flags once and then
// only read, so a step cannot quietly reconfigure the one after it.
type sweepConfig struct {
	Clips      []string
	RTSPServer string

	DetectorPath string
	OCRPath      string
	LibraryPath  string

	DeviceSets    [][]int
	SessionCounts []int
	OCRExtra      int

	Start, Step, Max int

	Warmup time.Duration
	Window time.Duration

	DetectEvery   time.Duration
	BorrowTimeout time.Duration
	DetectTimeout time.Duration
	OCRTimeout    time.Duration
	DedupCooldown time.Duration
	DedupDistance int

	IncludeCrop bool
	IncludeFull bool

	// HWAccel is recorded in the run's output as well as used, because a sweep with hardware
	// decode and one without are not the same experiment and their numbers must not be
	// compared as if they were.
	HWAccel     string
	JPEGQuality int

	SLOP95          float64
	SLOMaxBusyDrops float64
	RequireAllCUDA  bool

	Out  string
	Root string
}

func main() {
	clips := flag.String("clips", "", "directory of video clips, or a comma-separated list of files")
	rtsp := flag.String("rtsp-server", "rtsp://127.0.0.1:8554",
		"RTSP server to publish the clips to and read them back from (mediamtx, rtsp-simple-server, ...)")

	detector := flag.String("detector", "", "path to the detector .ort")
	ocr := flag.String("ocr", "", "path to the OCR .ort")
	library := flag.String("onnxruntime", libraryDefault(), "path to libonnxruntime.so; defaults to ONNXRUNTIME_LIB_PATH")

	devices := flag.String("devices", "0", "CUDA device ids to use, comma separated")
	gpuSweep := flag.Bool("gpu-sweep", false,
		"also sweep the number of GPUs, running every prefix of -devices. Off by default: it multiplies the wall clock by the number of devices")
	sessions := flag.String("sessions-per-gpu", "2",
		"detector sessions per device to sweep, comma separated. The OCR gets one more of each, which is the production ratio")
	ocrExtra := flag.Int("ocr-extra-sessions", 1,
		"how many more OCR sessions than detector sessions to build per device")

	start := flag.Int("start", 8, "cameras to begin the sweep at")
	step := flag.Int("step", 8, "how many cameras to add per step")
	maxCameras := flag.Int("max", 128, "stop the sweep here even if the SLO still holds")

	warmup := flag.Duration("warmup", 60*time.Second,
		"discarded at the start of every step, so the reported quantiles describe a settled fleet")
	window := flag.Duration("window", 10*time.Minute, "the measurement window")

	detectEvery := flag.Duration("detect-every", 200*time.Millisecond, "minimum gap between detection passes per camera")
	borrow := flag.Duration("borrow-timeout", 250*time.Millisecond, "how long a frame waits for a session before it is dropped")
	detectTimeout := flag.Duration("detect-timeout", 5*time.Second, "deadline on one detector pass")
	ocrTimeout := flag.Duration("ocr-timeout", 5*time.Second, "deadline on one OCR pass")
	dedup := flag.Duration("dedup-cooldown", 30*time.Second, "how long the same plate is suppressed")
	dedupDistance := flag.Int("dedup-distance", 1, "edit distance at which two readings are the same plate")

	includeCrop := flag.Bool("include-crop", true, "encode the plate crop on every accepted reading")
	includeFull := flag.Bool("include-full", false, "encode the whole frame on every accepted reading")
	quality := flag.Int("jpeg-quality", 70, "JPEG quality for the encoded images")

	sloP95 := flag.Float64("slo-p95", 1.0,
		"the bar, in seconds: the sweep stops when the p95 of alpr_pipeline_duration_seconds crosses it")
	sloBusy := flag.Float64("slo-max-busy-drops", 0.20,
		"the other half of the bar: the share of frames droppable as inference_busy before the fleet counts as over capacity")
	requireCUDA := flag.Bool("require-all-cuda", true,
		"abort if any node of any model lands on the CPU. A silent CPU fallback benchmarks as a hardware problem")

	hwaccel := flag.String("hwaccel", "", "hardware decoder: \"cuvid\" for NVDEC, empty for CPU. "+
		"Decoding one 1080p30 stream costs 5.6% of a core against 0.44% for the whole Go pipeline, "+
		"so on a card with NVDEC this is the largest single lever the sweep has")

	out := flag.String("out", "", "append the JSON lines here instead of stdout")

	flag.Parse()

	cfg, err := build(*clips, *rtsp, *detector, *ocr, *library, *devices, *gpuSweep, *sessions, *ocrExtra,
		*start, *step, *maxCameras, *warmup, *window, *detectEvery, *borrow, *detectTimeout, *ocrTimeout,
		*dedup, *dedupDistance, *includeCrop, *includeFull, *quality, *sloP95, *sloBusy, *requireCUDA, *hwaccel, *out)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fleet: %v\n\n", err)
		flag.PrintDefaults()
		os.Exit(2)
	}

	if err := run(cfg); err != nil {
		fmt.Fprintf(os.Stderr, "\nfleet: %v\n", err)
		os.Exit(1)
	}
}

// run owns main.main, and hands it to the ONNX builder.
//
// That inversion is not a style choice. A CUDA session created off the process's main thread
// crashes with SIGSEGV (ORT #26610), and the only way to reach that thread from Go is the
// runtime.LockOSThread in the onnx package's init. So Serve blocks here and the sweep runs
// beside it.
func run(cfg sweepConfig) error {
	if !onnx.OnMainThread() {
		return errors.New("this is not the process's main thread; the onnx package's init did not run first")
	}

	builder := onnx.NewBuilder(onnx.Config{
		LibraryPath:    cfg.LibraryPath,
		Devices:        cfg.DeviceSets[len(cfg.DeviceSets)-1],
		RequireAllCUDA: cfg.RequireAllCUDA,
	})
	if err := builder.Init(); err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	// The sweep runs beside Serve and ends it when it is done, so a finished sweep exits rather
	// than sitting on a builder nobody will ask anything of again.
	sweepCtx, endSweep := context.WithCancel(ctx)
	errs := make(chan error, 1)
	go func() {
		defer endSweep()
		errs <- sweep(sweepCtx, builder, cfg)
	}()

	if err := builder.Serve(sweepCtx); err != nil {
		return err
	}
	return <-errs
}

// build turns the flags into a config, failing early on anything that would only show up as a
// confusing failure ten minutes into a sweep.
func build(
	clips, rtsp, detector, ocr, library, devices string, gpuSweep bool, sessions string, ocrExtra,
	start, step, maxCameras int, warmup, window, detectEvery, borrow, detectTimeout, ocrTimeout,
	dedup time.Duration, dedupDistance int, includeCrop, includeFull bool, quality int,
	sloP95, sloBusy float64, requireCUDA bool, hwaccel, out string,
) (sweepConfig, error) {
	var cfg sweepConfig

	if detector == "" || ocr == "" {
		return cfg, errors.New("both -detector and -ocr are required")
	}
	found, err := resolveClips(clips)
	if err != nil {
		return cfg, err
	}

	deviceList, err := parseInts(devices)
	if err != nil || len(deviceList) == 0 {
		return cfg, fmt.Errorf("-devices %q is not a list of CUDA device ids", devices)
	}
	sessionList, err := parseInts(sessions)
	if err != nil || len(sessionList) == 0 {
		return cfg, fmt.Errorf("-sessions-per-gpu %q is not a list of counts", sessions)
	}
	sort.Ints(sessionList)

	// Sweeping GPUs means running every PREFIX of the device list, so one device is measured
	// before two. Measuring [1] alone would be a different question, and a benchmark that
	// silently changes which card it used is one nobody can reproduce.
	sets := [][]int{deviceList}
	if gpuSweep {
		sets = nil
		for n := 1; n <= len(deviceList); n++ {
			sets = append(sets, append([]int(nil), deviceList[:n]...))
		}
	}

	if start <= 0 || step <= 0 || maxCameras < start {
		return cfg, fmt.Errorf("the sweep %d..%d by %d has no steps in it", start, maxCameras, step)
	}
	if warmup <= 0 || window <= 0 {
		return cfg, errors.New("-warmup and -window must both be positive; the warmup is what makes the window mean anything")
	}

	root, _ := os.Getwd()

	return sweepConfig{
		Clips:           found,
		RTSPServer:      rtsp,
		DetectorPath:    detector,
		OCRPath:         ocr,
		LibraryPath:     library,
		DeviceSets:      sets,
		SessionCounts:   sessionList,
		OCRExtra:        ocrExtra,
		Start:           start,
		Step:            step,
		Max:             maxCameras,
		Warmup:          warmup,
		Window:          window,
		DetectEvery:     detectEvery,
		BorrowTimeout:   borrow,
		DetectTimeout:   detectTimeout,
		OCRTimeout:      ocrTimeout,
		DedupCooldown:   dedup,
		DedupDistance:   dedupDistance,
		HWAccel:         hwaccel,
		IncludeCrop:     includeCrop,
		IncludeFull:     includeFull,
		JPEGQuality:     quality,
		SLOP95:          sloP95,
		SLOMaxBusyDrops: sloBusy,
		RequireAllCUDA:  requireCUDA,
		Out:             out,
		Root:            root,
	}, nil
}

// resolveClips accepts a directory or a comma-separated list, and sorts what it finds.
//
// Sorted, because camera i replays clip i%len(clips): an unsorted directory listing would put
// different footage on different cameras between two runs of the same command, and the two
// throughput numbers would differ for a reason invisible in both reports.
func resolveClips(spec string) ([]string, error) {
	if spec == "" {
		return nil, errors.New("-clips is required: the sweep replays real footage, because a synthetic pattern has no plates in it and the OCR then costs nothing")
	}

	if strings.Contains(spec, ",") {
		parts := strings.Split(spec, ",")
		out := make([]string, 0, len(parts))
		for _, p := range parts {
			if p = strings.TrimSpace(p); p != "" {
				out = append(out, p)
			}
		}
		sort.Strings(out)
		return out, nil
	}

	info, err := os.Stat(spec)
	if err != nil {
		return nil, err
	}
	if !info.IsDir() {
		return []string{spec}, nil
	}

	entries, err := os.ReadDir(spec)
	if err != nil {
		return nil, err
	}
	var out []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		switch strings.ToLower(filepath.Ext(e.Name())) {
		case ".mp4", ".mkv", ".mov", ".ts", ".h264":
			out = append(out, filepath.Join(spec, e.Name()))
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no video files in %s", spec)
	}
	sort.Strings(out)
	return out, nil
}

func parseInts(s string) ([]int, error) {
	var out []int
	for _, part := range strings.Split(s, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		v, err := strconv.Atoi(part)
		if err != nil {
			return nil, err
		}
		out = append(out, v)
	}
	return out, nil
}

// postprocessConfig is the detector decoding the sweep runs with. It is the service's default
// rather than a set of numbers chosen here, so the benchmark measures what production does.
func postprocessConfig() engine.PostprocessConfig { return engine.DefaultPostprocess() }

// libraryDefault is ONNXRUNTIME_LIB_PATH when it is set, so the bench finds the library the same
// way the service does.
func libraryDefault() string {
	if v := os.Getenv("ONNXRUNTIME_LIB_PATH"); v != "" {
		return v
	}
	return onnx.DefaultLibraryPath
}
