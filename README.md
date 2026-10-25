<p align="center">
  <img src=".github/logo.svg" alt="Arandu" width="180">
</p>

# Arandu ALPR

License plate recognition for Arandu: a detector and an OCR on ONNX Runtime,
in INT8 on NVIDIA GPUs through TensorRT; live cameras read over RTSP, RTSPS,
HTTP or HTTPS with NVDEC decoding; a supervisor that spreads the cameras over
the cards and keeps them running; and the plate pipeline between them — the
quality gate, deduplication and the metrics of every camera.

It is the engine of [api.alpr.sh](https://github.com/alpr-sh/api.alpr.sh). It
knows no tenant, no database and no HTTP: the application decides where the
cameras come from and what happens to a reading, and wires the engine into its
kernel.

## Requirements

- Go 1.26 and `CGO_ENABLED=1`: `github.com/yalue/onnxruntime_go` v1.31.0 has a C
  wrapper.
- ONNX Runtime 1.27.1 with its CUDA and TensorRT providers, and TensorRT
  10.14 for INT8. The library path is `onnx.Config.LibraryPath`.
- `ffmpeg` and `ffprobe` built with `--enable-cuvid` for NVDEC.
- NVIDIA T4 (sm75) or A10 (sm86). The CPU runs the tests and nothing else:
  `RequireAllCUDA` refuses a model that falls back to it.

## The main thread

A CUDA session of ONNX Runtime created off the process's main thread ends in
SIGSEGV (onnxruntime issue #26610). `engine/onnx` pins the main goroutine to
the main thread in an `init`, and `Builder.Init` refuses to run anywhere else.
So `main.main` belongs to the builder, and the application runs in a goroutine
beside it:

```go
builder := onnx.NewBuilder(builderCfg)
if err := builder.Init(); err != nil { // refuses off the main thread
	return err
}
ctx, release := context.WithCancel(context.Background())
done := make(chan error, 1)
go func() {
	defer release() // only after the application has closed its sessions
	done <- runApplication(builder)
}()
if err := builder.Serve(ctx); err != nil { // blocks on the main thread
	return err
}
return <-done
```

## Wire it

Inside the application, with the builder above:

```go
eng, err := onnx.New(ctx, builder, onnx.EngineConfig{
	DetectorPath: "/models/detector.onnx",
	OCRPath:      "/models/ocr.onnx",
	Postprocess:  engine.DefaultPostprocess(),
	Metrics:      registry,
})

start, err := runner.New(runner.Config{
	Camera:   cameraTemplate, // camera.Config: dialer, gate, sink, observer, encoder
	NewFocus: func() (camera.Focus, error) { return imaging.NewScaler(engine.DetectorInputSize) },
})

fleet, err := alpr.NewModule(alpr.ModuleConfig{
	Source:     cameras,  // contract.Source: the cameras the application wants running
	Sink:       readings, // contract.Sink: where every accepted plate goes
	Observer:   events,   // contract.Observer: camera lifecycle events
	Watcher:    resync,   // contract.Watcher: an early reconcile
	Engine:     eng,
	Runner:     start,
	Supervisor: supervisor.Config{Devices: []int{0, 1}},
	Metrics:    registry,
	Logger:     log,
})
```

`fleet` has `Boot`, `Start`, `Close`, `Health` and `Diagnose`. All of this is
written by hand in the application's `bootstrap/app.go`, which wraps the fleet
in its own kernel module and registers it with `k.Register(...)`; there is no
provider and no discovery. `Close` stops the cameras and destroys the sessions.

TensorRT runs both models in INT8 from a float graph and a calibration table:

```go
onnx.Config{
	Devices:        []int{0, 1},
	RequireAllCUDA: true,
	TensorRT: onnx.TensorRTConfig{
		Enabled:       true,
		FP16:          true,
		CacheDir:      "/trt", // a volume: the first compile takes about 20 minutes on a T4
		DetectorTable: "/models/detector.int8.flatbuffers",
		OCRTable:      "/models/ocr.int8.flatbuffers",
	},
}
```

## One vehicle, one row

A car crossing a camera is read once per detection pass, each time from a different picture, and
no frame's confidence says which of them read the plate right. Left to the gate and the deduper
the same car is reported under several spellings. Give the camera a `Consensus` and it reports
the vehicle instead:

```go
cons := alpr.DefaultConsensus()
active := time.Duration(0)

cameraTemplate := camera.Config{
	// ...dialer, engine, gate, sink, encoder...
	Consensus:            &cons,   // readings of one plate are followed as a track and decided together
	ActiveDetectInterval: &active, // and every frame is read while one is followed
}
```

Each frame contributes the OCR head's whole distribution (`engine.Reading.Probabilities`), not its
top choice, weighted by the detector's score, the plate's size in the frame, its sharpness and the
reading's own confidence. The Detection that reaches the `Sink` is the track's verdict, with the
JPEGs of the best frame that read the winning text. It is decided early, with the car in view, when
enough frames agree, and otherwise when the car is gone. A plate nothing agrees on is dropped
without a row. There is no plate-format rule: the winner is a text some frame read, or the
slot-by-slot best of all of them. `alpr.Consensus` is also usable on its own, fed `Observation`s.

`streaming.Config.MaxFPS` only drops frames (a cap above the camera's rate does nothing), and
`MaxHeight` scales a taller picture down before the conversion.

## Packages

| package | what it holds |
| --- | --- |
| `alpr` | the fleet module, the quality gate, deduplication, the session pool |
| `contract` | the boundary with the application: `Source`, `Sink`, `Observer`, `Watcher`, `CameraSpec`, `Detection` |
| `engine` | the `Engine` interface, the detector's post-processing and `Fake` |
| `engine/onnx` | the session builder on the main thread, CUDA and TensorRT, placement and the model contract |
| `camera` | one camera's loop: frames in, plates out |
| `runner` | the camera factory the supervisor starts |
| `supervisor` | which camera runs on which card, reconcile, watchdog, retries |
| `source` | ffmpeg and ffprobe, with NVDEC and the public-address guard |
| `streaming` | the stream reader under `source` |
| `imaging` | scaling, letterbox, JPEG, brightness and contrast |
| `metrics` | counters and histograms with percentiles, and `Discard` |

`bench/`, `cmd/alpr-smoke` and `internal/corpus` are the tools that measure the
engine against real clips. `docs/MODELS.md` says which models run and why.

## Tests

```bash
GOWORK=off go test -race ./...
```

They run on a laptop with no GPU: `engine.Fake` stands in for the models, and
`tests/Unit/discipline_test.go` keeps the engine free of any Arandu import, of
package state and of `init` functions beyond the main-thread pin.

## Licence

MIT. See [LICENSE.md](LICENSE.md).
