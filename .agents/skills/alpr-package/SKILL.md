---
name: alpr-package
description: Install, wire and use the Alpr engine (Go, Arandu) in an application. Use when the request is to "install alpr", "go get github.com/hyz-is/arandu-alpr", "wire the cameras into bootstrap", "read plates from a photo", "the CUDA session segfaults", "SIGSEGV in onnxruntime", "it fell back to the CPU", "TensorRT compiles on every boot", "no h264_cuvid decoder", or when a project's go.mod already requires github.com/hyz-is/arandu-alpr. Covers the main-thread rule, the builder and the engine, the fleet module and its four interfaces, TensorRT INT8 and its cache, and what the application keeps for itself.
license: MIT
metadata:
  audience: app
---

# Using Alpr

An engine, not a feature: it reads plates and keeps cameras running, and
knows no tenant, database or route. The application decides where cameras come
from (`contract.Source`), what a reading becomes (`contract.Sink`), who hears
about a camera going down (`contract.Observer`), and wires the fleet into its
kernel by hand in `bootstrap/app.go`.

## Install

```bash
go get github.com/hyz-is/arandu-alpr@v0.1.0
```

The build needs `CGO_ENABLED=1`; the image needs ONNX Runtime 1.27.1 with its
CUDA and TensorRT providers, TensorRT 10.14, and ffmpeg built with cuvid.

## The main thread comes first

`engine/onnx` pins the main goroutine to the main thread in an `init`, and
`Builder.Init` refuses anywhere else: a CUDA session created on another thread
ends in SIGSEGV. `main.main` therefore calls `builder.Init()`, starts the
application in a goroutine, and blocks in `builder.Serve(ctx)`. The builder's
context is cancelled only after the application has returned, so `Close` still
has a loop to destroy its sessions on. README.md has the code.

## Wire it

In `bootstrap.Build`, in this order:

1. `onnx.New(ctx, builder, onnx.EngineConfig{...})` — validates both models
   against their contract and builds the sessions.
2. `runner.New(runner.Config{Camera: template, NewFocus: ...})` — `template` is a
   `camera.Config` whose dialer, gate, sink, observer and encoder the application
   fills.
3. `alpr.NewModule(alpr.ModuleConfig{Source, Sink, Observer, Watcher, Engine,
   Runner, Supervisor, Metrics, Logger})`.
4. Wrap the fleet in the application's kernel module (`Boot`, `Start` in the
   background, `Close`, `Health`, `Diagnose`) and `k.Register(...)` it, in
   `bootstrap/app.go`, where every line of wiring is written by hand.

The routes that read the fleet or recognise a photo are the application's
controllers and services; the engine has none.

## TensorRT

`onnx.TensorRTConfig{Enabled, FP16, CacheDir, DetectorTable, OCRTable}` runs
both models in INT8 from the float graph and its calibration table. Without
`CacheDir` on a volume every boot compiles both engines again (about 20 minutes
on a T4), and without a table nothing runs in INT8; the application's
configuration should refuse both.

## Do not

- Create a session, or call anything in `engine/onnx`, before `builder.Init`.
- Give the engine a tenant, a database handle or an HTTP client: it reports to
  the application through the contract interfaces.
- Read the environment inside the engine; the application's `config/` fills
  every `Config`.
