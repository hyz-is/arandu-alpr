# Changelog

Everything worth knowing about a release of Alpr is recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

A published module version is immutable: Go serves it from the proxy forever, so
a release is corrected by another release and never by moving a tag.

## [Unreleased]

## [0.1.0] - 2026-10-13

### Added

- The engine of api.alpr.sh, published as a package: `alpr.NewModule` (the
  fleet: `Boot`, `Start`, `Close`, `Health`, `Diagnose`), the quality gate,
  deduplication and the session pool.
- `engine/onnx`: the session builder on the process's main thread, CUDA with
  `RequireAllCUDA`, TensorRT INT8 from a float graph and a calibration table
  (`TensorRTConfig`), the engine cache, node placement and the model contract
  (any square detector input, uint8 or float OCR input).
- `source` and `streaming`: ffmpeg and ffprobe with NVDEC (`-c:v h264_cuvid
  -gpu N`), a protocol whitelist and a public-address check before every
  connection.
- `camera`, `runner` and `supervisor`: one loop per camera, placement over the
  cards, reconcile, watchdog and retries.
- `contract`, `engine.Fake`, `imaging` and `metrics`.
- `bench/`, `cmd/alpr-smoke` and `internal/corpus`, the measuring tools.

### Requirements

- Go 1.26, `CGO_ENABLED=1`, `github.com/yalue/onnxruntime_go` v1.31.0, ONNX
  Runtime 1.27.1, TensorRT 10.14 for INT8, ffmpeg with cuvid.

