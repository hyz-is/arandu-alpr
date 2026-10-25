# Changelog

Everything worth knowing about a release of Alpr is recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

A published module version is immutable: Go serves it from the proxy forever, so
a release is corrected by another release and never by moving a tag.

## [Unreleased]

## [0.3.0] - 2026-10-25

### Changed

- Require Go `1.26.10`, the release that fixes the standard library
  vulnerabilities a module built on the Arandu kernel reaches (CVE-2026-111943
  in crypto/x509 and the net/http HTTP/2 and CONNECT fixes); the manifest floor
  is `framework = ">= 0.66"`.

## [0.2.0] - 2026-10-23

### Added

- Cameras over RTMP: the stream guard accepts `rtmp://` and `rtmps://` from a public
  server, pinned to the checked address as RTSP is, and `rtmp` and `rtmps` are on ffmpeg's
  protocol whitelist.
- `streaming.Guard.Trusted` and `source.Config.Trusted`: the internal host:port addresses a
  camera may still be read from over RTMP -- the engine's own receiver, for cameras that push
  (github.com/hyz-is/arandu-rtmp).
- `contract.CameraSpec.ROI` and `contract.Region`: the rectangle of the frame plates are looked
  for in, as fractions. The camera crops it before the detector and maps the boxes back to the
  whole frame.
- `alpr.Deduper.WhilePresent` and `runner.Config.DedupWhilePresent`: a plate that stays in frame
  is reported once, and again only after it was out of sight for the whole cooldown. The default
  still reports a parked vehicle once per cooldown.

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

