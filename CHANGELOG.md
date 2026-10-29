# Changelog

Everything worth knowing about a release of Alpr is recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

A published module version is immutable: Go serves it from the proxy forever, so
a release is corrected by another release and never by moving a tag.

## [0.6.3] - 2026-10-29

### Added

- `streaming.Opener.MeasureRate`: counts the frames a camera actually sends in a window of stream
  time and answers frames per second. `Probe`'s FPS is ffprobe's `r_frame_rate`, which several
  recent Hikvision models report wrong (24.9 for a camera set to 10) because their stream carries
  no timing in its headers.

## [0.6.2] - 2026-10-29

### Added

- `streaming.Config.GPUFilters` and `source.Config.GPUFilters`: with the cuvid decoder, 8-bit 4:2:0
  H.264 or HEVC, and a frame cap or a height limit that changes the picture, the picture stays in
  GPU memory (`-hwaccel cuda -hwaccel_output_format cuda -c:v h264_cuvid`), the cap (`select`) and
  the limit (`scale_cuda`, bilinear, nv12) run there, and `hwdownload,format=nv12` copies only the
  chosen frames, already at their final size; `-pix_fmt bgr24` still converts on the CPU, because
  `scale_cuda` has no bgr output. The pipe, `Stream.Properties()` and every `Frame` are unchanged.
  The live-view tap (`-c:v copy`, fragmented MP4) keeps working beside it: checked on ffmpeg
  6.1.1, 250 of 250 frames copied. Any camera outside that case (10-bit, 4:2:2, MJPEG, a probe
  without a pixel format, software decode) is read as in v0.6.1. False (the zero value) reads as
  in v0.6.1 for every camera. Needs ffmpeg 6.1 or newer.
- `streaming.Properties.PixFmt`, from ffprobe's `pix_fmt`.

## [0.6.1] - 2026-10-29

Less CPU per camera and per node, measured on a 32-vCPU node reading 23 cameras: ffmpeg took
about 1900% of CPU and the Go process about 700%, with inference already on the GPU.

### Changed

- `streaming.Config.MaxHeight` with the cuvid decoder is applied by NVDEC (`-resize WxH` on the
  decoder, both sides even, proportional to the source) and the CPU `scale` filter is not added.
  Before, every frame came back to system memory at the camera's size, was converted by swscale
  and then scaled. Without cuvid nothing changes: the `scale` filter stays. NVDEC resamples with
  its own filter, not with `flags=area`.
- The reading ffmpeg runs with `-filter_threads 1` (15 to 18% less CPU per ffmpeg, measured).
- The pipe ffmpeg writes the frames to is grown to 1 MiB on Linux (`F_SETPIPE_SZ`), so a frame
  crosses it in fewer write and read round trips; a refusal from the host is ignored.
- `engine/onnx`: the detector's letterbox runs before a session is borrowed, into a scratch tensor
  of the engine, and is copied into the session's input. The GPU slot is no longer held while the
  CPU waits for a core. A frame that then finds no free session (`engine.ErrBusy`) has paid its
  letterbox.

## [0.6.0] - 2026-10-26

### Added

- `contract.Stills` and `camera.Config.Stills`: a consumer that is told every frame a camera receives
  (`Seen`, a few atomic operations) and is handed the pixels of the one it asks for (`Still`), lent
  for the call. It is how a page shows a still of a camera that is not worth a video: the picture the
  loop already decoded, with no new connection to the camera and no second decode. The hook runs
  before the brightness adjustment and before the stale test, and a camera with no consumer pays
  one nil check.

## [0.5.0] - 2026-10-26

### Added

- `alpr.Consensus`, `camera.Config.Consensus`: one vehicle, one row. The readings of one plate
  across frames are followed as a track (same place, same text) and decided together, from the
  OCR head's whole distribution per frame, weighted by detector score, plate size, sharpness and
  the reading's own confidence. The Detection that reaches the Sink is the track's verdict, with
  the JPEGs of the best frame that read the winning text; it is decided early, with the vehicle in
  view, once enough frames agree, and otherwise when the vehicle is gone. A track no frame backs
  up (a smear read as something different each frame) is dropped without a row. No plate-format
  check anywhere: the winner is a text the frames read, or the slot-by-slot best of all of them.
- `alpr.Reading.Probabilities` and `engine.Reading.Probabilities`: the OCR head's distribution
  (10 slots by the 37 classes), which the argmax used to throw away.
- `camera.Config.ActiveDetectInterval`: the detection budget while a plate is being followed. The
  camera idles at `Spec.MinDetectInterval` and reads every frame of a vehicle once it sees one.
- `streaming.Config.MaxHeight` and `source.Config.MaxHeight`: scale a taller picture down (1080
  for a camera that sends 1440) before the BGR conversion, with the area filter. Both filters
  belong to the frame output the plates are read from: the live-view tap keeps the camera's own
  packets and picture (`-c copy`), so neither the cap nor the limit reaches it.
- `alpr.GateConfig.MinWidthPct`: drops plates narrower than this fraction of the frame's width
  before the OCR pass (0 = off).
- `imaging.NewEncoderQualities(frame, crop)`: the plate crop gets its own JPEG quality;
  `NewEncoder(q)` is the same quality for both, as before.
- `alpr.ProbabilitiesFromText`: a distribution for an engine that kept only the text.

### Changed

- `streaming.Config.MaxFPS` and `source.Config.MaxFPS` only drop frames. The `fps` filter they
  used made a frame at every 1/N whether or not the camera sent one: a camera at 8 fps behind a
  cap of 15 delivered 15, seven of them copies. A cap above the camera's rate now does nothing.
- `engine/onnx` returns the OCR distribution in `engine.Reading.Probabilities` (a copy; the
  tensor is reused).

## [0.4.0] - 2026-10-25

### Added

- `transcode`: converts a camera's video to H.264 for a live view, as a separate ffmpeg
  process fed the container copy the tap already writes. At most 1080 lines and 15 pictures a
  second, a keyframe a second; NVENC (and NVDEC, with `HWAccel: "cuvid"`, scaling as it
  decodes) when the card has it, probed once with a real encode, and libx264 ultrafast when it
  does not or when NVENC fails at once. It never touches the plate reader's own decode, which
  keeps its resolution and rate whether anyone watches or not. `Transcode` has the shape
  arandu-rtc's Hub asks for, so the Hub can run it on demand.

- `contract.VideoTap` and `source.Config.VideoTap`: a copy of each camera's compressed
  video for a live view. The ffmpeg that already decodes the camera writes a second output,
  `-c copy` to fragmented MP4 on a descriptor of its own, cut every tenth of a second, so
  nothing is decoded or encoded for it and a camera is still read once. The tap is closed on
  every path, its pipe is drained whatever its writer does, and a camera with no tap runs the
  same ffmpeg arguments as before.
- `contract.Overlay` and `camera.Config.Overlay`: every detection pass tells a watched camera's
  overlay which plates the detector found, with the reading when it held up, whether or not
  the plate was reported again, and an empty pass when the plate has gone. A camera nobody
  watches asks once per pass and allocates nothing.

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

