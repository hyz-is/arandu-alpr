# Upgrade Guide

## v0.6.2

Nothing breaks. `source.Config.GPUFilters` is false by default and every camera is read as in v0.6.1.
An application that decodes with cuvid sets it from its own switch (alpr: `ALPR_GPU_FILTERS`, true
by default) to keep the picture on the GPU for the cap and the limit; the switch off is the
rollback, with no rebuild. It needs the ffmpeg of the engine image (6.1.1); an older ffmpeg refuses
`scale_cuda` with `interp_algo` and the camera does not open.

## v0.6.1

Nothing breaks and nothing has to be written. Set `MaxHeight` (1080 for cameras that send 1440 or
more) to get the decoder-side scaling on a node that decodes with cuvid; without it the frames keep
the camera's size, as before. On a node that decodes in software, `MaxHeight` still scales on the
CPU. `Stream.Properties()` and every `Frame` report the size after scaling, as in v0.5.0.

## v0.6.0

Nothing breaks. `camera.Config.Stills` is nil by default and a camera pays one nil check.

New and optional: `contract.Stills` (`Seen` is called for every frame a camera receives and says
whether anyone waits for pixels; `Still` is called only then and is handed the frame's pixels,
lent for the call, so copy before returning) and `camera.Config.Stills`. A node that serves still
pictures of its cameras sets it and copies what it needs; it needs `arandu-rtc` v0.2.0 for the
pass kind of those pictures.

## v0.5.0

Nothing breaks. `camera.Config.Consensus` is nil by default and the camera behaves as before.

New settings, all off or unchanged until set (the module reads no environment variable; an
application maps its own to these):

- `camera.Config.Consensus` (`*alpr.ConsensusConfig`, start from `alpr.DefaultConsensus()`): one
  row per vehicle instead of one per frame.
- `camera.Config.ActiveDetectInterval` (`*time.Duration`): the detection budget while a plate is
  followed; point it at a zero duration to read every frame of a vehicle.
- `source.Config.MaxHeight` and `streaming.Config.MaxHeight` (int, 0 = the camera's own size):
  scale a taller picture down before the conversion; `Stream.Properties()` and every `Frame`
  then report the scaled size.
- `alpr.GateConfig.MinWidthPct` (float, 0 = off): drops plates narrower than that fraction of the
  frame before the OCR pass.
- `imaging.NewEncoderQualities(frame, crop)`: separate JPEG qualities; `NewEncoder` is unchanged.
- `engine.Reading.Probabilities` and `alpr.Reading.Probabilities`: new field, filled by
  `engine/onnx`; an engine of your own that leaves it empty still works with a consensus.

`MaxFPS` and `MaxHeight` apply to the frame output only. The live-view tap
(`source.Config.VideoTap`) is a second `-c copy` output of the same ffmpeg and is neither
filtered, decoded nor encoded: a camera capped at 5 fps still feeds its live view at the camera's
own rate and size.

One behaviour changes for everybody: `MaxFPS` no longer repeats frames. A deployment that set it
to 10 on cameras slower than that used to read the same picture several times a second and now
reads each picture once; the frame rate it sees falls to the camera's own, which is what it was
always meant to be.

Turning the consensus on moves the moment a reading is delivered: a verdict arrives when the
track has enough agreeing frames (usually under a second after the third), or when the vehicle is
gone (`ConsensusConfig.Gap` after its last frame), instead of at the first frame that passes the
gate. `MinOCRConfidence` then judges the verdict, and no longer each frame.

## v0.4.0

Nothing to change: `source.Config.VideoTap` and `camera.Config.Overlay` are new and nil by
default, and a nil one leaves a camera exactly as it was. To show a camera live, give the
dialer a `VideoTap` (github.com/hyz-is/arandu-rtc's `Hub` is one) and the camera template an
`Overlay`.

## v0.3.0

Go 1.26.10 is the floor: `go.mod` declares `go 1.26.10`, because every
vulnerability an application reaches through this module's dependencies is in the
standard library and fixed in that release (crypto/x509, net/http HTTP/2 and
CONNECT, html/template, net/textproto, crypto/tls). Build with Go 1.26.10 or newer.

The engine still imports no Arandu package. The `framework` floor in
`arandu.mod.toml` rises to `>= 0.66`, the framework (v0.66.0, with hesape v0.85.0)
that declares the same Go floor; an application wiring the engine into an older
kernel moves its kernel first. Nothing in this module's API changed.

## v0.2.0

Nothing to change: every addition is a new field or method, and the defaults keep
v0.1.0's behaviour. A camera without `ROI` is read whole, and a parked vehicle is
still reported once per cooldown unless `runner.Config.DedupWhilePresent` is set.

## v0.1.0

The first public release. Nothing to upgrade from.

A checkout that held this code in `app/Services/alpr` and `app/Services/streaming`
replaces those import paths with `github.com/hyz-is/arandu-alpr` and
`github.com/hyz-is/arandu-alpr/streaming`, and requires `v0.1.0`.

