# Upgrade Guide

## Unreleased

Nothing breaks. `camera.Config.Consensus` is nil by default and the camera behaves as before.

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

