# Upgrade Guide

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

