---
name: alpr-module
description: Change the Alpr engine itself — the detector or the OCR, the TensorRT build, the camera loop, the supervisor, ffmpeg capture, the gate, deduplication or the metrics. Use when the request is to "change the detector", "support another model", "add a TensorRT option", "change how cameras reconnect", "add a metric", "the plate is read twice", "a new field in the detection", or when a change touches camera/, engine/, source/, streaming/, supervisor/ or contract/. Covers where each piece lives, the contract with the application, the discipline the tests enforce, and how a change is measured.
license: MIT
metadata:
  audience: maintainer
---

# Changing the engine

## Where each piece lives

| piece | package |
| --- | --- |
| the boundary with the application | `contract` |
| the models: sessions, placement, TensorRT, the model contract | `engine/onnx`, behind `engine.Engine` |
| what the detector's output means | `engine` (`Postprocess`) |
| one camera, frame to plate | `camera` |
| which camera runs where, and keeping it running | `supervisor`, started by `runner` |
| reading a stream | `source` over `streaming` |
| the gate, deduplication, the fleet | the root package `alpr` |

## The contract is the API

`contract.CameraSpec`, `Detection` and the four interfaces are what every
application compiles against. A field added there is a minor release; a field
renamed or removed is a break that needs a line in UPGRADE.md.

## The discipline

`tests/Unit/discipline_test.go` fails the build when the engine imports an
Arandu package, holds package-level mutable state, writes to a lookup table,
has an `init` beyond the main-thread pin, or forces a garbage collection. Each
of those once cost a production incident or a benchmark; keep them.

## Measuring a change

`engine.Fake` drives every test without a GPU. A change to the models, the
decode or the camera loop is measured on a GPU with `bench/fleet` and
`cmd/alpr-smoke` against real clips before it is released, and the numbers go
in the changelog entry.

## Gates

```sh
export GOWORK=off
gofmt -l $(find . -name '*.go' -not -path '*/testdata/*' -not -name '*.kyse.go')
go build ./...
go vet ./...
go test -race ./...
```
