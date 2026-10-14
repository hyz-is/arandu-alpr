# Working on Alpr

This is an Arandu package that is an engine: license plate recognition on ONNX
Runtime and TensorRT, live cameras over ffmpeg with NVDEC, and the supervisor
that keeps them running. It has no entity, no policy, no table and no route;
the application owns all four and hands the engine plain interfaces
(`contract.Source`, `Sink`, `Observer`, `Watcher`). The engine imports no Arandu
package at all, and `tests/Unit/discipline_test.go` keeps it that way.
It is a Go module somebody `go get`s and registers by hand in their own
`bootstrap/app.go`, which is the whole difference from working in an
application. There is no service provider, no container and no discovery — if a
line of wiring is not written in the installer's repository, it does not happen.

Read `.agents/skills/` before writing code. Each skill is a procedure, and the
one you need is named by the situation you are in.

## The gates

Nothing is finished until all four exit zero.

```sh
export GOWORK=off
gofmt -l $(find . -name '*.go' -not -path '*/testdata/*' -not -name '*.kyse.go')
go build ./...
go vet ./...
go test -race ./...
```

`GOWORK=off` is not borrowed from somewhere else, and here it is not a
preference either. This checkout may sit beside a Go workspace that lists the
framework repositories and does not list this one; when it does, every command
above fails before it compiles anything:

```
pattern ./...: directory prefix . does not contain modules listed in go.work
or their selected dependencies
```

With the workspace off, the module resolves the framework version in `go.mod` —
which is what CI compiles against, and what somebody's `go get` will get.

Both filters on `gofmt` are load-bearing in the toolchain even where this
repository has nothing for them to skip: `gofmt` is the only tool in the chain
that ignores build tags, and `testdata/` is where a fixture is allowed to be
invalid on purpose.

`aru doctor` is not one of the gates, and running it here costs a minute and
answers nothing:

```
this is not an Arandu project: no go.mod, main.go and arandu.toml together.
Run it from inside a project, or create one with `aru new`
```

It exits 1. It reads applications, and this is a library.

It does not read this one after it is installed either, and that is why
`tests/Unit/audit_test.go` exists. The doctor walks the application's own tree,
skips `vendor/`, and opens the one `arandu.mod.toml` at its root; it never loads
a dependency. So `tenant-from-request`, `system-grant-without-tenant` and
`permission-not-declared` never see a line of an installed package, and whatever
this one must prove about itself it proves in its own suite or nowhere.

## What this repository holds

| | measured with |
| --- | --- |
| 7 Go files, one per role and one generated, all in one package at the root | `grep -l '^package alpr' *.go` |
| 6 test files, 44 tests | `find tests -name '*_test.go'` · `go test -count=1 ./... -v \| grep -c '^--- PASS'` |
| 2 direct dependencies, both under `arandu-io` | `go list -m -f '{{if and (not .Indirect) (not .Main)}}{{.Path}} {{.Version}}{{end}}' all` |

The layout is by package, each with one job; README.md has the table. The root
package `alpr` is the fleet, the gate, deduplication and the session pool;
`contract` is everything an application compiles against; `engine/onnx` is the
only code that touches ONNX Runtime, and only on the main thread.

## What does not belong here

| A model reaches for | What is here instead |
| --- | --- |
| an Arandu import in the engine, to log through the kernel or read config | plain Go and `*slog.Logger`, handed in. `TestCoreDoesNotImportArandu` fails the build |
| a tenant, a database handle or an HTTP client in a camera | nothing. A reading leaves through `contract.Sink`, a camera event through `contract.Observer`; the application decides what they become |
| reading the environment at call time | the typed `Config` of each package, filled once by the application's `config/` |
| a CUDA session created where it is needed | `onnx.Builder`, the only thing that creates or destroys a session, on the main thread |
| a model that quietly runs on the CPU | `RequireAllCUDA`, which refuses it at boot |
| package-level mutable state, an `init` | parameters. The discipline suite allows one `init`: the main-thread pin |
| a third dependency | an argument, first. This module is imported into other people's builds |

## What a change has to keep true

`tests/Unit/discipline_test.go` holds the engine to five rules by reading it: no
Arandu import, no package-level mutable state, no write to a lookup table, no
`init` beyond the main-thread pin, and no forced garbage collection.
`tests/Unit/audit_test.go` compares `arandu.mod.toml` against what the code
calls -- the cameras over the network, the models and the TensorRT cache on
disk, ffmpeg as a process -- and fails in both directions: used and not
declared, declared and not used.

Everything else is held by the packages' own tests with `engine.Fake`, which
needs no GPU. A change to the models, the decode or the camera loop is also
measured on a GPU with `bench/fleet` before it is released.
