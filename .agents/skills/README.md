# Skills

Procedures an assistant follows when working on this package.

They live in `.agents/skills/<name>/SKILL.md`, which is the path the coding
assistants read from — Cursor, Codex, Cline, Copilot, Gemini CLI, Amp, OpenCode,
Warp, Zed and the rest all look there. It is one directory rather than a file
per vendor, so a skill written once is read by whatever this package is being
written with.

Each file opens with frontmatter carrying a `name` and a `description`. The
`name` has to equal the directory name exactly, or the skill is not loaded. The
description is what a tool reads to decide whether the skill is relevant, so it
names the situation you are in rather than the topic it covers.

| skill | when it fires |
| --- | --- |
| `alpr-module` | changing the engine: the models, TensorRT, the camera loop, the supervisor, capture, the gate |
| `alpr-release` | the gates, the manifest, a dependency, a version, a tag |
| `alpr-vault-notes` | writing the note, the gap or the journal entry, when this checkout sits inside the Arandu Obsidian vault |
| `alpr-package` | installing and wiring this package **into an application** |

The last one has a different audience from the other three, and that is on
purpose: it travels with the package so that an assistant working in somebody
else's project — the one running `go get` — has the main-thread rule and the
wiring in front of it instead of guessing.

Its frontmatter says so, under `metadata`, with `audience: app`. That line is
what `aru skills:sync` reads to copy the skill into an application whose
`go.mod` requires this package, and no other skill here carries it: an
application that received the release procedure would follow it.

`alpr-vault-notes` fires on a condition rather than on a task: it applies
only when `MOC-arandu.md`, `plans/cmd/audit-vault/` and `45-modules/` are actually
beside this checkout. Outside the vault it is inert, and it says so first, so a
package cloned somewhere else never grows a folder tree imitating one.

## Why these exist

The audience of `alpr-module` is somebody changing the engine. The common
failure modes are an Arandu import in the engine, package-level state, a CUDA
session created off the main thread, a model that quietly falls back to the
CPU, and a reading pushed anywhere but through `contract.Sink`. The discipline
tests catch the first two; `Builder.Init` and `RequireAllCUDA` refuse the next
two at boot.

The package is built to be checked rather than trusted: the discipline suite reads
the engine's source, the audit compares the manifest with what the code calls,
and `engine.Fake` drives the rest without a GPU. Running `go test -race ./...`
exercises all of it.

## Adding your own

A skill in this directory is yours and travels with the repository. Keep it a
procedure rather than a description: a file that says "read the documentation"
never changes what anybody does.
