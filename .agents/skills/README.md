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
| `alpr-policy` | opening an action, adding an authorized Model-backed use case, or anything answering 403 |
| `alpr-module` | adding a route, a handler, a config field, a response field or a migration |
| `alpr-release` | the gates, the manifest, a dependency, a version, a tag |
| `alpr-vault-notes` | writing the note, the gap or the journal entry, when this checkout sits inside the Arandu Obsidian vault |
| `alpr-package` | installing and wiring this package **into an application** |

The last one has a different audience from the other three, and that is on
purpose: it travels with the package so that an assistant working in somebody
else's project — the one running `go get` — has the wiring, the migration step
and the closed policy in front of it instead of guessing.

Its frontmatter says so, under `metadata`, with `audience: app`. That line is
what `aru skills:sync` reads to copy the skill into an application whose
`go.mod` requires this package, and no other skill here carries it: an
application that received the release procedure would follow it.

`alpr-vault-notes` fires on a condition rather than on a task: it applies
only when `MOC-arandu.md`, `plans/cmd/audit-vault/` and `45-modules/` are actually
beside this checkout. Outside the vault it is inert, and it says so first, so a
package cloned somewhere else never grows a folder tree imitating one.

## Why these exist

The audience of the first three is somebody changing the package. The common
failure modes are a provider, a container lookup, a hand edit to the generated
`AlprQuery.go`, a rule of the entity written into the Service instead of the
custom block of `model.go`, a handler that loads the session or maps an error to
a status itself, a CRUD Repository beside the Model, a tenant read from the URL,
or a Policy branch that returns nil for administrators "for now". None belongs
here, and the last three are security failures rather than style disagreements.

The package is built to be checked rather than trusted. Its denial tests use a
nil database, so reaching even the generated query constructor before
authorization panics. The structural twin reads every exported Service method
and checks the same order on the allowed path, and reads every handler for the
subject and the status it must leave to the framework. Running
`go test -race ./...` exercises all of it.

## Adding your own

A skill in this directory is yours and travels with the repository. Keep it a
procedure rather than a description: a file that says "read the documentation"
never changes what anybody does.
