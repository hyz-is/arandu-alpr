<p align="center">
  <img src=".github/logo.svg" alt="Arandu" width="180">
</p>

# Alpr

An Arandu package. It registers its own routes, owns its own table, and decides
for itself who may reach either.

## Install

```bash
go get github.com/hyz-is/arandu-alpr
```

## Wire it

An Arandu application registers a module explicitly. There is no service
provider, no container and no discovery, so these are the lines to paste into
`bootstrap/app.go` and there are no others.

The import, with the other module imports:

```go
import (
	alpr "github.com/hyz-is/arandu-alpr"
)
```

The construction, in `Build`, after the session store exists and before
`k.Register`:

```go
	alprModule, err := alpr.New(alpr.Config{}, db, sessions)
	if err != nil {
		return App{}, err
	}
```

And the registration, inside the `k.Register(...)` call already there:

```go
		alprModule,
```

Then, once, before the application serves:

```bash
aru migrate
```

This package owns a table, which is why the migration step is not optional and
why `arandu.mod.toml` says `migrations = true`.

## Publish the views

This package carries the markup of its own pages and hands it over instead of
rendering it from the inside, because a page you cannot edit is a page that says
the wrong thing in your product.

Look at what would be written, then write it:

```bash
aru vendor:publish --tag=view
aru vendor:publish --tag=view --apply
```

Nothing is written without `--apply`. The preview lists every file as `create`,
`update`, `unchanged` or `conflict`, and running the command a second time
writes nothing. A file changed outside its `arandu:begin custom` markers is
reported as a conflict and left alone; `--force` publishes over one, and even
then what is inside the markers is carried forward.

The files land under `resources/views/modules/alpr/`, and from that point
they are yours. Nothing of this package is compiled beside them, so no view name
is registered twice and no rule has to decide which of two files won — the
consequence being that a view of this package that changes later does not reach
a project that already published it.

Two steps are left to you, and they are left to you because a command that
edited `bootstrap/app.go` behind your back is a command whose output nobody can
explain. Compile what was written:

```bash
aru view:build
```

and import the directory it wrote into, with the other imports:

```go
	_ "your/module/path/storage/framework/views/modules/alpr"
```

Without that import the views are not in the binary, and the module refuses to
boot rather than answering the first request that reaches one of them with a
500. The refusal names the view, the command and the import.

## Configuration

| field | required | meaning |
| --- | --- | --- |
| `Prefix` | no | where the routes are mounted. Defaults to `/alpr`. |
| `PageSize` | no | how many records one page answers with. Defaults to 25, refused above 200. |

The zero `Config{}` is a valid configuration. There is no tenant to set: every
route needs a session, and the tenant of every statement is the one on the
Grant of whoever is signed in.

`New` returns an error rather than starting half-wired, so a setting that
cannot work fails where it is written instead of on the first request that
needed it.

## Routes

| method | path | name |
| --- | --- | --- |
| `GET` | `/alpr` | `alpr.index` |
| `GET` | `/alpr/{id}` | `alpr.show` |
| `POST` | `/alpr` | `alpr.store` |

Every one of them sits behind `RequireAuth`: a visitor with no session is sent
to `/auth/login`, and the handler of a request that has one reads who is asking
with `ctx.User()`. Behind the guard, every route is refused until the policy is
opened. That is the state the package ships in, and it is deliberate.

A handler returns the error the service returned, and the router answers it: a
policy refusal is 403, a record that is not there is 404, and a rejected input
goes back where it came from with the messages in the flash. A request that asks
for JSON gets the same statuses as `application/problem+json` documents, and a
rejected input as a 422 with the messages keyed by field. No handler maps an
error to a status itself.

## Open the policy

`policy.go` denies every action and has no branch that allows one. Open what
this package needs, one action at a time, inside the custom block:

```go
	// arandu:begin custom
	if a == AlprView && (s.ID == record.ID || s.HasRole("admin")) {
		return nil
	}
	// arandu:end custom
```

What is not written there stays closed, including every action added later.

## The data path

`Alpr` embeds `model.Model`, and `alprTable`, beside it in `model.go`,
declares its table once. `aru model:build` generates `AlprQuery.go` from
it: `Alprs(db)`, which starts a `*AlprQuery`, and `AlprCollection`.
The file is committed and never edited by hand; regenerate it after changing
the entity, and `aru model:build --check` fails when it is out of date.

`AlprService` owns `*data.DB` and follows
`validate -> security.Authorize -> Grant -> Model terminal`; handlers never hold
the database or start a query. Every query starts at `Alprs(s.db)`: the
constructor returns one mutable query, so a second chain begun from a value held
in a variable would carry the clauses of the first.

Create builds the row with `Alprs(s.db).New()` and writes `TenantID` from
`data.Tenant(g)`. Find authorizes before reading and again against the row it
found. List authorizes before building its scoped, allowlisted query. The table
keeps its default `tenant_id` scope on every terminal.

Terminals return `*Alpr` and `[]*Alpr`. Keep those pointers: a copy of a
row keeps the model of the original, and the model refuses to write through it
with `model.ErrUnwired`. `Resource` and `Collection` are explicit response
snapshots and do not expose tenant or Model internals.

The rules of the entity itself — an invariant, a derived value, a transition
such as `Rename` guarded by `CanRename` — go in the custom block of `model.go`,
as pure methods: no database, no network, no clock read inside, no Grant. The
service orchestrates: it validates, authorizes, calls those rules and saves
with the Grant.

There is no CRUD Repository. Add one only for a complex query, read model,
report, export or raw SQL contract that the generated query cannot express.

## Layout

```
module.go         registration, routes, handlers and migrations
config.go         what the application passes in
model.go          the entity, its table and its own rules, and what it may answer with
AlprQuery.go  the query aru model:build generates from model.go
policy.go         who may do what
service.go        the use cases, and Model access after authorization
views.go          the files the application takes ownership of
```

## What is already correct, and has to stay that way

**The policy denies everything.** There is no permit-all branch to delete
later. The Service calls `security.Authorize` before its first `Alprs(db)`
reach, and every Model terminal requires the Grant that call produced.

**Authorization precedes the Model.** The package audit checks that order in
every exported Service method. A Model terminal enforces tenant scope; the
preceding Policy call decides whether the action itself is allowed.

**The tenant comes from `data.Tenant(g)`.** Never from the path, the body, the
query string or a header. The value on the Grant came from the session; a value
that arrived with the request is a value the caller chose.

**Who is asking comes from the guard.** `RequireAuth` loads the session once and
puts the subject on the request; nothing in the module loads it again.

**`arandu.mod.toml` declares what the package does** — network, filesystem,
exec, migrations — and the suite compares the declaration against what the code
*calls*, not against what it imports: `net/http` is imported by everything with
a route and says nothing. A package that says it makes no outbound calls and
then opens one fails its own tests, and that is the only place the comparison
happens. `aru doctor` audits the application it is run inside and never loads a
dependency, so nothing audits an installed package except the package itself.

## Tests

```bash
go test -race ./...
```

The denial suite constructs the Service with a nil database, so even building
`Alprs(nil)` would panic. The structural twin reads the allowed path and
rejects any Service method that reaches the Model before `Authorize`, and any
handler that loads the session or picks a status itself. The route tests drive
the module with no session, with a signed-in subject and with a rejected input,
on that same handle over no database.

## Licence

MIT. See [LICENSE.md](LICENSE.md). Copyright Paulo R. Lima.
