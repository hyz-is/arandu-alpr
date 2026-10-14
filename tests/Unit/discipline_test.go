package unit_test

// The ALPR service is meant to be embedded by an application and reused. Three properties make
// that true, and none of them survives on good intentions: they are one careless import or one
// package-level `var` away from being false, and nothing about the code would look wrong.
//
// So they are tests. Each one costs a fraction of a second and each one has already been
// violated once -- twice in production, in the Python service these rules come from.

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// coreDir is the module root: every package of the engine must stay framework-free. The
// directories in notCore are the module's tests and tools, not the engine.
const coreDir = "."

var notCore = map[string]bool{"tests": true, "bench": true, "cmd": true, "internal": true, ".git": true, ".github": true, ".agents": true}

// corePackages lists the import paths of the core, resolved from disk so a new subpackage is
// covered the day it is created rather than the day someone remembers to add it here.
func corePackages(t *testing.T) []string {
	t.Helper()
	root := packageRoot(t)
	var pkgs []string
	err := filepath.WalkDir(filepath.Join(root, coreDir), func(path string, d os.DirEntry, err error) error {
		if err != nil || !d.IsDir() {
			return err
		}
		if notCore[d.Name()] {
			return filepath.SkipDir
		}
		entries, err := os.ReadDir(path)
		if err != nil {
			return err
		}
		for _, e := range entries {
			if strings.HasSuffix(e.Name(), ".go") {
				rel, _ := filepath.Rel(root, path)
				pkgs = append(pkgs, "./"+filepath.ToSlash(rel))
				return nil
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(pkgs) == 0 {
		t.Fatalf("no packages found under %s", coreDir)
	}
	return pkgs
}

// TestCoreDoesNotImportArandu.
//
// The service is a service: it has no HTTP server, no database, no authentication and no
// tenancy of its own, because all four belong to the application that embeds it. The way that
// stays true is that the core cannot reach the framework at all -- not directly and not through
// a dependency.
//
// The payoff is concrete. The core can be tested without booting an application, benchmarked
// without one, and lifted into a different one unchanged. The first import that crosses this
// line takes all three away, and it takes them away quietly: everything still compiles.
func TestCoreDoesNotImportArandu(t *testing.T) {
	root := packageRoot(t)
	args := append([]string{"list", "-deps"}, corePackages(t)...)
	cmd := exec.Command("go", args...)
	cmd.Dir = root
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("go list -deps: %v\n%s", err, out)
	}

	for _, line := range strings.Split(string(out), "\n") {
		dep := strings.TrimSpace(line)
		if strings.HasPrefix(dep, "github.com/arandu-io/") || (strings.HasPrefix(dep, "github.com/hyz-is/") && !strings.HasPrefix(dep, "github.com/hyz-is/arandu-alpr")) {
			t.Errorf("the core reaches %s\n\n"+
				"The service must stay embeddable without the framework. Put whatever needs\n"+
				"the framework in the adapter (module.go) and hand the core a plain interface.", dep)
		}
	}
}

// TestNoPackageLevelMutableStateInCore.
//
// A package-level `var` is shared by every instance in the process, which means two of them
// silently share whatever it holds -- a semaphore, a counter, a cache.
//
// This is the shape of a bug that reached production in the Python service twice. Once as a
// global lock around opening a camera, where one unreachable camera held it for the full 30s
// connect timeout and blocked every other camera from opening. Once as an asyncio.Semaphore
// built at import time, which bound itself to the wrong event loop.
//
// `const`, `var _ Interface = (*T)(nil)` assertions and error sentinels are fine: none of them
// carries state.
func TestNoPackageLevelMutableStateInCore(t *testing.T) {
	forEachCoreFile(t, func(t *testing.T, rel string, file *ast.File, fset *token.FileSet) {
		for _, decl := range file.Decls {
			gen, ok := decl.(*ast.GenDecl)
			if !ok || gen.Tok != token.VAR {
				continue
			}
			for _, spec := range gen.Specs {
				vs := spec.(*ast.ValueSpec)
				for _, name := range vs.Names {
					if name.Name == "_" {
						continue // interface satisfaction assertion
					}
					if isErrorSentinel(vs) || isLookupTable(vs) || isCompiledPattern(vs) {
						continue
					}
					if varAllowed[rel][name.Name] {
						continue
					}
					t.Errorf("%s:%d: package-level var %q\n\n"+
						"Two instances of the service in one process would share it. Move it "+
						"into the struct that owns it and pass that struct around.",
						rel, fset.Position(name.Pos()).Line, name.Name)
				}
			}
		}
	})
}

// isErrorSentinel reports whether a var declaration is `var ErrX = errors.New(...)`.
//
// Sentinels are immutable in practice and are the idiomatic way to let a caller compare against
// a specific failure. Excluding them keeps the rule about state rather than about syntax.
func isErrorSentinel(vs *ast.ValueSpec) bool {
	if len(vs.Values) != 1 {
		return false
	}
	call, ok := vs.Values[0].(*ast.CallExpr)
	if !ok {
		return false
	}
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	pkg, ok := sel.X.(*ast.Ident)
	if !ok {
		return false
	}
	return (pkg.Name == "errors" && sel.Sel.Name == "New") ||
		(pkg.Name == "fmt" && sel.Sel.Name == "Errorf")
}

// varAllowed names the package-level variables that are allowed to exist, per file.
//
// One entry so far, and it is the shape of exception this list is for: a mutex guarding a
// resource that genuinely IS one per process. Everything else on this list would be a bug.
var varAllowed = map[string]map[string]bool{
	"engine/onnx/placement.go": {
		// captureMu serialises redirection of file descriptor 2. There is exactly one stderr
		// per process, so two Builders redirecting it at the same time would interleave and
		// each would parse the other's log. Making this a field would model a world with one
		// stderr per instance, which is not the world.
		"captureMu": true,
	},
}

// isCompiledPattern reports whether a var is a regexp compiled at package level.
//
// A compiled *regexp.Regexp is immutable and safe for concurrent use by design, so it is a
// lookup table that happens to be built by a function call rather than written as a literal.
// Compiling it per call would be the actual mistake.
func isCompiledPattern(vs *ast.ValueSpec) bool {
	if len(vs.Values) != 1 {
		return false
	}
	call, ok := vs.Values[0].(*ast.CallExpr)
	if !ok {
		return false
	}
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	pkg, ok := sel.X.(*ast.Ident)
	if !ok || pkg.Name != "regexp" {
		return false
	}
	return strings.HasPrefix(sel.Sel.Name, "MustCompile")
}

// isLookupTable reports whether a var is a fixed table written out as a literal.
//
// Go has no const array, so a lookup table -- the 66 region names, a set of bucket bounds -- has
// to be a var even though nothing is supposed to write to it. Excluding these keeps the rule
// about shared *state*, which is what actually breaks when two instances share a process.
//
// TestCoreNeverWritesToALookupTable covers the other half: that nothing assigns into one.
func isLookupTable(vs *ast.ValueSpec) bool {
	if len(vs.Values) != 1 {
		return false
	}
	lit, ok := vs.Values[0].(*ast.CompositeLit)
	if !ok {
		return false
	}
	switch lit.Type.(type) {
	case *ast.ArrayType, *ast.MapType:
		return true
	}
	return false
}

// TestCoreNeverWritesToALookupTable.
//
// The region table is the case this exists for. Its index IS the class the model emits, so the
// positions are the mapping -- and a write to it, or a reassignment of it, changes what every
// past and future reading is called. Nothing at runtime would look wrong: plates would still
// come back with country names, just the wrong ones.
//
// Sorting the table alphabetically would look like tidying and would do exactly this.
func TestCoreNeverWritesToALookupTable(t *testing.T) {
	tables := map[string]string{} // name -> "file:line" where it was declared

	forEachCoreFile(t, func(t *testing.T, rel string, file *ast.File, fset *token.FileSet) {
		for _, decl := range file.Decls {
			gen, ok := decl.(*ast.GenDecl)
			if !ok || gen.Tok != token.VAR {
				continue
			}
			for _, spec := range gen.Specs {
				vs := spec.(*ast.ValueSpec)
				if !isLookupTable(vs) {
					continue
				}
				for _, name := range vs.Names {
					tables[name.Name] = rel
				}
			}
		}
	})

	if len(tables) == 0 {
		return
	}

	forEachCoreFile(t, func(t *testing.T, rel string, file *ast.File, fset *token.FileSet) {
		ast.Inspect(file, func(n ast.Node) bool {
			assign, ok := n.(*ast.AssignStmt)
			if !ok {
				return true
			}
			for _, lhs := range assign.Lhs {
				target := lhs
				if idx, ok := target.(*ast.IndexExpr); ok {
					target = idx.X
				}
				ident, ok := target.(*ast.Ident)
				if !ok {
					continue
				}
				if decl, isTable := tables[ident.Name]; isTable {
					t.Errorf("%s:%d: writes to the lookup table %q (declared in %s)\n\n"+
						"Its positions are the mapping. Changing one changes what every "+
						"reading is called, and nothing at runtime would look wrong.",
						rel, fset.Position(assign.Pos()).Line, ident.Name, decl)
				}
			}
			return true
		})
	})
}

// TestNoInitFunctionsInCore.
//
// An init() runs before anyone has decided anything: before config is read, before a test has
// set up its fixtures, and in an order the language chooses. Whatever it builds is built once
// per process and cannot be built differently for a second instance or for a test.
//
// New(cfg) is the alternative, and it is strictly better in every case this service has: it can
// fail, it can be called twice, and it can be handed a different clock.
//
// Exactly one init() in this service is allowed, and it is the one below.
//
// Creating a CUDA session off the process's main thread crashes with SIGSEGV (ONNX Runtime
// issue #26610), and Go gives no way to reach that thread except one: the main goroutine is
// pinned to the main thread for the duration of package init and released just before
// main.main, so a runtime.LockOSThread from an init keeps the pin. runtime.LockOSThread called
// anywhere else pins to SOME thread, which is not the same thing and does not help.
//
// It is listed by path rather than waived by a comment convention, so adding a second one is a
// decision somebody has to make here, in the open.
var initAllowed = map[string]bool{
	"engine/onnx/mainthread.go": true,
}

func TestNoInitFunctionsInCore(t *testing.T) {
	forEachCoreFile(t, func(t *testing.T, rel string, file *ast.File, fset *token.FileSet) {
		if initAllowed[rel] {
			return
		}
		for _, decl := range file.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Recv != nil || fn.Name.Name != "init" {
				continue
			}
			t.Errorf("%s:%d: init() in the core\n\n"+
				"It runs before configuration exists and cannot be run twice. Do the work in "+
				"New(cfg) instead, where it can fail and be tested.",
				rel, fset.Position(fn.Pos()).Line)
		}
	})
}

// TestCoreNeverForcesGarbageCollection.
//
// The Python service called gc.collect() every time a camera stopped. It pauses the whole
// process, and the watchdog restarts processors every 30 seconds -- so on a fleet with any
// churn the pause was close to continuous, and it landed on the cameras that were working.
//
// Freeing a session's VRAM is a real need and the real fix is an explicit Destroy, which this
// service does. Reaching for the collector to make that happen trades a leak for a stall.
func TestCoreNeverForcesGarbageCollection(t *testing.T) {
	forEachCoreFile(t, func(t *testing.T, rel string, file *ast.File, fset *token.FileSet) {
		ast.Inspect(file, func(n ast.Node) bool {
			sel, ok := n.(*ast.SelectorExpr)
			if !ok {
				return true
			}
			pkg, ok := sel.X.(*ast.Ident)
			if !ok {
				return true
			}
			bad := (pkg.Name == "runtime" && sel.Sel.Name == "GC") ||
				(pkg.Name == "debug" && sel.Sel.Name == "FreeOSMemory")
			if bad {
				t.Errorf("%s:%d: %s.%s stops the world\n\n"+
					"Release the resource explicitly instead -- a session has Destroy, a "+
					"buffer goes back to its pool.",
					rel, fset.Position(sel.Pos()).Line, pkg.Name, sel.Sel.Name)
			}
			return true
		})
	})
}

// forEachCoreFile parses every non-test .go file of the core.
//
// Test files are excluded on purpose: a test may legitimately hold package state for a fixture
// or force a collection to prove something about memory.
func forEachCoreFile(t *testing.T, fn func(*testing.T, string, *ast.File, *token.FileSet)) {
	t.Helper()
	root := packageRoot(t)
	fset := token.NewFileSet()
	err := filepath.WalkDir(filepath.Join(root, coreDir), func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if notCore[d.Name()] {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
			return nil
		}
		rel, _ := filepath.Rel(root, path)
		rel = filepath.ToSlash(rel)
		file, perr := parser.ParseFile(fset, path, nil, parser.SkipObjectResolution)
		if perr != nil {
			t.Errorf("parsing %s: %v", rel, perr)
			return nil
		}
		fn(t, rel, file, fset)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}
