// Package onnx runs the models through ONNX Runtime.
//
// It is the only package in the service that imports onnxruntime_go, and it is deliberately not
// behind a build tag: the library reaches its shared object through dlopen at runtime, so this
// code compiles and type-checks on a laptop with no ONNX Runtime and no GPU. A build tag would
// buy nothing and would let this package rot unnoticed between visits to a CUDA machine.
//
// # Why main.main has to cooperate
//
// Creating a CUDA session on a thread that is not the process's main thread crashes with
// SIGSEGV (ONNX Runtime issue #26610). Go makes that awkward on purpose: goroutines migrate
// between OS threads, and runtime.LockOSThread pins a goroutine to SOME thread, not to the main
// one.
//
// The only way to reach the main thread is a documented property of the runtime: the main
// goroutine stays on the main thread for the duration of package init, and is released just
// before main.main is called. Calling runtime.LockOSThread from an init keeps that pin, so
// main.main then runs on the process's main thread.
//
// That is why this package has an init() -- the one in the service -- and why the application
// has to hand main.main to Builder.Serve rather than blocking on its own server.
package onnx

import "runtime"

// init pins the main goroutine to the process's main thread.
//
// It has to be an init, and it has to run before main.main. See the package comment; there is
// no other way to reach that thread from Go.
func init() {
	runtime.LockOSThread()
}

// OnMainThread reports whether the caller is running on the process's main thread.
//
// The property this whole design rests on -- that an init-time LockOSThread carries the main
// thread into main.main -- is an implementation detail of the runtime. It is long-standing and
// commented in the runtime source, but it is not part of the language specification.
//
// So it is checked rather than trusted. The failure it prevents is a SIGSEGV somewhere inside
// CUDA with no Go stack worth reading; the check turns that into one sentence at boot.
func OnMainThread() bool { return onMainThread() }
