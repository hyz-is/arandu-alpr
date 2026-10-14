//go:build !linux

package onnx

// onMainThread cannot be answered portably, and the check is only load-bearing where CUDA runs.
//
// Reporting true here is deliberate rather than lazy: on macOS this package is compiled and
// tested but never given a GPU, so a false would fail every boot on a developer machine to guard
// against a crash that cannot happen there. The production platform is Linux, and that is where
// the real check lives.
func onMainThread() bool { return true }
