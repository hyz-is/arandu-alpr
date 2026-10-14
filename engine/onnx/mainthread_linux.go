//go:build linux

package onnx

import (
	"os"

	"golang.org/x/sys/unix"
)

// onMainThread uses the one property that identifies the main thread on Linux: it is the thread
// whose id equals the process id. Every other thread the runtime creates gets a different tid.
func onMainThread() bool { return unix.Gettid() == os.Getpid() }
