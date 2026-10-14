//go:build unix

package onnx

import (
	"os"

	"golang.org/x/sys/unix"
)

// The three primitives capturePlacement needs. They are separated by build tag because file
// descriptor duplication has no portable form, and the placement report is only load-bearing on
// the platform that has a GPU.

func dupStderr() (int, error) { return unix.Dup(int(os.Stderr.Fd())) }

func redirectStderr(w *os.File) error { return unix.Dup2(int(w.Fd()), int(os.Stderr.Fd())) }

func restoreStderr(saved int) error { return unix.Dup2(saved, int(os.Stderr.Fd())) }

func closeFD(fd int) { _ = unix.Close(fd) }
