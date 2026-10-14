//go:build !unix

package onnx

import "os"

// Without file descriptor duplication there is no way to capture what the C library writes, so
// the placement report is simply unavailable. AllCUDA treats an empty report as "not measured",
// never as success, so nothing downstream mistakes this for a clean run.

func dupStderr() (int, error) { return 0, errNoCapture }

func redirectStderr(*os.File) error { return errNoCapture }

func restoreStderr(int) error { return errNoCapture }

func closeFD(int) {}
