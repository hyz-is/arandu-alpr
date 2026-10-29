//go:build !linux

package streaming

import "io"

// framePipeSize is only meaningful on Linux, where the pipe buffer can be resized.
const framePipeSize = 1 << 20

// growPipe does nothing off Linux: F_SETPIPE_SZ is a Linux call.
func growPipe(io.Reader, int) {}
