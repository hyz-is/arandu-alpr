package streaming

import (
	"io"
	"syscall"
)

// framePipeSize is the pipe buffer asked for the frames: 1 MiB, which is the default ceiling of
// /proc/sys/fs/pipe-max-size for a process without CAP_SYS_RESOURCE.
const framePipeSize = 1 << 20

// fSetPipeSz is F_SETPIPE_SZ, which package syscall does not name.
const fSetPipeSz = 1031

// growPipe asks the kernel for a larger buffer on the pipe r reads from.
//
// ffmpeg writes a raw frame of several megabytes into a pipe that holds 64 KiB, so every frame
// crosses it in dozens of write and read round trips, each one a wakeup and a copy through the
// pipe's lock. A larger buffer cuts the round trips. It is best effort: when r is not a file, or
// the kernel refuses (the limit is lower, or the descriptor is not a pipe), the pipe keeps its
// size and the stream works the same, only slower.
func growPipe(r io.Reader, size int) {
	f, ok := r.(interface {
		SyscallConn() (syscall.RawConn, error)
	})
	if !ok {
		return
	}
	raw, err := f.SyscallConn()
	if err != nil {
		return
	}
	_ = raw.Control(func(fd uintptr) {
		_, _, _ = syscall.Syscall(syscall.SYS_FCNTL, fd, fSetPipeSz, uintptr(size))
	})
}
