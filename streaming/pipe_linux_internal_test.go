package streaming

import (
	"os"
	"syscall"
	"testing"
)

func TestFramePipeGrowsWhenTheKernelAllows(t *testing.T) {
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	defer w.Close()

	size := func() int {
		raw, err := r.SyscallConn()
		if err != nil {
			t.Fatal(err)
		}
		n := 0
		_ = raw.Control(func(fd uintptr) {
			v, _, errno := syscall.Syscall(syscall.SYS_FCNTL, fd, 1032, 0) // F_GETPIPE_SZ
			if errno == 0 {
				n = int(v)
			}
		})
		return n
	}

	before := size()
	growPipe(r, framePipeSize)
	after := size()
	if after < before {
		t.Errorf("pipe shrank from %d to %d", before, after)
	}
	// An unprivileged process may be held below 1 MiB by /proc/sys/fs/pipe-max-size; then the
	// request is refused and the pipe must keep working at its old size.
	if after != framePipeSize {
		t.Logf("pipe is %d bytes, not %d: the host limit refused the request", after, framePipeSize)
	}
	if _, err := w.Write([]byte("x")); err != nil {
		t.Errorf("pipe unusable after resize: %v", err)
	}
}

func TestGrowPipeIgnoresWhatIsNotAPipe(t *testing.T) {
	growPipe(nopReader{}, framePipeSize) // must not panic
	f, err := os.Open(os.DevNull)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	growPipe(f, framePipeSize) // fcntl fails with EBADF/EINVAL; ignored
}

type nopReader struct{}

func (nopReader) Read([]byte) (int, error) { return 0, nil }
