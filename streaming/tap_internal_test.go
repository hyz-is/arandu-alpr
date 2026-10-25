package streaming

import (
	"bytes"
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// These tests run the real ffmpeg: what is under test is that one process feeds two readers, the
// frames the plates are read from and the compressed copy a live view is made of, and that the
// second cannot hurt the first. A fake would prove the arguments and nothing about the pipe.

const (
	tapW = 160
	tapH = 120
)

// tapSource serves a short clip over HTTP, the way a camera is read, and says where.
func tapSource(t *testing.T) string {
	t.Helper()
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg is not installed")
	}
	dir := t.TempDir()
	clip := filepath.Join(dir, "clip.mp4")
	out, err := exec.Command("ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i",
		"testsrc=size="+strconv.Itoa(tapW)+"x"+strconv.Itoa(tapH)+":rate=10", "-t", "3",
		"-c:v", "libx264", "-preset", "ultrafast", "-g", "10", "-pix_fmt", "yuv420p",
		"-movflags", "+faststart", clip).CombinedOutput()
	if err != nil {
		t.Skipf("ffmpeg cannot make the test clip (no libx264?): %v\n%s", err, out)
	}
	srv := httptest.NewServer(http.FileServer(http.Dir(dir)))
	t.Cleanup(srv.Close)
	return srv.URL + "/clip.mp4"
}

// tapRunner answers ffprobe with the clip's geometry and runs everything else for real.
func tapRunner(record *[][]string) Runner {
	var mu sync.Mutex
	return func(ctx context.Context, name string, args ...string) *exec.Cmd {
		if name == "ffprobe" {
			return exec.CommandContext(ctx, "sh", "-c", "printf %s "+strconv.Quote(
				`{"streams":[{"width":160,"height":120,"r_frame_rate":"10/1","codec_name":"h264"}]}`))
		}
		mu.Lock()
		if record != nil {
			*record = append(*record, args)
		}
		mu.Unlock()
		return exec.CommandContext(ctx, name, args...)
	}
}

type memoryTap struct {
	mu     sync.Mutex
	buf    bytes.Buffer
	closed chan struct{}
	failAt int // fail every write after this many bytes; 0 never
}

func newMemoryTap() *memoryTap { return &memoryTap{closed: make(chan struct{})} }

func (m *memoryTap) Write(p []byte) (int, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.failAt > 0 && m.buf.Len() >= m.failAt {
		return 0, errors.New("tap: refused")
	}
	return m.buf.Write(p)
}

func (m *memoryTap) Close() error { close(m.closed); return nil }

func (m *memoryTap) bytes() []byte {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]byte(nil), m.buf.Bytes()...)
}

// readAll takes every frame of a stream until it ends, and says how many there were.
func readAll(t *testing.T, s *Stream) int {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	n := 0
	for {
		f, ok := s.Next(ctx)
		if !ok {
			return n
		}
		f.Release()
		n++
	}
}

func boxes(b []byte) []string {
	var out []string
	for len(b) >= 8 {
		size := int(b[0])<<24 | int(b[1])<<16 | int(b[2])<<8 | int(b[3])
		if size < 8 || size > len(b) {
			break
		}
		out = append(out, string(b[4:8]))
		b = b[size:]
	}
	return out
}

func TestTheTapGetsTheCompressedVideoWhileTheFramesStillFlow(t *testing.T) {
	url := tapSource(t)
	var cmds [][]string
	o := NewOpener(2, tapRunner(&cmds))
	o.SetGuard(Guard{AllowPrivate: true})
	tap := newMemoryTap()

	s, err := o.Open(context.Background(), Config{URL: url, Tap: tap})
	if err != nil {
		t.Fatal(err)
	}
	frames := readAll(t, s)
	s.Close()

	if frames < 20 {
		t.Errorf("%d frames reached the reader, want about 30: the tap must not take them", frames)
	}
	select {
	case <-tap.closed:
	case <-time.After(5 * time.Second):
		t.Fatal("the tap was not closed when the stream ended")
	}
	got := boxes(tap.bytes())
	if len(got) < 4 || got[0] != "ftyp" || got[1] != "moov" {
		t.Fatalf("the tap received boxes %v, want ftyp, moov and fragments", got)
	}
	moofs := 0
	for _, b := range got {
		if b == "moof" {
			moofs++
		}
	}
	if moofs < 10 {
		t.Errorf("%d fragments in 3 seconds: they should be cut every tenth of a second, not once per group", moofs)
	}
	joined := strings.Join(cmds[0], " ")
	if !strings.Contains(joined, "-c:v copy") || !strings.HasSuffix(joined, "pipe:3") {
		t.Errorf("ffmpeg was run as %q: the copy is a second output to pipe:3", joined)
	}
}

func TestATapThatFailsNeverStopsTheFrames(t *testing.T) {
	url := tapSource(t)
	o := NewOpener(2, tapRunner(nil))
	o.SetGuard(Guard{AllowPrivate: true})
	tap := newMemoryTap()
	tap.failAt = 1 // the first write is taken, every later one is refused

	s, err := o.Open(context.Background(), Config{URL: url, Tap: tap})
	if err != nil {
		t.Fatal(err)
	}
	frames := readAll(t, s)
	s.Close()
	if frames < 20 {
		t.Errorf("%d frames after the tap failed: a full pipe stalls ffmpeg and the plates with it", frames)
	}
	select {
	case <-tap.closed:
	case <-time.After(5 * time.Second):
		t.Fatal("a failed tap was not closed")
	}
}

func TestAStreamThatNeverOpensClosesItsTap(t *testing.T) {
	o := NewOpener(1, func(ctx context.Context, name string, args ...string) *exec.Cmd {
		return exec.CommandContext(ctx, "sh", "-c", "exit 1") // ffprobe finds nothing
	})
	o.SetGuard(Guard{AllowPrivate: true})
	tap := newMemoryTap()
	if _, err := o.Open(context.Background(), Config{URL: "http://127.0.0.1:9/x", Tap: tap}); err == nil {
		t.Fatal("a stream that cannot be probed opened")
	}
	select {
	case <-tap.closed:
	default:
		t.Fatal("the tap of a stream that never opened stays open: the hub would keep a source nobody writes")
	}
}

func TestWithoutATapFfmpegHasOneOutput(t *testing.T) {
	var cmds [][]string
	o := NewOpener(1, func(ctx context.Context, name string, args ...string) *exec.Cmd {
		if name == "ffmpeg" {
			cmds = append(cmds, args)
			return exec.CommandContext(ctx, "sh", "-c", "exit 0")
		}
		return tapRunner(nil)(ctx, name, args...)
	})
	o.SetGuard(Guard{AllowPrivate: true})
	s, err := o.Open(context.Background(), Config{URL: "http://127.0.0.1:9/x"})
	if err != nil {
		t.Fatal(err)
	}
	s.Close()
	if joined := strings.Join(cmds[0], " "); strings.Contains(joined, "pipe:3") || strings.Contains(joined, "copy") {
		t.Errorf("ffmpeg was run as %q, and nothing asked for a tap", joined)
	}
}
