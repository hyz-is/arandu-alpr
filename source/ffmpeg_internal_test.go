package source

import (
	"context"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/hyz-is/arandu-alpr/contract"
	"github.com/hyz-is/arandu-alpr/engine"
)

// This adapter is twelve lines of pump loop and it is the single most dangerous seam in the
// service, because the failure it can cause has no error attached to it.
//
// Frame pixels are borrowed from a ring of three buffers. If the borrow does not travel with the
// frame, or a frame is dropped on the floor without releasing, the ring runs dry and the camera
// stops producing: no panic, no log line, no metric that says why. A feed that simply goes quiet
// looks like a network problem, and that is where someone will go looking.

// fakeFFmpeg produces a fixed number of solid-colour frames and then ends.
//
// It has to write real bytes to a real pipe, because what is being tested is the handoff between
// two goroutines and a subprocess, not arithmetic.
func fakeFFmpeg(t *testing.T, width, height, frames int) Config {
	t.Helper()
	const probe = `{"streams":[{"width":%d,"height":%d,"r_frame_rate":"25/1","codec_name":"h264"}]}`

	return Config{
		MaxConcurrentOpens: 4,
		// The fake cameras have no address; the guard has its own tests in streaming.
		AllowPrivate: true,
		Runner: func(ctx context.Context, name string, args ...string) *exec.Cmd {
			if name == "ffprobe" {
				return exec.CommandContext(ctx, "sh", "-c",
					"printf '%s' '"+sprintfProbe(probe, width, height)+"'")
			}
			// Each frame is a distinct non-uniform pattern: a solid frame would be dropped as
			// blank before it ever reached the ring, which would test the wrong thing.
			size := width * height * 3
			script := "for i in $(seq 1 " + itoa(frames) + "); do " +
				"head -c " + itoa(size) + " /dev/urandom; done"
			return exec.CommandContext(ctx, "sh", "-c", script)
		},
	}
}

func TestBorrowedBufferTravelsWithTheFrameAndIsReleased(t *testing.T) {
	// The whole contract in one test: every frame the loop hands out carries a Release, and
	// releasing it keeps the ring supplying frames past its own depth of three.
	d := NewFFmpegDialer(fakeFFmpeg(t, 32, 24, 12))

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	s, err := d.Dial(ctx, contract.CameraSpec{URL: "rtsp://cam/live"})
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	defer s.Close()

	seen := 0
	for f := range s.Frames() {
		if f.Release == nil {
			t.Fatalf("frame %d arrived without a Release; its buffer can never go back", seen)
		}
		if len(f.Pix) != 32*24*3 {
			t.Fatalf("frame %d has %d bytes, want %d", seen, len(f.Pix), 32*24*3)
		}
		engine.ReleaseFrame(f)
		seen++
	}

	// More than the ring is deep. Without releases this stops at three and the channel never
	// closes on its own, which is exactly the silent stall being guarded against.
	if seen < 4 {
		t.Fatalf("only %d frames got through; the ring is not being replenished", seen)
	}
}

func TestFramesStopWhenBuffersAreNeverReturned(t *testing.T) {
	// The inverse, stated as a test so the consequence of forgetting is written down somewhere
	// rather than learned from a quiet camera. Holding frames without releasing must starve the
	// producer -- and must NOT deadlock the reader, block ffmpeg, or allocate its way around the
	// bound.
	d := NewFFmpegDialer(fakeFFmpeg(t, 32, 24, 200))

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	s, err := d.Dial(ctx, contract.CameraSpec{URL: "rtsp://cam/live"})
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	defer s.Close()

	var held []engine.Frame
	deadline := time.After(2 * time.Second)
	for {
		select {
		case f, ok := <-s.Frames():
			if !ok {
				goto done
			}
			held = append(held, f)
			if len(held) > 20 {
				t.Fatalf("received %d frames while holding every buffer; the ring is not bounded",
					len(held))
			}
		case <-deadline:
			goto done
		}
	}
done:
	if len(held) == 0 {
		t.Fatal("no frames arrived at all")
	}
	for _, f := range held {
		engine.ReleaseFrame(f)
	}
}

func TestClosingWithAFramePendingDoesNotLoseABuffer(t *testing.T) {
	// A shutdown that races a frame in flight must not leave that buffer borrowed forever. The
	// ring would come back one buffer short on the next connection, and a camera that reconnects
	// often would lose them one at a time until it stopped.
	d := NewFFmpegDialer(fakeFFmpeg(t, 32, 24, 200))

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	s, err := d.Dial(ctx, contract.CameraSpec{URL: "rtsp://cam/live"})
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}

	// Take one, then close without taking the next -- which is almost certainly already waiting
	// to be handed over.
	select {
	case f := <-s.Frames():
		engine.ReleaseFrame(f)
	case <-time.After(3 * time.Second):
		t.Fatal("no frame arrived")
	}

	s.Close()
	// Draining is what lets the pump goroutine finish rather than block forever on a handover
	// nobody is taking; a leaked goroutine per reconnect is its own slow failure.
	for f := range s.Frames() {
		engine.ReleaseFrame(f)
	}
}

func TestDialFailsWhenTheCameraCannotBeProbed(t *testing.T) {
	d := NewFFmpegDialer(Config{
		Runner: func(ctx context.Context, name string, args ...string) *exec.Cmd {
			return exec.CommandContext(ctx, "sh", "-c", "exit 1")
		},
	})
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if _, err := d.Dial(ctx, contract.CameraSpec{URL: "rtsp://cam/live"}); err == nil {
		t.Fatal("Dial succeeded against a camera that cannot be probed")
	}
}

func TestDialNeverPutsCredentialsInAnError(t *testing.T) {
	// Errors from here reach the operator panel and the connectivity log. A stream URL carries
	// the camera's username and password, and this is the boundary where it would escape.
	d := NewFFmpegDialer(Config{
		Runner: func(ctx context.Context, name string, args ...string) *exec.Cmd {
			return exec.CommandContext(ctx, "sh", "-c", "exit 1")
		},
	})
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	_, err := d.Dial(ctx, contract.CameraSpec{
		URL: "rtsp://cam.local/live", Username: "admin", Password: "sup3rs3cret",
	})
	if err == nil {
		t.Fatal("expected an error")
	}
	if strings.Contains(err.Error(), "sup3rs3cret") || strings.Contains(err.Error(), "admin") {
		t.Fatalf("the error carries the camera's credentials: %v", err)
	}
}

// Small helpers, kept local so the test file has no dependency of its own.

func sprintfProbe(format string, w, h int) string {
	out := strings.Replace(format, "%d", itoa(w), 1)
	return strings.Replace(out, "%d", itoa(h), 1)
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}

func TestMain(m *testing.M) {
	// Every test here drives a real /bin/sh. Skipping rather than failing keeps the suite honest
	// on a machine that has no shell, instead of reporting a pipeline fault.
	if _, err := exec.LookPath("sh"); err != nil {
		os.Exit(0)
	}
	os.Exit(m.Run())
}

func TestDecodeLandsOnTheCameraOwnCard(t *testing.T) {
	// Without this every camera decodes on device 0, because that is what ffmpeg picks when
	// nobody says otherwise. On a two-card box that puts the whole fleet's decoding on one
	// card's NVDEC block while the other's sits idle -- and both cards report healthy.
	for placement, want := range map[string]int{
		"cuda:0": 0,
		"cuda:1": 1,
		"":       -1,
		"gpu:1":  -1, // not a form we emit; better unset than guessed
		"cuda:x": -1,
	} {
		if got := deviceFor(contract.CameraSpec{Placement: placement}); got != want {
			t.Errorf("placement %q gave device %d, want %d", placement, got, want)
		}
	}
}
