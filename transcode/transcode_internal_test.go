package transcode

import (
	"bytes"
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestSizeNeverScalesUpAndKeepsSidesEven(t *testing.T) {
	tr := New(Config{})
	for _, c := range []struct{ w, h, tw, th int }{
		{2560, 1440, 1920, 1080},
		{3840, 2160, 1920, 1080},
		{1920, 1080, 1920, 1080},
		{1280, 720, 1280, 720},
		{704, 576, 704, 576},
		{2688, 1520, 1908, 1080}, // 1909 down to even
		{1081, 1081, 1080, 1080},
	} {
		if w, h := tr.Size(c.w, c.h); w != c.tw || h != c.th {
			t.Errorf("Size(%d, %d) = %dx%d, want %dx%d", c.w, c.h, w, h, c.tw, c.th)
		}
	}
}

func joined(args []string) string { return " " + strings.Join(args, " ") + " " }

func TestTheCardDecodesScalesAndEncodesWhenItHasTheEncoder(t *testing.T) {
	tr := New(Config{HWAccel: "cuvid", Devices: []int{1}})
	a := joined(tr.Args(NVENC, "hvc1.1.6.L153.B0", 2560, 1440, 1))
	for _, want := range []string{
		" -c:v hevc_cuvid -gpu 1 -resize 1920x1080 -f mp4 -i pipe:0 ",
		" -vf fps=15 ", " -c:v h264_nvenc -preset p1 -tune ll ", " -b:v 3M ", " -g 15 -bf 0 ",
		" -movflags empty_moov+default_base_moof+frag_keyframe ", " pipe:1 ", " -an ", " -map 0:v:0 ",
	} {
		if !strings.Contains(a, want) {
			t.Errorf("the arguments lack %q:%s", want, a)
		}
	}
	if strings.Contains(a, "scale=") || strings.Contains(a, "libx264") {
		t.Errorf("the card path scales on the CPU or encodes on it:%s", a)
	}
}

func TestWithoutTheCardTheCPUDecodesScalesAndEncodes(t *testing.T) {
	tr := New(Config{FPS: 10, Bitrate: "2M", MaxHeight: 720})
	a := joined(tr.Args(X264, "hvc1.1.6.L153.B0", 2560, 1440, -1))
	for _, want := range []string{" -vf fps=10,scale=1280:720 ", " -c:v libx264 -preset ultrafast -tune zerolatency ", " -b:v 2M ", " -g 10 "} {
		if !strings.Contains(a, want) {
			t.Errorf("the arguments lack %q:%s", want, a)
		}
	}
	if strings.Contains(a, "cuvid") || strings.Contains(a, "nvenc") || strings.Contains(a, "-gpu") {
		t.Errorf("the CPU path asks for the card:%s", a)
	}
	// A camera already within the limit is not scaled.
	if a := joined(tr.Args(X264, "hvc1.1.6.L93.B0", 1280, 720, -1)); strings.Contains(a, "scale=") {
		t.Errorf("a camera within the limit is scaled:%s", a)
	}
}

func TestConversionsAreSpreadOverTheCardsInTurn(t *testing.T) {
	var mu sync.Mutex
	var seen []string
	tr := New(Config{Encoder: NVENC, Devices: []int{0, 1}, Runner: func(ctx context.Context, name string, args ...string) *exec.Cmd {
		mu.Lock()
		seen = append(seen, joined(args))
		mu.Unlock()
		return exec.CommandContext(ctx, "sh", "-c", "exit 0")
	}})
	for i := 0; i < 4; i++ {
		_ = tr.Transcode(context.Background(), "cam", "hvc1.1.6.L93.B0", 1280, 720, strings.NewReader(""), &bytes.Buffer{})
	}
	for i, a := range seen {
		want := " -gpu " + strconv.Itoa(i%2) + " "
		if !strings.Contains(a, want) {
			t.Errorf("conversion %d: %s lacks %q", i, a, want)
		}
	}
}

func TestAutoFallsBackToTheCPUWhenTheCardHasNoEncoder(t *testing.T) {
	tr := New(Config{Runner: func(ctx context.Context, name string, args ...string) *exec.Cmd {
		if strings.Contains(strings.Join(args, " "), "h264_nvenc") {
			return exec.CommandContext(ctx, "sh", "-c", "echo 'No NVENC capable devices found' >&2; exit 1")
		}
		return exec.CommandContext(ctx, "sh", "-c", "exit 0")
	}})
	if got := tr.Encoder(context.Background()); got != X264 {
		t.Fatalf("a machine whose NVENC does not work chose %s", got)
	}
}

func TestAutoUsesTheCardWhenItEncodes(t *testing.T) {
	tr := New(Config{Runner: func(ctx context.Context, name string, args ...string) *exec.Cmd {
		return exec.CommandContext(ctx, "sh", "-c", "exit 0")
	}})
	if got := tr.Encoder(context.Background()); got != NVENC {
		t.Fatalf("a machine whose NVENC works chose %s", got)
	}
}

func TestAnEncoderThatFailsAtOnceIsLeftAndTheErrorSaysWhy(t *testing.T) {
	calls := 0
	tr := New(Config{Runner: func(ctx context.Context, name string, args ...string) *exec.Cmd {
		calls++
		if calls == 1 {
			return exec.CommandContext(ctx, "sh", "-c", "exit 0") // the probe
		}
		return exec.CommandContext(ctx, "sh", "-c", "echo 'OpenEncodeSessionEx failed: out of sessions' >&2; exit 1")
	}})
	err := tr.Transcode(context.Background(), "cam", "hvc1.1.6.L93.B0", 1280, 720, strings.NewReader(""), &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "out of sessions") {
		t.Fatalf("error %v, want ffmpeg's own words", err)
	}
	if tr.Encoder(context.Background()) != X264 {
		t.Fatal("the next conversion tries the encoder that just failed")
	}
}

func TestCancellingStopsTheConversionAtOnce(t *testing.T) {
	tr := New(Config{Encoder: X264, Runner: func(ctx context.Context, name string, args ...string) *exec.Cmd {
		return exec.CommandContext(ctx, "sleep", "30")
	}})
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	r, w := io_pipe()
	defer w.Close()
	go func() { done <- tr.Transcode(ctx, "cam", "hvc1", 1280, 720, r, &bytes.Buffer{}) }()
	time.Sleep(100 * time.Millisecond)
	cancel()
	select {
	case err := <-done:
		if err != context.Canceled {
			t.Fatalf("error %v, want context.Canceled", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the conversion did not stop when it was cancelled")
	}
}

// The real thing: an H.265 camera's container in, an H.264 picture of at most the limit out.
func TestARealH265StreamComesOutAsH264OfTheSizeAndRateAsked(t *testing.T) {
	for _, name := range []string{"ffmpeg", "ffprobe"} {
		if _, err := exec.LookPath(name); err != nil {
			t.Skip(name + " is not installed")
		}
	}
	dir := t.TempDir()
	in := filepath.Join(dir, "camera.mp4")
	if out, err := exec.Command("ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=25", "-t", "4",
		"-c:v", "libx265", "-preset", "ultrafast", "-g", "25", "-pix_fmt", "yuv420p", "-tag:v", "hvc1",
		"-f", "mp4", "-movflags", "empty_moov+default_base_moof+frag_keyframe", "-frag_duration", "100000", in).CombinedOutput(); err != nil {
		t.Skipf("no libx265: %v %s", err, out)
	}
	data, err := os.ReadFile(in)
	if err != nil {
		t.Fatal(err)
	}
	tr := New(Config{Encoder: X264, MaxHeight: 360, FPS: 10})
	var out bytes.Buffer
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if err := tr.Transcode(ctx, "cam", "hvc1.1.6.L93.B0", 1280, 720, bytes.NewReader(data), &out); err != nil && !strings.Contains(err.Error(), "ffmpeg stopped") {
		t.Fatalf("Transcode: %v", err)
	}
	if out.Len() < 1000 {
		t.Fatalf("only %d bytes came out", out.Len())
	}
	file := filepath.Join(dir, "out.mp4")
	if err := os.WriteFile(file, out.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	probe, err := exec.Command("ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
		"-show_entries", "stream=codec_name,width,height,nb_read_frames", "-show_entries", "format=duration", "-of", "csv=p=0", file).CombinedOutput()
	if err != nil {
		t.Fatalf("ffprobe: %v %s", err, probe)
	}
	fields := strings.Fields(strings.ReplaceAll(string(probe), ",", " "))
	if len(fields) < 5 || fields[0] != "h264" || fields[1] != "640" || fields[2] != "360" {
		t.Fatalf("ffprobe says %q, want h264 at 640x360", probe)
	}
	frames, _ := strconv.Atoi(fields[3])
	seconds, _ := strconv.ParseFloat(fields[4], 64)
	if seconds <= 0 || float64(frames)/seconds > 10.6 {
		t.Errorf("%d pictures in %.2fs: more than the 10 a second asked for", frames, seconds)
	}
	if frames < 20 {
		t.Errorf("only %d pictures came out of four seconds", frames)
	}
}

func io_pipe() (*os.File, *os.File) {
	r, w, _ := os.Pipe()
	return r, w
}
