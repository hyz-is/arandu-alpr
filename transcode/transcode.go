// Package transcode converts a camera's video to H.264 for a live view, on demand.
//
// A browser decodes H.264 everywhere and H.265 on some machines, and the cameras that matter most
// send H.265. The conversion is a separate ffmpeg process fed the copy of the camera's stream the
// plate reader's own ffmpeg already writes (contract.VideoTap), and it writes a smaller picture for
// a person: at most 1080 lines and fifteen pictures a second. It is never on the path the plates
// are read from. That path decodes the camera's full-size stream once, and keeps doing so at the
// rate it always did whether anyone is watching or not; this package only costs while somebody is,
// on the encoder of the card (NVENC) when there is one and on the CPU when there is not.
//
// Whoever owns the viewers decides when to run it and how many at once; this package runs one
// conversion each time it is asked.
package transcode

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Encoders.
const (
	// Auto uses NVENC when the card has it and x264 when it does not, decided once.
	Auto = "auto"
	// NVENC is the card's encoder: no CPU for the encode, and a decode on NVDEC as well when
	// Config.HWAccel is "cuvid".
	NVENC = "nvenc"
	// X264 is the CPU: libx264, ultrafast, zero latency.
	X264 = "x264"
)

// Defaults of Config.
const (
	DefaultMaxHeight = 1080
	DefaultFPS       = 15
	DefaultBitrate   = "3M"
)

// Runner builds the command for ffmpeg. It exists so tests can stand in for it.
type Runner func(ctx context.Context, name string, args ...string) *exec.Cmd

// Config is what a Transcoder takes.
type Config struct {
	// FFmpegPath is the binary; empty is "ffmpeg" on the PATH.
	FFmpegPath string
	// Encoder is Auto, NVENC or X264. Empty is Auto.
	Encoder string
	// HWAccel is "cuvid" to decode on the card too (NVDEC), and empty to decode on the CPU. It is
	// the same setting the plate reader uses, and it only applies when the encoder is NVENC.
	HWAccel string
	// Devices are the cards the conversions are spread over, in turn. Empty leaves the choice to
	// ffmpeg, which is card 0.
	Devices []int
	// MaxHeight is the tallest picture sent, in lines; a camera of a larger picture is scaled down.
	// Zero takes DefaultMaxHeight.
	MaxHeight int
	// FPS is the most pictures a second sent. Zero takes DefaultFPS.
	FPS float64
	// Bitrate is the target of the encoder, in ffmpeg's spelling. Empty takes DefaultBitrate.
	Bitrate string

	// Runner substitutes for ffmpeg in tests. Nil uses os/exec.
	Runner Runner
	Logger *slog.Logger
}

// Transcoder runs the conversions.
type Transcoder struct {
	cfg  Config
	log  *slog.Logger
	next atomic.Uint32

	probe     sync.Once
	nvenc     bool
	nvencDead atomic.Bool
}

// New returns a Transcoder.
func New(cfg Config) *Transcoder {
	if cfg.FFmpegPath == "" {
		cfg.FFmpegPath = "ffmpeg"
	}
	if cfg.Encoder == "" {
		cfg.Encoder = Auto
	}
	if cfg.MaxHeight <= 0 {
		cfg.MaxHeight = DefaultMaxHeight
	}
	if cfg.FPS <= 0 {
		cfg.FPS = DefaultFPS
	}
	if cfg.Bitrate == "" {
		cfg.Bitrate = DefaultBitrate
	}
	if cfg.Runner == nil {
		cfg.Runner = func(ctx context.Context, name string, args ...string) *exec.Cmd {
			return exec.CommandContext(ctx, name, args...)
		}
	}
	if cfg.Logger == nil {
		cfg.Logger = slog.New(slog.DiscardHandler)
	}
	return &Transcoder{cfg: cfg, log: cfg.Logger}
}

// Encoder says which encoder a conversion started now would use: NVENC or X264.
func (t *Transcoder) Encoder(ctx context.Context) string {
	switch t.cfg.Encoder {
	case NVENC:
		return NVENC
	case X264:
		return X264
	}
	t.probe.Do(func() { t.nvenc = t.nvencWorks(ctx) })
	if t.nvenc && !t.nvencDead.Load() {
		return NVENC
	}
	return X264
}

// nvencWorks encodes one picture to nothing, which is the only honest way to learn whether the
// card has the encoder: the binary may list h264_nvenc on a machine with no GPU.
func (t *Transcoder) nvencWorks(ctx context.Context) bool {
	ctx, cancel := context.WithTimeout(ctx, 20*time.Second)
	defer cancel()
	cmd := t.cfg.Runner(ctx, t.cfg.FFmpegPath, "-hide_banner", "-loglevel", "error",
		"-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.2:r=10", "-frames:v", "1",
		"-c:v", "h264_nvenc", "-f", "null", "-")
	if err := cmd.Run(); err != nil {
		t.log.Info("transcode: the card has no H.264 encoder, using x264", "error", err)
		return false
	}
	return true
}

// Size is the picture sent for a camera of w by h: scaled down to MaxHeight, never up, and always
// of even sides, which the encoders need.
func (t *Transcoder) Size(w, h int) (int, int) {
	if h <= 0 || w <= 0 {
		return w, h
	}
	th := min(h, t.cfg.MaxHeight)
	tw := w * th / h
	return tw &^ 1, th &^ 1
}

// Args are ffmpeg's arguments for one conversion, with the encoder and the card already chosen.
func (t *Transcoder) Args(encoder, codec string, w, h, device int) []string {
	tw, th := t.Size(w, h)
	scale := tw != w || th != h
	fps := strconv.FormatFloat(t.cfg.FPS, 'g', -1, 64)
	gop := strconv.Itoa(max(1, int(t.cfg.FPS+0.5))) // a keyframe a second: a viewer starts within one

	// The container comes on stdin as the plate reader's ffmpeg wrote it: low delay, and no waiting
	// to see more of it than the header says.
	args := []string{"-hide_banner", "-loglevel", "error", "-fflags", "nobuffer", "-flags", "low_delay",
		"-probesize", "262144", "-analyzeduration", "0"}
	filters := []string{"fps=" + fps}
	hwDecode := encoder == NVENC && t.cfg.HWAccel == "cuvid" && cuvid(codec) != ""
	if hwDecode {
		args = append(args, "-c:v", cuvid(codec))
		if device >= 0 {
			args = append(args, "-gpu", strconv.Itoa(device))
		}
		if scale {
			// NVDEC scales as it decodes, which is cheaper than any filter after it.
			args = append(args, "-resize", fmt.Sprintf("%dx%d", tw, th))
		}
	} else if scale {
		filters = append(filters, fmt.Sprintf("scale=%d:%d", tw, th))
	}
	args = append(args, "-f", "mp4", "-i", "pipe:0", "-map", "0:v:0", "-an", "-sn", "-vf", strings.Join(filters, ","))

	switch encoder {
	case NVENC:
		args = append(args, "-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ll", "-zerolatency", "1",
			"-rc", "cbr", "-b:v", t.cfg.Bitrate)
		if device >= 0 {
			args = append(args, "-gpu", strconv.Itoa(device))
		}
	default:
		args = append(args, "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
			"-b:v", t.cfg.Bitrate, "-maxrate", t.cfg.Bitrate, "-bufsize", t.cfg.Bitrate)
	}
	return append(args, "-g", gop, "-bf", "0", "-pix_fmt", "yuv420p",
		"-f", "mp4", "-movflags", "empty_moov+default_base_moof+frag_keyframe", "-frag_duration", "100000", "pipe:1")
}

func cuvid(codec string) string {
	switch {
	case strings.HasPrefix(codec, "hvc1"), strings.HasPrefix(codec, "hev1"):
		return "hevc_cuvid"
	case strings.HasPrefix(codec, "avc1"), strings.HasPrefix(codec, "avc3"):
		return "h264_cuvid"
	}
	return ""
}

// Transcode runs one conversion of one camera: the camera's container in, an H.264 container out,
// until ctx is cancelled, the input ends or ffmpeg stops. It is what arandu-live's Hub asks for.
func (t *Transcoder) Transcode(ctx context.Context, camera, codec string, width, height int, in io.Reader, out io.Writer) error {
	encoder := t.Encoder(ctx)
	device := -1
	if n := len(t.cfg.Devices); n > 0 {
		device = t.cfg.Devices[int(t.next.Add(1)-1)%n]
	}
	cmd := t.cfg.Runner(ctx, t.cfg.FFmpegPath, t.Args(encoder, codec, width, height, device)...)
	cmd.Stdin = in
	cmd.Stdout = out
	var stderr tail
	cmd.Stderr = &stderr
	// A conversion that ends leaves its stdin copier waiting on a reader nobody feeds any more.
	cmd.WaitDelay = time.Second

	started := time.Now()
	t.log.Info("transcode: started", "camera", camera, "encoder", encoder, "codec", codec, "device", device)
	err := cmd.Run()
	if ctx.Err() != nil {
		return ctx.Err()
	}
	if err == nil {
		err = errors.New("ffmpeg stopped")
	}
	if encoder == NVENC && t.cfg.Encoder == Auto && time.Since(started) < 5*time.Second {
		// It never got going: the card's encoder is not usable now (a session limit, a driver), and
		// the next conversion uses the CPU rather than failing the same way.
		t.nvencDead.Store(true)
		t.log.Warn("transcode: NVENC failed at once, using x264 from now on", "camera", camera, "error", stderr.String())
	}
	return fmt.Errorf("transcode: %w: %s", err, stderr.String())
}

// tail keeps the last bytes ffmpeg wrote to its error output, for the message when it fails.
type tail struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (t *tail) Write(p []byte) (int, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.buf.Write(p)
	if over := t.buf.Len() - 2048; over > 0 {
		t.buf.Next(over)
	}
	return len(p), nil
}

func (t *tail) String() string {
	t.mu.Lock()
	defer t.mu.Unlock()
	return strings.TrimSpace(t.buf.String())
}
