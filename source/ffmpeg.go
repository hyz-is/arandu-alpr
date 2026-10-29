// Package source connects the ffmpeg capture to the camera loop.
//
// It exists because the two halves have different shapes on purpose. The capture side hands out
// borrowed buffers and wants an explicit Next/Release pair, so a forgotten release is visible at
// the call site. The camera loop wants a channel it can select on alongside its context. This
// package is the one place that reconciles them, and it is small enough to read in one sitting --
// which matters, because a mistake here drains the buffer ring and the symptom is a camera that
// simply stops, with no error anywhere.
package source

import (
	"context"
	"io"
	"net"
	"strconv"
	"strings"
	"time"

	"github.com/hyz-is/arandu-alpr/contract"
	"github.com/hyz-is/arandu-alpr/engine"
	"github.com/hyz-is/arandu-alpr/streaming"
)

// FFmpegDialer opens camera streams through ffmpeg.
//
// One Dialer serves a whole fleet: it holds the semaphore that bounds how many cameras may be
// connecting at once, which is what keeps one unreachable camera from delaying the others.
type FFmpegDialer struct {
	opener *streaming.Opener
	cfg    Config
}

// Config tunes the capture.
type Config struct {
	// MaxConcurrentOpens bounds simultaneous connects. Zero uses the streaming default.
	MaxConcurrentOpens int

	// Runner substitutes for ffmpeg and ffprobe in tests.
	Runner streaming.Runner

	// OpenTimeout caps the probe and the first frame; ReadTimeout is how long a running stream
	// may go silent before it counts as dead. Zero uses the streaming defaults.
	//
	// They were declared here and never passed through, which is the worst kind of knob: setting
	// it changes nothing and the camera keeps the default timeout, so a deployment tuned for slow
	// cameras behaves as though it had not been.
	OpenTimeout time.Duration
	ReadTimeout time.Duration

	// DebugBufferOwnership poisons released buffers so a use-after-release reads as obvious
	// nonsense instead of a picture from a moment ago.
	DebugBufferOwnership bool

	// HWAccel decodes in hardware -- "cuvid" for NVDEC. Empty decodes on the CPU.
	//
	// Worth turning on wherever the hardware has it: decoding one 1080p30 stream costs 5.6% of a
	// CPU core, against 0.44% for everything this service does with those frames. On a T4 the
	// NVDEC block is separate silicon from the SMs, so the decode competes with nothing.
	HWAccel string

	// GPUFilters keeps the picture on the GPU for the frame cap and the height limit (cuvid, 8-bit
	// H.264 and HEVC; see streaming.Config.GPUFilters). False reads as before.
	GPUFilters bool

	// MaxFPS caps the frames ffmpeg converts and pipes, after the decoder, by dropping: it never
	// repeats a frame to reach the rate, so a camera slower than the cap is read as it is. Zero
	// passes every one.
	MaxFPS float64

	// MaxHeight scales a taller picture down to this height, keeping its proportions (1080 for a
	// camera that sends 1440). Zero keeps the camera's own size.
	MaxHeight int

	// AllowPrivate lets cameras on internal addresses through: an engine that runs inside the
	// customer's network. Off, every camera address has to be public, checked at each connection
	// (streaming.Guard).
	AllowPrivate bool

	// Trusted names the internal host:port addresses a camera may still be read from over RTMP:
	// the engine's own receiver, for cameras that push (streaming.Guard).
	Trusted []string

	// Lookup resolves camera host names for that check; nil uses the system resolver.
	Lookup func(ctx context.Context, host string) ([]net.IPAddr, error)

	// VideoTap, when set, is asked for a writer each time a camera opens, and receives that
	// stream's compressed video as fragmented MP4 from the same ffmpeg that decodes it, with no
	// re-encoding (contract.VideoTap). Nil adds nothing to any camera.
	VideoTap contract.VideoTap
}

func NewFFmpegDialer(cfg Config) *FFmpegDialer {
	opener := streaming.NewOpener(cfg.MaxConcurrentOpens, cfg.Runner)
	opener.SetGuard(streaming.Guard{AllowPrivate: cfg.AllowPrivate, Trusted: cfg.Trusted, Lookup: cfg.Lookup})
	return &FFmpegDialer{opener: opener, cfg: cfg}
}

// Dial opens one camera.
func (d *FFmpegDialer) Dial(ctx context.Context, spec contract.CameraSpec) (*Stream, error) {
	var tap io.WriteCloser
	if d.cfg.VideoTap != nil {
		tap = d.cfg.VideoTap.Open(spec.ID)
	}
	s, err := d.opener.Open(ctx, streaming.Config{
		Tap:                  tap,
		URL:                  spec.URL,
		Username:             spec.Username,
		Password:             spec.Password,
		OpenTimeout:          d.cfg.OpenTimeout,
		ReadTimeout:          d.cfg.ReadTimeout,
		DebugBufferOwnership: d.cfg.DebugBufferOwnership,
		HWAccel:              d.cfg.HWAccel,
		GPUFilters:           d.cfg.GPUFilters,
		MaxFPS:               d.cfg.MaxFPS,
		MaxHeight:            d.cfg.MaxHeight,
		// Decode on the card that will run this camera's inference.
		//
		// The alternative is to let ffmpeg choose, and ffmpeg always chooses device 0 -- so
		// every camera in a two-card fleet decodes on one of them while the other's decoder
		// block sits idle. Both cards then report healthy, and the one carrying the fleet is
		// simply slower.
		HWDevice: deviceFor(spec),
	})
	if err != nil {
		return nil, err
	}

	st := &Stream{
		inner:  s,
		frames: make(chan engine.Frame),
		done:   make(chan struct{}),
	}
	go st.pump()
	return st, nil
}

// deviceFor reads the device out of a camera's placement hint.
//
// The hint is set by the supervisor, which balances cameras across cards. A camera with no hint
// gets -1, which leaves the choice to ffmpeg.
func deviceFor(spec contract.CameraSpec) int {
	const prefix = "cuda:"
	if !strings.HasPrefix(spec.Placement, prefix) {
		return -1
	}
	n, err := strconv.Atoi(strings.TrimPrefix(spec.Placement, prefix))
	if err != nil {
		return -1
	}
	return n
}

// Stream adapts a streaming.Stream to the channel the camera loop selects on.
type Stream struct {
	inner  *streaming.Stream
	frames chan engine.Frame
	done   chan struct{}
}

// Frames yields decoded pictures, newest-wins.
//
// The channel is UNBUFFERED, and that is the whole design. A buffer here would hold frames that
// have already been superseded -- and, worse, each held frame keeps a borrowed buffer out of a
// ring of three, so a buffer of two would leave the producer with one and no way to say why it
// started dropping.
//
// Newest-wins is enforced upstream by the capture, which overwrites its pending frame rather
// than queueing. This side only has to hand over whatever the capture currently holds.
func (s *Stream) Frames() <-chan engine.Frame { return s.frames }

func (s *Stream) pump() {
	defer close(s.frames)
	defer close(s.done)

	ctx := context.Background()
	for {
		f, ok := s.inner.Next(ctx)
		if !ok {
			return
		}
		frame := engine.Frame{
			Pix:        f.Bytes,
			Width:      f.Width,
			Height:     f.Height,
			CapturedAt: f.CapturedAt,
			// The borrow travels with the frame. Whoever receives it releases it, which is the
			// only arrangement that survives the receiver dropping the frame for its own
			// reasons -- a stale frame, a full pool -- without leaking a buffer.
			Release: f.Release,
		}
		select {
		case s.frames <- frame:
		case <-s.done:
			// Nobody is going to take it. Release here rather than let the ring shrink by one
			// on every shutdown that races a frame.
			f.Release()
			return
		}
	}
}

// Close stops the feed. It is safe to call twice.
func (s *Stream) Close() { s.inner.Close() }
