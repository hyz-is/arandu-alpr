// Package streaming reads video from a camera and hands out the most recent frame.
//
// It shells out to ffmpeg rather than binding OpenCV. Two reasons: the codec matrix a camera
// can throw at us is ffmpeg's problem, not ours, and a subprocess that wedges is killable in a
// way a wedged cgo call is not. The cost is a process per camera, which is cheap next to the
// GPU work that follows.
//
// The behaviour here is ported from the Python service, which earned it in production. Two of
// its fixes are load-bearing and are called out where they live:
//
//   - opening a stream must never serialise across cameras (`openSlots`)
//   - an empty frame must never reach the recogniser (`readFrame`)
//
// One more rule is this port's own, and it is why frames are borrowed rather than owned
// (`bufferRing`): decoding must not allocate.
package streaming

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// DefaultMaxConcurrentOpens bounds how many cameras may be in the middle of opening at once.
const DefaultMaxConcurrentOpens = 16

// Runner builds the command for an external tool. It exists so tests can stand in for ffmpeg
// and ffprobe.
//
// Without it the two failures worth testing here -- one dead camera blocking the others, and an
// empty frame reaching the recogniser -- can only be reproduced with real cameras that misbehave
// on demand, which means they are never tested at all. Both have happened in production.
type Runner func(ctx context.Context, name string, args ...string) *exec.Cmd

// Opener opens streams, bounding how many opens run at once.
//
// The bound is a SEMAPHORE, and the difference from a lock is the whole reason this type exists.
// The Python service wrapped the capture open in a global lock, so one unreachable camera
// blocking on connect held every other camera's open behind it -- 30 seconds of connect timeout,
// observed, with a watchdog retrying every 30 seconds so the queue never drained. A single dead
// stream took the fleet down, and the symptom ("cameras stopped working") pointed nowhere near
// the cause.
//
// With a semaphore, unreachable cameras fail *next to each other* instead of in a line. The
// bound exists only so a fleet-wide restart does not open hundreds of sockets at once. It is not
// mutual exclusion and must never become one.
//
// It is a type rather than package state because two Openers in one process must not share
// slots: an embedded service that quietly borrowed another instance's budget would stall for
// reasons invisible in its own configuration.
type Opener struct {
	slots  chan struct{}
	runner Runner
	guard  Guard
}

// SetGuard sets the check every camera address passes before ffprobe or ffmpeg is started for it.
// The zero Guard refuses internal addresses; see Guard.
func (o *Opener) SetGuard(g Guard) { o.guard = g }

// NewOpener returns an Opener. A maxConcurrent of zero or less uses the default; a nil runner
// uses exec.CommandContext.
func NewOpener(maxConcurrent int, runner Runner) *Opener {
	if maxConcurrent <= 0 {
		maxConcurrent = DefaultMaxConcurrentOpens
	}
	if runner == nil {
		runner = func(ctx context.Context, name string, args ...string) *exec.Cmd {
			return exec.CommandContext(ctx, name, args...)
		}
	}
	return &Opener{slots: make(chan struct{}, maxConcurrent), runner: runner}
}

// acquire takes an open slot, or gives up when the context does.
func (o *Opener) acquire(ctx context.Context) (release func(), err error) {
	select {
	case o.slots <- struct{}{}:
		return func() { <-o.slots }, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// Properties describe the stream as ffprobe sees it, before any frame is read.
type Properties struct {
	Width  int
	Height int
	FPS    float64
	Codec  string
}

// buffersPerStream is the size of every stream's ring, and three is the smallest number that
// works: one buffer the producer is filling, one holding the frame that is pending for the
// consumer, one the consumer is still looking at. Two would make the producer overwrite the
// pending frame while the consumer reads it; a fourth buys nothing -- the consumer never wants
// an older picture -- and costs another 6.2 MB per camera, which is 620 MB across a fleet.
const buffersPerStream = 3

// poisonByte is what a released buffer is filled with under Config.DebugBufferOwnership.
//
// It is chosen to be loud rather than plausible: a uniform buffer is a picture no camera
// produces, isBlank already calls that signal loss, and 0xDE reads at a glance in a hex dump.
// Use-after-release then looks wrong on sight instead of looking like a moment ago.
const poisonByte = 0xDE

// bufferRing is the fixed set of frame buffers belonging to one stream.
//
// It exists instead of a per-frame make(): 1080p BGR24 is 6.2 MB, so 30 fps across 100 cameras
// is roughly 18 GB/s of garbage, which is the largest single source of GC pressure in the
// pipeline and lands as pauses on the recogniser.
//
// It is a ring rather than a sync.Pool for two reasons. A sync.Pool is CLEARED on every GC
// cycle, so a 6 MB buffer is reallocated moments after each collection -- precisely the cost
// this is meant to remove, and worst exactly when the heap is already under pressure. And a
// ring's footprint is exact: 3*frameSize per camera, knowable before a fleet is deployed, where
// a pool's high-water mark is whatever the last burst happened to need.
type bufferRing struct {
	bufs [][]byte

	// free carries the indices of the buffers nobody holds. A channel rather than a mutex and a
	// stack because the producer must be able to ask without blocking, which `select`/`default`
	// gives for free.
	free chan int

	// gen[i] changes every time buffer i comes back. A Frame carries the generation it was
	// handed, so a second Release of the same frame -- or of a copy of it made before the first
	// -- loses the compare-and-swap and cannot put one buffer in the ring twice.
	gen []atomic.Uint64

	poison bool
}

func newBufferRing(frameSize int, poison bool) *bufferRing {
	r := &bufferRing{
		bufs:   make([][]byte, buffersPerStream),
		free:   make(chan int, buffersPerStream),
		gen:    make([]atomic.Uint64, buffersPerStream),
		poison: poison,
	}
	for i := range r.bufs {
		r.bufs[i] = make([]byte, frameSize)
		r.free <- i
	}
	return r
}

// acquire takes a free buffer, reporting false when every one of them is out.
//
// It never waits: a producer that blocked here would back-pressure ffmpeg on behalf of a slow
// consumer, and the frames it eventually delivered would describe the past.
func (r *bufferRing) acquire() (slot int, buf []byte, gen uint64, ok bool) {
	select {
	case slot = <-r.free:
		return slot, r.bufs[slot], r.gen[slot].Load(), true
	default:
		return 0, nil, 0, false
	}
}

func (r *bufferRing) release(slot int, gen uint64) {
	if !r.gen[slot].CompareAndSwap(gen, gen+1) {
		return // already released; a second return would hand the same buffer to two readers
	}
	if r.poison {
		for i := range r.bufs[slot] {
			r.bufs[slot][i] = poisonByte
		}
	}
	// free has room for every slot and each slot is in it at most once, so this cannot block.
	r.free <- slot
}

// Frame is one decoded picture, in packed BGR24.
//
// Bytes is BORROWED, not owned. It points into the stream's fixed ring of three buffers, and the
// producer refills it as soon as Release hands it back -- so a frame is valid only between
// receiving it and releasing it. Anything that must outlive that window (a crop kept for the
// recogniser, a JPEG queued for upload, a frame stored in a struct) must be copied out first.
//
// The old contract was the opposite -- a fresh buffer per frame, owned by the receiver -- and it
// cost an allocation of width*height*3 bytes per frame. See bufferRing for why that is not
// affordable at fleet scale.
//
// Two things make the new rule survivable. Release is idempotent and safe on the zero Frame, so
// a `defer frame.Release()` is always correct. And forgetting it does not corrupt anything and
// does not leak: the producer drops frames rather than allocate a fourth buffer, so a missing
// Release shows up as a camera that goes quiet with Dropped climbing -- loud, and local to the
// one consumer at fault. Config.DebugBufferOwnership turns reading after Release into visibly
// wrong bytes as well.
type Frame struct {
	Bytes      []byte
	Width      int
	Height     int
	CapturedAt time.Time

	ring *bufferRing
	slot int
	gen  uint64
}

// Release returns the frame's buffer to the stream that lent it. After Release, Bytes may be
// overwritten at any moment and must not be read.
func (f Frame) Release() {
	if f.ring != nil {
		f.ring.release(f.slot, f.gen)
	}
}

// Config is everything needed to open one stream.
type Config struct {
	URL      string
	Username string
	Password string

	// OpenTimeout caps the probe and the first frame. A camera that is powered on but not
	// serving will otherwise hold the connection open indefinitely.
	OpenTimeout time.Duration

	// ReadTimeout is how long a running stream may go silent before it is considered dead.
	ReadTimeout time.Duration

	// DebugBufferOwnership fills every released buffer with 0xDE.
	//
	// Frame.Bytes is borrowed, so a consumer that reads after Release sees whatever the producer
	// has written since -- which is a plausible-looking picture from a moment ago, and therefore
	// a bug that produces slightly wrong plates instead of an obvious failure. Poisoning turns it
	// into a frame that is visibly not a picture. Off by default: it costs a full pass over the
	// buffer per frame.
	DebugBufferOwnership bool

	// HWAccel moves decoding onto a dedicated hardware block instead of the CPU.
	//
	// This is the largest CPU cost in the pipeline, and it is not our code. Decoding one 1080p30
	// stream and converting it to BGR24 measures at 5.6% of a core; everything the Go service
	// does with those same frames measures at 0.44%. Thirteen to one.
	//
	// The sibling Rust and C++ engines reached the same conclusion and carry the same fix under
	// `STREAM_HWACCEL`; the note in alpr-rs cites NVIDIA's own guidance that in multi-camera
	// pipelines the video decoder, not inference, is usually what saturates first.
	//
	// Values: "cuvid" for NVDEC, "cuda", or anything else ffmpeg accepts. Empty means CPU.
	//
	// On a T4 the NVDEC block is separate silicon from the SMs, so hardware decode costs neither
	// CPU nor any inference throughput -- it is capacity that is otherwise simply unused.
	HWAccel string

	// HWDevice picks which GPU decodes. Negative means "let ffmpeg choose".
	//
	// The sibling engines do not have this, and with one card it does not matter. With two it
	// does: without it every camera decodes on device 0, which turns a card meant to share the
	// fleet into the one holding all of it.
	HWDevice int

	// MaxFPS drops frames inside ffmpeg, after the decoder and before the bgr24 conversion and
	// the pipe. Zero passes every frame.
	//
	// It only ever DROPS. The fps filter this used to be makes a frame at every 1/MaxFPS whether
	// or not the camera sent one, so a camera that delivers 8 fps behind a cap of 15 yielded 15,
	// seven of them copies -- measured on a real stream: fps=10 on 8 fps gave 600 frames a
	// minute where the camera sent 480, and fps=15 gave 900. Each copy cost a conversion, a pipe
	// write and, under a consensus that counts readings, a duplicate vote for a picture it had
	// already counted. Dropping cannot invent a frame, so a cap above the camera's rate is a cap
	// that does nothing, which is what a cap is for.
	//
	// The cap, like MaxHeight, belongs to the raw frame output alone. The Tap is a second output of
	// the same ffmpeg that copies the camera's packets, so no frame is dropped from it.
	MaxFPS float64

	// MaxHeight scales a taller picture down to this height, keeping its proportions, before the
	// bgr24 conversion. Zero keeps the camera's own size.
	//
	// The plate reads the same from 1080 lines as from 1440 -- measured, the OCR takes a 128x64
	// crop and a plate that is 160 pixels wide at 1440p is 120 at 1080p, which it reads as well --
	// while a frame is 44% smaller to convert, to pipe and to hold three of per camera. A camera
	// no taller than this is never scaled UP. It does not touch the Tap, which keeps the camera's
	// own picture.
	//
	// With the cuvid decoder the scaling is done by NVDEC (-resize) and the smaller picture is
	// the only one that reaches system memory; the CPU scale filter is not added. Its resampling
	// is the decoder's, not the area filter's. Any other decoder scales with the filter.
	MaxHeight int

	// Tap receives a copy of the camera's compressed video as fragmented MP4, written by the same
	// ffmpeg that decodes it as a second output with -c copy: nothing is decoded or encoded for it.
	// Nil adds no second output.
	//
	// Open takes ownership: it closes the Tap on every path, when the stream ends or when it never
	// starts. Write must not block. The pipe it is fed from is drained whatever Write does, because
	// ffmpeg waits on a full pipe and the raw frames the plates are read from would wait with it.
	Tap io.WriteCloser
}

// tapArgs is the second output: the camera's own packets, remuxed, to the descriptor ffmpeg was
// given as fd 3.
//
// Fragments are cut every tapFragment and at every keyframe, so a viewer is at most one fragment
// behind and not a whole group of pictures. Audio is left out: a plate reader has no use for it and
// a camera's audio codec is one more thing a browser may not play.
func tapArgs() []string {
	return []string{
		"-map", "0:v:0", "-an", "-c:v", "copy",
		"-f", "mp4",
		"-movflags", "empty_moov+default_base_moof+frag_keyframe",
		"-frag_duration", strconv.Itoa(int(tapFragment / time.Microsecond)),
		"pipe:3",
	}
}

// tapFragment is the longest a fragment of the tap may be.
const tapFragment = 100 * time.Millisecond

// filterArgs returns the output filters for a source of srcW x srcH, and the size of what they
// produce. With neither a cap nor a limit it returns no arguments and the source's own size.
func (c Config) filterArgs(srcW, srcH int) (args []string, outW, outH int) {
	outW, outH = srcW, srcH
	var chain []string
	if c.MaxFPS > 0 {
		gap := strconv.FormatFloat(0.9/c.MaxFPS, 'f', 4, 64)
		// 0.9 of the interval and not the whole of it: a source whose frames are exactly 1/MaxFPS
		// apart would otherwise lose every second one to rounding in the timestamps. A picture
		// without a timestamp passes -- an unknown time is not a reason to drop.
		chain = append(chain, "select='isnan(prev_selected_t)+isnan(t)+gte(t-prev_selected_t,"+gap+")'")
	}
	if c.MaxHeight > 0 && srcH > c.MaxHeight && srcW > 0 {
		outH = c.MaxHeight
		// Even, because the pixel formats a camera sends are subsampled in pairs.
		outW = (srcW*c.MaxHeight/srcH + 1) &^ 1
		// area: every output pixel is the mean of the source pixels it covers. The default
		// bicubic samples a few and skips the rest, which turns thin strokes into noise at the
		// ratios a plate is shrunk by.
		chain = append(chain, "scale="+strconv.Itoa(outW)+":"+strconv.Itoa(outH)+":flags=area")
	}
	if len(chain) == 0 {
		return nil, outW, outH
	}
	return []string{"-vf", strings.Join(chain, ",")}, outW, outH
}

// LogValue prints how the stream is opened, not the camera's credentials.
//
// The URL is left out with the password, because a stream URL routinely carries the credentials
// in its own userinfo.
func (c Config) LogValue() slog.Value {
	return slog.GroupValue(
		slog.Duration("open_timeout", c.OpenTimeout),
		slog.Duration("read_timeout", c.ReadTimeout),
		slog.String("hwaccel", c.HWAccel),
		slog.Int("hwdevice", c.HWDevice),
		slog.Float64("max_fps", c.MaxFPS),
		slog.Int("max_height", c.MaxHeight),
	)
}

// MarshalJSON keeps the credentials out of anything that serialises a stream configuration.
func (c Config) MarshalJSON() ([]byte, error) {
	// A local type with no methods: converting to it is what stops MarshalJSON calling itself.
	type plain Config
	safe := plain(c)
	for _, field := range []*string{&safe.URL, &safe.Username, &safe.Password} {
		if *field != "" {
			*field = "[redacted]"
		}
	}
	return json.Marshal(safe)
}

// hwaccelArgs returns the decode flags, and nothing at all when hardware decoding is off.
//
// They go BEFORE -i, because they decide how the input is opened; ffmpeg ignores them after it.
//
// The decoder is named explicitly (-c:v h264_cuvid) and not left to -hwaccel alone, which is what
// the Rust and C++ engines do and is the part that actually engages NVDEC -- `-hwaccel cuvid` by
// itself will silently fall back to the software decoder on some builds, which looks like hardware
// decode that simply did not help.
//
// The codec comes from the probe rather than from configuration. Both siblings make the operator
// choose between cuvid and hevc_cuvid, and choosing wrong means the camera does not open; ffprobe
// already told us which it is.
//
// For cuvid the decoder alone is named, with -gpu, and -hwaccel is left OUT. With `-hwaccel cuvid`
// the frames stay in GPU memory, and the bgr24 conversion this pipe asks for cannot reach them:
// ffmpeg 6.1 (Ubuntu 24.04, this image) stops on the first frame with "Impossible to convert
// between the formats", exit 218, and so did 4.4 under the Python worker. `-c:v h264_cuvid -gpu N`
// decodes on NVDEC and hands the frame back in system memory -- measured on a T4, and the form the
// Python worker runs on every camera in production.
//
// With a MaxHeight below the picture's own height, the cuvid decoder also gets -resize WxH: NVDEC
// scales on the GPU, and only the smaller picture crosses to system memory. Without it the full
// frame came back to RAM, was converted to bgr24 by swscale on the CPU and was then scaled again
// by the scale filter; the swscale conversion alone was 43% of the ffmpeg profile on a loaded node.
func (c Config) hwaccelArgs(props Properties) []string {
	if c.HWAccel == "" {
		return nil
	}

	if c.HWAccel == "cuvid" {
		dec := cuvidDecoder(props.Codec)
		if dec == "" {
			return nil // a codec NVDEC does not decode: software, for this camera only
		}
		args := []string{"-c:v", dec}
		if c.HWDevice >= 0 {
			args = append(args, "-gpu", strconv.Itoa(c.HWDevice))
		}
		if w, h, ok := c.decoderResize(props); ok {
			args = append(args, "-resize", strconv.Itoa(w)+"x"+strconv.Itoa(h))
		}
		return args
	}

	args := []string{"-hwaccel", c.HWAccel}
	if c.HWDevice >= 0 {
		args = append(args, "-hwaccel_device", strconv.Itoa(c.HWDevice))
	}
	return args
}

// decoderResize reports the size NVDEC is asked to decode to, and false when the picture is left
// at its own size or is scaled by the filter chain instead.
//
// It holds only for the cuvid decoder, only when MaxHeight is set, and only for a source taller
// than MaxHeight: a smaller picture is never scaled up. Both sides are even, as the pixel formats a
// camera sends are subsampled in pairs, and the width keeps the source's proportions.
func (c Config) decoderResize(props Properties) (w, h int, ok bool) {
	if c.HWAccel != "cuvid" || cuvidDecoder(props.Codec) == "" {
		return 0, 0, false
	}
	if c.MaxHeight <= 0 || props.Height <= c.MaxHeight || props.Width <= 0 {
		return 0, 0, false
	}
	h = c.MaxHeight &^ 1
	w = (props.Width*h/props.Height + 1) &^ 1
	if w < 2 || h < 2 || h >= props.Height {
		return 0, 0, false
	}
	return w, h, true
}

// cuvidDecoder maps a probed codec to its NVDEC decoder, and returns "" for anything NVDEC does
// not handle -- in which case ffmpeg falls back to software for that camera rather than failing.
func cuvidDecoder(codec string) string {
	switch codec {
	case "h264":
		return "h264_cuvid"
	case "hevc", "h265":
		return "hevc_cuvid"
	case "mjpeg":
		return "mjpeg_cuvid"
	case "vp9":
		return "vp9_cuvid"
	default:
		return ""
	}
}

func (c Config) withDefaults() Config {
	if c.OpenTimeout <= 0 {
		c.OpenTimeout = 10 * time.Second
	}
	if c.ReadTimeout <= 0 {
		c.ReadTimeout = 15 * time.Second
	}
	return c
}

// authenticatedURL folds the credentials into the URL.
//
// Credentials are percent-encoded because camera passwords routinely contain `@` and `/`, and an
// unescaped one silently reshapes the URL: everything after the last `@` becomes the host. The
// stream then fails to open with a DNS error naming a host nobody configured.
func (c Config) authenticatedURL() (string, error) {
	if c.Username == "" && c.Password == "" {
		return c.URL, nil
	}
	scheme, rest, found := strings.Cut(c.URL, "://")
	if !found {
		return "", fmt.Errorf("streaming: url has no scheme: %q", c.URL)
	}
	// If the URL already carries credentials, leave it alone rather than stacking a second set.
	if strings.Contains(rest, "@") {
		return c.URL, nil
	}
	return scheme + "://" + escapeUserinfo(c.Username) + ":" + escapeUserinfo(c.Password) + "@" + rest, nil
}

func escapeUserinfo(s string) string {
	const hex = "0123456789ABCDEF"
	var b strings.Builder
	for i := 0; i < len(s); i++ {
		ch := s[i]
		unreserved := (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
			(ch >= '0' && ch <= '9') || strings.IndexByte("-._~", ch) >= 0
		if unreserved {
			b.WriteByte(ch)
			continue
		}
		b.WriteByte('%')
		b.WriteByte(hex[ch>>4])
		b.WriteByte(hex[ch&0x0F])
	}
	return b.String()
}

// transportArgs returns the flags that only make sense for an RTSP source.
//
// TCP rather than the default UDP because a dropped UDP packet is a torn picture the decoder
// then has to guess at, and the recogniser reads the guess as a plate.
//
// It is conditional because ffmpeg refuses the option outright for any other input -- "Option
// rtsp_transport not found", before it opens anything -- so passing it unconditionally makes
// every non-RTSP source (an HTTP/MJPEG camera, a file being replayed for a test) impossible to
// open at all, with an error that says nothing about transports.
func transportArgs(url string) []string {
	if strings.HasPrefix(url, "rtsp://") || strings.HasPrefix(url, "rtsps://") {
		return []string{"-rtsp_transport", "tcp"}
	}
	return nil
}

// Probe asks ffprobe for the stream geometry.
//
// It takes an open slot for the same reason opening does: probing an unreachable camera blocks
// for the full timeout, and a fleet restart would otherwise spend that time serially.
func (o *Opener) Probe(ctx context.Context, cfg Config) (Properties, error) {
	cfg = cfg.withDefaults()
	url, err := cfg.authenticatedURL()
	if err != nil {
		return Properties{}, err
	}
	url, guarded, err := o.guard.Address(ctx, url)
	if err != nil {
		return Properties{}, err
	}

	release, err := o.acquire(ctx)
	if err != nil {
		return Properties{}, err
	}
	defer release()

	ctx, cancel := context.WithTimeout(ctx, cfg.OpenTimeout)
	defer cancel()

	args := append(append(guarded, transportArgs(url)...),
		"-select_streams", "v:0",
		"-show_entries", "stream=width,height,r_frame_rate,codec_name",
		"-of", "json",
		url,
	)
	cmd := o.runner(ctx, "ffprobe", append([]string{"-v", "error"}, args...)...)
	out, err := cmd.Output()
	if err != nil {
		return Properties{}, fmt.Errorf("streaming: probe failed: %w", err)
	}

	var parsed struct {
		Streams []struct {
			Width     int    `json:"width"`
			Height    int    `json:"height"`
			FrameRate string `json:"r_frame_rate"`
			Codec     string `json:"codec_name"`
		} `json:"streams"`
	}
	if err := json.Unmarshal(out, &parsed); err != nil {
		return Properties{}, fmt.Errorf("streaming: probe output: %w", err)
	}
	if len(parsed.Streams) == 0 {
		return Properties{}, errors.New("streaming: probe found no video stream")
	}
	s := parsed.Streams[0]
	if s.Width <= 0 || s.Height <= 0 {
		return Properties{}, fmt.Errorf("streaming: probe gave no geometry (%dx%d)", s.Width, s.Height)
	}
	return Properties{Width: s.Width, Height: s.Height, FPS: parseRational(s.FrameRate), Codec: s.Codec}, nil
}

// parseRational reads ffprobe's `num/den` frame rate.
//
// A stream that reports `0/0` is not an error: many cameras do it, and the pipeline measures the
// real rate from arrival times anyway. Returning 0 says "unknown", never "no frames".
func parseRational(v string) float64 {
	num, den, found := strings.Cut(v, "/")
	if !found {
		f, _ := strconv.ParseFloat(v, 64)
		return f
	}
	n, err1 := strconv.ParseFloat(num, 64)
	d, err2 := strconv.ParseFloat(den, 64)
	if err1 != nil || err2 != nil || d == 0 {
		return 0
	}
	return n / d
}

// Stream is one camera being read. Use Open, then Next.
type Stream struct {
	props Properties
	ring  *bufferRing

	// latest holds the newest frame nobody has taken yet, and signal wakes a waiting consumer.
	//
	// This replaces a `chan Frame` of capacity 1 whose producer replaced the pending frame in
	// three steps: try to send, else receive the old one, else send again. That dance is correct
	// today only by an argument nobody should have to make -- with a single producer the final
	// send cannot fail, because the only way the slot refills between the receive and the send is
	// another producer, and a consumer waiting on the empty channel is handed the value directly.
	// Measured over 2M frames it never lost one. Add a second publisher -- a reconnect draining
	// the old pump while the new one starts -- and the same loop drops the frame it was in the
	// middle of publishing about 3 times in 1000, keeping the older one. Newest-wins is the whole
	// point of the design, and it should not rest on a proof about select branches.
	//
	// A pointer swap makes it true by construction: the store IS the replacement, one operation,
	// no window and no failure branch, whoever is publishing. It is also the shape the borrowed
	// buffers need -- displacement happens in exactly one place, so exactly one Release has to
	// exist -- and it lets the stream end with a frame still pending, which closing a channel
	// cannot do without stranding that frame's buffer.
	//
	// The signal only wakes the consumer; the frame never travels through it. Because the
	// producer stores before it signals, a wakeup cannot be lost: a consumer that saw no frame
	// either takes a pending token and looks again, or is still waiting when one is delivered.
	latest atomic.Pointer[Frame]
	signal chan struct{}

	dropped atomic.Int64

	cancel context.CancelFunc

	// tapDone closes when the tap's pipe has been read to its end and the tap closed. Nil without
	// a tap.
	tapDone chan struct{}

	once sync.Once
	// err is atomic because the watchdog goroutine and the read loop can both fail the stream,
	// and first failure wins: the watchdog's "no frame for 15s" explains the read error that
	// follows it, not the other way round.
	err  atomic.Pointer[error]
	done chan struct{}
}

// newStream builds the stream that pump will fill.
func newStream(props Properties, cfg Config, cancel context.CancelFunc) *Stream {
	return &Stream{
		props:  props,
		ring:   newBufferRing(props.Width*props.Height*3, cfg.DebugBufferOwnership),
		signal: make(chan struct{}, 1),
		cancel: cancel,
		done:   make(chan struct{}),
	}
}

// Open probes the camera and starts decoding into a channel of frames.
func (o *Opener) Open(ctx context.Context, cfg Config) (*Stream, error) {
	cfg = cfg.withDefaults()
	// The tap is the stream's from here: whatever stops it from starting closes the tap.
	started := false
	defer func() {
		if !started && cfg.Tap != nil {
			_ = cfg.Tap.Close()
		}
	}()

	props, err := o.Probe(ctx, cfg)
	if err != nil {
		return nil, err
	}
	url, err := cfg.authenticatedURL()
	if err != nil {
		return nil, err
	}
	url, guarded, err := o.guard.Address(ctx, url)
	if err != nil {
		return nil, err
	}

	// What the pipe carries is what the filters make of the picture, and the ring, the reads and
	// every frame handed out are sized by that. Properties reports it, not the camera's own size.
	//
	// When the decoder scales (cuvid with a MaxHeight), the picture reaches the filter chain
	// already at its final size, so the chain sees that size and adds no scale of its own.
	source := props
	chainW, chainH := source.Width, source.Height
	if w, h, ok := cfg.decoderResize(source); ok {
		chainW, chainH = w, h
	}
	filter, outW, outH := cfg.filterArgs(chainW, chainH)
	props.Width, props.Height = outW, outH

	runCtx, cancel := context.WithCancel(ctx)

	// One pending frame, and the producer replaces it rather than queueing or blocking. The
	// consumer wants the most recent picture, never a queue of stale ones: buffering frames would
	// grow latency without bound against a slow recogniser, and every frame it eventually got
	// would describe the past.
	s := newStream(props, cfg, cancel)

	args := append(append(cfg.hwaccelArgs(source), guarded...), transportArgs(url)...)
	args = append(args, "-i", url, "-an")
	// One thread for the filter graph. The graph is a select and a pixel-format conversion on a
	// single stream; the extra threads ffmpeg starts per filter cost more in scheduling and
	// synchronisation than they save -- measured, 15 to 18% less CPU per camera.
	args = append(args, "-filter_threads", "1")
	args = append(args, filter...)
	args = append(args,
		"-vsync", "passthrough",
		"-f", "rawvideo",
		"-pix_fmt", "bgr24",
		"pipe:1",
	)
	if cfg.Tap != nil {
		args = append(args, tapArgs()...)
	}
	cmd := o.runner(runCtx, "ffmpeg", append([]string{"-loglevel", "error"}, args...)...)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		cancel()
		return nil, fmt.Errorf("streaming: stdout: %w", err)
	}
	// Best effort: a refusal leaves the kernel's default of 64 KiB, which is slower and works.
	growPipe(stdout, framePipeSize)
	var tap *os.File
	if cfg.Tap != nil {
		var write *os.File
		if tap, write, err = os.Pipe(); err != nil {
			cancel()
			return nil, fmt.Errorf("streaming: tap pipe: %w", err)
		}
		cmd.ExtraFiles = []*os.File{write}
		// The child has its own copy once it starts; the parent's must go, or the pipe never
		// reports the end of the stream.
		defer write.Close()
	}
	if err := cmd.Start(); err != nil {
		cancel()
		if tap != nil {
			_ = tap.Close()
		}
		return nil, fmt.Errorf("streaming: start ffmpeg: %w", err)
	}
	started = true

	if tap != nil {
		s.tapDone = make(chan struct{})
		go s.drainTap(tap, cfg.Tap)
	}
	go s.pump(runCtx, cmd, stdout, cfg)
	return s, nil
}

// drainTap hands the tap pipe to the tap until ffmpeg closes it, and then closes the tap.
//
// The pipe is read to its end whatever the writer does. A writer that fails is dropped and the
// rest is thrown away, because the other side of this pipe is ffmpeg: it blocks when the pipe is
// full, and the raw frames the plates are read from come from the same process.
func (s *Stream) drainTap(pipe *os.File, tap io.WriteCloser) {
	defer close(s.tapDone)
	defer pipe.Close()
	defer func() { _ = tap.Close() }()
	buf := make([]byte, 64<<10)
	live := true
	for {
		n, err := pipe.Read(buf)
		if n > 0 && live {
			if _, werr := tap.Write(buf[:n]); werr != nil {
				live = false
			}
		}
		if err != nil {
			return
		}
	}
}

// Properties returns the geometry ffprobe reported.
func (s *Stream) Properties() Properties { return s.props }

// Next returns the most recent frame, waiting until one arrives. It reports false once the
// stream has ended (check Err) or the context is done.
//
// It replaces the previous `Frames() <-chan Frame`, and the reason is the buffer ring rather
// than taste. A channel is a queue, and a queue is exactly the wrong shape for both new rules:
// the frames sitting in it are stale by definition, and each one is holding a borrowed buffer
// that the producer cannot reuse and no one has promised to Release -- so a consumer that stops
// ranging, or a stream that ends with a frame still in flight, silently strands a third of the
// ring. Handing the frame back through a call makes the borrow visible at the point where the
// caller can pair it with `defer frame.Release()`, which is the only place that pairing can be
// made obvious.
//
// The caller must Release the frame it gets. Next is safe to call from one consumer at a time;
// two consumers would race for the same pending frame, which is meaningless for a design where
// the answer is always "the newest one".
func (s *Stream) Next(ctx context.Context) (Frame, bool) {
	for {
		if f := s.latest.Swap(nil); f != nil {
			return *f, true
		}
		select {
		case <-s.signal:
			// Loop: the token may be stale (the frame it announced was already taken), so the
			// pointer is the truth and the token is only a nudge to go look at it.
		case <-s.done:
			// A frame published just before the stream ended is still worth delivering.
			if f := s.latest.Swap(nil); f != nil {
				return *f, true
			}
			return Frame{}, false
		case <-ctx.Done():
			return Frame{}, false
		}
	}
}

// Dropped counts frames decoded but never delivered: replaced by a newer one before the consumer
// took them, or discarded because the consumer was holding every buffer in the ring.
//
// It is the number that says whether a consumer is keeping up, and it is the symptom a forgotten
// Frame.Release produces -- so it is worth exporting rather than leaving to a profiler.
func (s *Stream) Dropped() int64 { return s.dropped.Load() }

// Err reports why the stream ended. It is only meaningful once Next has reported false.
func (s *Stream) Err() error {
	<-s.done
	if err := s.err.Load(); err != nil {
		return *err
	}
	return nil
}

// fail records why the stream is ending, keeping the first reason.
//
// The watchdog's "no frame for 15s" is the cause and the read error that follows is the effect,
// and they are set from different goroutines.
func (s *Stream) fail(err error) { s.err.CompareAndSwap(nil, &err) }

// publish makes a frame the pending one, returning the buffer of whatever it displaced.
func (s *Stream) publish(f Frame) {
	if old := s.latest.Swap(&f); old != nil {
		// Nobody took it. Its buffer has to go back now or the producer starves itself out of
		// the ring one frame later.
		old.Release()
		s.dropped.Add(1)
	}
	select {
	case s.signal <- struct{}{}:
	default: // a wakeup is already pending; one is enough
	}
}

// Close stops the stream and reaps the subprocess.
func (s *Stream) Close() {
	s.once.Do(func() { s.cancel() })
	<-s.done
}

func (s *Stream) pump(ctx context.Context, cmd *exec.Cmd, stdout io.ReadCloser, cfg Config) {
	defer close(s.done)
	defer func() {
		s.cancel()
		_ = cmd.Wait()
		// ffmpeg has exited, so the tap's pipe is at its end and the reader follows at once.
		if s.tapDone != nil {
			<-s.tapDone
		}
	}()

	frameSize := s.props.Width * s.props.Height * 3
	reader := bufio.NewReaderSize(stdout, frameSize)

	// A watchdog rather than a read deadline: a pipe has no deadline, and ffmpeg holding a
	// half-open TCP connection produces no error and no bytes. Without this the goroutine would
	// wait forever on a camera that is, from the network's point of view, simply quiet.
	lastFrame := time.Now()
	var mu sync.Mutex
	sawBytes := func() {
		mu.Lock()
		lastFrame = time.Now()
		mu.Unlock()
	}
	watchdog := time.NewTicker(time.Second)
	defer watchdog.Stop()
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case <-watchdog.C:
				mu.Lock()
				silent := time.Since(lastFrame)
				mu.Unlock()
				if silent > cfg.ReadTimeout {
					s.fail(fmt.Errorf("streaming: no frame for %s", silent.Round(time.Second)))
					s.cancel()
					return
				}
			}
		}
	}()

	readFailed := func(err error) {
		if errors.Is(err, context.Canceled) {
			return
		}
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			s.fail(fmt.Errorf("streaming: stream ended: %w", err))
			return
		}
		s.fail(fmt.Errorf("streaming: read: %w", err))
	}

	for {
		slot, buf, gen, ok := s.ring.acquire()
		if !ok {
			// The consumer is holding every buffer. Drop this frame: blocking would back-pressure
			// ffmpeg until the camera's own buffers overflow, and allocating a fourth buffer would
			// make a slow consumer cost memory without bound -- which is the failure the ring is
			// here to prevent. Discard reads the frame off the pipe without a buffer of its own;
			// leaving it there would only make the next frame we do deliver older.
			if _, err := reader.Discard(frameSize); err != nil {
				readFailed(err)
				return
			}
			// Bytes arrived, so the camera is alive even though nothing was delivered. Without
			// this the watchdog would kill a healthy stream to punish a slow consumer.
			sawBytes()
			s.dropped.Add(1)
			continue
		}

		if _, err := io.ReadFull(reader, buf); err != nil {
			s.ring.release(slot, gen)
			readFailed(err)
			return
		}

		sawBytes()

		if isBlank(buf) {
			// Ported deliberately from the Python service, where it is
			// `fix(detection): drop empty frames before they reach the ALPR`.
			//
			// A camera that loses signal keeps the connection open and emits solid black. Those
			// frames decode fine and cost a full detector pass to produce nothing, so under a
			// fleet outage the GPU saturates on pictures of nothing — starving the cameras that
			// are still working. Dropping here is what keeps a partial outage partial.
			s.ring.release(slot, gen)
			continue
		}

		s.publish(Frame{
			Bytes:      buf,
			Width:      s.props.Width,
			Height:     s.props.Height,
			CapturedAt: time.Now(),
			ring:       s.ring,
			slot:       slot,
			gen:        gen,
		})

		select {
		case <-ctx.Done():
			return
		default:
		}
	}
}

// isBlank reports whether a frame carries no picture.
//
// Sampled, not exhaustive: a 4K frame is 24 MB and running the whole buffer per frame would cost
// more than the check saves. The stride is prime relative to the 3-byte pixel so the samples do
// not land on one colour channel.
func isBlank(buf []byte) bool {
	const stride = 997
	if len(buf) == 0 {
		return true
	}
	first := buf[0]
	for i := 0; i < len(buf); i += stride {
		if buf[i] != first {
			return false
		}
	}
	return true
}
