package streaming

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestAuthenticatedURLEscapesCredentials(t *testing.T) {
	// A camera password containing `@` is the failure this escaping exists for. Unescaped, the
	// host becomes whatever follows the last `@`, and the stream fails to open against a host
	// nobody configured -- with a DNS error that points nowhere near the password.
	cfg := Config{URL: "rtsp://cam.local:554/live", Username: "admin", Password: "p@ss/w0rd"}
	got, err := cfg.authenticatedURL()
	if err != nil {
		t.Fatalf("authenticatedURL: %v", err)
	}
	const want = "rtsp://admin:p%40ss%2Fw0rd@cam.local:554/live"
	if got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
	// The host must survive as the last `@` segment.
	if _, after, _ := strings.Cut(got, "@"); after != "cam.local:554/live" {
		t.Fatalf("host reshaped by credentials: %q", after)
	}
}

func TestAuthenticatedURLLeavesExistingCredentials(t *testing.T) {
	// Stacking a second set would produce `user:pass@user:pass@host`, whose host is still the
	// last segment but whose userinfo is nonsense.
	cfg := Config{URL: "rtsp://u:p@cam.local/live", Username: "admin", Password: "secret"}
	got, err := cfg.authenticatedURL()
	if err != nil {
		t.Fatalf("authenticatedURL: %v", err)
	}
	if got != "rtsp://u:p@cam.local/live" {
		t.Fatalf("existing credentials replaced: %q", got)
	}
}

func TestAuthenticatedURLWithoutCredentialsIsUntouched(t *testing.T) {
	cfg := Config{URL: "rtsp://cam.local/live"}
	got, err := cfg.authenticatedURL()
	if err != nil || got != cfg.URL {
		t.Fatalf("got %q, %v", got, err)
	}
}

func TestParseRationalHandlesUnknownRate(t *testing.T) {
	// `0/0` is what many cameras report. It means "unknown", and the pipeline measures the real
	// rate from arrival times. Treating it as an error would refuse working cameras.
	cases := map[string]float64{"30/1": 30, "30000/1001": 29.97002997002997, "0/0": 0, "25": 25, "": 0}
	for in, want := range cases {
		if got := parseRational(in); got != want {
			t.Errorf("parseRational(%q) = %v, want %v", in, got, want)
		}
	}
}

func TestIsBlankDetectsSignalLoss(t *testing.T) {
	// A camera that loses signal keeps the connection and emits solid frames. Those cost a full
	// detector pass to produce nothing.
	blank := make([]byte, 1920*1080*3) // all zero
	if !isBlank(blank) {
		t.Error("solid black not detected as blank")
	}

	grey := make([]byte, 1920*1080*3)
	for i := range grey {
		grey[i] = 0x80
	}
	if !isBlank(grey) {
		t.Error("solid grey not detected as blank")
	}
}

func TestIsBlankKeepsRealPictures(t *testing.T) {
	// The check must not eat a dark night frame. Sampling every 997th byte, one differing
	// sample is enough to keep the frame -- so the risk is a picture whose sampled bytes all
	// happen to match, not a picture with visible content.
	frame := make([]byte, 1920*1080*3)
	for i := range frame {
		frame[i] = byte(i % 251)
	}
	if isBlank(frame) {
		t.Error("textured frame dropped as blank")
	}

	// A frame that is black except for one bright region, landing on a sampled byte.
	mostlyBlack := make([]byte, 1920*1080*3)
	mostlyBlack[997] = 0xFF
	if isBlank(mostlyBlack) {
		t.Error("frame with content dropped as blank")
	}
}

// probeJSON is what ffprobe prints for a healthy 1080p30 h264 camera.
const probeJSON = `{"streams":[{"width":1920,"height":1080,"r_frame_rate":"30/1","codec_name":"h264"}]}`

// fakeRunner stands in for ffprobe. URLs containing "dead" hang until the context ends, the way
// a camera that is powered on but not serving does; everything else answers immediately.
func fakeRunner(t *testing.T) Runner {
	t.Helper()
	return func(ctx context.Context, name string, args ...string) *exec.Cmd {
		for _, a := range args {
			if strings.Contains(a, "dead") {
				return exec.CommandContext(ctx, "sh", "-c", "sleep 30")
			}
		}
		return exec.CommandContext(ctx, "sh", "-c", "printf %s "+strconv.Quote(probeJSON))
	}
}

func TestOneDeadCameraDoesNotBlockTheOthers(t *testing.T) {
	// The regression: the Python service wrapped the capture open in a global lock, so one
	// unreachable camera held every other camera's open behind it for the full connect timeout
	// -- 30 seconds, observed -- while a watchdog retried every 30 seconds and refilled the
	// queue. A single dead stream took the fleet down.
	//
	// One camera hangs here for the whole test. Every healthy one must still open.
	o := NewOpener(8, fakeRunner(t))
	o.SetGuard(Guard{AllowPrivate: true}) // fake cameras; the guard is tested on its own

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Hold slots with dead cameras, but leave the pool short of full: a semaphore lets the
	// healthy ones through the remaining slots, a mutex lets nobody through at all.
	var hung sync.WaitGroup
	for i := 0; i < 4; i++ {
		hung.Add(1)
		go func() {
			defer hung.Done()
			_, _ = o.Probe(ctx, Config{URL: "rtsp://dead/stream", OpenTimeout: 8 * time.Second})
		}()
	}

	// Give the dead ones time to take their slots before the healthy ones ask.
	time.Sleep(50 * time.Millisecond)

	const healthy = 20
	var ok atomic.Int64
	var wg sync.WaitGroup
	for i := 0; i < healthy; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := o.Probe(ctx, Config{URL: "rtsp://good/stream"}); err == nil {
				ok.Add(1)
			}
		}()
	}

	done := make(chan struct{})
	go func() { wg.Wait(); close(done) }()

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatalf("only %d of %d healthy cameras opened while dead ones were connecting; "+
			"opening is serialised", ok.Load(), healthy)
	}
	if got := ok.Load(); got != healthy {
		t.Errorf("%d of %d healthy cameras opened", got, healthy)
	}

	cancel()
	hung.Wait()
}

func TestOpenSlotsIsASemaphoreNotAMutex(t *testing.T) {
	// The distinction is the entire fix, and it is one character away from being undone: a
	// capacity of 1 is a mutex wearing a channel's clothes, and everything still compiles.
	o := NewOpener(0, nil)
	if cap(o.slots) < 2 {
		t.Fatalf("open slots capacity is %d; that is mutual exclusion, which is the bug", cap(o.slots))
	}
	if cap(o.slots) != DefaultMaxConcurrentOpens {
		t.Errorf("default capacity = %d, want %d", cap(o.slots), DefaultMaxConcurrentOpens)
	}
}

func TestTwoOpenersDoNotShareSlots(t *testing.T) {
	// The semaphore used to be a package-level var, which meant two instances of the service
	// in one process silently split one budget -- each stalling for a reason invisible in its
	// own configuration.
	a := NewOpener(2, nil)
	b := NewOpener(2, nil)

	ctx := context.Background()
	relA, err := a.acquire(ctx)
	if err != nil {
		t.Fatal(err)
	}
	relA2, err := a.acquire(ctx)
	if err != nil {
		t.Fatal(err)
	}
	// a is now full. b must be untouched.
	short, cancel := context.WithTimeout(ctx, 200*time.Millisecond)
	defer cancel()
	relB, err := b.acquire(short)
	if err != nil {
		t.Fatal("the second opener was starved by the first; they share slots")
	}
	relB()
	relA()
	relA2()
}

// --- the frame handoff -------------------------------------------------------------------
//
// Two properties are under test here and neither is visible with an idle consumer: that the
// newest frame always wins, and that frames cost no allocation. Both only misbehave under
// contention, which is exactly when the pipeline is under load.

// stubStream builds a stream with no subprocess behind it.
//
// The handoff and the buffer ring are what these tests are about, and ffmpeg has nothing to do
// with either; driving the producer by hand is what makes the contention window reproducible
// instead of a matter of timing luck.
func stubStream(t *testing.T, props Properties, cfg Config) *Stream {
	t.Helper()
	return newStream(props, cfg, func() {})
}

// emit does what pump does for one frame: take a buffer, fill it, publish it. It reports false
// when the ring is empty, which is the producer's cue to drop.
func emit(s *Stream, mark byte) bool {
	slot, buf, gen, ok := s.ring.acquire()
	if !ok {
		return false
	}
	for i := range buf {
		buf[i] = mark
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
	return true
}

func TestProducerReplacesRatherThanQueues(t *testing.T) {
	// A queue of frames is worse than no frame: against a slow recogniser it grows latency
	// without bound, and every picture it eventually delivers describes the past. The consumer
	// must see the newest frame and nothing else.
	s := stubStream(t, Properties{Width: 8, Height: 8}, Config{})

	const frames = 20
	for i := 1; i <= frames; i++ {
		if !emit(s, byte(i)) {
			t.Fatalf("ring exhausted after %d frames although nothing was held", i-1)
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	f, ok := s.Next(ctx)
	if !ok {
		t.Fatal("no frame delivered")
	}
	if f.Bytes[0] != frames {
		t.Errorf("delivered frame %d, want the newest (%d): the producer is queueing", f.Bytes[0], frames)
	}
	f.Release()

	// The frames it replaced must be gone, not waiting behind it.
	short, cancelShort := context.WithTimeout(ctx, 100*time.Millisecond)
	defer cancelShort()
	if stale, ok := s.Next(short); ok {
		t.Errorf("a second frame (%d) was queued behind the newest", stale.Bytes[0])
	}
	if got := s.Dropped(); got != frames-1 {
		t.Errorf("Dropped() = %d, want %d", got, frames-1)
	}
}

func TestNewFrameIsNeverDroppedInFavourOfTheOldOne(t *testing.T) {
	// The property, stated as the thing that must never happen: a consumer takes the OLD frame
	// and the NEW one vanishes with it.
	//
	// The publish path used to be a three-step dance on a capacity-1 channel (send, else receive
	// the old, else send again) whose last step could give up. It never actually did with one
	// producer -- 2M frames, zero losses -- because a consumer waiting on the empty channel is
	// handed the value directly and nothing else can refill the slot. With two publishers the
	// same loop discards the frame it is publishing roughly 3 times in 1000. So this test is a
	// lock, not a reproduction: the pointer swap makes the loss impossible by construction, and
	// this is what fails if publishing ever goes back to something with a failure branch.
	//
	// Each round publishes an old frame, then races a consumer against the publication of a new
	// one. The new frame must always survive -- taken by the consumer, or still pending.
	s := stubStream(t, Properties{Width: 8, Height: 8}, Config{})

	const (
		oldMark = 0x01
		newMark = 0x02
		rounds  = 500
	)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	for round := 0; round < rounds; round++ {
		if !emit(s, oldMark) {
			t.Fatalf("round %d: ring exhausted", round)
		}

		var got byte
		taken := make(chan struct{})
		go func() {
			defer close(taken)
			f, ok := s.Next(ctx)
			if !ok {
				return
			}
			got = f.Bytes[0]
			f.Release()
		}()

		if !emit(s, newMark) {
			t.Fatalf("round %d: ring exhausted publishing the new frame", round)
		}
		<-taken

		if got == newMark {
			continue // the consumer took the new one; the old one was correctly discarded
		}
		if got != oldMark {
			t.Fatalf("round %d: consumer got mark %#x, which is neither frame", round, got)
		}

		// The consumer took the old frame, so the new one must still be pending. If it is not,
		// the producer threw away the newer picture to keep the older one.
		short, cancelShort := context.WithTimeout(ctx, 2*time.Second)
		f, ok := s.Next(short)
		cancelShort()
		if !ok {
			t.Fatalf("round %d: the consumer got the OLD frame and the NEW one was dropped", round)
		}
		if f.Bytes[0] != newMark {
			t.Fatalf("round %d: pending frame is %#x, want the new one", round, f.Bytes[0])
		}
		f.Release()
	}
}

func TestFrameBuffersAreReused(t *testing.T) {
	// 1080p BGR24 is 6.2 MB. At 30 fps across 100 cameras a buffer per frame is ~18 GB/s of
	// garbage, so N frames must come out of the same handful of buffers, not N of them.
	props := Properties{Width: 640, Height: 480}
	frameSize := props.Width * props.Height * 3
	s := stubStream(t, props, Config{})

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	const frames = 200
	seen := make(map[*byte]bool)

	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	for i := 0; i < frames; i++ {
		if !emit(s, byte(i)) {
			t.Fatalf("ring exhausted at frame %d although every frame was released", i)
		}
		f, ok := s.Next(ctx)
		if !ok {
			t.Fatalf("frame %d never arrived", i)
		}
		seen[&f.Bytes[0]] = true
		f.Release()
	}
	runtime.ReadMemStats(&after)

	if len(seen) > buffersPerStream {
		t.Errorf("%d frames came from %d distinct buffers, want at most %d",
			frames, len(seen), buffersPerStream)
	}

	// The only per-frame allocation left is the Frame header the atomic pointer publishes -- tens
	// of bytes. A per-frame buffer would show up here as frames*frameSize, which is 184 MB.
	grew := after.TotalAlloc - before.TotalAlloc
	if budget := uint64(frames) * 256; grew > budget {
		t.Errorf("%d frames allocated %d bytes (budget %d, one buffer is %d): buffers are not reused",
			frames, grew, budget, frameSize)
	}
}

func TestProducerDropsWhenNoBufferIsFree(t *testing.T) {
	// A consumer holding every buffer must cost frames, never memory and never the stream. The
	// producer may not block (ffmpeg would back-pressure until the camera's own buffers overflow)
	// and may not allocate a fourth buffer (a slow consumer would then cost memory without
	// bound, which is what the fixed ring exists to prevent).
	props := Properties{Width: 64, Height: 64}
	frameSize := props.Width * props.Height * 3

	t.Run("ring", func(t *testing.T) {
		s := stubStream(t, props, Config{})
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		var held []Frame
		for i := 0; i < buffersPerStream; i++ {
			if !emit(s, byte(i)) {
				t.Fatalf("ring exhausted early, at frame %d", i)
			}
			f, ok := s.Next(ctx)
			if !ok {
				t.Fatalf("frame %d never arrived", i)
			}
			held = append(held, f) // deliberately not released
		}

		// Asking with nothing free must answer immediately and allocate nothing.
		allocs := testing.AllocsPerRun(100, func() {
			if _, _, _, ok := s.ring.acquire(); ok {
				t.Fatal("a buffer was handed out although the consumer holds all of them")
			}
		})
		if allocs != 0 {
			t.Errorf("a failed acquire allocated %v objects, want 0", allocs)
		}
		if emit(s, 0xFF) {
			t.Error("the producer published although no buffer was free")
		}

		for _, f := range held {
			f.Release()
		}
		if !emit(s, 0xAB) {
			t.Error("the ring did not recover once the frames were released")
		}
	})

	t.Run("pump", func(t *testing.T) {
		// The same thing end to end: a consumer that holds two frames while ffmpeg keeps feeding.
		// If the producer blocked, the pipe would fill and this test would time out rather than
		// fail; if it overwrote a held buffer, the held frames would no longer match.
		const frames = 400
		o := NewOpener(4, rawRunner(t, props, frames))
		o.SetGuard(Guard{AllowPrivate: true}) // fake cameras; the guard is tested on its own

		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
		defer cancel()

		s, err := o.Open(ctx, Config{URL: "rtsp://cam/live"})
		if err != nil {
			t.Fatalf("open: %v", err)
		}
		defer s.Close()

		var held []Frame
		var copies [][]byte
		for len(held) < 2 {
			f, ok := s.Next(ctx)
			if !ok {
				t.Fatalf("only %d frames arrived before the stream ended: %v", len(held), s.Err())
			}
			if len(f.Bytes) != frameSize {
				t.Fatalf("frame is %d bytes, want %d", len(f.Bytes), frameSize)
			}
			held = append(held, f)
			copies = append(copies, append([]byte(nil), f.Bytes...))
		}

		// With two frames held and one pending, the producer has nothing left and must drop its
		// way to the end of the stream.
		select {
		case <-s.done:
		case <-ctx.Done():
			t.Fatal("the producer blocked instead of dropping frames")
		}

		for i, f := range held {
			if string(f.Bytes) != string(copies[i]) {
				t.Errorf("held frame %d was overwritten while the consumer still had it", i)
			}
			f.Release()
		}
		if s.Dropped() == 0 {
			t.Error("no frame was reported dropped although the consumer held the ring")
		}
	})
}

func TestReleasedBufferIsPoisonedWhenDebugging(t *testing.T) {
	// Reading a frame after releasing it otherwise yields a plausible picture from a moment ago,
	// so the bug shows up as slightly wrong plates rather than as a failure. Poisoning makes it
	// look as wrong as it is.
	props := Properties{Width: 16, Height: 16}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	s := stubStream(t, props, Config{DebugBufferOwnership: true})
	if !emit(s, 0x11) {
		t.Fatal("nothing published")
	}
	f, ok := s.Next(ctx)
	if !ok {
		t.Fatal("no frame delivered")
	}
	stale := f.Bytes // the mistake this catches: keeping the slice past Release
	f.Release()

	for i, b := range stale {
		if b != poisonByte {
			t.Fatalf("byte %d of the released buffer is %#x, want %#x", i, b, poisonByte)
		}
	}

	// Off by default, because it costs a full pass over 6 MB per frame.
	plain := stubStream(t, props, Config{})
	if !emit(plain, 0x22) {
		t.Fatal("nothing published")
	}
	g, ok := plain.Next(ctx)
	if !ok {
		t.Fatal("no frame delivered")
	}
	kept := g.Bytes
	g.Release()
	if kept[0] == poisonByte {
		t.Error("buffers are poisoned without DebugBufferOwnership; that cost is not free")
	}
}

func TestReleasingTwiceDoesNotLendOneBufferTwice(t *testing.T) {
	// Frame is a value, so copies of it exist wherever it was passed along. If each copy's
	// Release returned the buffer, the ring would hand the same memory to two readers at once and
	// the corruption would look like a decoder bug.
	s := stubStream(t, Properties{Width: 8, Height: 8}, Config{})
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if !emit(s, 0x01) {
		t.Fatal("nothing published")
	}
	f, ok := s.Next(ctx)
	if !ok {
		t.Fatal("no frame delivered")
	}
	copyOfFrame := f
	f.Release()
	copyOfFrame.Release()
	f.Release()

	// The ring must hold exactly its buffers, no more.
	if got := len(s.ring.free); got != buffersPerStream {
		t.Fatalf("ring holds %d free buffers after repeated releases, want %d", got, buffersPerStream)
	}

	seen := make(map[*byte]bool)
	for i := 0; i < buffersPerStream; i++ {
		_, buf, _, ok := s.ring.acquire()
		if !ok {
			t.Fatalf("only %d buffers were available", i)
		}
		if seen[&buf[0]] {
			t.Fatal("the same buffer was handed out twice")
		}
		seen[&buf[0]] = true
	}
}

// rawRunner stands in for both tools: ffprobe answers with the geometry, ffmpeg replays a file of
// raw BGR24 frames. A file rather than a generator so the bytes are identical on every run --
// these tests compare frame contents, and a race that only reproduces sometimes is not a test.
func rawRunner(t *testing.T, props Properties, frames int) Runner {
	t.Helper()
	frameSize := props.Width * props.Height * 3
	raw := make([]byte, frames*frameSize)
	for i := range raw {
		// Anything but a solid frame: isBlank samples every 997th byte and would drop it.
		raw[i] = byte(i % 251)
	}
	path := filepath.Join(t.TempDir(), "frames.bgr24")
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	probe := fmt.Sprintf(`{"streams":[{"width":%d,"height":%d,"r_frame_rate":"30/1","codec_name":"h264"}]}`,
		props.Width, props.Height)

	return func(ctx context.Context, name string, args ...string) *exec.Cmd {
		if strings.Contains(name, "ffprobe") {
			return exec.CommandContext(ctx, "sh", "-c", "printf %s "+strconv.Quote(probe))
		}
		return exec.CommandContext(ctx, "cat", path)
	}
}

func TestPumpDeliversTheNewestFrameAndKeepsTheRingBounded(t *testing.T) {
	// The end-to-end shape of both fixes: frames arrive newest-first out of three buffers, and
	// the stream ends by itself.
	props := Properties{Width: 64, Height: 64}
	o := NewOpener(4, rawRunner(t, props, 300))
	o.SetGuard(Guard{AllowPrivate: true}) // fake cameras; the guard is tested on its own

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	s, err := o.Open(ctx, Config{URL: "rtsp://cam/live"})
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer s.Close()

	seen := make(map[*byte]bool)
	received := 0
	for {
		f, ok := s.Next(ctx)
		if !ok {
			break
		}
		seen[&f.Bytes[0]] = true
		received++
		f.Release()
	}
	if received == 0 {
		t.Fatal("no frames arrived")
	}
	if len(seen) > buffersPerStream {
		t.Errorf("%d frames came from %d buffers, want at most %d", received, len(seen), buffersPerStream)
	}
	if err := s.Err(); err == nil || !strings.Contains(err.Error(), "stream ended") {
		t.Errorf("Err() = %v, want the end of the stream", err)
	}
}

// TestRealFFmpegStreamReusesBuffers runs the pump against real ffmpeg.
//
// It is opt-in through ALPR_TEST_CLIP rather than pointed at a checked-in fixture: a video file
// is the one input this package cannot mock away, and requiring it would make the suite depend
// on a binary and a blob that most machines running `go test ./...` do not have.
func TestRealFFmpegStreamReusesBuffers(t *testing.T) {
	clip := os.Getenv("ALPR_TEST_CLIP")
	if clip == "" {
		t.Skip("set ALPR_TEST_CLIP to a video file to run the ffmpeg integration test")
	}
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg not installed")
	}
	if _, err := exec.LookPath("ffprobe"); err != nil {
		t.Skip("ffprobe not installed")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	o := NewOpener(4, nil)
	s, err := o.Open(ctx, Config{URL: clip})
	if err != nil {
		t.Fatalf("open %s: %v", clip, err)
	}
	defer s.Close()

	props := s.Properties()
	frameSize := props.Width * props.Height * 3
	if frameSize == 0 {
		t.Fatalf("probe gave no geometry: %+v", props)
	}

	seen := make(map[*byte]bool)
	received := 0
	for {
		f, ok := s.Next(ctx)
		if !ok {
			break
		}
		if len(f.Bytes) != frameSize {
			t.Fatalf("frame is %d bytes, want %d", len(f.Bytes), frameSize)
		}
		seen[&f.Bytes[0]] = true
		received++
		f.Release()
	}
	if received == 0 {
		t.Fatalf("no frames decoded from %s (%v)", clip, s.Err())
	}
	if len(seen) > buffersPerStream {
		t.Errorf("%d frames came from %d buffers, want at most %d", received, len(seen), buffersPerStream)
	}
	t.Logf("%d frames of %dx%d from %d buffers, %d dropped",
		received, props.Width, props.Height, len(seen), s.Dropped())
}

// BenchmarkFrameHandoff is the number the ring exists for, at 1080p BGR24.
//
// Before the ring: 6226019 B/op, 2 allocs/op, ~60us -- one 6.2 MB buffer per frame, which is
// ~18 GB/s of garbage across 100 cameras at 30 fps. After: 96 B/op, 1 alloc/op, ~80ns. The one
// allocation left is the Frame the atomic pointer publishes, not the picture.
func BenchmarkFrameHandoff(b *testing.B) {
	props := Properties{Width: 1920, Height: 1080}
	s := newStream(props, Config{}, func() {})
	ctx := context.Background()

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		slot, buf, gen, ok := s.ring.acquire()
		if !ok {
			b.Fatal("ring exhausted although every frame was released")
		}
		buf[0] = byte(i) // touch it; pump reads the picture in here
		s.publish(Frame{
			Bytes: buf, Width: props.Width, Height: props.Height,
			CapturedAt: time.Now(), ring: s.ring, slot: slot, gen: gen,
		})
		f, ok := s.Next(ctx)
		if !ok {
			b.Fatal("no frame")
		}
		f.Release()
	}
}

func TestRTSPTransportIsOnlyPassedForRTSP(t *testing.T) {
	// ffmpeg rejects -rtsp_transport for any other input before it opens anything, so passing it
	// unconditionally made an HTTP camera or a replayed file impossible to open -- and the error
	// ("Option rtsp_transport not found") points at the flag, not at the URL that provoked it.
	for _, url := range []string{"rtsp://cam/live", "rtsps://cam/live"} {
		if got := transportArgs(url); len(got) != 2 || got[0] != "-rtsp_transport" || got[1] != "tcp" {
			t.Errorf("transportArgs(%q) = %v, want TCP transport", url, got)
		}
	}
	for _, url := range []string{"http://cam/mjpeg", "/var/lib/clips/gate.mp4", "file:///clip.mp4"} {
		if got := transportArgs(url); got != nil {
			t.Errorf("transportArgs(%q) = %v, want nothing: ffmpeg refuses the option here", url, got)
		}
	}
}

func TestHardwareDecodeIsOffByDefault(t *testing.T) {
	// CPU decode has to stay the default. A machine without NVDEC that silently got cuvid flags
	// would fail to open every camera, and the error names a decoder rather than a setting.
	if args := (Config{}).hwaccelArgs("h264"); args != nil {
		t.Errorf("an unconfigured Config asked for hardware decode: %v", args)
	}
}

func TestCuvidNamesTheDecoderNotJustTheAccelerator(t *testing.T) {
	// `-hwaccel cuvid` on its own silently falls back to the software decoder on some ffmpeg
	// builds. That is the worst outcome available: it looks like hardware decode that did not
	// help, and the CPU cost it was meant to remove is still there.
	//
	// The Rust and C++ engines both pass -c:v explicitly for the same reason.
	args := Config{HWAccel: "cuvid", HWDevice: -1}.hwaccelArgs("h264")
	joined := strings.Join(args, " ")
	if !strings.Contains(joined, "-c:v h264_cuvid") {
		t.Errorf("cuvid did not name the decoder: %v", args)
	}
}

func TestDecoderFollowsTheProbedCodec(t *testing.T) {
	// Both sibling engines make the operator pick between cuvid and hevc_cuvid, and picking
	// wrong means the camera does not open. ffprobe already answered that question.
	for codec, want := range map[string]string{
		"h264":  "h264_cuvid",
		"hevc":  "hevc_cuvid",
		"mjpeg": "mjpeg_cuvid",
	} {
		args := Config{HWAccel: "cuvid", HWDevice: -1}.hwaccelArgs(codec)
		if !strings.Contains(strings.Join(args, " "), want) {
			t.Errorf("codec %q gave %v, want %s", codec, args, want)
		}
	}
}

func TestUnknownCodecFallsBackToSoftwareRatherThanFailing(t *testing.T) {
	// A codec NVDEC cannot handle must cost that one camera its hardware decode, never its
	// connection. Naming a decoder that does not exist makes ffmpeg refuse to start.
	args := Config{HWAccel: "cuvid", HWDevice: -1}.hwaccelArgs("av1")
	if strings.Contains(strings.Join(args, " "), "-c:v") {
		t.Errorf("a codec NVDEC does not decode still got a decoder: %v", args)
	}
}

func TestHardwareDeviceIsSelectable(t *testing.T) {
	// With one card this does not matter. With two it decides whether the fleet shares them or
	// whether device 0 quietly decodes for all of it while device 1 idles.
	args := Config{HWAccel: "cuvid", HWDevice: 1}.hwaccelArgs("h264")
	if !strings.Contains(strings.Join(args, " "), "-gpu 1") {
		t.Errorf("device 1 was not selected: %v", args)
	}
}

func TestCuvidKeepsTheFrameInSystemMemory(t *testing.T) {
	// `-hwaccel cuvid` leaves the decoded frame on the GPU, and the bgr24 pipe cannot reach it:
	// ffmpeg 6.1 stops on the first frame with "Impossible to convert between the formats". The
	// decoder named alone, with -gpu, decodes on NVDEC and hands back a frame the pipe can carry.
	args := strings.Join(Config{HWAccel: "cuvid", HWDevice: 0}.hwaccelArgs("h264"), " ")
	if strings.Contains(args, "-hwaccel") {
		t.Errorf("cuvid passed -hwaccel, which fails every camera on this image's ffmpeg: %s", args)
	}
	if args != "-c:v h264_cuvid -gpu 0" {
		t.Errorf("cuvid args = %q, want %q", args, "-c:v h264_cuvid -gpu 0")
	}
}

func TestMaxFPSFiltersAfterTheDecoder(t *testing.T) {
	if args := (Config{}).filterArgs(); args != nil {
		t.Errorf("an unconfigured Config filtered frames: %v", args)
	}
	if got := strings.Join(Config{MaxFPS: 10}.filterArgs(), " "); got != "-vf fps=10" {
		t.Errorf("filterArgs = %q, want %q", got, "-vf fps=10")
	}
}

func TestHardwareFlagsComeBeforeTheInput(t *testing.T) {
	// They choose how the input is opened, so ffmpeg ignores them after -i. The failure is
	// silent: the camera opens and decodes on the CPU.
	cfg := Config{HWAccel: "cuvid", HWDevice: 0}
	args := append(cfg.hwaccelArgs("h264"), transportArgs("rtsp://cam/live")...)
	args = append(args, "-i", "rtsp://cam/live")

	hw, input := -1, -1
	for i, a := range args {
		if a == "-c:v" && hw < 0 {
			hw = i
		}
		if a == "-i" {
			input = i
		}
	}
	if hw < 0 || input < 0 || hw > input {
		t.Fatalf("hardware flags are not before -i: %v", args)
	}
}
