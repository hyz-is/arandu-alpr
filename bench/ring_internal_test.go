package bench

import (
	"context"
	"testing"
	"time"

	"github.com/hyz-is/arandu-alpr/streaming"
)

// ringSize is the number of buffers a stream lends out, restated here because it is unexported
// where it lives. The tests below assert against the CONSEQUENCE of that number rather than
// against the number itself, so a change to it changes what they measure and not whether they
// pass.
const ringSize = 3

// openReplayStream opens a stream over the fake ffmpeg, replaying raw frames as fast as the
// pipe allows.
func openReplayStream(t *testing.T) *streaming.Stream {
	t.Helper()

	clip, err := rawClip(t.TempDir(), 16)
	if err != nil {
		t.Fatal(err)
	}
	opener := streaming.NewOpener(4, replayRunner(clip))
	opener.SetGuard(streaming.Guard{AllowPrivate: true}) // a local clip, which the camera guard refuses by design
	stream, err := opener.Open(context.Background(), streaming.Config{URL: "file://" + clip})
	if err != nil {
		t.Fatal(err)
	}
	return stream
}

// TestRingKeepsASlowConsumerOnTheNewestFrame.
//
// The whole reason the capture holds a fixed ring and overwrites rather than queues is what a
// slow consumer gets: the newest picture, always, at the cost of the ones it missed. A queue
// would hand it the oldest one and the frames would get steadily older -- which is the failure
// that reads as "the ALPR is slow" and is actually "the ALPR is answering about the past".
//
// So this measures the AGE of what a deliberately slow consumer receives, at the start of a run
// and at the end of it, and requires that it does not climb.
func TestRingKeepsASlowConsumerOnTheNewestFrame(t *testing.T) {
	const (
		takes     = 60
		consumeAt = 10 * time.Millisecond
	)

	stream := openReplayStream(t)
	defer stream.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	ages := make([]time.Duration, 0, takes)
	for i := 0; i < takes; i++ {
		frame, ok := stream.Next(ctx)
		if !ok {
			t.Fatalf("the stream ended after %d frames: %v", i, stream.Err())
		}
		ages = append(ages, time.Since(frame.CapturedAt))

		// The slow consumer. It holds the borrow for the whole sleep, which is exactly what a
		// camera doing a GPU pass does.
		time.Sleep(consumeAt)
		frame.Release()
	}

	early := meanAge(ages[:10])
	late := meanAge(ages[len(ages)-10:])

	// Two producers' worth of slack: the age of the newest frame is bounded by how long the
	// producer takes to decode one, and that varies with the machine. What must not happen is
	// growth proportional to how long the consumer has been behind.
	if late > early+5*consumeAt {
		t.Errorf("delivered frames aged from %s to %s over %d takes at %s each\n\n"+
			"A slow consumer must keep receiving the NEWEST frame, not the front of a queue. "+
			"Age that grows with how long the consumer has been behind means frames are being "+
			"kept, and every kept frame answers a question that has already gone stale.",
			early.Round(time.Millisecond), late.Round(time.Millisecond), takes, consumeAt)
	}

	if stream.Dropped() == 0 {
		t.Errorf("nothing was dropped while the consumer took %s per frame\n\n"+
			"Either the producer is blocking on the consumer -- which back-pressures ffmpeg and "+
			"eventually the camera's own buffers -- or frames are being held somewhere.",
			consumeAt)
	}

	t.Logf("%d takes at %s: age %s -> %s, %d frames dropped by the producer",
		takes, consumeAt, early.Round(time.Millisecond), late.Round(time.Millisecond), stream.Dropped())
}

// TestAForgottenReleaseStopsTheCameraLoudly.
//
// The borrow contract has one rule and it is easy to break: whoever receives a frame releases
// it. This pins what breaking it does, because the answer is the reason the rule is survivable
// at all.
//
// A consumer that never releases cannot drain the ring past its size, and then the feed stops.
// It does not corrupt a picture, it does not leak, and it does not slowly degrade -- it goes
// quiet, with the producer's dropped counter climbing. That is a diagnosable failure. Handing
// out a fresh buffer instead would turn it into unbounded memory growth on the busiest camera
// in the fleet.
func TestAForgottenReleaseStopsTheCameraLoudly(t *testing.T) {
	stream := openReplayStream(t)
	defer stream.Close()

	held := make([]streaming.Frame, 0, ringSize+1)
	defer func() {
		for _, f := range held {
			f.Release()
		}
	}()

	starved := false
	for i := 0; i < ringSize+3; i++ {
		ctx, cancel := context.WithTimeout(context.Background(), 750*time.Millisecond)
		frame, ok := stream.Next(ctx)
		cancel()
		if !ok {
			starved = true
			break
		}
		held = append(held, frame)
	}

	if len(held) > ringSize {
		t.Errorf("a consumer that released nothing was lent %d buffers, more than the ring's %d\n\n"+
			"The ring's footprint is what makes a fleet's memory knowable before it is deployed: "+
			"three buffers per camera, and no fourth however far behind the consumer falls.",
			len(held), ringSize)
	}
	if !starved {
		t.Errorf("the feed kept delivering after %d unreleased borrows\n\n"+
			"With every buffer out, the producer must discard rather than allocate. Allocating a "+
			"fresh buffer would make a slow consumer cost memory without bound, which is the "+
			"failure the ring exists to prevent.", len(held))
	}
	if stream.Dropped() == 0 {
		t.Errorf("nothing was counted as dropped while the ring was empty\n\n" +
			"A stalled camera with no counter moving is a camera that stopped for no visible " +
			"reason, and this counter is the only thing that names the cause.")
	}

	t.Logf("held %d buffers without releasing; feed stopped, %d frames dropped by the producer",
		len(held), stream.Dropped())
}

func meanAge(ages []time.Duration) time.Duration {
	if len(ages) == 0 {
		return 0
	}
	var sum time.Duration
	for _, a := range ages {
		sum += a
	}
	return sum / time.Duration(len(ages))
}
