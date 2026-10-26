package camera

import (
	"context"
	"testing"
	"time"

	"github.com/hyz-is/arandu-alpr/contract"
	"github.com/hyz-is/arandu-alpr/engine"
)

// recordingStills hears every frame and asks for the pixels of the ones a test wants.
type recordingStills struct {
	want     bool
	seen     []time.Time
	pixels   []contract.Still
	snapshot [][]byte
}

func (s *recordingStills) Seen(_ string, at time.Time, _, _ int) bool {
	s.seen = append(s.seen, at)
	w := s.want
	s.want = false
	return w
}

func (s *recordingStills) Still(_ string, st contract.Still) {
	s.pixels = append(s.pixels, st)
	// A consumer copies what it keeps: the pixels are lent for the call.
	s.snapshot = append(s.snapshot, append([]byte(nil), st.Pix[:16]...))
}

// A still is asked of every frame, a stale one included, and the pixels it gets are the camera's
// own: before the brightness adjustment rewrites them in place.
func TestStillsHearEveryFrameAndGetThePixelsBeforeTheyAreAdjusted(t *testing.T) {
	stills := &recordingStills{}
	r := newRig(t, func(c *Config) {
		c.Stills = stills
		c.Adjust = func(pix []byte, _, _ int) {
			for i := range pix {
				pix[i] = 0xFF
			}
		}
		c.Spec.Brightness = 150
	})
	ctx := context.Background()

	// Nobody asked: the frame is heard and its pixels are not handed over.
	f := r.freshFrame()
	f.Pix[0] = 7
	r.cam.step(ctx, f)
	if len(stills.seen) != 1 || len(stills.pixels) != 0 {
		t.Fatalf("seen=%d pixels=%d, want one frame heard and none handed over", len(stills.seen), len(stills.pixels))
	}

	// Somebody asked: this frame's pixels, as they came.
	stills.want = true
	f = r.freshFrame()
	f.Pix[0] = 9
	r.clock.Advance(time.Second)
	r.cam.step(ctx, f)
	if len(stills.pixels) != 1 {
		t.Fatalf("pixels handed over %d times, want once", len(stills.pixels))
	}
	got := stills.pixels[0]
	if got.Width != frameW || got.Height != frameH || stills.snapshot[0][0] != 9 {
		t.Fatalf("still = %dx%d, first byte %d: want the frame as it came", got.Width, got.Height, stills.snapshot[0][0])
	}

	// A frame too late for a GPU pass is still a frame the camera sent.
	stills.want = true
	r.cam.step(ctx, r.staleFrame())
	if len(stills.seen) != 3 || len(stills.pixels) != 2 {
		t.Errorf("seen=%d pixels=%d after a stale frame, want 3 and 2", len(stills.seen), len(stills.pixels))
	}
	if stills.seen[2].IsZero() {
		t.Error("the stale frame was heard without the time it was captured")
	}
}

// Without a consumer, and with one that never asks, the loop allocates nothing for stills.
func TestStillsCostAFrameNothingWhenNobodyAsks(t *testing.T) {
	r := newRig(t, func(c *Config) { c.Stills = &countingStills{} })
	f := engine.Frame{Pix: make([]byte, frameW*frameH*3), Width: frameW, Height: frameH, CapturedAt: r.clock.Now()}
	ctx := context.Background()
	r.cam.step(ctx, f)
	allocs := testing.AllocsPerRun(50, func() { r.cam.step(ctx, f) })
	if allocs > 0 {
		t.Errorf("a frame nobody asked a still of allocated %.0f times", allocs)
	}
}

type countingStills struct{ n int }

func (s *countingStills) Seen(string, time.Time, int, int) bool { s.n++; return false }
func (s *countingStills) Still(string, contract.Still)          {}
