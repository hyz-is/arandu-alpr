package camera

import (
	"bytes"
	"context"
	"image/jpeg"
	"sync"
	"testing"
	"time"

	"github.com/hyz-is/arandu-alpr"
	"github.com/hyz-is/arandu-alpr/contract"
	"github.com/hyz-is/arandu-alpr/engine"
	"github.com/hyz-is/arandu-alpr/imaging"
)

// platesByBox answers each box with the plate that sits there, which the one-reading fakeEngine
// cannot: several vehicles in view at once is the case this file is about.
type platesByBox struct {
	*fakeEngine
	plates map[float64]engine.Reading
}

func (e platesByBox) Recognize(_ context.Context, _ engine.Frame, box contract.BBox) (engine.Reading, error) {
	return e.plates[box.X*10000+box.Y], nil
}

// allDetections keeps every delivery; the fakeSink keeps only the last.
type allDetections struct {
	mu  sync.Mutex
	got []contract.Detection
}

func (s *allDetections) Detect(_ context.Context, d contract.Detection) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.got = append(s.got, d)
	return nil
}

func (s *allDetections) all() []contract.Detection {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]contract.Detection(nil), s.got...)
}

// crowd is six vehicles in view at once, far enough apart to be six tracks, one more than two
// rows of maxKeptFrames hold.
func crowd() (boxes []contract.BBox, plates map[float64]engine.Reading, texts []string) {
	texts = []string{"ABC1D23", "DEF2G34", "HIJ3K45", "LMN4P56", "QRS5T67", "UVW6X78"}
	plates = map[float64]engine.Reading{}
	i := 0
	for _, y := range []float64{100, 800} {
		for _, x := range []float64{100, 800, 1500} {
			b := contract.BBox{X: x, Y: y, Width: 100, Height: 30}
			boxes = append(boxes, b)
			plates[x*10000+y] = readingOf(probs(texts[i], 0.95, -1, 0))
			i++
		}
	}
	return boxes, plates, texts
}

func TestEveryPlateOfACrowdReachesTheSinkWithItsPictures(t *testing.T) {
	// More vehicles undecided at once than maxKeptFrames hold raw. The one whose frame is let go
	// first still decides later, and went to the Sink with no picture at all: 275 of 1461 reads of
	// the busiest camera in an hour.
	cfg := alpr.DefaultConsensus()
	enc, err := imaging.NewEncoder(80)
	if err != nil {
		t.Fatal(err)
	}
	boxes, plates, texts := crowd()
	sink := &allDetections{}
	r := newRig(t, func(c *Config) {
		c.Consensus = &cfg
		c.Encoder = enc
		c.IncludeFullImage = true
		c.IncludeCropImage = true
		c.Focus = fakeFocus{variance: 300}
		c.Sink = sink
	})
	r.cam.cfg.Engine = platesByBox{fakeEngine: r.engine, plates: plates}
	r.engine.setCandidates([]engine.Candidate{
		{Box: boxes[0], Score: 0.9}, {Box: boxes[1], Score: 0.9}, {Box: boxes[2], Score: 0.9},
		{Box: boxes[3], Score: 0.9}, {Box: boxes[4], Score: 0.9}, {Box: boxes[5], Score: 0.9},
	})

	pix := make([]byte, frameW*frameH*3)
	for i := range pix {
		pix[i] = 200
	}
	for i := 0; i < 6; i++ {
		r.cam.step(context.Background(), engine.Frame{Pix: pix, Width: frameW, Height: frameH, CapturedAt: r.clock.Now()})
		r.clock.Advance(125 * time.Millisecond)
	}
	r.clock.Advance(5 * time.Second)
	r.engine.setCandidates(nil)
	r.cam.step(context.Background(), r.freshFrame()) // the clock the tracks are judged by is the frames'

	got := sink.all()
	if len(got) != len(texts) {
		t.Fatalf("sink calls = %d for %d vehicles, want one each", len(got), len(texts))
	}
	for _, d := range got {
		if len(d.FullJPEG) == 0 {
			t.Errorf("%s reached the sink with no full picture", d.Plate)
		}
		if len(d.CropJPEG) == 0 {
			t.Errorf("%s reached the sink with no crop", d.Plate)
			continue
		}
		img, err := jpeg.Decode(bytes.NewReader(d.CropJPEG))
		if err != nil {
			t.Errorf("%s: the crop is not a JPEG: %v", d.Plate, err)
			continue
		}
		// The crop is cut from the best reading's box, evicted frame or not.
		if w := img.Bounds().Dx(); w < int(d.Box.Width) || w > 2*int(d.Box.Width) {
			t.Errorf("%s: crop is %d px wide for a %v px plate", d.Plate, w, d.Box.Width)
		}
	}
	if n := r.errorCount(stageImages); n != 0 {
		t.Errorf("%v verdicts went out without pictures", n)
	}
	if len(r.cam.kept) != 0 {
		t.Errorf("%d frames still held after every verdict", len(r.cam.kept))
	}
}

func TestAnEncodedFrameIsReplacedByTheNextBestRawOneOnlyWhenOneIsFree(t *testing.T) {
	// A track whose frame was encoded and let go and then reads better takes a raw frame back only
	// when one is free: taking one from another track would encode that one, and a crowd would
	// encode a frame per reading. When it does take it, the pictures encoded from the old frame
	// must not travel with the new one.
	cfg := alpr.DefaultConsensus()
	enc, _ := imaging.NewEncoder(80)
	r := newRig(t, func(c *Config) {
		c.Consensus = &cfg
		c.Encoder = enc
		c.IncludeFullImage = true
	})
	pix := make([]byte, frameW*frameH*3)
	box := contract.BBox{X: 100, Y: 100, Width: 100, Height: 30}
	frameAt := func() engine.Frame {
		return engine.Frame{Pix: pix, Width: frameW, Height: frameH, CapturedAt: r.clock.Now()}
	}
	for i := 0; i < maxKeptFrames+1; i++ {
		r.cam.keep(uint64(i+1), frameAt(), box)
		r.clock.Advance(time.Millisecond)
	}
	old := r.cam.kept[1]
	if len(old.pix) != 0 || len(old.jpeg.full) == 0 {
		t.Fatalf("track 1 holds %d raw bytes and %d encoded, want its frame encoded and let go", len(old.pix), len(old.jpeg.full))
	}
	encoded := old.jpeg.full

	r.cam.keep(1, frameAt(), box)
	if k := r.cam.kept[1]; len(k.pix) != 0 || len(k.jpeg.full) != len(encoded) {
		t.Errorf("track 1 took a raw frame with none free")
	}
	if r.cam.raw != maxKeptFrames {
		t.Errorf("%d raw frames held, want %d", r.cam.raw, maxKeptFrames)
	}

	r.cam.drop(2)
	r.cam.keep(1, frameAt(), box)
	if k := r.cam.kept[1]; len(k.pix) == 0 || k.jpeg.full != nil {
		t.Errorf("track 1 holds %d raw bytes and %d encoded, want the new raw frame alone", len(k.pix), len(k.jpeg.full))
	}
	if r.cam.raw != maxKeptFrames {
		t.Errorf("%d raw frames held, want %d", r.cam.raw, maxKeptFrames)
	}
}

func TestTheEncodedPicturesOfUndecidedTracksAreBounded(t *testing.T) {
	cfg := alpr.DefaultConsensus()
	enc, _ := imaging.NewEncoder(80)
	r := newRig(t, func(c *Config) {
		c.Consensus = &cfg
		c.Encoder = enc
		c.IncludeFullImage = true
	})
	pix := make([]byte, frameW*frameH*3)
	box := contract.BBox{X: 100, Y: 100, Width: 100, Height: 30}
	for i := 0; i < maxEncodedFrames+maxKeptFrames+10; i++ {
		f := engine.Frame{Pix: pix, Width: frameW, Height: frameH, CapturedAt: r.clock.Now()}
		r.cam.keep(uint64(i+1), f, box)
		r.clock.Advance(time.Millisecond)
	}
	if got := r.cam.raw; got != maxKeptFrames {
		t.Errorf("%d raw frames held, want %d", got, maxKeptFrames)
	}
	if got := len(r.cam.kept); got != maxKeptFrames+maxEncodedFrames {
		t.Errorf("%d tracks held, want the raw %d and the encoded %d", got, maxKeptFrames, maxEncodedFrames)
	}
	if r.cam.kept[1] != nil {
		t.Errorf("the oldest track still holds pictures past the bound")
	}
}
