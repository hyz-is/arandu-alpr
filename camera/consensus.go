package camera

import (
	"context"
	"time"

	"github.com/hyz-is/arandu-alpr"
	"github.com/hyz-is/arandu-alpr/contract"
	"github.com/hyz-is/arandu-alpr/engine"
)

// keptFrame is a copy of the frame a track's best reading came from.
//
// The frame the loop is given is borrowed from a ring of three and goes back the moment step
// returns, and a track decides frames or seconds later. What the Sink receives has to be the
// picture the plate was read from, so the loop copies it when a reading becomes the track's best
// -- one memmove of the frame, not an encode, which happens once per verdict instead of once per
// improvement.
type keptFrame struct {
	pix  []byte
	w, h int
	at   time.Time
}

// observe hands one reading to the consensus and acts on what it says.
func (c *Camera) observe(
	ctx context.Context,
	f engine.Frame,
	cand engine.Candidate,
	reading engine.Reading,
	confidence, focus float64,
	now time.Time,
) {
	table := reading.Probabilities
	if len(table) == 0 {
		// An engine that kept only the text: a vote is still more than a row per frame.
		table = alpr.ProbabilitiesFromText(reading.Plate, reading.CharConfidences)
	}
	res := c.consensus.Observe(alpr.Observation{
		At:               now,
		CapturedAt:       f.CapturedAt,
		Box:              alpr.BBox{X: cand.Box.X, Y: cand.Box.Y, Width: cand.Box.Width, Height: cand.Box.Height},
		DetectorScore:    cand.Score,
		Sharpness:        focus,
		FrameWidth:       f.Width,
		Plate:            reading.Plate,
		Confidence:       confidence,
		Probabilities:    table,
		Region:           reading.Region,
		RegionConfidence: reading.RegionConfidence,
	})
	if res.Keep {
		c.keep(res.Track, f)
	}
	if res.Verdict != nil {
		c.emitVerdict(ctx, *res.Verdict, now)
	}
}

// flushConsensus decides the tracks whose vehicle has gone, and lets go of their frames.
func (c *Camera) flushConsensus(ctx context.Context, now time.Time) {
	if c.consensus == nil {
		return
	}
	verdicts, closed := c.consensus.Flush(now)
	for _, v := range verdicts {
		c.emitVerdict(ctx, v, now)
	}
	for _, id := range closed {
		c.drop(id)
	}
}

// emitVerdict delivers a track's decision the way a single reading used to be delivered: through
// the deduper, then to the Sink with the images of the best frame.
//
// The deduper stays in front of the Sink. A track reports its vehicle once, but the same car can
// leave and come back inside the cooldown, and that is a second track with the same plate.
func (c *Camera) emitVerdict(ctx context.Context, v alpr.Verdict, now time.Time) {
	defer c.drop(v.Track)

	if c.cfg.Dedup != nil && !c.cfg.Dedup.Accept(v.Plate, now) {
		c.m.deduplicated.Inc()
		return
	}

	cand := engine.Candidate{
		Box:   contract.BBox{X: v.Best.Box.X, Y: v.Best.Box.Y, Width: v.Best.Box.Width, Height: v.Best.Box.Height},
		Score: v.Best.DetectorScore,
	}
	reading := engine.Reading{
		Plate:            v.Plate,
		CharConfidences:  v.CharConfidences,
		Confidence:       v.Confidence,
		Region:           v.Best.Region,
		RegionConfidence: v.Best.RegionConfidence,
	}
	frame := engine.Frame{CapturedAt: v.Best.CapturedAt}
	if k := c.kept[v.Track]; k != nil {
		frame = engine.Frame{Pix: k.pix, Width: k.w, Height: k.h, CapturedAt: k.at}
	}

	c.m.detected.Inc()
	c.lastDetectionAt = now
	c.noDetectionSent = false
	c.remember(cand.Box, now)
	if !frame.CapturedAt.IsZero() {
		c.m.pipelineDuration.Observe(c.now().Sub(frame.CapturedAt).Seconds())
	}

	c.deliver(ctx, frame, cand, reading, v.Confidence, now)
}

// needsPixels says whether anything downstream will ask for a picture.
func (c *Camera) needsPixels() bool {
	return c.cfg.Encoder != nil && (c.cfg.IncludeFullImage || c.cfg.IncludeCropImage || c.cfg.Preview != nil)
}

// keep copies f as the best frame of a track, reusing the track's own buffer when it has one.
func (c *Camera) keep(track uint64, f engine.Frame) {
	if !c.needsPixels() || len(f.Pix) == 0 {
		return
	}
	k := c.kept[track]
	if k == nil {
		if len(c.kept) >= maxKeptFrames {
			c.evictOldestKept()
		}
		k = &keptFrame{}
		c.kept[track] = k
	}
	n := len(f.Pix)
	if cap(k.pix) < n {
		k.pix = c.takeSpare(n)
	}
	k.pix = k.pix[:n]
	copy(k.pix, f.Pix)
	k.w, k.h, k.at = f.Width, f.Height, f.CapturedAt
}

// drop gives a track's frame back for the next track to use.
func (c *Camera) drop(track uint64) {
	k := c.kept[track]
	if k == nil {
		return
	}
	delete(c.kept, track)
	if len(c.spare) < maxSpareFrames && cap(k.pix) > 0 {
		c.spare = append(c.spare, k.pix[:0])
	}
}

// maxKeptFrames bounds the pictures a camera holds for tracks that have not decided: one frame is
// 11 MB at 1440p, and a car park can show a dozen plates at once. A track that loses its picture to
// a newer one is still decided and reported; it just goes without images.
const maxKeptFrames = 4

// evictOldestKept gives back the picture of the track whose best frame is the oldest.
func (c *Camera) evictOldestKept() {
	var oldest uint64
	var at time.Time
	for id, k := range c.kept {
		if oldest == 0 || k.at.Before(at) {
			oldest, at = id, k.at
		}
	}
	if oldest != 0 {
		c.drop(oldest)
	}
}

// maxSpareFrames is how many buffers a camera keeps for the next track: enough for the vehicles
// a lane shows at once, few enough that a quiet camera does not sit on a gigabyte.
const maxSpareFrames = 3

func (c *Camera) takeSpare(n int) []byte {
	for i, b := range c.spare {
		if cap(b) >= n {
			c.spare = append(c.spare[:i], c.spare[i+1:]...)
			return b
		}
	}
	return make([]byte, n)
}
