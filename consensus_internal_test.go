package alpr

import (
	"math"
	"strings"
	"testing"
	"time"
)

// table builds the OCR head's answer for a text: each character of the text gets `conf` at its
// slot, the padding class takes the slots after it, and every other class shares what is left.
// `alt` moves part of one slot's probability to another character, which is how a frame misreads.
type alt struct {
	slot int
	ch   byte
	p    float32
}

func table(text string, conf float32, alts ...alt) []float32 {
	a := len(Alphabet)
	out := make([]float32, MaxPlateSlots*a)
	for slot := 0; slot < MaxPlateSlots; slot++ {
		ch := byte(PadChar)
		if slot < len(text) {
			ch = text[slot]
		}
		rest := (1 - conf) / float32(a-1)
		for k := 0; k < a; k++ {
			out[slot*a+k] = rest
		}
		out[slot*a+strings.IndexByte(Alphabet, ch)] = conf
	}
	for _, x := range alts {
		row := out[x.slot*a : (x.slot+1)*a]
		best := 0
		for k, v := range row {
			if v > row[best] {
				best = k
			}
		}
		row[strings.IndexByte(Alphabet, x.ch)] = x.p
		row[best] -= x.p
		if row[best] < 0 {
			row[best] = 0
		}
	}
	return out
}

// argmaxText is what a frame reads on its own, the way DecodePlate does.
func argmaxText(p []float32) (string, float64) {
	r := DecodePlate(p)
	return r.Plate, r.Confidence()
}

func obs(at time.Time, x float64, p []float32) Observation {
	plate, conf := argmaxText(p)
	return Observation{
		At: at, CapturedAt: at,
		Box:           BBox{X: x, Y: 400, Width: 150, Height: 80},
		DetectorScore: 0.7, Sharpness: 150, FrameWidth: 2560,
		Plate: plate, Confidence: conf, Probabilities: p,
	}
}

func TestFramesThatEachMisreadOneCharacterAgreeOnTheRightPlate(t *testing.T) {
	// The failure this exists for. Five frames of one car, each wrong in a different slot: taken one
	// at a time they are five different plates, and the deduper cannot tell them from five cars. The
	// right character is the runner-up of every wrong slot, and the frames that read it correctly
	// outvote the one that did not.
	c := NewConsensus(DefaultConsensus())
	frames := [][]float32{
		table("ABC1D23", 0.80, alt{slot: 2, ch: 'G', p: 0.55}),
		table("ABC1D23", 0.80, alt{slot: 4, ch: 'O', p: 0.52}),
		table("ABC1D23", 0.80, alt{slot: 6, ch: '8', p: 0.50}),
		table("ABC1D23", 0.80),
		table("ABC1D23", 0.80, alt{slot: 0, ch: '8', p: 0.50}),
	}
	var got *Verdict
	for i, p := range frames {
		res := c.Observe(obs(t0.Add(time.Duration(i)*125*time.Millisecond), 400+float64(i)*20, p))
		if res.Verdict != nil {
			got = res.Verdict
		}
	}
	if got == nil {
		verdicts, _ := c.Flush(t0.Add(time.Minute))
		if len(verdicts) != 1 {
			t.Fatalf("no verdict from five frames of one car")
		}
		got = &verdicts[0]
	}
	if got.Plate != "ABC1D23" {
		t.Errorf("verdict %q, want ABC1D23", got.Plate)
	}
	if got.Observations < 3 {
		t.Errorf("verdict on %d readings", got.Observations)
	}
}

func TestOneCarIsReportedOnceNotOncePerFrame(t *testing.T) {
	c := NewConsensus(DefaultConsensus())
	var verdicts int
	for i := 0; i < 12; i++ {
		res := c.Observe(obs(t0.Add(time.Duration(i)*125*time.Millisecond), 400+float64(i)*15, table("ABC1D23", 0.9)))
		if res.Verdict != nil {
			verdicts++
		}
	}
	v, closed := c.Flush(t0.Add(time.Minute))
	verdicts += len(v)
	if verdicts != 1 {
		t.Errorf("%d verdicts for one car, want 1", verdicts)
	}
	if len(closed) != 1 {
		t.Errorf("%d tracks closed, want 1", len(closed))
	}
}

func TestDecidesEarlyOnlyWithEnoughReadings(t *testing.T) {
	cfg := DefaultConsensus()
	cfg.MinObservations = 3
	c := NewConsensus(cfg)
	for i := 0; i < 2; i++ {
		if res := c.Observe(obs(t0.Add(time.Duration(i)*100*time.Millisecond), 400, table("ABC1D23", 0.95))); res.Verdict != nil {
			t.Fatalf("decided on %d readings, the floor is %d", i+1, cfg.MinObservations)
		}
	}
	res := c.Observe(obs(t0.Add(300*time.Millisecond), 400, table("ABC1D23", 0.95)))
	if res.Verdict == nil || !res.Verdict.Early {
		t.Fatalf("no early verdict on the third agreeing reading: %+v", res.Verdict)
	}
}

func TestDisagreementDoesNotDecideEarly(t *testing.T) {
	// Two readings that split evenly must not be called a plate while the car is still in view.
	c := NewConsensus(DefaultConsensus())
	a, b := table("ABC1D23", 0.8), table("ABD1D23", 0.8)
	for i, p := range [][]float32{a, b, a, b} {
		if res := c.Observe(obs(t0.Add(time.Duration(i)*100*time.Millisecond), 400, p)); res.Verdict != nil {
			t.Fatalf("decided early on an even split at reading %d: %q (posterior %.2f)",
				i+1, res.Verdict.Plate, res.Verdict.Posterior)
		}
	}
}

func TestATrackThatEndsUndecidedDecidesWithWhatItHas(t *testing.T) {
	// A car seen twice by a slow camera never reaches three readings, and is still a car.
	c := NewConsensus(DefaultConsensus())
	c.Observe(obs(t0, 400, table("ABC1D23", 0.9)))
	c.Observe(obs(t0.Add(125*time.Millisecond), 430, table("ABC1D23", 0.9)))

	if v, _ := c.Flush(t0.Add(time.Second)); len(v) != 0 {
		t.Fatalf("decided %d tracks inside the gap", len(v))
	}
	v, closed := c.Flush(t0.Add(3 * time.Second))
	if len(v) != 1 || v[0].Plate != "ABC1D23" || v[0].Early {
		t.Fatalf("verdicts after the gap = %+v", v)
	}
	if len(closed) != 1 || closed[0] != v[0].Track {
		t.Errorf("closed %v, want the verdict's track %d", closed, v[0].Track)
	}
	if c.Tracks() != 0 {
		t.Errorf("%d tracks left after the vehicle went", c.Tracks())
	}
}

func TestAPlateSeenOnceMustBeSure(t *testing.T) {
	// One frame has nothing to be checked against, so it is held to a higher bar than a track.
	cfg := DefaultConsensus()
	c := NewConsensus(cfg)
	c.Observe(obs(t0, 400, table("ABC1D23", 0.66)))
	if v, _ := c.Flush(t0.Add(5 * time.Second)); len(v) != 0 {
		t.Errorf("a single 0.66 reading was reported: %+v", v)
	}
	c.Observe(obs(t0.Add(10*time.Second), 400, table("ABC1D23", 0.9)))
	if v, _ := c.Flush(t0.Add(15 * time.Second)); len(v) != 1 {
		t.Errorf("a single 0.90 reading was not reported")
	}
}

func TestLowConfidenceReadingsAreNotEvidence(t *testing.T) {
	c := NewConsensus(DefaultConsensus())
	res := c.Observe(obs(t0, 400, table("ABC1D23", 0.2)))
	if res.Track != 0 || c.Tracks() != 0 {
		t.Errorf("a 0.2 reading opened a track")
	}
}

func TestTracksThatNeverReachTheBarAreDroppedQuietly(t *testing.T) {
	// Not a plate: an object the detector liked and the OCR could not read. Three frames of it
	// must not become a row.
	c := NewConsensus(DefaultConsensus())
	for i := 0; i < 3; i++ {
		c.Observe(obs(t0.Add(time.Duration(i)*100*time.Millisecond), 400, table("QZK4", 0.45)))
	}
	if v, closed := c.Flush(t0.Add(time.Minute)); len(v) != 0 || len(closed) != 1 {
		t.Errorf("verdicts %d closed %d, want 0 and 1", len(v), len(closed))
	}
}

func TestTwoCarsBothReported(t *testing.T) {
	c := NewConsensus(DefaultConsensus())
	plates := map[string]bool{}
	for i := 0; i < 4; i++ {
		at := t0.Add(time.Duration(i) * 125 * time.Millisecond)
		if r := c.Observe(obs(at, 300, table("ABC1D23", 0.9))); r.Verdict != nil {
			plates[r.Verdict.Plate] = true
		}
		o := obs(at, 1700, table("XYZ9K88", 0.9))
		o.Box.Y = 800
		if r := c.Observe(o); r.Verdict != nil {
			plates[r.Verdict.Plate] = true
		}
	}
	if len(plates) != 2 || !plates["ABC1D23"] || !plates["XYZ9K88"] {
		t.Errorf("verdicts for %v, want both plates", plates)
	}
}

func TestADifferentPlateInTheSamePlaceIsANewTrack(t *testing.T) {
	// A car replaces another at the stop line, and the second is read where the first was. The
	// text is what tells them apart -- when the new reading is sure of its own, which a badly read
	// tail of the same car, at 0.85, is not.
	c := NewConsensus(DefaultConsensus())
	for i := 0; i < 4; i++ {
		c.Observe(obs(t0.Add(time.Duration(i)*100*time.Millisecond), 400, table("ABC1D23", 0.97)))
	}
	if c.Tracks() != 1 {
		t.Fatalf("tracks = %d", c.Tracks())
	}
	c.Observe(obs(t0.Add(500*time.Millisecond), 410, table("XYZ9K88", 0.97)))
	if c.Tracks() != 2 {
		t.Errorf("tracks = %d after a different plate arrived in the same place, want 2", c.Tracks())
	}
}

func TestADecidedTrackDoesNotReportAgain(t *testing.T) {
	c := NewConsensus(DefaultConsensus())
	n := 0
	for i := 0; i < 30; i++ {
		if r := c.Observe(obs(t0.Add(time.Duration(i)*100*time.Millisecond), 400, table("ABC1D23", 0.9))); r.Verdict != nil {
			n++
		}
	}
	if n != 1 {
		t.Errorf("%d verdicts from one parked car", n)
	}
	if v, _ := c.Flush(t0.Add(time.Minute)); len(v) != 0 {
		t.Errorf("the track reported again when it ended")
	}
}

func TestKeepFlagsTheBestFrameUntilTheTrackDecides(t *testing.T) {
	c := NewConsensus(DefaultConsensus())
	r1 := c.Observe(obs(t0, 400, table("ABC1D23", 0.6)))
	if !r1.Keep {
		t.Fatal("the first reading of a track is its best so far")
	}
	r2 := c.Observe(obs(t0.Add(100*time.Millisecond), 410, table("ABC1D23", 0.9)))
	if !r2.Keep {
		t.Error("a better reading was not flagged to keep")
	}
	r3 := c.Observe(obs(t0.Add(200*time.Millisecond), 420, table("ABC1D23", 0.62)))
	if r3.Keep {
		t.Error("a worse reading was flagged to keep")
	}
}

func TestTheVerdictCarriesTheBestFramesBox(t *testing.T) {
	c := NewConsensus(DefaultConsensus())
	var v *Verdict
	for i, conf := range []float32{0.7, 0.95, 0.7, 0.7} {
		o := obs(t0.Add(time.Duration(i)*100*time.Millisecond), 400+float64(i)*10, table("ABC1D23", conf))
		if r := c.Observe(o); r.Verdict != nil {
			v = r.Verdict
		}
	}
	if v == nil {
		t.Fatal("no verdict")
	}
	if v.Best.Box.X != 410 {
		t.Errorf("best box x = %v, want 410 (the 0.95 reading)", v.Best.Box.X)
	}
}

func TestSharperFramesWeighMore(t *testing.T) {
	// Same confidence, one frame soft and one sharp: the sharp one decides a 1-1 split.
	c := NewConsensus(DefaultConsensus())
	soft := obs(t0, 400, table("ABC1D23", 0.8))
	soft.Sharpness = 10
	sharp := obs(t0.Add(100*time.Millisecond), 400, table("ABD1D23", 0.8))
	sharp.Sharpness = 300
	c.Observe(soft)
	c.Observe(sharp)
	v, _ := c.Flush(t0.Add(time.Minute))
	if len(v) != 1 || v[0].Plate != "ABD1D23" {
		t.Errorf("verdict %+v, want the sharp frame's ABD1D23", v)
	}
}

func TestVerdictTextIsSomethingTheFramesCouldRead(t *testing.T) {
	// The winner is a text some frame read, or the slot-by-slot best of all of them. Never an
	// invention: every character of it has to be the top choice of a frame at that slot.
	c := NewConsensus(DefaultConsensus())
	frames := [][]float32{
		table("ABC1D23", 0.8, alt{slot: 2, ch: 'G', p: 0.6}),
		table("ABC1D23", 0.8, alt{slot: 4, ch: 'O', p: 0.6}),
		table("ABC1D23", 0.8),
	}
	for i, p := range frames {
		c.Observe(obs(t0.Add(time.Duration(i)*100*time.Millisecond), 400, p))
	}
	v, _ := c.Flush(t0.Add(time.Minute))
	if len(v) != 1 {
		t.Fatalf("verdicts %d", len(v))
	}
	for slot, ch := range []byte(v[0].Plate) {
		ok := false
		for _, p := range frames {
			row := p[slot*len(Alphabet) : (slot+1)*len(Alphabet)]
			best := 0
			for k, x := range row {
				if x > row[best] {
					best = k
				}
			}
			if Alphabet[best] == ch {
				ok = true
			}
		}
		if !ok {
			t.Errorf("slot %d of %q is %q, which no frame read there", slot, v[0].Plate, ch)
		}
	}
}

func TestNoNationalFormatIsAssumed(t *testing.T) {
	// A foreign plate, longer than any Mercosur one, decides like any other. There is no format
	// check in this engine and the consensus must not smuggle one in through the back.
	c := NewConsensus(DefaultConsensus())
	var v *Verdict
	for i := 0; i < 4; i++ {
		if r := c.Observe(obs(t0.Add(time.Duration(i)*100*time.Millisecond), 400, table("KA01AB1234", 0.9))); r.Verdict != nil {
			v = r.Verdict
		}
	}
	if v == nil || v.Plate != "KA01AB1234" {
		t.Errorf("verdict %+v for a ten-character plate", v)
	}
}

func TestVerdictConfidenceIsTheMeanOfItsCharacters(t *testing.T) {
	c := NewConsensus(DefaultConsensus())
	var got *Verdict
	for i := 0; i < 4; i++ {
		if r := c.Observe(obs(t0.Add(time.Duration(i)*100*time.Millisecond), 400, table("ABC1D23", 0.9))); r.Verdict != nil {
			got = r.Verdict
		}
	}
	if got == nil {
		t.Fatal("no verdict")
	}
	if len(got.CharConfidences) != len(got.Plate) {
		t.Fatalf("%d confidences for %d characters", len(got.CharConfidences), len(got.Plate))
	}
	var sum float64
	for _, x := range got.CharConfidences {
		sum += x
	}
	if math.Abs(sum/float64(len(got.CharConfidences))-got.Confidence) > 1e-9 {
		t.Errorf("confidence %v is not the mean of %v", got.Confidence, got.CharConfidences)
	}
	if math.Abs(got.Confidence-0.9) > 0.01 {
		t.Errorf("confidence %v for frames that all said 0.9", got.Confidence)
	}
}

func TestMalformedReadingsAreIgnored(t *testing.T) {
	c := NewConsensus(DefaultConsensus())
	o := obs(t0, 400, table("ABC1D23", 0.9))
	o.Probabilities = o.Probabilities[:10]
	if r := c.Observe(o); r.Track != 0 || c.Tracks() != 0 {
		t.Error("a reading with a truncated table opened a track")
	}
	o = obs(t0, 400, table("ABC1D23", 0.9))
	o.Plate = ""
	if r := c.Observe(o); r.Track != 0 {
		t.Error("an empty plate opened a track")
	}
}

func TestTheOldestTrackGoesWhenTheCameraSeesTooMany(t *testing.T) {
	cfg := DefaultConsensus()
	cfg.MaxTracks = 3
	c := NewConsensus(cfg)
	for i := 0; i < 5; i++ {
		o := obs(t0.Add(time.Duration(i)*10*time.Millisecond), 100+float64(i)*600, table("ABC1D23", 0.9))
		c.Observe(o)
	}
	if c.Tracks() > 3 {
		t.Errorf("%d tracks, the bound is 3", c.Tracks())
	}
}

func TestASmallPlateIsAWeakWitness(t *testing.T) {
	// Same confidence and sharpness, one reading of a plate that is 3% of the frame and one of a
	// plate that is 7%: the big one decides. They are the same car a moment apart, so the track
	// has to be one for the comparison to mean anything.
	c := NewConsensus(DefaultConsensus())
	far := obs(t0, 400, table("ABC1D23", 0.8))
	far.Box.Width = 77 // 3% of 2560
	near := obs(t0.Add(100*time.Millisecond), 420, table("ABD1D23", 0.8))
	near.Box.Width = 120
	c.Observe(far)
	c.Observe(near)
	v, _ := c.Flush(t0.Add(time.Minute))
	if len(v) != 1 || v[0].Plate != "ABD1D23" {
		t.Errorf("verdict %+v, want the big plate's ABD1D23", v)
	}
}

func TestAPlateTheDetectorDoubtsIsNotReported(t *testing.T) {
	// Agreeing with itself on every frame is not enough: a bumper that reads as a plate does that.
	c := NewConsensus(DefaultConsensus())
	for i := 0; i < 6; i++ {
		o := obs(t0.Add(time.Duration(i)*100*time.Millisecond), 400, table("ABC1D23", 0.9))
		o.DetectorScore = 0.33
		if r := c.Observe(o); r.Verdict != nil {
			t.Fatalf("a box the detector scored 0.33 was reported: %+v", r.Verdict)
		}
	}
	if v, _ := c.Flush(t0.Add(time.Minute)); len(v) != 0 {
		t.Errorf("reported at the end: %+v", v)
	}
}

func TestTheMotionOfATrackKeepsAnotherCarsPlateOut(t *testing.T) {
	// A near car going right and a far car's plate that appears between the near car's last
	// position and where it is going. Nearest-last-box would hand the far plate to the near car's
	// track and orphan the near car; the prediction does not.
	c := NewConsensus(DefaultConsensus())
	at := func(i int) time.Time { return t0.Add(time.Duration(i) * 125 * time.Millisecond) }
	for i := 0; i < 4; i++ {
		o := obs(at(i), 300+float64(i)*150, table("ABC1D23", 0.95))
		o.Box.Y, o.Box.Width = 700, 160
		c.Observe(o)
	}
	// Frame 4: the near car is predicted at x = 900, the far plate is at x = 600, a hundred pixels
	// from the near car's last box and a third of a plate away from nothing it should join.
	far := obs(at(4), 600, table("XYZ9K88", 0.5))
	far.Box.Y, far.Box.Width = 650, 100
	near := obs(at(4), 900, table("ABC1D23", 0.95))
	near.Box.Y, near.Box.Width = 700, 160
	c.Observe(far)
	res := c.Observe(near)
	if res.Verdict != nil {
		t.Fatalf("the near car reported a second time as %q: its track was taken over", res.Verdict.Plate)
	}
	if c.Tracks() != 2 {
		t.Errorf("tracks = %d, want the near car's and the far plate's", c.Tracks())
	}
}

func TestTheTailOfACarIsNotAnotherCar(t *testing.T) {
	// The last frames of a vehicle are small and soft, and the OCR reads them as something else
	// at a confidence that looks sure. One edit distance apart too many and the old gate cut them
	// into a track of their own, which reported the car a second time.
	c := NewConsensus(DefaultConsensus())
	for i := 0; i < 4; i++ {
		c.Observe(obs(t0.Add(time.Duration(i)*125*time.Millisecond), 300+float64(i)*150, table("IXT0G70", 0.97)))
	}
	tail := obs(t0.Add(4*125*time.Millisecond), 900, table("1ITG670", 0.89)) // four edits away
	c.Observe(tail)
	if c.Tracks() != 1 {
		t.Errorf("tracks = %d, want the tail to join the car it is the tail of", c.Tracks())
	}
}

// BenchmarkConsensusObserve is what a reading costs on the camera's own goroutine, with a track of
// forty readings: the candidates, their scores over every reading and the fused text. It has to be
// small beside the OCR pass it follows (milliseconds on a GPU).
func BenchmarkConsensusObserve(b *testing.B) {
	c := NewConsensus(DefaultConsensus())
	tables := [][]float32{
		table("ABC1D23", 0.85, alt{slot: 2, ch: 'G', p: 0.55}),
		table("ABC1D23", 0.85, alt{slot: 4, ch: 'O', p: 0.52}),
		table("ABC1D23", 0.9),
		table("ABD1D23", 0.8),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// A new track every forty readings, as a camera sees vehicles come and go.
		if i%40 == 0 {
			c.Flush(t0.Add(time.Hour * time.Duration(i+1)))
		}
		at := t0.Add(time.Hour*time.Duration(i/40) + time.Duration(i%40)*100*time.Millisecond)
		c.Observe(obs(at, 400+float64(i%40)*10, tables[i%len(tables)]))
	}
}
