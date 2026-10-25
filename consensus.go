package alpr

import (
	"math"
	"slices"
	"strings"
	"time"
)

// Consensus turns the many readings one vehicle produces into one plate.
//
// A car crossing a camera is read once per detection pass -- four, eight, fifteen times, each from
// a slightly different picture. Judging every one of them alone and deduplicating the survivors by
// edit distance reports the car several times with several spellings: the same plate read OJ33194,
// CLR3194 and 8JB5194 from three frames is three rows in the database, none of which is
// wrong enough to look like an error and one of which may be right. That is the failure this file
// removes, and it is not a threshold problem: no cut-off on a single frame's confidence separates
// the frame that read the plate correctly from the one that read it wrong at the same confidence.
//
// What separates them is the other frames. So readings are grouped into a track (the same plate in
// the same place, frame after frame), and the track's decision is taken from all of them together:
//
//   - each frame contributes the OCR head's whole distribution, not its argmax. A frame that read
//     "1" with 55% and "I" with 45% is evidence for both, and a text vote counts it as a full vote
//     for one of them;
//   - frames are weighted by how much they can be trusted: the detector's score, the size and the
//     sharpness of the plate, and the OCR's own confidence (to the fourth power, because a 0.9
//     frame is worth far more than three 0.5 frames of the same car);
//   - the candidates are the texts the frames read, plus the slot-by-slot best of all of them, and
//     the winner is the one with the most evidence. It is never a text nobody could have read.
//
// There is no plate-format check anywhere in here, as nowhere in this engine: a track that reads
// as ABC1D23 on every frame wins because the frames said so, not because Brazil's plates look that
// way.
//
// A Consensus is not safe for concurrent use. It belongs to one camera's goroutine, like the
// deduper it sits in front of.
type Consensus struct {
	cfg    ConsensusConfig
	tracks []*track
	nextID uint64
}

// ConsensusConfig tunes the track and the decision. The zero value is not valid; use
// DefaultConsensus and change what a deployment needs.
type ConsensusConfig struct {
	// Gap is how long a track may go without a reading before the vehicle counts as gone. A track
	// that has not decided by then decides with what it has.
	Gap time.Duration

	// MinObservations and Threshold decide a track EARLY, with the vehicle still in view: that many
	// readings, and a posterior probability for the winning text at or above Threshold (0..1).
	// Deciding early is what keeps a plate seen by a slow camera from arriving after the car has
	// left; the track keeps absorbing readings afterwards and never reports the same car twice.
	MinObservations int
	Threshold       float64

	// MinEvidence is the lowest single-reading Confidence that counts as evidence at all. Below it
	// the frame is a plate-shaped smear, and following it only opens a track that will be dropped.
	MinEvidence float64

	// MinConfidence is the floor on the verdict's Confidence, the weighted mean of the winning
	// text's per-character probabilities -- the same quantity GateConfig.MinOCRConfidence judges a
	// single reading by, so one number means one thing. A track that never reaches it is dropped
	// without a word, which is how an object that is not a plate stays out of the database.
	MinConfidence float64

	// EndPosterior is the lowest posterior a track that did NOT reach Threshold may end with and
	// still be reported. Two frames that disagree evenly are not a reading.
	EndPosterior float64

	// MinAgreement is the share of the winning text's characters, averaged over the readings and
	// weighted like them, that the readings themselves put in the same slot. It is what tells a
	// plate that is read a little differently each time from a smear that is read as something
	// else each time: the second can have a middling confidence on every frame and still agree
	// with itself nowhere.
	MinAgreement float64

	// SingleConfidence is the Confidence a track of ONE reading must reach. A plate seen once, by
	// one frame, has nothing to be checked against, so it is held to a higher bar than a track.
	SingleConfidence float64

	// MinScore is the detector score the best reading of a track must reach. A box the detector is
	// only half sure of (a bumper, a sticker, a mounting bracket) can read as something confident,
	// frame after frame, and agree with itself the whole time: what it lacks is a plate under it.
	MinScore float64

	// MatchDistance is how far, in plate widths, a reading's centre may be from the last one of its
	// track and still belong to it. MatchTextDistance is the edit distance its text may be from the
	// track's last confident text; zero turns the text check off.
	MatchDistance     float64
	MatchTextDistance int

	// Floor is the lowest probability a character is scored at, so one frame that gave the right
	// character a near-zero probability cannot veto it alone.
	Floor float64

	// MaxObservations bounds the readings a track keeps (the weakest are dropped first), and
	// MaxTracks the tracks a camera follows at once.
	MaxObservations int
	MaxTracks       int

	// ReferenceWidth is the plate width, as a fraction of the frame's width, from which a frame is
	// not penalised for the plate being small; ReferenceSharpness the Laplacian variance from which
	// it is not penalised for being soft. Both fall off quadratically below: a plate at half the
	// reference width is a quarter as good a witness, which is about what was measured -- on a real
	// camera the readings of plates under 4.5% of the frame were right 0 times in 70, between 4.5%
	// and 6% 12 times in 100, and from 6% up 53 in 100.
	ReferenceWidth     float64
	ReferenceSharpness float64
}

// DefaultConsensus is the configuration the engine ships with.
func DefaultConsensus() ConsensusConfig {
	return ConsensusConfig{
		Gap:                1500 * time.Millisecond,
		MinObservations:    4,
		Threshold:          0.90,
		MinEvidence:        0.30,
		MinConfidence:      0.60,
		MinAgreement:       0.50,
		EndPosterior:       0.50,
		SingleConfidence:   0.85,
		MinScore:           0.50,
		MatchDistance:      4,
		MatchTextDistance:  3,
		Floor:              0.02,
		MaxObservations:    48,
		MaxTracks:          16,
		ReferenceWidth:     0.065,
		ReferenceSharpness: 100,
	}
}

func (c ConsensusConfig) withDefaults() ConsensusConfig {
	d := DefaultConsensus()
	if c.Gap <= 0 {
		c.Gap = d.Gap
	}
	if c.MinObservations <= 0 {
		c.MinObservations = d.MinObservations
	}
	if c.Threshold <= 0 {
		c.Threshold = d.Threshold
	}
	if c.MinEvidence <= 0 {
		c.MinEvidence = d.MinEvidence
	}
	if c.MinConfidence <= 0 {
		c.MinConfidence = d.MinConfidence
	}
	if c.MinAgreement <= 0 {
		c.MinAgreement = d.MinAgreement
	}
	if c.EndPosterior <= 0 {
		c.EndPosterior = d.EndPosterior
	}
	if c.SingleConfidence <= 0 {
		c.SingleConfidence = d.SingleConfidence
	}
	if c.MinScore <= 0 {
		c.MinScore = d.MinScore
	}
	if c.MatchDistance <= 0 {
		c.MatchDistance = d.MatchDistance
	}
	if c.MatchTextDistance < 0 {
		c.MatchTextDistance = 0
	}
	if c.Floor <= 0 {
		c.Floor = d.Floor
	}
	if c.MaxObservations <= 0 {
		c.MaxObservations = d.MaxObservations
	}
	if c.MaxTracks <= 0 {
		c.MaxTracks = d.MaxTracks
	}
	if c.ReferenceWidth <= 0 {
		c.ReferenceWidth = d.ReferenceWidth
	}
	if c.ReferenceSharpness <= 0 {
		c.ReferenceSharpness = d.ReferenceSharpness
	}
	return c
}

// NewConsensus builds a Consensus. Zero fields of cfg take the defaults.
func NewConsensus(cfg ConsensusConfig) *Consensus {
	return &Consensus{cfg: cfg.withDefaults()}
}

// Observation is one reading of one plate in one frame.
type Observation struct {
	// At is when the reading was made, on the clock that decides how long a track may stay silent;
	// CapturedAt is when its frame left the camera.
	At         time.Time
	CapturedAt time.Time
	Box        BBox

	// DetectorScore is the detector's confidence in the box, and Sharpness the Laplacian variance
	// of its pixels (zero when unknown, which costs the frame nothing).
	DetectorScore float64
	Sharpness     float64

	// FrameWidth is the width in pixels of the picture Box is in. It makes the plate's size a
	// fraction of the frame, which means the same on a camera scaled to 1080p and one left at 1440p;
	// zero compares Box.Width with ReferenceWidth as if that were pixels.
	FrameWidth int

	Plate         string
	Confidence    float64
	Probabilities []float32 // MaxPlateSlots * len(Alphabet); a reading without it is ignored

	// Region and RegionConfidence are what the OCR's region head said about this frame, carried
	// through to the Verdict untouched: the track's best frame speaks for it.
	Region           string
	RegionConfidence float64
}

// Verdict is a track's decision.
type Verdict struct {
	Track uint64

	Plate string
	// CharConfidences is, per character of Plate, the weighted mean probability the frames gave
	// it, and Confidence their mean: the same meaning as a single reading's.
	CharConfidences []float64
	Confidence      float64
	// Posterior is the winning text's share of the evidence among the candidates (0..1), and
	// Agreement the weighted share of its characters the readings put in the same slot.
	Posterior float64
	Agreement float64

	Observations int
	// Early is true when the track decided with the vehicle still in view.
	Early bool

	// Best is the observation the frame to keep belongs to: the one the evidence images come from.
	Best Observation
	// First is when the track's first reading arrived.
	First time.Time
}

// ObserveResult is what Observe says about a reading.
type ObserveResult struct {
	Track uint64
	// Keep is true when this reading is now the best of a track that has not decided yet: the
	// caller should hold on to its frame, because the Verdict will point at it.
	Keep bool
	// Verdict is non-nil when this reading made the track decide.
	Verdict *Verdict
}

type slotKey [MaxPlateSlots]uint8

// frameEvidence is one reading as the track remembers it.
type frameEvidence struct {
	logp   []float32 // log of the floored probabilities, MaxPlateSlots*len(Alphabet)
	key    slotKey   // what the frame reads on its own: the argmax of each slot
	weight float64
}

type track struct {
	id      uint64
	box     BBox
	first   time.Time
	last    time.Time
	decided bool

	obs []frameEvidence

	// best is the reading whose frame the caller was told to keep, bestText what it read and bestW
	// its weight. It follows the leading text: a picture of the plate that the verdict reports,
	// not of a misreading of it.
	best     Observation
	bestText string
	bestW    float64
	bestSeen bool

	// bestScore is the highest detector score any of its readings had.
	bestScore float64

	// vx, vy are the plate's velocity in pixels a second, from its last two readings, and known
	// says there were two to measure it from. stable is the text the track is sure of, when it is.
	vx, vy float64
	known  bool
	stable string

	// probSum[slot*A+k] is sum(weight * p) and wSum the sum of the weights, for the per-character
	// confidence of the winner.
	probSum []float64
	wSum    float64
	count   int
}

const slotCount = MaxPlateSlots

const alphabetLen = len(Alphabet)

// Observe adds a reading and says whether it is the best so far and whether it decided its track.
func (c *Consensus) Observe(o Observation) ObserveResult {
	if len(o.Probabilities) != slotCount*alphabetLen || o.Plate == "" || o.Confidence < c.cfg.MinEvidence {
		return ObserveResult{}
	}
	t := c.match(o)
	if t == nil {
		t = c.open(o)
	}
	w := c.weight(o)
	t.add(o, w, c.cfg)
	t.bestScore = math.Max(t.bestScore, o.DetectorScore)
	t.follow(o)

	res := ObserveResult{Track: t.id}
	if t.decided {
		return res
	}

	lead, ok := c.lead(t)
	leadText := o.Plate
	if ok {
		leadText = lead.plate
		// Sure enough of its text to tell a different plate from a badly read one.
		t.stable = ""
		if t.count >= c.cfg.MinObservations && lead.posterior >= c.cfg.Threshold && lead.agreement >= 0.7 {
			t.stable = lead.plate
		}
	}
	switch reads := o.Plate == leadText; {
	case !t.bestSeen,
		reads && (t.bestText != leadText || w > t.bestW),
		!reads && t.bestText != leadText && w > t.bestW:
		t.best, t.bestText, t.bestW, t.bestSeen = o, o.Plate, w, true
		res.Keep = true
	}

	if ok && t.count >= c.cfg.MinObservations && lead.posterior >= c.cfg.Threshold {
		if v, pass := c.verdict(t, lead, true); pass {
			t.decided = true
			res.Verdict = &v
		}
	}
	return res
}

// confidentReading is the single-reading Confidence from which a reading is trusted to say that a
// track is NOT the plate it is following. It is high on purpose: the OCR reads a car's last, small
// frames as something else at 0.85 and 0.89, and a gate that believed those would cut the tail of
// every vehicle into a track of its own and report it again.
const confidentReading = 0.95

// Flush ends the tracks whose vehicle is gone and returns what they decided, plus the ids of every
// track it closed, deciding or not, so the caller can let go of the frames it kept for them.
func (c *Consensus) Flush(now time.Time) (verdicts []Verdict, closed []uint64) {
	live := c.tracks[:0]
	for _, t := range c.tracks {
		if now.Sub(t.last) <= c.cfg.Gap {
			live = append(live, t)
			continue
		}
		closed = append(closed, t.id)
		if t.decided {
			continue
		}
		if lead, ok := c.lead(t); ok {
			if v, pass := c.verdict(t, lead, false); pass {
				verdicts = append(verdicts, v)
			}
		}
	}
	for i := len(live); i < len(c.tracks); i++ {
		c.tracks[i] = nil
	}
	c.tracks = live
	return verdicts, closed
}

// Tracks reports how many vehicles are being followed.
func (c *Consensus) Tracks() int { return len(c.tracks) }

func (c *Consensus) open(o Observation) *track {
	if len(c.tracks) >= c.cfg.MaxTracks {
		// The oldest goes. A camera pointed at a car park can hold more plates in view than any
		// bound, and the one that has been silent longest is the one least likely to come back.
		oldest := 0
		for i, t := range c.tracks {
			if t.last.Before(c.tracks[oldest].last) {
				oldest = i
			}
		}
		c.tracks = slices.Delete(c.tracks, oldest, oldest+1)
	}
	c.nextID++
	t := &track{
		id: c.nextID, box: o.Box, first: o.At, last: o.At,
		probSum: make([]float64, slotCount*alphabetLen),
	}
	c.tracks = append(c.tracks, t)
	return t
}

// match finds the track a reading continues, or nil.
//
// Where the plate should be, then how big, then what it says. A vehicle moves, so the plate is
// looked for where the track's last two readings say it is going, not where it last was: at eight
// frames a second a plate crosses most of its own width between two of them, and the plate of a
// second car in the other lane can be nearer to the last position than the car it belongs to --
// which is how a near car's tail was once taken over by a far car's plate, and the near car then
// reported twice. A plate is also about as wide as it was; one that is half the size is another
// car. And the text only counts when the track is sure of its own: the edit distance is there to
// tell a different plate that happens to be in the same place (a car replacing another at a stop
// line) from the same plate read badly, and a track still unsure of its plate has nothing to
// compare with.
//
// Two readings of one frame are two plates, so a track that already took one this frame is passed
// over.
func (c *Consensus) match(o Observation) *track {
	var best *track
	bestDist := math.MaxFloat64
	cx, cy := o.Box.X+o.Box.Width/2, o.Box.Y+o.Box.Height/2
	for _, t := range c.tracks {
		dt := o.At.Sub(t.last)
		if dt > c.cfg.Gap || dt <= 0 {
			continue
		}
		if r := o.Box.Width / math.Max(t.box.Width, 1); r < 1/maxPlateSizeRatio || r > maxPlateSizeRatio {
			continue
		}
		tx, ty := t.box.X+t.box.Width/2, t.box.Y+t.box.Height/2
		radius := c.cfg.MatchDistance
		if t.known {
			sec := dt.Seconds()
			tx, ty = tx+t.vx*sec, ty+t.vy*sec
			// Once the motion is known the plate is where it was predicted, to within what a
			// vehicle's speed changes in the time since.
			radius = math.Min(radius, 1.5+sec)
		}
		scale := math.Max(math.Max(o.Box.Width, t.box.Width), 1)
		dist := math.Hypot(cx-tx, cy-ty) / scale
		if dist > radius && iou(o.Box, t.box) <= 0.1 {
			continue
		}
		if t.stable != "" && c.cfg.MatchTextDistance > 0 && o.Confidence >= confidentReading &&
			!levenshteinWithin(normalisePlate(o.Plate), normalisePlate(t.stable), c.cfg.MatchTextDistance) {
			continue
		}
		if dist < bestDist {
			best, bestDist = t, dist
		}
	}
	return best
}

// maxPlateSizeRatio is how much wider or narrower than the track's last plate a reading may be and
// still be that plate: a vehicle changes its plate's size by a few percent a frame.
const maxPlateSizeRatio = 1.6

// follow moves the track to where the reading is, and works out how fast it is going.
func (t *track) follow(o Observation) {
	dt := o.At.Sub(t.last).Seconds()
	if dt > 0.02 {
		cx, cy := o.Box.X+o.Box.Width/2, o.Box.Y+o.Box.Height/2
		px, py := t.box.X+t.box.Width/2, t.box.Y+t.box.Height/2
		vx, vy := (cx-px)/dt, (cy-py)/dt
		if t.known {
			// A running mean: one frame's box wobbles by a few pixels, and ten times a second that
			// is a velocity that is not there.
			vx, vy = (t.vx+vx)/2, (t.vy+vy)/2
		}
		t.vx, t.vy, t.known = vx, vy, true
	}
	t.box, t.last = o.Box, o.At
}

// weight is how much one frame counts, in (0, 1].
func (c *Consensus) weight(o Observation) float64 {
	width := o.Box.Width
	if o.FrameWidth > 0 {
		width /= float64(o.FrameWidth)
	}
	size := math.Min(1, width/c.cfg.ReferenceWidth)
	sharp := 1.0
	if o.Sharpness > 0 {
		sharp = math.Min(1, o.Sharpness/c.cfg.ReferenceSharpness)
	}
	conf := math.Min(math.Max(o.Confidence, 0), 1)
	score := math.Min(math.Max(o.DetectorScore, 0.05), 1)
	w := score * size * size * (0.5 + 0.5*sharp) * math.Pow(conf, 4)
	return math.Max(w, 1e-6)
}

func (t *track) add(o Observation, w float64, cfg ConsensusConfig) {
	logp := make([]float32, len(o.Probabilities))
	floor := float32(cfg.Floor)
	var key slotKey
	for slot := 0; slot < slotCount; slot++ {
		bi, bp := 0, float32(-1)
		for k := 0; k < alphabetLen; k++ {
			i := slot*alphabetLen + k
			p := o.Probabilities[i]
			if p > bp {
				bi, bp = k, p
			}
			if p < floor {
				p = floor
			}
			logp[i] = float32(math.Log(float64(p)))
			t.probSum[i] += w * float64(o.Probabilities[i])
		}
		key[slot] = uint8(bi)
	}
	t.wSum += w
	t.count++
	t.obs = append(t.obs, frameEvidence{logp: logp, key: key, weight: w})
	if len(t.obs) > cfg.MaxObservations {
		// Drop the weakest. The sums above keep their contribution, which is deliberate: they
		// only feed the per-character confidences, where a faint frame is a faint frame.
		weakest := 0
		for i, e := range t.obs {
			if e.weight < t.obs[weakest].weight {
				weakest = i
			}
		}
		t.obs = slices.Delete(t.obs, weakest, weakest+1)
	}
}

// leader is the text a track currently favours and how well the evidence supports it.
type leader struct {
	key       slotKey
	plate     string
	confs     []float64
	conf      float64
	posterior float64
	agreement float64
}

// lead scores the candidates and returns the favourite.
func (c *Consensus) lead(t *track) (leader, bool) {
	if len(t.obs) == 0 {
		return leader{}, false
	}
	keys := t.candidates()
	scores := make([]float64, len(keys))
	for j, key := range keys {
		for _, e := range t.obs {
			var s float64
			for slot, k := range key {
				s += float64(e.logp[slot*alphabetLen+int(k)])
			}
			scores[j] += e.weight * s
		}
	}
	bestJ := 0
	for j := range scores {
		if scores[j] > scores[bestJ] {
			bestJ = j
		}
	}
	// Softmax over the candidates, shifted by the maximum so the exponentials cannot overflow.
	var z float64
	for _, s := range scores {
		z += math.Exp(s - scores[bestJ])
	}

	key := keys[bestJ]
	plate, confs := t.read(key)
	if plate == "" {
		return leader{}, false
	}
	var sum float64
	for _, v := range confs {
		sum += v
	}
	return leader{
		key: key, plate: plate, confs: confs, conf: sum / float64(len(confs)),
		posterior: 1 / z, agreement: t.agreement(key),
	}, true
}

// agreement is the weighted share of the winner's characters that each reading puts in the same
// slot, averaged over the readings.
func (t *track) agreement(key slotKey) float64 {
	var chars int
	for _, k := range key {
		if Alphabet[k] != PadChar {
			chars++
		}
	}
	if chars == 0 {
		return 0
	}
	var sum, wsum float64
	for _, e := range t.obs {
		match := 0
		for slot, k := range key {
			if Alphabet[k] != PadChar && e.key[slot] == k {
				match++
			}
		}
		sum += e.weight * float64(match) / float64(chars)
		wsum += e.weight
	}
	if wsum == 0 {
		return 0
	}
	return sum / wsum
}

// verdict applies the bars a decision has to clear.
func (c *Consensus) verdict(t *track, l leader, early bool) (Verdict, bool) {
	floor := c.cfg.MinConfidence
	if t.count == 1 {
		floor = math.Max(floor, c.cfg.SingleConfidence)
	}
	if t.best.DetectorScore < c.cfg.MinScore {
		return Verdict{}, false
	}
	if l.conf < floor || l.agreement < c.cfg.MinAgreement {
		return Verdict{}, false
	}
	if !early && l.posterior < c.cfg.EndPosterior {
		return Verdict{}, false
	}
	return Verdict{
		Track: t.id, Plate: l.plate, CharConfidences: l.confs, Confidence: l.conf,
		Posterior: l.posterior, Agreement: l.agreement,
		Observations: t.count, Early: early, Best: t.best, First: t.first,
	}, true
}

// candidates are the slot-by-slot readings the track can defend: what each frame read, and what
// all of them together read at each slot.
func (t *track) candidates() []slotKey {
	var keys []slotKey
	seen := map[slotKey]bool{}
	add := func(k slotKey) {
		if !seen[k] {
			seen[k] = true
			keys = append(keys, k)
		}
	}
	for _, e := range t.obs {
		add(e.key)
	}
	var fused slotKey
	for slot := 0; slot < slotCount; slot++ {
		bestScore := math.Inf(-1)
		for ch := 0; ch < alphabetLen; ch++ {
			var s float64
			for _, e := range t.obs {
				s += e.weight * float64(e.logp[slot*alphabetLen+ch])
			}
			if s > bestScore {
				bestScore, fused[slot] = s, uint8(ch)
			}
		}
	}
	add(fused)
	// Keep the work bounded: a track with dozens of distinct texts is a track of noise, and the
	// fused text is the last one added.
	if len(keys) > 12 {
		keys = keys[len(keys)-12:]
	}
	return keys
}

// read renders a key as text with the weighted mean probability each frame gave its characters.
func (t *track) read(key slotKey) (string, []float64) {
	var b strings.Builder
	var confs []float64
	for slot, k := range key {
		ch := Alphabet[k]
		if ch == PadChar {
			continue
		}
		b.WriteByte(ch)
		p := 0.0
		if t.wSum > 0 {
			p = t.probSum[slot*alphabetLen+int(k)] / t.wSum
		}
		confs = append(confs, p)
	}
	return b.String(), confs
}

func iou(a, b BBox) float64 {
	ix := math.Max(0, math.Min(a.X2(), b.X2())-math.Max(a.X, b.X))
	iy := math.Max(0, math.Min(a.Y2(), b.Y2())-math.Max(a.Y, b.Y))
	inter := ix * iy
	union := a.Width*a.Height + b.Width*b.Height - inter
	if union <= 0 {
		return 0
	}
	return inter / union
}
