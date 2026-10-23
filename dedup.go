package alpr

import (
	"strings"
	"sync"
	"time"
)

// Deduper suppresses repeat readings of the same vehicle inside a cooldown window.
//
// A car sitting in front of a camera is read once per frame. Without this, one parked vehicle
// produces thousands of detections, thousands of webhook deliveries and a bill to match.
//
// Matching is edit distance, not equality, because the OCR is not stable across frames: the same
// plate arrives as ABC1D23 and A8C1D23 as the vehicle moves through focus. Requiring an exact
// match would let every misread through as a "new" plate -- which is the opposite of what the
// filter is for.
//
// **The comparison is plain Levenshtein, with no country-specific normalisation.** An earlier
// system in this family expanded each reading into national format variants before comparing;
// that is meaningless for a product that reads plates worldwide, and it hid the fact that the
// distance check already covered the case.
type Deduper struct {
	cooldown    time.Duration
	maxDistance int
	// whilePresent makes a suppressed reading refresh the timer: see WhilePresent.
	whilePresent bool

	mu     sync.Mutex
	lastAt map[string]time.Time
}

// NewDeduper builds a deduper. maxDistance 2 is what the Python service runs with: it absorbs
// the two-character drift the OCR shows between frames without merging genuinely different
// plates, which in a seven-character plate would need three or more changes to collide.
func NewDeduper(cooldown time.Duration, maxDistance int) *Deduper {
	if maxDistance < 0 {
		maxDistance = 0
	}
	return &Deduper{cooldown: cooldown, maxDistance: maxDistance, lastAt: make(map[string]time.Time)}
}

// WhilePresent makes the window run from the last time the plate was SEEN rather than from the
// last time it was emitted: a car that stays in frame is reported once, and again only after it
// has been out of sight for the whole cooldown.
//
// Without it a car parked in front of a camera is reported every cooldown for as long as it
// stays -- a 30-second cooldown turned one parked car into 120 readings an hour, 17% of every
// reading the fleet made (Olho de Águia, 22/10/2026). That default stays for installers who want
// a periodic reminder that a vehicle is still there; this is the one a history of passages wants.
func (d *Deduper) WhilePresent() *Deduper {
	d.whilePresent = true
	return d
}

// Accept reports whether this reading should be emitted, and records it if so.
//
// By default a rejected reading does **not** refresh the timer, so a vehicle parked in frame is
// reported again once per cooldown. WhilePresent turns that into "once while it stays".
func (d *Deduper) Accept(plate string, now time.Time) bool {
	key := normalisePlate(plate)
	if key == "" {
		return false
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	// Prune first, so the distance scan below only walks live entries.
	//
	// The Python service skips this: it appends to the history and never removes, so a camera
	// running for a week carries every plate it ever saw. Memory grows without bound and the
	// scan -- which is O(entries) with a string comparison each -- slows down as the day goes on.
	cutoff := now.Add(-d.cooldown)
	for k, t := range d.lastAt {
		if !t.After(cutoff) {
			delete(d.lastAt, k)
		}
	}

	for known := range d.lastAt {
		if levenshteinWithin(key, known, d.maxDistance) {
			if d.whilePresent {
				d.lastAt[known] = now
			}
			return false
		}
	}

	d.lastAt[key] = now
	return true
}

// Len reports how many plates are being remembered. Exposed for tests and metrics.
func (d *Deduper) Len() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return len(d.lastAt)
}

// normalisePlate strips separators and case so the same plate written two ways compares equal.
//
// Only separators are removed. Letters and digits pass through untouched, including non-ASCII:
// a plate is whatever the OCR read, and deciding which characters are "valid" is a national rule
// this engine does not have.
func normalisePlate(plate string) string {
	var b strings.Builder
	b.Grow(len(plate))
	for _, r := range plate {
		switch r {
		case '-', ' ', '.', '\t':
			continue
		}
		b.WriteRune(r)
	}
	return strings.ToUpper(b.String())
}

// levenshteinWithin reports whether the edit distance between a and b is at most max.
//
// It bails out as soon as the answer is certain rather than computing the full distance: the
// caller only ever asks a yes/no question, and this runs per reading against every remembered
// plate.
func levenshteinWithin(a, b string, max int) bool {
	ra, rb := []rune(a), []rune(b)
	// Length alone can settle it, and does for most pairs.
	if diff := len(ra) - len(rb); diff > max || -diff > max {
		return false
	}
	if len(ra) == 0 {
		return len(rb) <= max
	}

	prev := make([]int, len(rb)+1)
	curr := make([]int, len(rb)+1)
	for j := range prev {
		prev[j] = j
	}

	for i := 1; i <= len(ra); i++ {
		curr[0] = i
		rowMin := curr[0]
		for j := 1; j <= len(rb); j++ {
			cost := 1
			if ra[i-1] == rb[j-1] {
				cost = 0
			}
			curr[j] = min3(curr[j-1]+1, prev[j]+1, prev[j-1]+cost)
			if curr[j] < rowMin {
				rowMin = curr[j]
			}
		}
		// Every remaining edit can only add: if the whole row already exceeds the budget, no
		// completion of this alignment can come back under it.
		if rowMin > max {
			return false
		}
		prev, curr = curr, prev
	}
	return prev[len(rb)] <= max
}

func min3(a, b, c int) int {
	if b < a {
		a = b
	}
	if c < a {
		a = c
	}
	return a
}
