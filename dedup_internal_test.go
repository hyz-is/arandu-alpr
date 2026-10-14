package alpr

import (
	"fmt"
	"testing"
	"time"
)

var t0 = time.Date(2026, 8, 26, 12, 0, 0, 0, time.UTC)

func TestSameVehicleIsReportedOnce(t *testing.T) {
	// A car standing in front of a camera is read every frame. Without this, one parked vehicle
	// becomes thousands of detections, thousands of webhook deliveries, and a bill to match.
	d := NewDeduper(30*time.Second, 2)
	if !d.Accept("ABC1D23", t0) {
		t.Fatal("first reading suppressed")
	}
	if d.Accept("ABC1D23", t0.Add(time.Second)) {
		t.Error("same plate emitted twice inside the cooldown")
	}
	if !d.Accept("ABC1D23", t0.Add(31*time.Second)) {
		t.Error("plate not emitted again after the cooldown")
	}
}

func TestOcrDriftIsTreatedAsTheSamePlate(t *testing.T) {
	// The OCR is not stable across frames: the same plate arrives as ABC1D23 and A8C1D23 as the
	// vehicle moves through focus. Exact matching would let every misread through as new.
	d := NewDeduper(30*time.Second, 2)
	d.Accept("ABC1D23", t0)
	for _, drift := range []string{"A8C1D23", "ABC1D28", "ABCID23"} {
		if d.Accept(drift, t0.Add(time.Second)) {
			t.Errorf("%q emitted as a new plate; it is one edit from the original", drift)
		}
	}
}

func TestDifferentPlatesBothGetThrough(t *testing.T) {
	// The distance budget must not merge genuinely different vehicles.
	d := NewDeduper(30*time.Second, 2)
	if !d.Accept("ABC1D23", t0) {
		t.Fatal("first plate suppressed")
	}
	if !d.Accept("XYZ9K88", t0.Add(time.Second)) {
		t.Error("an unrelated plate was suppressed")
	}
}

func TestSeparatorsAndCaseDoNotCreateNewPlates(t *testing.T) {
	d := NewDeduper(30*time.Second, 2)
	d.Accept("ABC-1D23", t0)
	for _, same := range []string{"abc1d23", "ABC 1D23", "ABC1D23"} {
		if d.Accept(same, t0.Add(time.Second)) {
			t.Errorf("%q treated as a different plate", same)
		}
	}
}

func TestForeignPlatesDedupWithoutNationalRules(t *testing.T) {
	// No country-specific normalisation runs here. An earlier system in this family expanded each
	// reading into national format variants before comparing; edit distance already covers it,
	// and for a worldwide product the variants are meaningless.
	d := NewDeduper(30*time.Second, 2)
	if !d.Accept("M-AB 1234", t0) {
		t.Fatal("German plate suppressed on first sight")
	}
	if d.Accept("MAB1234", t0.Add(time.Second)) {
		t.Error("same plate written without separators emitted again")
	}
	if !d.Accept("京A12345", t0.Add(2*time.Second)) {
		t.Error("non-ASCII plate suppressed")
	}
}

func TestSuppressedReadingDoesNotExtendTheWindow(t *testing.T) {
	// If a rejected reading refreshed the timer, a vehicle parked in frame would hold its own
	// cooldown open forever and never be reported again: the window would restart on every
	// suppressed frame instead of on the last emission.
	d := NewDeduper(10*time.Second, 2)
	d.Accept("ABC1D23", t0)
	for i := 1; i <= 9; i++ {
		d.Accept("ABC1D23", t0.Add(time.Duration(i)*time.Second))
	}
	if !d.Accept("ABC1D23", t0.Add(11*time.Second)) {
		t.Error("suppressed readings pushed the cooldown forward")
	}
}

func TestHistoryIsPrunedNotAccumulated(t *testing.T) {
	// The Python service appends to its history and never removes, so a camera running for a week
	// carries every plate it ever saw: memory grows without bound and the distance scan -- which
	// walks every entry -- slows down as the day goes on.
	d := NewDeduper(30*time.Second, 2)
	for i := 0; i < 500; i++ {
		d.Accept(fmt.Sprintf("PLATE%03d", i), t0)
	}
	if d.Len() == 0 {
		t.Fatal("nothing remembered")
	}
	// One reading well past the window must clear everything that expired.
	d.Accept("ZZZ0000", t0.Add(time.Hour))
	if got := d.Len(); got != 1 {
		t.Errorf("history holds %d entries after the window passed, want 1", got)
	}
}

func TestEmptyPlateIsNeverEmitted(t *testing.T) {
	d := NewDeduper(30*time.Second, 2)
	if d.Accept("", t0) {
		t.Error("empty reading emitted")
	}
	if d.Accept("---", t0) {
		t.Error("a reading of only separators emitted")
	}
}

func TestDistanceZeroRequiresExactMatch(t *testing.T) {
	// maxDistance is configuration; at 0 the filter must degrade to equality, not to "everything
	// matches" or "nothing matches".
	d := NewDeduper(30*time.Second, 0)
	d.Accept("ABC1D23", t0)
	if d.Accept("ABC1D23", t0.Add(time.Second)) {
		t.Error("identical plate emitted twice at distance 0")
	}
	if !d.Accept("ABC1D24", t0.Add(time.Second)) {
		t.Error("one-edit plate suppressed at distance 0")
	}
}

func TestLevenshteinWithinMatchesKnownDistances(t *testing.T) {
	cases := []struct {
		a, b string
		max  int
		want bool
	}{
		{"ABC1D23", "ABC1D23", 0, true},
		{"ABC1D23", "ABC1D24", 0, false},
		{"ABC1D23", "ABC1D24", 1, true},
		{"ABC1D23", "ABC1D45", 1, false},
		{"ABC1D23", "ABC1D45", 2, true},
		{"ABC1D23", "ABC1D2", 1, true},   // deletion
		{"ABC1D23", "ABC1D233", 1, true}, // insertion
		{"ABC1D23", "XYZ9K88", 2, false},
		{"", "AB", 2, true},
		{"", "ABC", 2, false},
	}
	for _, c := range cases {
		if got := levenshteinWithin(c.a, c.b, c.max); got != c.want {
			t.Errorf("levenshteinWithin(%q,%q,%d) = %v, want %v", c.a, c.b, c.max, got, c.want)
		}
	}
}
