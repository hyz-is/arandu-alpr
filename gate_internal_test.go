package alpr

import (
	"math"
	"testing"
)

func TestForeignPlatesAreNotRejected(t *testing.T) {
	// The check this guards against is `_is_valid_brazilian_plate`, which the Python service runs
	// as the last step of its gate. It accepts AAA9999 and AAA9A99 and drops everything else.
	//
	// This product reads plates worldwide. If that check ever reappears, every correct foreign
	// reading is discarded with no log and no metric -- the plates simply never arrive, and the
	// symptom is "the engine does not work abroad" with nothing pointing at the cause.
	g := DefaultGate()
	for _, plate := range []string{
		"ABC1D23",   // Brazil, Mercosul
		"ABC1234",   // Brazil, old
		"AB12CDE",   // United Kingdom
		"M-AB 1234", // Germany
		"1-ABC-234", // Belgium
		"7ABC123",   // California
		"京A12345",   // China
		"ABC 123",   // spaces
	} {
		if !g.AcceptReading(plate, 0.9) {
			t.Errorf("plate %q rejected; the gate must not judge plate format", plate)
		}
	}
}

func TestReadingNeedsTextAndConfidence(t *testing.T) {
	g := DefaultGate()
	if g.AcceptReading("", 0.99) {
		t.Error("empty plate accepted")
	}
	if g.AcceptReading("ABC1D23", 0.59) {
		t.Error("reading below MinOCRConfidence accepted")
	}
	if !g.AcceptReading("ABC1D23", 0.60) {
		t.Error("reading exactly at the threshold rejected")
	}
}

func TestNaNScoreIsRejected(t *testing.T) {
	// `score < min` would let NaN through: every ordered comparison with NaN is false. A
	// degenerate model would then push a garbage crop into the OCR, and the reading that came
	// back would look like any other.
	g := DefaultGate()
	b := BBox{X: 100, Y: 100, Width: 200, Height: 60}
	if g.AcceptDetection(math.NaN(), b, 1920, 1080) {
		t.Error("NaN score accepted")
	}
}

func TestEdgeFilterDropsCroppedPlates(t *testing.T) {
	// A plate half outside the frame does not produce a missed read -- it produces a confident
	// wrong one, because the model invents the characters it cannot see.
	g := DefaultGate()
	const w, h = 1920, 1080

	touchingLeft := BBox{X: 5, Y: 500, Width: 200, Height: 60} // margin is 1% = 19.2px
	if g.AcceptDetection(0.9, touchingLeft, w, h) {
		t.Error("plate touching the left border accepted")
	}

	touchingBottom := BBox{X: 800, Y: 1040, Width: 200, Height: 60}
	if g.AcceptDetection(0.9, touchingBottom, w, h) {
		t.Error("plate crossing the bottom border accepted")
	}

	middle := BBox{X: 800, Y: 500, Width: 200, Height: 60}
	if !g.AcceptDetection(0.9, middle, w, h) {
		t.Error("plate well inside the frame rejected")
	}
}

func TestBlurThresholdScalesWithPlateSize(t *testing.T) {
	// Without normalisation a small distant plate is judged against the sharpness only a close
	// one can reach, and every far plate is dropped. This is what BlurNormalizeByArea buys.
	g := DefaultGate()

	small := BBox{Width: 100, Height: 30} // 3000 px^2, well under the 27000 reference
	// 27 is below the flat threshold of 80, but above the scaled one for this size.
	if !g.AcceptFocus(27, small) {
		t.Error("small plate rejected by a threshold calibrated for a large one")
	}

	large := BBox{Width: 300, Height: 90} // exactly the reference area
	if g.AcceptFocus(27, large) {
		t.Error("blurred large plate accepted")
	}
	if !g.AcceptFocus(80, large) {
		t.Error("sharp large plate rejected at the exact threshold")
	}
}

func TestBlurSkipsTinyCrops(t *testing.T) {
	// Variance over a handful of pixels is noise. Judging on it would drop distant plates the
	// OCR can still read.
	g := DefaultGate()
	if !g.AcceptFocus(0, BBox{Width: 4, Height: 4}) {
		t.Error("tiny crop judged for focus")
	}
}

func TestDisabledFiltersAcceptEverything(t *testing.T) {
	g := DefaultGate()
	g.EdgeFilterEnabled = false
	g.BlurFilterEnabled = false

	atBorder := BBox{X: 0, Y: 0, Width: 200, Height: 60}
	if !g.AcceptDetection(0.9, atBorder, 1920, 1080) {
		t.Error("edge filter still applied when disabled")
	}
	if !g.AcceptFocus(0, BBox{Width: 300, Height: 90}) {
		t.Error("blur filter still applied when disabled")
	}
}

func TestMeanConfidenceIsMeanNotMin(t *testing.T) {
	// The Python service averages the per-character confidences. Switching to the minimum would
	// judge every reading by its worst character and, at the same threshold, accept far fewer
	// plates -- a silent re-tune of the gate disguised as a refactor.
	perChar := []float64{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.30}
	got := MeanConfidence(perChar)
	if math.Abs(got-0.9) > 1e-9 {
		t.Fatalf("MeanConfidence = %v, want 0.9", got)
	}

	g := DefaultGate()
	if !g.AcceptReading("ABC1D23", got) {
		t.Error("reading rejected; the mean is above the threshold even with one weak character")
	}
	if g.AcceptReading("ABC1D23", 0.30) {
		t.Error("the minimum would have rejected it -- confirming the two differ")
	}
}

func TestMeanConfidenceOfNothingIsZero(t *testing.T) {
	if MeanConfidence(nil) != 0 {
		t.Error("no characters read must not produce a confident reading")
	}
}
