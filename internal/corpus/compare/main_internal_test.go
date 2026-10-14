package main

import (
	"math"
	"testing"

	"github.com/hyz-is/arandu-alpr/internal/corpus"
)

// The comparator is the accuracy gate, so its own failure mode is the worst one available: a
// comparator that passes everything makes the corpus a folder of pictures. Each test below is a
// real divergence it must not let through.

func reading(plate string, box [4]float64, conf float64, region string) corpus.Reading {
	return corpus.Reading{Plate: plate, Confidence: conf, Region: region, Box: box, Score: 0.9}
}

func sets(g, p []corpus.Reading) (map[string]corpus.Result, map[string]corpus.Result) {
	return map[string]corpus.Result{"f1": {FrameID: "f1", Impl: "go", Readings: g, GatesApplied: false}},
		map[string]corpus.Result{"f1": {FrameID: "f1", Impl: "py", Readings: p}}
}

func TestIdenticalResultsAgree(t *testing.T) {
	box := [4]float64{100, 100, 200, 60}
	g, p := sets(
		[]corpus.Reading{reading("ABC1D23", box, 0.97, "Brazil")},
		[]corpus.Reading{reading("ABC1D23", box, 0.97, "Brazil")},
	)
	r := compare(g, p, true)
	if r.plateExact != 1 || r.plateTotal != 1 {
		t.Fatalf("exact %d/%d", r.plateExact, r.plateTotal)
	}
	if fail := r.verdict(0.9, 0.95); fail != "" {
		t.Errorf("identical results failed: %s", fail)
	}
}

func TestSystematicCharacterConfusionIsReported(t *testing.T) {
	// This is the alphabet indexed at an offset: every plate still looks like a plate, and the
	// only signal is that the same substitution repeats. A comparator that reported "89% exact"
	// and nothing else would leave that undiagnosed.
	box := [4]float64{10, 10, 100, 30}
	var gs, ps []corpus.Reading
	for i := 0; i < 5; i++ {
		gs = append(gs, reading("ABC1D23", box, 0.9, "Brazil"))
		ps = append(ps, reading("ABC1O23", box, 0.9, "Brazil"))
	}
	g, p := sets(gs, ps)
	r := compare(g, p, true)

	if r.confusions["D->O"] == 0 {
		t.Fatalf("the repeated D-for-O substitution was not recorded: %v", r.confusions)
	}
	if r.plateExact != 0 {
		t.Errorf("plates differ but %d were counted exact", r.plateExact)
	}
	if fail := r.verdict(0.9, 0.95); fail == "" {
		t.Error("a total plate disagreement passed the gate")
	}
}

func TestRegionDisagreementFailsEvenWhenPlatesMatch(t *testing.T) {
	// The region table is positional: its index IS the class the model emits. Reordering it
	// relabels every reading and changes no plate, so plate agreement cannot be the only gate.
	box := [4]float64{10, 10, 100, 30}
	g, p := sets(
		[]corpus.Reading{reading("ABC1D23", box, 0.9, "Argentina")},
		[]corpus.Reading{reading("ABC1D23", box, 0.9, "Brazil")},
	)
	r := compare(g, p, true)
	if len(r.regionWrong) != 1 {
		t.Fatalf("region disagreement not recorded: %v", r.regionWrong)
	}
	if fail := r.verdict(0.0, 0.0); fail == "" {
		t.Error("a region disagreement passed while the plates matched")
	}
}

func TestConfidenceDeltaBeyondRoundingIsFlagged(t *testing.T) {
	// Same weights, same input. A delta this size is a preprocessing difference -- padding
	// colour, interpolation, channel order, input scale -- not arithmetic.
	box := [4]float64{10, 10, 100, 30}
	g, p := sets(
		[]corpus.Reading{reading("ABC1D23", box, 0.9000, "Brazil")},
		[]corpus.Reading{reading("ABC1D23", box, 0.9400, "Brazil")},
	)
	r := compare(g, p, true)
	if len(r.confDeltas) != 1 {
		t.Fatalf("a %.2f confidence gap was not flagged", 0.04)
	}
	if math.Abs(r.worstConf-0.04) > 1e-9 {
		t.Errorf("worst delta = %v, want 0.04", r.worstConf)
	}
}

func TestRoundingDifferenceIsNotFlagged(t *testing.T) {
	// The other half of the same rule: two resize implementations WILL differ in the last bits,
	// and a comparator that called that a bug would cry wolf on every run.
	box := [4]float64{10, 10, 100, 30}
	g, p := sets(
		[]corpus.Reading{reading("ABC1D23", box, 0.900000, "Brazil")},
		[]corpus.Reading{reading("ABC1D23", box, 0.900001, "Brazil")},
	)
	if r := compare(g, p, true); len(r.confDeltas) != 0 {
		t.Errorf("a 1e-6 difference was reported as a bug: %v", r.confDeltas)
	}
}

func TestShiftedBoxDoesNotMatch(t *testing.T) {
	// A box in letterbox coordinates instead of frame coordinates is plausible and points at
	// the wrong part of the picture. Low IoU is exactly how that surfaces.
	g, p := sets(
		[]corpus.Reading{reading("ABC1D23", [4]float64{100, 100, 200, 60}, 0.9, "Brazil")},
		[]corpus.Reading{reading("ABC1D23", [4]float64{400, 300, 200, 60}, 0.9, "Brazil")},
	)
	r := compare(g, p, true)
	if r.iouMatched != 0 {
		t.Errorf("boxes 300px apart were matched (IoU %v)", r.iouSum)
	}
	if r.onlyGo != 1 || r.onlyPy != 1 {
		t.Errorf("unmatched counts wrong: go %d, py %d", r.onlyGo, r.onlyPy)
	}
}

func TestGatesOffRejectsAGatedGoResult(t *testing.T) {
	// Model-against-model is only meaningful when the Go side carried everything the model
	// produced. Comparing a gated Go result to an ungated Python one measures the gates and
	// reports it as a model difference, which is the exact confusion the two modes exist for.
	g := map[string]corpus.Result{"f1": {FrameID: "f1", GatesApplied: true}}
	p := map[string]corpus.Result{"f1": {FrameID: "f1"}}
	r := compare(g, p, true)
	if len(r.mismatches) == 0 {
		t.Fatal("a gated Go result was accepted for a model-against-model comparison")
	}
}

func TestDifferentWeightsAreCalledOut(t *testing.T) {
	// A result set produced against other weights is not stale, it is meaningless. Saying so by
	// hash beats letting it surface as a mysterious accuracy drop weeks later.
	g := map[string]corpus.Result{"f1": {FrameID: "f1", DetectorSHA: "aaaa1111"}}
	p := map[string]corpus.Result{"f1": {FrameID: "f1", DetectorSHA: "bbbb2222"}}
	if r := compare(g, p, false); len(r.shaWarnings) == 0 {
		t.Fatal("mismatched detector weights were not reported")
	}
}

func TestMissingPythonResultIsNotSilentlyPassed(t *testing.T) {
	g := map[string]corpus.Result{"f1": {FrameID: "f1"}, "f2": {FrameID: "f2"}}
	p := map[string]corpus.Result{"f1": {FrameID: "f1"}}
	r := compare(g, p, false)
	if r.frames != 1 {
		t.Errorf("compared %d frames, want 1", r.frames)
	}
	if len(r.mismatches) == 0 {
		t.Error("the missing Python result was not reported")
	}
}

func TestIoUIsSymmetricAndBounded(t *testing.T) {
	a := [4]float64{0, 0, 10, 10}
	b := [4]float64{5, 0, 10, 10}
	if got := iou(a, a); math.Abs(got-1) > 1e-9 {
		t.Errorf("iou(a,a) = %v, want 1", got)
	}
	if iou(a, b) != iou(b, a) {
		t.Error("iou is not symmetric")
	}
	if got := iou(a, [4]float64{100, 100, 10, 10}); got != 0 {
		t.Errorf("disjoint boxes gave iou %v", got)
	}
}
