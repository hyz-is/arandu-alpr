package alpr

import (
	"math"
	"testing"
)

// oneHot builds a plate tensor where each slot puts `conf` on the given character and spreads the
// rest evenly, mimicking the softmaxed output the model actually emits.
func oneHot(chars string, conf float32) []float32 {
	out := make([]float32, MaxPlateSlots*len(Alphabet))
	rest := (1 - conf) / float32(len(Alphabet)-1)
	for slot := 0; slot < MaxPlateSlots; slot++ {
		ch := byte(PadChar)
		if slot < len(chars) {
			ch = chars[slot]
		}
		idx := 0
		for i := 0; i < len(Alphabet); i++ {
			if Alphabet[i] == ch {
				idx = i
				break
			}
		}
		for i := 0; i < len(Alphabet); i++ {
			if i == idx {
				out[slot*len(Alphabet)+i] = conf
			} else {
				out[slot*len(Alphabet)+i] = rest
			}
		}
	}
	return out
}

func TestAlphabetMatchesTheModelHead(t *testing.T) {
	// The head emits [1, 10, 37]. A different alphabet length here would silently remap every
	// character the engine reads -- the plates would still look like plates.
	if len(Alphabet) != 37 {
		t.Fatalf("alphabet has %d characters, the model head has 37", len(Alphabet))
	}
	if MaxPlateSlots != 10 {
		t.Fatalf("MaxPlateSlots is %d, the model head emits 10", MaxPlateSlots)
	}
}

func TestDecodePlateStripsPadding(t *testing.T) {
	// The model always emits ten slots and fills the unused ones with `_`. Keeping them would put
	// literal underscores in the plate text.
	r := DecodePlate(oneHot("ABC1D23", 0.99))
	if r.Plate != "ABC1D23" {
		t.Fatalf("plate = %q, want ABC1D23", r.Plate)
	}
	if len(r.CharConfidences) != 7 {
		t.Errorf("got %d confidences for a 7-character plate", len(r.CharConfidences))
	}
}

func TestDecodePlateDropsPaddingFromTheMiddle(t *testing.T) {
	// A model emitting `AB_1D23` has lost a character. Keeping the underscore would put it inside
	// the plate; dropping it yields a short plate, which the confidence gate can then judge.
	r := DecodePlate(oneHot("AB_1D23", 0.9))
	if r.Plate != "AB1D23" {
		t.Errorf("plate = %q, want AB1D23 with the padding removed", r.Plate)
	}
	if len(r.CharConfidences) != len(r.Plate) {
		t.Errorf("%d confidences for %d characters", len(r.CharConfidences), len(r.Plate))
	}
}

func TestDecodePlateOutputIsAlreadySoftmaxed(t *testing.T) {
	// The model ends in a softmax. Applying another one here would flatten the distribution and
	// return confidences that are all wrong in the same direction -- never obviously wrong.
	r := DecodePlate(oneHot("ABC1D23", 0.87))
	for i, c := range r.CharConfidences {
		if math.Abs(c-0.87) > 1e-5 {
			t.Fatalf("confidence[%d] = %v, want the raw 0.87 from the tensor", i, c)
		}
	}
	if math.Abs(r.Confidence()-0.87) > 1e-5 {
		t.Errorf("Confidence() = %v, want 0.87", r.Confidence())
	}
}

func TestDecodePlateRejectsWrongTensorSize(t *testing.T) {
	// A model whose head does not match returns nothing rather than reading garbage out of a
	// short buffer.
	if r := DecodePlate(make([]float32, 5)); r.Plate != "" {
		t.Errorf("decoded %q from a malformed tensor", r.Plate)
	}
}

func TestRegionTableIsPositionalAndComplete(t *testing.T) {
	// The class is the argmax index, and this table is the only thing turning it into a name.
	// Sorting the list would look like tidying and would relabel every reading ever produced.
	if len(regions) != RegionCount {
		t.Fatalf("region table has %d entries, the head emits %d", len(regions), RegionCount)
	}
	// Pinned positions, taken from the model's own config.
	for idx, want := range map[int]string{0: "Albania", 2: "Argentina", 65: "Unknown"} {
		if got := RegionName(idx); got != want {
			t.Errorf("RegionName(%d) = %q, want %q", idx, got, want)
		}
	}
}

func TestRegionNameOutOfRangeIsUnknown(t *testing.T) {
	// A model with a different head must degrade to "country not known", not kill the worker
	// mid-frame. The mismatch is caught at startup instead.
	if RegionName(-1) != "Unknown" || RegionName(999) != "Unknown" {
		t.Error("out-of-range class did not fall back to Unknown")
	}
}

func TestCheckRegionOutputCatchesTheWrongModel(t *testing.T) {
	if err := CheckRegionOutput(RegionCount); err != nil {
		t.Errorf("matching head rejected: %v", err)
	}
	err := CheckRegionOutput(40)
	if err == nil {
		t.Fatal("a 40-class head was accepted; every country name would be wrong")
	}
	if got := err.Error(); got == "" {
		t.Error("error message is empty")
	}
}

func TestDecodeRegionPicksTheArgmax(t *testing.T) {
	logits := make([]float32, RegionCount)
	logits[2] = 0.93 // Argentina
	logits[7] = 0.04
	name, conf := DecodeRegion(logits)
	if name != "Argentina" {
		t.Errorf("region = %q, want Argentina", name)
	}
	if math.Abs(conf-0.93) > 1e-6 {
		t.Errorf("confidence = %v, want 0.93", conf)
	}
}

func TestDecodeRegionRejectsWrongSize(t *testing.T) {
	name, conf := DecodeRegion(make([]float32, 10))
	if name != "Unknown" || conf != 0 {
		t.Errorf("got %q/%v from a malformed tensor", name, conf)
	}
}
