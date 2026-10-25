package alpr

import (
	"math"
	"strings"
)

// Alphabet is the character set the OCR head emits, one class per index.
//
// 36 characters plus `_`, the padding class -- 37 in total, which is the last dimension of the
// `plate` output. Copied from the model's own plate config; the head is positional, so a
// different string here silently remaps every character the engine reads.
const Alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"

// PadChar marks an unused slot. The model always emits MaxPlateSlots positions and fills the
// unused ones with this class, so a six-character plate arrives padded to ten.
const PadChar = '_'

// MaxPlateSlots is the number of character positions the head emits.
const MaxPlateSlots = 10

// Reading is one decoded plate.
type Reading struct {
	Plate string
	// CharConfidences holds one probability per emitted character, in reading order. Padding
	// slots are excluded, so this is as long as Plate.
	CharConfidences []float64
	// Region is the country the region head chose, and RegionConfidence its probability.
	Region           string
	RegionConfidence float64

	// Probabilities is the head's whole answer: MaxPlateSlots rows of len(Alphabet) probabilities,
	// row-major, padding class included -- the distribution Plate was read from, before the argmax.
	//
	// It is kept because the argmax throws away what several frames of the same car need to agree.
	// A frame that read "1" with 55% and "I" with 45% says something about both, and a text vote
	// counts it as a full vote for the one. Consensus adds these tables up instead.
	Probabilities []float32
}

// DecodePlate turns the `plate` output into text.
//
// The tensor is [1, MaxPlateSlots, len(Alphabet)] and is **already softmaxed** -- the model ends
// in a softmax, and running another one here would flatten the distribution and hand back
// confidences that are all wrong in the same direction, which is the kind of error that never
// looks like an error.
//
// Padding is dropped from the middle as well as the end. A model that emits `AB_1D23` has lost a
// character, and keeping the underscore would put a literal `_` inside the plate text; dropping
// it yields a plate one character short, which the confidence gate can then judge.
func DecodePlate(logits []float32) Reading {
	var r Reading
	if len(logits) != MaxPlateSlots*len(Alphabet) {
		return r
	}

	var text strings.Builder
	text.Grow(MaxPlateSlots)
	confs := make([]float64, 0, MaxPlateSlots)

	for slot := 0; slot < MaxPlateSlots; slot++ {
		base := slot * len(Alphabet)
		best, bestIdx := float32(-1), 0
		for i := 0; i < len(Alphabet); i++ {
			if v := logits[base+i]; v > best {
				best, bestIdx = v, i
			}
		}
		ch := Alphabet[bestIdx]
		if ch == PadChar {
			continue
		}
		text.WriteByte(ch)
		confs = append(confs, float64(best))
	}

	r.Plate = text.String()
	r.CharConfidences = confs
	r.Probabilities = append([]float32(nil), logits...)
	return r
}

// DecodeRegion turns the `region` output into a country name.
//
// Like the plate head, this is already softmaxed, so the winning value is a probability and can
// be reported as confidence without further work.
func DecodeRegion(logits []float32) (string, float64) {
	if len(logits) != RegionCount {
		return "Unknown", 0
	}
	best, bestIdx := float32(-1), 0
	for i, v := range logits {
		if v > best {
			best, bestIdx = v, i
		}
	}
	return RegionName(bestIdx), float64(best)
}

// Confidence is the reading's overall confidence: the mean of its per-character values.
//
// See MeanConfidence for why the mean and not the minimum.
func (r Reading) Confidence() float64 { return MeanConfidence(r.CharConfidences) }

// ProbabilitiesFromText rebuilds an OCR distribution from what a reading kept of it: the text and
// one probability per character.
//
// It is for an engine that reports a reading and not the head's whole answer (the fake the tests
// use, or a backend that never kept the tensor). Each character gets its probability in its slot
// and the rest of the slot is shared evenly; the slots after the text say padding with the
// reading's own mean confidence. That is the best table the reading supports, no better: a
// consensus fed these counts frames the way a text vote would, which is still a vote, and still
// better than none.
//
// A character outside the alphabet is the padding class, and a text longer than the head's slots
// is cut at them.
func ProbabilitiesFromText(plate string, confs []float64) []float32 {
	a := len(Alphabet)
	out := make([]float32, MaxPlateSlots*a)
	var mean float64
	for _, c := range confs {
		mean += c
	}
	if len(confs) > 0 {
		mean /= float64(len(confs))
	}
	for slot := 0; slot < MaxPlateSlots; slot++ {
		top, p := a-1, mean // padding
		if slot < len(plate) {
			if i := strings.IndexByte(Alphabet, plate[slot]); i >= 0 {
				top, p = i, 0.5
				if slot < len(confs) {
					p = confs[slot]
				}
			}
		}
		p = math.Min(math.Max(p, 0), 1)
		rest := (1 - p) / float64(a-1)
		for k := 0; k < a; k++ {
			out[slot*a+k] = float32(rest)
		}
		out[slot*a+top] = float32(p)
	}
	return out
}
