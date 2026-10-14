// Package alpr holds the recognition pipeline: which detections survive, and in what order the
// checks run.
//
// The order is the whole point of this file. Every check here throws work away, and the cheap
// ones must run before the expensive one, because the expensive one is a GPU pass:
//
//	score  ->  edge  ->  blur  ->  OCR  ->  confidence  ->  non-empty  ->  dedup
//	         (arithmetic)        (GPU)
//
// The Python service this is ported from could not do that. It called fast_alpr, which detects
// and reads in a single call, and only then filtered -- so every blurred, edge-cropped plate
// still cost a full OCR pass. Splitting detection from reading is what lets the geometry checks
// run first, and under a busy fleet that difference is GPU capacity.
//
// One check from the Python service is deliberately absent: `_is_valid_brazilian_plate`. See
// AcceptReading.
package alpr

import "math"

// BBox is a detection in frame coordinates.
type BBox struct {
	X, Y, Width, Height float64
}

func (b BBox) X2() float64 { return b.X + b.Width }
func (b BBox) Y2() float64 { return b.Y + b.Height }

// GateConfig carries the thresholds. Every one is an environment knob in the running service;
// they are values here so the decisions stay pure and testable.
type GateConfig struct {
	// MinDetectionScore rejects weak detections before anything is cropped.
	MinDetectionScore float64

	// MinOCRConfidence rejects weak readings after the OCR pass.
	MinOCRConfidence float64

	// EdgeFilterEnabled and EdgeMarginPct drop plates touching the frame border.
	EdgeFilterEnabled bool
	EdgeMarginPct     float64

	// BlurFilterEnabled and BlurThreshold drop out-of-focus crops.
	BlurFilterEnabled bool
	BlurThreshold     float64
	// BlurNormalizeByArea scales the threshold with crop size, so a small distant plate is not
	// judged against the sharpness a close one can reach.
	BlurNormalizeByArea bool
}

// DefaultGate mirrors the defaults the Python service runs with in production.
func DefaultGate() GateConfig {
	return GateConfig{
		MinDetectionScore:   0.24,
		MinOCRConfidence:    0.60,
		EdgeFilterEnabled:   true,
		EdgeMarginPct:       0.01,
		BlurFilterEnabled:   true,
		BlurThreshold:       80,
		BlurNormalizeByArea: true,
	}
}

// AcceptDetection reports whether a detection is worth cropping and reading.
//
// Negated comparisons (`!(score >= min)`) rather than `score < min`: a NaN score fails every
// ordered comparison, so `<` would let it through. A degenerate model producing NaN would then
// send a garbage crop to the OCR, and the reading that came back would look like any other.
func (g GateConfig) AcceptDetection(score float64, b BBox, frameW, frameH int) bool {
	if !(score >= g.MinDetectionScore) {
		return false
	}
	return g.withinSafeZone(b, frameW, frameH)
}

// withinSafeZone rejects a plate touching the frame border.
//
// A plate half outside the frame is worse than no plate: the model receives an incomplete crop
// and invents the missing characters, so the reading arrives with ordinary confidence and a
// wrong plate. That is the failure mode this check exists for -- not a missed read, a confident
// wrong one.
func (g GateConfig) withinSafeZone(b BBox, frameW, frameH int) bool {
	if !g.EdgeFilterEnabled {
		return true
	}
	if frameW <= 0 || frameH <= 0 {
		return true
	}
	marginX := float64(frameW) * g.EdgeMarginPct
	marginY := float64(frameH) * g.EdgeMarginPct
	return b.X > marginX &&
		b.Y > marginY &&
		b.X2() < float64(frameW)-marginX &&
		b.Y2() < float64(frameH)-marginY
}

// referenceArea is the area of a plate filling a typical crop, ~300x90 px.
//
// It is the size at which BlurThreshold was calibrated. Normalisation scales by the square root
// of the ratio because Laplacian variance falls roughly with linear scale, not with area.
const referenceArea = 27000.0

// AcceptFocus reports whether a crop is sharp enough to read, given the Laplacian variance of
// its grayscale pixels.
//
// Variance is passed in rather than computed here: it needs the pixels, and keeping this package
// free of image decoding is what lets the whole gate be tested without a frame.
//
// A crop smaller than 5px on a side is accepted without judging. The variance of a handful of
// pixels is noise, and rejecting on noise would drop distant plates that the OCR can still read.
func (g GateConfig) AcceptFocus(variance float64, b BBox) bool {
	if !g.BlurFilterEnabled {
		return true
	}
	if b.Width <= 4 || b.Height <= 4 {
		return true
	}
	threshold := g.BlurThreshold
	if g.BlurNormalizeByArea {
		area := b.Width * b.Height
		if area > 0 {
			threshold *= math.Sqrt(area / referenceArea)
		}
	}
	return variance >= threshold
}

// AcceptReading reports whether an OCR result becomes a detection.
//
// **There is no plate-format check here, and its absence is deliberate.** The Python service
// ends this gate with `_is_valid_brazilian_plate`, which accepts `AAA9999` and `AAA9A99` and
// drops everything else. This product reads plates worldwide, so that check would silently
// discard every correct foreign reading -- no log, no metric, just plates that never arrive.
//
// The confidence is the only quality bar. The engine returns what the OCR read; deciding what a
// plate should look like belongs to whoever consumes the reading.
func (g GateConfig) AcceptReading(plate string, ocrConfidence float64) bool {
	if plate == "" {
		return false
	}
	return ocrConfidence >= g.MinOCRConfidence
}

// MeanConfidence collapses per-character confidences into one number.
//
// The OCR returns one value per character. The Python service averages them, and that choice
// matters: taking the minimum instead would judge every reading by its worst character, and at
// the same threshold would accept far fewer plates. Changing this silently re-tunes the gate.
//
// An empty slice returns 0, which the threshold then rejects. That is the right answer -- no
// characters read is not a confident reading.
func MeanConfidence(perChar []float64) float64 {
	if len(perChar) == 0 {
		return 0
	}
	var sum float64
	for _, c := range perChar {
		sum += c
	}
	return sum / float64(len(perChar))
}
