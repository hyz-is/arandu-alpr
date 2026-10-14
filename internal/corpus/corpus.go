// Package corpus is the shared vocabulary of the Python-versus-Go accuracy gate.
//
// The migration's whole risk is here. Throughput is easy to see -- a number goes up or it does
// not -- but accuracy fails quietly: a letterbox padded with the wrong grey, a resize that
// interpolates differently, a channel order swapped, and the service still returns plates. They
// are just slightly wrong plates, at ordinary confidence, and no test that only reads Go code
// can tell.
//
// So both implementations run the same frames and write the same shape, and a comparator that
// knows what each field is allowed to differ by reads them.
package corpus

// Manifest is the list of frames both implementations must process.
type Manifest struct {
	// Alphabet and the model hashes are recorded so a result set can be attributed. A golden
	// file produced against different weights is not stale, it is meaningless, and saying which
	// is the difference between a five-minute answer and an afternoon.
	Alphabet string  `json:"alphabet"`
	Frames   []Frame `json:"frames"`
}

// Frame is one picture with its provenance.
type Frame struct {
	ID     string `json:"id"`
	File   string `json:"file"`
	SHA256 string `json:"sha256"`
	Width  int    `json:"width"`
	Height int    `json:"height"`

	// Notes says what this frame is FOR -- "plate at the frame edge", "IR illuminator stuck
	// white". A corpus without it decays into a folder of pictures nobody dares delete.
	Notes string `json:"notes"`
}

// Result is one implementation's answer for one frame.
type Result struct {
	FrameID string `json:"frame_id"`
	Impl    string `json:"impl"` // "go" or "py"

	DetectorSHA string `json:"detector_sha256"`
	OCRSHA      string `json:"ocr_sha256"`
	AlphabetSHA string `json:"alphabet_sha256"`

	// GatesApplied says whether the cheap filters ran before the OCR.
	//
	// It is recorded rather than assumed because the two implementations cannot be compared
	// without it. The Python service detects and reads in ONE call and filters afterwards; the
	// Go service splits them and drops a candidate before the OCR ever sees it. Go therefore
	// produces legitimately FEWER readings, and without this flag every gate looks like a bug.
	GatesApplied bool `json:"gates_applied"`

	Readings []Reading `json:"readings"`
}

// Reading is one plate, in a fixed order so a diff is a diff.
type Reading struct {
	Plate           string    `json:"plate"`
	CharConfidences []float64 `json:"char_confidences"`
	Confidence      float64   `json:"confidence"`

	Region           string  `json:"region"`
	RegionConfidence float64 `json:"region_confidence"`

	// Box is in ORIGINAL frame coordinates: x, y, width, height.
	Box   [4]float64 `json:"box"`
	Score float64    `json:"detector_score"`
}
