// Package engine is the inference boundary: what the camera loop asks for, and what a backend
// has to provide.
//
// The interfaces here are what makes the rest of the service testable on a laptop. Fake
// implements all of them without a GPU, an ONNX Runtime or a model file, so the camera loop,
// the supervisor, the backpressure policy and the allocation budgets can all be exercised in an
// ordinary `go test` -- and the one thing that genuinely needs hardware, the ONNX session, is
// isolated behind a build tag in engine/onnx.
package engine

import (
	"context"
	"errors"
	"time"

	"github.com/hyz-is/arandu-alpr/contract"
)

// Frame is one decoded picture in packed BGR24, as ffmpeg emits it.
//
// Pix is BORROWED. A capture source hands out buffers from a small fixed ring rather than
// allocating six megabytes per frame, so Pix is valid only until Release is called, and a
// Detector or Recognizer must not keep a reference past the call it was passed to.
type Frame struct {
	Pix           []byte
	Width, Height int
	CapturedAt    time.Time

	// Release returns the buffer to whoever lent it. It is nil for a frame that owns its
	// memory, so ReleaseFrame is the safe way to call it.
	//
	// Whoever RECEIVES a frame releases it -- exactly once, when done with the pixels. Forget,
	// and the ring runs dry after three frames and the camera silently stops producing: no
	// error, no crash, just a feed that stops. The capture side counts those drops for exactly
	// that reason.
	Release func()
}

// ReleaseFrame returns a frame's buffer if it borrowed one.
//
// It is nil-safe and idempotent-safe so that `defer engine.ReleaseFrame(f)` is always the right
// thing to write, including for frames built in a test that own their own memory.
func ReleaseFrame(f Frame) {
	if f.Release != nil {
		f.Release()
	}
}

// Candidate is a box the detector proposes, in ORIGINAL frame coordinates.
//
// Detections never leave the detector in letterbox coordinates. A box in letterbox space has
// plausible numbers that point at the wrong part of the picture, which produces a crop of
// nothing and an OCR result that looks like an ordinary bad read.
type Candidate struct {
	Box   contract.BBox
	Score float64
}

// Reading is what the OCR produced for one crop.
type Reading struct {
	Plate           string
	CharConfidences []float64
	Confidence      float64

	Region           string
	RegionConfidence float64

	// Probabilities is the OCR head's whole answer, [alpr.MaxPlateSlots][len(alpr.Alphabet)]
	// row-major, or nil from a backend that does not keep it (alpr.Reading.Probabilities).
	Probabilities []float32
}

// Detector finds plates in a frame.
//
// dst is appended to and returned, so a caller reusing a slice between frames allocates
// nothing. This is the stdlib's AppendFormat shape, and without it the per-frame allocation
// budget leaks out of this package and into every caller.
type Detector interface {
	Detect(ctx context.Context, f Frame, dst []Candidate) ([]Candidate, error)
}

// Recognizer reads one plate.
//
// It takes the frame and the box rather than a finished crop, so the crop buffer stays inside
// the implementation and can be reused. The crop is taken from the ORIGINAL frame, never from
// the detector's letterboxed input: cropping the letterbox throws away resolution exactly on
// the distant plates, which are the hard ones.
type Recognizer interface {
	Recognize(ctx context.Context, f Frame, box contract.BBox) (Reading, error)
}

// ProviderReport says where each model actually ran.
//
// It answers a question the C++ engine's equivalent could not: there, the "active provider" was
// recorded before the session was created, so it reported the provider that was *asked for*. It
// detects a provider failing to attach at all; it does not detect the model quietly falling back
// to CPU node by node, which is the failure that matters and the one this reports.
type ProviderReport struct {
	Models []ModelReport `json:"models"`
	// AllCUDA is true when node placement was measured and no model runs real work on the CPU.
	// The shape arithmetic ONNX Runtime moves to the CPU on purpose does not count against it;
	// see ModelReport.CPUFallback.
	AllCUDA bool `json:"all_cuda"`
	// CPUNodes counts the nodes in CPUFallback, across every measured model.
	CPUNodes int `json:"cpu_nodes"`
}

type ModelReport struct {
	Role   string `json:"role"` // "detector" or "ocr"
	Device int    `json:"device"`
	SHA256 string `json:"sha256"`

	// NodesByProvider counts graph nodes per execution provider. Only the first session of each
	// (role, device) is measured; the others place the same way and report nothing here.
	NodesByProvider map[string]int `json:"nodes_by_provider"`

	// CPUFallback counts, by operator, the nodes of a GPU session that run on the CPU and are
	// not shape arithmetic. The CUDA provider places small shape computations (Shape, Gather,
	// Concat over int64) on the CPU deliberately, because copying a four-element tensor to the
	// GPU costs more than computing it; anything else here is an operator CUDA has no kernel
	// for, which is what a model quantized for the CPU produces.
	CPUFallback map[string]int `json:"cpu_fallback,omitempty"`

	// Warnings are things the operator should know that are not reasons to refuse the model --
	// a format built for a phone rather than a server, for instance.
	Warnings []string `json:"warnings,omitempty"`

	// The three numbers that separate cold start from steady state. A SteadyRunMS in the
	// hundreds is the signature of a CPU fallback, whatever NodesByProvider claims.
	BuildMS     float64 `json:"build_ms"`
	FirstRunMS  float64 `json:"first_run_ms"`
	SteadyRunMS float64 `json:"steady_run_ms"`
}

// Engine is a Detector and a Recognizer with a lifecycle.
type Engine interface {
	Detector
	Recognizer

	// Ready reports whether inference can serve traffic. It is non-nil during cold start,
	// when sessions are still being built and the first pass would take seconds.
	Ready() error

	// Providers reports where the models ran.
	Providers() ProviderReport

	// ModelInfo identifies the loaded artefacts, for attribution on every detection.
	ModelInfo() contract.ModelInfo

	// Close releases sessions and the memory they hold.
	//
	// It must be explicit. An ONNX session that is merely dropped keeps its VRAM until the
	// process exits, and in a redeploy that means the container coming up finds the GPU still
	// occupied by the one going down.
	Close(ctx context.Context) error
}

// ErrNotReady is returned by Detect and Recognize before warmup finishes.
var ErrNotReady = errors.New("engine: not ready")

// ErrBusy is returned when no session became free before the caller's deadline.
//
// The camera loop turns this into a dropped frame, not a queued one. That is the whole
// backpressure policy at the GPU boundary: under saturation the service loses frames, which is
// recoverable, rather than growing latency, which is not.
var ErrBusy = errors.New("engine: no session available")
