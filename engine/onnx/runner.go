package onnx

import (
	"fmt"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/hyz-is/arandu-alpr"
)

// A runner is one ONNX session plus every buffer one inference needs, owned exclusively while it
// is checked out of the pool.
//
// The exclusivity is not tidiness. Sharing one of these between goroutines is what took
// production down on 04/08/2026 with `CUDA failure 700: an illegal memory access was encountered
// ; GPU=1`, and the cause is structural: ONNX Runtime's Run() is thread-safe, but AdvancedSession
// holds the bound input and output OrtValues from construction and reuses them on every call.
// Two callers overwrite each other's tensors, the GPU reads memory that is no longer what it
// expected, and the CUDA context stays corrupt until the process restarts.
//
// Nothing in the type system says so -- Run() has a pointer receiver but looks pure. The pool is
// what enforces it, and no *AdvancedSession may escape this package.

// detectorRunner holds a detector session and its tensors.
type detectorRunner struct {
	session *ort.AdvancedSession

	input  *ort.Tensor[float32]
	boxes  *ort.Tensor[float32]
	scores *ort.Tensor[float32]

	contract ModelContract
	device   int
}

// newDetectorRunner builds the session and binds its tensors once.
//
// Tensors are bound here rather than per call because AdvancedSession.Run uses the OrtValues it
// was constructed with -- so a bound session allocates nothing in Go per frame. The dynamic
// variant would allocate two slices on every inference, which at fleet rates is measurable
// garbage for no benefit.
func newDetectorRunner(c ModelContract, opts *ort.SessionOptions, device int) (*detectorRunner, error) {
	side := int64(c.InputSize)
	inShape := ort.NewShape(1, 3, side, side)
	input, err := ort.NewEmptyTensor[float32](inShape)
	if err != nil {
		return nil, fmt.Errorf("detector input tensor: %w", err)
	}
	anchors := int64(c.Anchors)
	boxes, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 4, anchors))
	if err != nil {
		input.Destroy()
		return nil, fmt.Errorf("detector boxes tensor: %w", err)
	}
	scores, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 1, anchors))
	if err != nil {
		input.Destroy()
		boxes.Destroy()
		return nil, fmt.Errorf("detector scores tensor: %w", err)
	}

	session, err := ort.NewAdvancedSession(
		c.Path,
		[]string{c.InputName},
		[]string{c.BoxesOutput, c.ScoresOutput},
		[]ort.Value{input},
		[]ort.Value{boxes, scores},
		opts,
	)
	if err != nil {
		input.Destroy()
		boxes.Destroy()
		scores.Destroy()
		return nil, fmt.Errorf("detector session (%s sha256 %s): %w", c.Path, short(c.SHA256), err)
	}

	return &detectorRunner{
		session: session, input: input, boxes: boxes, scores: scores,
		contract: c, device: device,
	}, nil
}

// inputBuffer is where the caller writes the letterboxed frame.
//
// It is the tensor's own memory, handed out so preprocessing can write straight into it with no
// intermediate float slice. The caller must never append to it: ONNX Runtime holds a pointer to
// this backing array for the life of the tensor, and a reallocation would leave the session
// reading freed memory.
func (r *detectorRunner) inputBuffer() []float32 { return r.input.GetData() }

func (r *detectorRunner) run() (boxes, scores []float32, err error) {
	if err := r.session.Run(); err != nil {
		return nil, nil, err
	}
	return r.boxes.GetData(), r.scores.GetData(), nil
}

func (r *detectorRunner) destroy() {
	// Order matters only in that everything must go: a session merely dropped keeps its VRAM
	// until the process exits, and in a redeploy the container coming up finds the GPU still
	// held by the one going down.
	if r.session != nil {
		r.session.Destroy()
	}
	if r.input != nil {
		r.input.Destroy()
	}
	if r.boxes != nil {
		r.boxes.Destroy()
	}
	if r.scores != nil {
		r.scores.Destroy()
	}
}

// ocrRunner holds an OCR session and its tensors.
type ocrRunner struct {
	session *ort.AdvancedSession

	// Exactly one of input and inputU8 is set, by the model's input type. A uint8 model is fed
	// through scratch: preprocessing writes float pixels there as it does for a float model, and
	// run rounds them into the tensor -- 24576 values a crop, against a model pass.
	input   *ort.Tensor[float32]
	inputU8 *ort.Tensor[uint8]
	scratch []float32

	plate  *ort.Tensor[float32]
	region *ort.Tensor[float32]

	contract ModelContract
	device   int
}

func newOCRRunner(c ModelContract, opts *ort.SessionOptions, device int) (*ocrRunner, error) {
	// NHWC. The shape is taken from the validated contract rather than written again here, so
	// there is exactly one place in the service that states the layout.
	r := &ocrRunner{contract: c, device: device}
	var in ort.Value
	var err error
	if c.InputUint8 {
		r.inputU8, err = ort.NewEmptyTensor[uint8](ort.NewShape(c.InputShape...))
		if err == nil {
			r.scratch = make([]float32, len(r.inputU8.GetData()))
			in = r.inputU8
		}
	} else {
		r.input, err = ort.NewEmptyTensor[float32](ort.NewShape(c.InputShape...))
		in = r.input
	}
	if err != nil {
		return nil, fmt.Errorf("ocr input tensor: %w", err)
	}
	input := in
	plate, err := ort.NewEmptyTensor[float32](
		ort.NewShape(1, int64(c.PlateSlots), int64(c.AlphabetSize)))
	if err != nil {
		input.Destroy()
		return nil, fmt.Errorf("ocr plate tensor: %w", err)
	}

	outNames := []string{c.PlateOutput}
	outVals := []ort.Value{plate}

	var region *ort.Tensor[float32]
	if c.RegionOutput != "" {
		region, err = ort.NewEmptyTensor[float32](ort.NewShape(1, int64(c.RegionCount)))
		if err != nil {
			input.Destroy()
			plate.Destroy()
			return nil, fmt.Errorf("ocr region tensor: %w", err)
		}
		outNames = append(outNames, c.RegionOutput)
		outVals = append(outVals, region)
	}

	session, err := ort.NewAdvancedSession(
		c.Path, []string{c.InputName}, outNames,
		[]ort.Value{input}, outVals, opts,
	)
	if err != nil {
		input.Destroy()
		plate.Destroy()
		if region != nil {
			region.Destroy()
		}
		return nil, fmt.Errorf("ocr session (%s sha256 %s): %w", c.Path, short(c.SHA256), err)
	}

	r.session, r.plate, r.region = session, plate, region
	return r, nil
}

func (r *ocrRunner) inputBuffer() []float32 {
	if r.scratch != nil {
		return r.scratch
	}
	return r.input.GetData()
}

// run reads one crop and decodes it.
//
// The plate tensor is ALREADY softmaxed -- the model ends in one. DecodePlate takes it as
// probabilities, and applying another softmax here would flatten the distribution and return
// confidences that are all wrong in the same direction, which is the kind of error that passes
// every threshold and never looks like a bug.
func (r *ocrRunner) run() (alpr.Reading, error) {
	if r.inputU8 != nil {
		toUint8(r.inputU8.GetData(), r.scratch)
	}
	if err := r.session.Run(); err != nil {
		return alpr.Reading{}, err
	}
	reading := alpr.DecodePlate(r.plate.GetData())
	if r.region != nil {
		reading.Region, reading.RegionConfidence = alpr.DecodeRegion(r.region.GetData())
	}
	return reading, nil
}

func (r *ocrRunner) destroy() {
	if r.session != nil {
		r.session.Destroy()
	}
	if r.input != nil {
		r.input.Destroy()
	}
	if r.inputU8 != nil {
		r.inputU8.Destroy()
	}
	if r.plate != nil {
		r.plate.Destroy()
	}
	if r.region != nil {
		r.region.Destroy()
	}
}

// toUint8 rounds float pixels in [0, 255] into bytes, clamping what the bilinear filter can
// overshoot. Rounding, not truncation: it is what a uint8 resize produces, which is what the
// model and its calibration table saw.
func toUint8(dst []uint8, src []float32) {
	for i, v := range src {
		switch {
		case v <= 0:
			dst[i] = 0
		case v >= 255:
			dst[i] = 255
		default:
			dst[i] = uint8(v + 0.5)
		}
	}
}

func short(sum string) string {
	if len(sum) > 8 {
		return sum[:8]
	}
	return sum
}
