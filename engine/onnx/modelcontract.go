package onnx

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"strings"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/hyz-is/arandu-alpr"
	"github.com/hyz-is/arandu-alpr/engine"
)

// Model roles. Every message names one, so an error says which file is wrong.
const (
	RoleDetector = "detector"
	RoleOCR      = "ocr"
)

// ModelContract is what a validated model file promises.
type ModelContract struct {
	Role   string
	Path   string
	SHA256 string

	InputName  string
	InputShape []int64

	// Detector: the side of its square input, read from the model. The 640 export and the 608
	// one decode the same way; only the anchor grid differs, and it is derived from this.
	InputSize int

	// Detector: resolved BY SHAPE, never by name. See resolveDetectorOutputs.
	BoxesOutput  string
	ScoresOutput string
	Anchors      int

	// OCR. InputUint8 is a model that takes uint8 pixels instead of float ones.
	InputUint8   bool
	PlateOutput  string
	RegionOutput string
	PlateSlots   int
	AlphabetSize int
	RegionCount  int

	// Warnings are things worth saying at boot that are not reasons to refuse the model.
	Warnings []string
}

// checkFormat warns about a model in ORT format.
//
// The .ort format exists to shrink the runtime binary on mobile: it bakes in a fixed set of
// kernels and carries runtime optimizations computed for one execution provider. Both production
// files here have CPUExecutionProvider baked into them, and the only .onnx-to-.ort converter in
// this workspace runs with `--target_platform arm --optimization_style Runtime
// --enable_type_reduction` -- a mobile ARM configuration.
//
// On a server with the full runtime, .onnx loads directly and lets the graph optimizer work
// against the provider that is actually going to run it. There is nothing .ort buys here, and
// a type-reduced build can only take options away.
//
// It is a warning and not an error: a .ort that works is not worth refusing to boot over.
func checkFormat(path string, device int) string {
	if device < 0 || !strings.HasSuffix(strings.ToLower(path), ".ort") {
		return ""
	}
	return fmt.Sprintf(
		"%s is in .ort format and this session runs on CUDA. The .ort format is for shrinking "+
			"mobile runtimes: it carries optimizations computed for one execution provider "+
			"(both of the models shipped here carry CPUExecutionProvider) and may have been "+
			"built with type reduction, which removes kernels the CUDA build could otherwise "+
			"use. On a server the .onnx loads directly and optimizes against the provider that "+
			"will actually run it.", path)
}

// ContractError collects every mismatch found in one boot.
//
// It reports all of them rather than the first, because a model swapped for the wrong export
// usually fails several checks at once, and fixing them one boot at a time turns a two-minute
// diagnosis into an afternoon.
type ContractError struct {
	Problems []string
}

func (e *ContractError) Error() string {
	return "alpr: model contract violated:\n  - " + strings.Join(e.Problems, "\n  - ")
}

func (e *ContractError) add(format string, args ...any) {
	e.Problems = append(e.Problems, fmt.Sprintf(format, args...))
}

func (e *ContractError) orNil() error {
	if len(e.Problems) == 0 {
		return nil
	}
	return e
}

// fileSHA256 identifies the artefact a contract was checked against.
//
// Without it, "the contract passed" is a claim about a path, and a path is not an artefact. With
// it, the health endpoint and the benchmark record name the exact bytes -- which is what makes
// two benchmark runs comparable and what turns "the accuracy changed" into a question with an
// answer.
func fileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// ValidateDetector checks a detector .ort against what the pipeline assumes.
//
// It loads the file with no execution provider options, which means a CPU-only throwaway
// session: milliseconds on a 3MB model, and it never touches CUDA. What it validates is
// therefore a CPU twin of the same file -- so the SHA is recorded here and the real session is
// built from the same path in the same boot.
func ValidateDetector(path string) (ModelContract, error) {
	c := ModelContract{Role: RoleDetector, Path: path}
	e := &ContractError{}

	sum, err := fileSHA256(path)
	if err != nil {
		e.add("detector: cannot read %s: %v", path, err)
		return c, e
	}
	c.SHA256 = sum

	inputs, outputs, err := ort.GetInputOutputInfo(path)
	if err != nil {
		e.add("detector: cannot load %s (sha256 %s): %v", path, sum[:8], err)
		return c, e
	}

	if len(inputs) != 1 {
		e.add("detector: %d inputs, want exactly 1", len(inputs))
	} else {
		in := inputs[0]
		c.InputName = in.Name
		c.InputShape = in.Dimensions
		requireTensor(e, RoleDetector, "input", in)
		requireFloat(e, RoleDetector, "input", in)
		c.InputSize = detectorSide(in.Dimensions)
		c.Anchors = engine.AnchorCount(c.InputSize)
		if c.Anchors == 0 {
			e.add("detector: input %s has shape %v, want [1 3 S S] with S a multiple of 32.\n"+
				"    NCHW, three channels, square. A different layout here is silent: the tensor is "+
				"the right size and every pixel lands in the wrong plane.", in.Name, in.Dimensions)
		}
	}

	if c.Anchors > 0 {
		boxes, scores, err := resolveDetectorOutputs(outputs, c.InputSize)
		if err != nil {
			e.add("detector: %v", err)
		} else {
			c.BoxesOutput, c.ScoresOutput = boxes, scores
		}
	}

	return c, e.orNil()
}

// detectorSide returns S for an input of shape [1 3 S S], and 0 for anything else.
//
// A dynamic spatial dimension is refused: the tensor is allocated once at a concrete size, and a
// model that accepts any size was exported for a size nobody wrote down.
func detectorSide(dims []int64) int {
	if len(dims) != 4 || !(dims[0] == 1 || dims[0] == -1) || dims[1] != 3 {
		return 0
	}
	if dims[2] <= 0 || dims[2] != dims[3] {
		return 0
	}
	return int(dims[2])
}

// resolveDetectorOutputs picks the boxes and scores tensors BY SHAPE.
//
// The C++ engine matches the output names against "mul" and "sigmoid", and that is a trap: the
// exported graph names every SiLU activation in the backbone `.../act/Mul`, so a re-export whose
// output names shift by one matches an activation instead of the head. The boxes it decodes then
// look like ordinary boxes in ordinary places, and nothing downstream can tell.
//
// The shapes cannot collide the same way: exactly one output is [1 4 anchors] and exactly one is
// [1 1 anchors].
func resolveDetectorOutputs(outputs []ort.InputOutputInfo, inputSize int) (boxes, scores string, err error) {
	anchors := engine.AnchorCount(inputSize)
	want4 := []int64{1, 4, int64(anchors)}
	want1 := []int64{1, 1, int64(anchors)}

	for _, o := range outputs {
		if o.OrtValueType != ort.ONNXTypeTensor || o.DataType != ort.TensorElementDataTypeFloat {
			continue
		}
		switch {
		case shapeEquals(o.Dimensions, want4):
			if boxes != "" {
				return "", "", fmt.Errorf("two outputs have the box shape %v (%q and %q)", want4, boxes, o.Name)
			}
			boxes = o.Name
		case shapeEquals(o.Dimensions, want1):
			if scores != "" {
				return "", "", fmt.Errorf("two outputs have the score shape %v (%q and %q)", want1, scores, o.Name)
			}
			scores = o.Name
		}
	}

	if boxes == "" || scores == "" {
		var got []string
		for _, o := range outputs {
			got = append(got, fmt.Sprintf("%s%v", o.Name, o.Dimensions))
		}
		return "", "", fmt.Errorf(
			"no output pair matching boxes %v and scores %v; found %s.\n"+
				"    %d anchors is what a %d model emits ((S/8)^2+(S/16)^2+(S/32)^2). If this model "+
				"was exported at a different input size, the decoder would read the wrong region of "+
				"both tensors and still return plausible boxes",
			want4, want1, strings.Join(got, ", "), anchors, inputSize)
	}
	return boxes, scores, nil
}

// ValidateOCR checks an OCR .ort against what the decoder assumes.
//
// The dimensions of the model are the source of truth here, and the constants in the recognition
// package are checked against them -- not the other way round. A model with a longer alphabet or
// more plate slots must fail at boot, because DecodePlate indexes the alphabet directly and would
// otherwise emit characters from the wrong offsets: real-looking plates, wrong letters.
func ValidateOCR(path string) (ModelContract, error) {
	c := ModelContract{Role: RoleOCR, Path: path}
	e := &ContractError{}

	sum, err := fileSHA256(path)
	if err != nil {
		e.add("ocr: cannot read %s: %v", path, err)
		return c, e
	}
	c.SHA256 = sum

	inputs, outputs, err := ort.GetInputOutputInfo(path)
	if err != nil {
		e.add("ocr: cannot load %s (sha256 %s): %v", path, sum[:8], err)
		return c, e
	}

	if len(inputs) != 1 {
		e.add("ocr: %d inputs, want exactly 1", len(inputs))
	} else {
		in := inputs[0]
		c.InputName = in.Name
		c.InputShape = in.Dimensions
		requireTensor(e, RoleOCR, "input", in)
		// float32 or uint8, both holding pixel values in [0, 255]: the rescale is inside the graph
		// either way. The fast-plate-ocr export (cct_s_v2_global, the model the INT8 calibration
		// table was made for) takes uint8; the app's re-export takes float.
		switch in.DataType {
		case ort.TensorElementDataTypeFloat:
		case ort.TensorElementDataTypeUint8:
			c.InputUint8 = true
		default:
			e.add("ocr: input %q is %v, want float32 or uint8 pixels in [0, 255]", in.Name, in.DataType)
		}
		if len(in.Dimensions) == 4 && in.Dimensions[0] == -1 {
			c.InputShape = append([]int64{1}, in.Dimensions[1:]...) // the tensor is built at batch 1
		}
		// NHWC, asserted position by position. Feeding NCHW to an NHWC graph is silent and
		// total: the tensor is exactly the right size, the model runs, and every reading is
		// nonsense with ordinary-looking confidence.
		requireShape(e, RoleOCR, "input "+in.Name, in.Dimensions, []int64{1, 64, 128, 3},
			"NHWC: height 64, width 128, three channels, in that order. Swapping to NCHW keeps "+
				"the element count identical and destroys every reading.")
	}

	var plate, region *ort.InputOutputInfo
	for i := range outputs {
		switch outputs[i].Name {
		case "plate":
			plate = &outputs[i]
		case "region":
			region = &outputs[i]
		}
	}

	if plate == nil {
		e.add("ocr: no output named \"plate\"")
	} else {
		requireFloat(e, RoleOCR, "plate", *plate)
		if len(plate.Dimensions) != 3 || !batchOne(plate.Dimensions[0]) {
			e.add("ocr: plate shape is %v, want [1 slots alphabet]", plate.Dimensions)
		} else {
			c.PlateSlots = int(plate.Dimensions[1])
			c.AlphabetSize = int(plate.Dimensions[2])
			if c.PlateSlots != alpr.MaxPlateSlots {
				e.add("ocr: the model emits %d plate slots, the decoder expects %d "+
					"(alpr.MaxPlateSlots)", c.PlateSlots, alpr.MaxPlateSlots)
			}
			if c.AlphabetSize != len(alpr.Alphabet) {
				e.add("ocr: the model has %d character classes, the alphabet has %d "+
					"(%q). The decoder indexes the alphabet directly, so a mismatch "+
					"reads real-looking plates with the wrong letters",
					c.AlphabetSize, len(alpr.Alphabet), alpr.Alphabet)
			}
		}
		c.PlateOutput = plate.Name
	}

	if region == nil {
		// A model without a region head is a degradation, not a failure: the service can still
		// read plates. But it must be visible, never inferred from empty fields downstream.
		c.RegionCount = 0
	} else {
		requireFloat(e, RoleOCR, "region", *region)
		if len(region.Dimensions) != 2 || !batchOne(region.Dimensions[0]) {
			e.add("ocr: region shape is %v, want [1 classes]", region.Dimensions)
		} else {
			c.RegionCount = int(region.Dimensions[1])
			if err := alpr.CheckRegionOutput(c.RegionCount); err != nil {
				e.add("ocr: %v. The region table is POSITIONAL -- its index is the class the "+
					"model emits -- so a size mismatch relabels every reading and nothing at "+
					"runtime looks wrong", err)
			}
		}
		c.RegionOutput = region.Name
	}

	return c, e.orNil()
}

func requireTensor(e *ContractError, role, what string, info ort.InputOutputInfo) {
	if info.OrtValueType != ort.ONNXTypeTensor {
		e.add("%s: %s %q is not a tensor", role, what, info.Name)
	}
}

// batchOne accepts a batch dimension of 1 or a dynamic one: every tensor here is built at batch 1.
func batchOne(d int64) bool { return d == 1 || d == -1 }

func requireFloat(e *ContractError, role, what string, info ort.InputOutputInfo) {
	if info.DataType != ort.TensorElementDataTypeFloat {
		e.add("%s: %s %q is %v, want float32.\n"+
			"    These models are weight-only INT8: the WEIGHTS are quantised, the activations "+
			"and the input stay float. Feeding uint8 here is not an optimisation, it is the "+
			"wrong tensor", role, what, info.Name, info.DataType)
	}
}

func requireShape(e *ContractError, role, what string, got, want []int64, why string) {
	if shapeEquals(got, want) {
		return
	}
	e.add("%s: %s has shape %v, want %v.\n    %s", role, what, got, want, why)
}

func shapeEquals(a []int64, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		// A -1 is a dynamic dimension. Accepting it where the pipeline pins the value to 1 is
		// deliberate: batch-1 export and dynamic-batch export are both usable, and the tensor is
		// built with the concrete size either way.
		if a[i] != b[i] && !(a[i] == -1 && b[i] == 1) {
			return false
		}
	}
	return true
}
