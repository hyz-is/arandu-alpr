package onnx

import (
	"strings"
	"testing"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/hyz-is/arandu-alpr/engine"
)

// These tests cover everything in this package that does not need ONNX Runtime loaded, which is
// deliberately most of what can go wrong. The session lifecycle needs a GPU and is exercised on
// the CUDA machine; the decisions that produce plausible-but-wrong output are all here.

func TestAnchorCountIsDerivedFromInputSize(t *testing.T) {
	// 8400 is only true for a 640 model. Writing the literal makes a re-export at another size
	// decode the wrong region of both tensors -- and still return boxes that look reasonable.
	if got := engine.AnchorCount(640); got != 8400 {
		t.Errorf("engine.AnchorCount(640) = %d, want 8400", got)
	}
	for size, want := range map[int]int{
		320: 40*40 + 20*20 + 10*10,
		512: 64*64 + 32*32 + 16*16,
		960: 120*120 + 60*60 + 30*30,
	} {
		if got := engine.AnchorCount(size); got != want {
			t.Errorf("engine.AnchorCount(%d) = %d, want %d", size, got, want)
		}
	}
}

func floatTensor(name string, dims ...int64) ort.InputOutputInfo {
	return ort.InputOutputInfo{
		Name:         name,
		OrtValueType: ort.ONNXTypeTensor,
		Dimensions:   ort.NewShape(dims...),
		DataType:     ort.TensorElementDataTypeFloat,
	}
}

func TestDetectorOutputsResolveByShapeNotName(t *testing.T) {
	// The C++ engine matches output names against "mul" and "sigmoid". Every SiLU activation in
	// the YOLO backbone is named `.../act/Mul`, so a re-export whose outputs shift matches an
	// activation instead of the head -- and the boxes it decodes look entirely ordinary.
	//
	// Here the names are deliberately misleading: the tensor called "Mul" is the SCORES one.
	outputs := []ort.InputOutputInfo{
		floatTensor("/model.9/act/Mul_output_0", 1, 1, 8400),
		floatTensor("/model.22/Sigmoid_output_0", 1, 4, 8400),
	}
	boxes, scores, err := resolveDetectorOutputs(outputs, 640)
	if err != nil {
		t.Fatalf("resolveDetectorOutputs: %v", err)
	}
	if boxes != "/model.22/Sigmoid_output_0" {
		t.Errorf("boxes resolved to %q; the name won over the shape", boxes)
	}
	if scores != "/model.9/act/Mul_output_0" {
		t.Errorf("scores resolved to %q; the name won over the shape", scores)
	}
}

func TestDetectorOutputsRejectWrongAnchorCount(t *testing.T) {
	// A 512 export against a 640 decoder. It has to fail loudly here, because nothing
	// downstream can tell: the boxes are well-formed, they just describe part of the picture.
	outputs := []ort.InputOutputInfo{
		floatTensor("boxes", 1, 4, 5376),
		floatTensor("scores", 1, 1, 5376),
	}
	_, _, err := resolveDetectorOutputs(outputs, 640)
	if err == nil {
		t.Fatal("a 5376-anchor model was accepted by an 8400-anchor decoder")
	}
	if !strings.Contains(err.Error(), "8400") {
		t.Errorf("the error does not name the expected anchor count: %v", err)
	}
}

func TestDetectorSideIsReadFromTheModel(t *testing.T) {
	// The 608 export and the 640 one decode the same way; the size is the model's, not a constant.
	cases := []struct {
		dims []int64
		want int
	}{
		{[]int64{1, 3, 640, 640}, 640},
		{[]int64{1, 3, 608, 608}, 608},
		{[]int64{-1, 3, 608, 608}, 608},
		{[]int64{1, 3, -1, -1}, 0},   // dynamic: no size to allocate
		{[]int64{1, 3, 608, 640}, 0}, // not square
		{[]int64{1, 608, 608, 3}, 0}, // NHWC
	}
	for _, c := range cases {
		if got := detectorSide(c.dims); got != c.want {
			t.Errorf("detectorSide(%v) = %d, want %d", c.dims, got, c.want)
		}
	}
	if got := engine.AnchorCount(608); got != 7581 {
		t.Errorf("engine.AnchorCount(608) = %d, want 7581", got)
	}
}

func TestDetectorOutputsRejectAmbiguousShapes(t *testing.T) {
	// Two tensors with the box shape means the head is not what we think it is. Picking either
	// one would be a coin flip resolved silently.
	outputs := []ort.InputOutputInfo{
		floatTensor("a", 1, 4, 8400),
		floatTensor("b", 1, 4, 8400),
		floatTensor("c", 1, 1, 8400),
	}
	if _, _, err := resolveDetectorOutputs(outputs, 640); err == nil {
		t.Fatal("two box-shaped outputs were accepted")
	}
}

func TestNonFloatOutputsAreNotConsidered(t *testing.T) {
	// A quantised export with a uint8 head would otherwise match on shape alone.
	outputs := []ort.InputOutputInfo{
		{Name: "q", OrtValueType: ort.ONNXTypeTensor, Dimensions: ort.NewShape(1, 4, 8400),
			DataType: ort.TensorElementDataTypeUint8},
		floatTensor("scores", 1, 1, 8400),
	}
	if _, _, err := resolveDetectorOutputs(outputs, 640); err == nil {
		t.Fatal("a uint8 tensor was accepted as the box output")
	}
}

func TestDynamicBatchDimensionIsAccepted(t *testing.T) {
	// A -1 batch is an ordinary export choice, and the tensor is built with a concrete 1 either
	// way. Rejecting it would refuse a model that works.
	if !shapeEquals([]int64{-1, 3, 640, 640}, []int64{1, 3, 640, 640}) {
		t.Error("a dynamic batch dimension was rejected")
	}
	// But a dynamic dimension anywhere the pipeline pins a real size is not the same thing.
	if shapeEquals([]int64{1, 3, -1, 640}, []int64{1, 3, 640, 640}) {
		t.Error("a dynamic spatial dimension was accepted as 640")
	}
}

func TestContractErrorReportsEveryProblemNotJustTheFirst(t *testing.T) {
	// A model swapped for the wrong export usually fails several checks at once. Reporting one
	// per boot turns a two-minute diagnosis into an afternoon of rebuilds.
	e := &ContractError{}
	e.add("first")
	e.add("second")
	e.add("third")
	msg := e.Error()
	for _, want := range []string{"first", "second", "third"} {
		if !strings.Contains(msg, want) {
			t.Errorf("error text is missing %q: %s", want, msg)
		}
	}
	if e.orNil() == nil {
		t.Error("orNil swallowed a non-empty error")
	}
	if (&ContractError{}).orNil() != nil {
		t.Error("orNil returned a non-nil error with no problems")
	}
}

// The fixtures below are the lines ONNX Runtime 1.27 prints, prefix included
// (core/framework/session_state.cc, VerifyEachNodeIsAssignedToAnEp).
const ortPrefix = "2026-10-02 12:00:00.000000000 [V:onnxruntime:, session_state.cc:1350 VerifyEachNodeIsAssignedToAnEp] "

func ortLog(lines ...string) string {
	out := ""
	for _, l := range lines {
		out += ortPrefix + l + "\n"
	}
	return out
}

func TestParsePlacementReadsTheHomogeneousSummary(t *testing.T) {
	p := parsePlacement(ortLog("Node placements", " All nodes placed on [CUDAExecutionProvider]. Number of nodes: 846"))
	if !p.allCUDA() {
		t.Errorf("allCUDA = false for a fully-CUDA graph: %+v", p)
	}
	if p.nodes["CUDAExecutionProvider"] != 846 || p.cpuNodes() != 0 {
		t.Errorf("nodes = %v, cpuNodes = %d; want 846 on CUDA and none on the CPU", p.nodes, p.cpuNodes())
	}
}

func TestShapeArithmeticOnTheCPUIsNotAFallback(t *testing.T) {
	// The CUDA provider keeps small int64 shape computations on the CPU on purpose. Counting them
	// would make every OCR graph with attention read as broken.
	p := parsePlacement(ortLog(
		"Node placements",
		" Node(s) placed on [CPUExecutionProvider]. Number of nodes: 3",
		"  Shape (/Shape)",
		"  Gather (/Gather)",
		"  Concat (/Concat)",
		" Node(s) placed on [CUDAExecutionProvider]. Number of nodes: 497",
		"  MatMul (/attn/MatMul)",
	))
	if !p.allCUDA() || p.cpuNodes() != 0 || p.fallback() != nil {
		t.Errorf("shape ops on the CPU read as a fallback: %+v, fallback %v", p, p.fallback())
	}
}

func TestACPUQuantizedModelIsAFallback(t *testing.T) {
	// This is the case the C++ equivalent cannot see, and the one the mobile .ort files produce
	// on a GPU: the provider attached fine, and the convolutions still ran on the CPU.
	p := parsePlacement(ortLog(
		"Node placements",
		" Node(s) placed on [CPUExecutionProvider]. Number of nodes: 4",
		"  QLinearConv (/model.0/conv/Conv_quant)",
		"  QLinearConv (/model.1/conv/Conv_quant)",
		"  NhwcMaxPool (/model.2/MaxPool_nhwc)",
		"  Shape (/Shape)",
		" Node(s) placed on [CUDAExecutionProvider]. Number of nodes: 2",
		"  Sigmoid (/Sigmoid)",
	))
	if p.allCUDA() {
		t.Errorf("allCUDA = true with convolutions on the CPU: %+v", p)
	}
	want := map[string]int{"QLinearConv": 2, "NhwcMaxPool": 1}
	if got := p.fallback(); len(got) != len(want) || got["QLinearConv"] != 2 || got["NhwcMaxPool"] != 1 {
		t.Errorf("fallback = %v, want %v", got, want)
	}
	if p.cpuNodes() != 3 {
		t.Errorf("cpuNodes = %d, want 3", p.cpuNodes())
	}
	if got := describeOps(p.fallback()); got != "NhwcMaxPool x1, QLinearConv x2" {
		t.Errorf("describeOps = %q", got)
	}
}

func TestAGraphEntirelyOnTheCPUIsAFallback(t *testing.T) {
	p := parsePlacement(ortLog("Node placements", " All nodes placed on [CPUExecutionProvider]. Number of nodes: 500"))
	if p.allCUDA() || p.cpuNodes() != 500 {
		t.Errorf("a graph entirely on the CPU: allCUDA %v, cpuNodes %d", p.allCUDA(), p.cpuNodes())
	}
}

func TestLinesAfterThePlacementSectionAreNotPlacements(t *testing.T) {
	// The allocator and memory-pattern lines that follow name providers too.
	p := parsePlacement(ortLog(
		"Node placements",
		" All nodes placed on [CUDAExecutionProvider]. Number of nodes: 846",
		"Allocator [CPUExecutionProvider] reserves  Conv (x)",
		" Node(s) placed on [CPUExecutionProvider]. Number of nodes: 9",
	))
	if !p.allCUDA() || p.nodes[cpuProvider] != 0 {
		t.Errorf("lines after the section were counted: %+v", p)
	}
}

func TestEmptyPlacementIsNotSuccess(t *testing.T) {
	// Nothing was measured. Answering "all CUDA" to a question never asked is precisely the
	// mistake this package exists to avoid -- the C++ version records the provider it ASKED
	// for, before the session is built, and reports that.
	if (placement{}).allCUDA() {
		t.Error("allCUDA of an empty placement = true; an unmeasured graph must not read as a clean one")
	}
	if parsePlacement("nothing about placement here").allCUDA() {
		t.Error("allCUDA = true for a log with no placement section")
	}
}

func TestProvidersIgnoreTheSessionsThatWereNotMeasured(t *testing.T) {
	// Only the first session of each (role, device) is built verbosely. The others carry no
	// placement, and counting them as failures made all_cuda false on every real fleet.
	measured := engine.ModelReport{Role: RoleDetector, NodesByProvider: map[string]int{"CUDAExecutionProvider": 846}}
	unmeasured := engine.ModelReport{Role: RoleDetector}
	e := &Engine{reports: []engine.ModelReport{measured, unmeasured, unmeasured}}
	if rep := e.Providers(); !rep.AllCUDA || rep.CPUNodes != 0 {
		t.Errorf("Providers = %+v; want all_cuda with no CPU nodes", rep)
	}

	fallback := engine.ModelReport{
		Role:            RoleOCR,
		NodesByProvider: map[string]int{"CUDAExecutionProvider": 2, cpuProvider: 4},
		CPUFallback:     map[string]int{"QLinearMatMul": 3},
	}
	e = &Engine{reports: []engine.ModelReport{measured, unmeasured, fallback}}
	if rep := e.Providers(); rep.AllCUDA || rep.CPUNodes != 3 {
		t.Errorf("Providers = %+v; want all_cuda false and 3 CPU nodes", rep)
	}

	if rep := (&Engine{reports: []engine.ModelReport{unmeasured}}).Providers(); rep.AllCUDA {
		t.Error("all_cuda = true with nothing measured")
	}
}

func TestConfigDefaultsPickTheLibraryMicrosoftShips(t *testing.T) {
	// The binding's own default is "onnxruntime.so"; the official releases contain
	// libonnxruntime.so. Without this the very first call fails at dlopen.
	c := Config{}.withDefaults()
	if !strings.HasSuffix(c.LibraryPath, "libonnxruntime.so") {
		t.Errorf("default library path is %q, which is not what ONNX Runtime ships", c.LibraryPath)
	}
	if c.WarmupRuns < 2 {
		t.Errorf("WarmupRuns = %d; one run only measures the cold path", c.WarmupRuns)
	}
}

func TestDestroyRequestsJumpTheQueue(t *testing.T) {
	// Destroying frees VRAM and building consumes it. Serving them in arrival order is how a
	// fleet that is churning cameras runs out of memory while holding memory it has already
	// finished with.
	pending := []buildRequest{
		{kind: kindBuild},
		{kind: kindBuild},
		{kind: kindDestroy},
		{kind: kindBuild},
	}
	got := takeHighestPriority(&pending)
	if got.kind != kindDestroy {
		t.Fatal("a build was served while a destroy was waiting")
	}
	if len(pending) != 3 {
		t.Errorf("%d requests left, want 3", len(pending))
	}
	for _, r := range pending {
		if r.kind == kindDestroy {
			t.Error("the destroy was left in the queue")
		}
	}
}

func TestBuildFailuresBackOffPerDeviceAndRole(t *testing.T) {
	// Without this, a GPU in a bad state turns every Borrow into a fresh multi-second attempt
	// on the single serial loop -- starving the devices that still work. The pool cannot know:
	// it hands the slot back as unbuilt and the next caller tries at once.
	b := NewBuilder(Config{})
	spec := SessionSpec{Role: RoleDetector, Device: 1}

	if _, backing := b.backoffRemaining(spec); backing {
		t.Fatal("a fresh builder is already backing off")
	}
	for i := 0; i < 3; i++ {
		b.recordOutcome(spec, errTest)
	}
	wait, backing := b.backoffRemaining(spec)
	if !backing {
		t.Fatal("three consecutive failures did not start a backoff")
	}
	if wait < 30e9 {
		t.Errorf("backoff is %s; from the third failure the cause is rarely transient", wait)
	}

	// A different device must be unaffected: one bad GPU cannot quarantine a good one.
	if _, backing := b.backoffRemaining(SessionSpec{Role: RoleDetector, Device: 0}); backing {
		t.Error("device 0 was backed off because device 1 failed")
	}

	// Success clears it.
	b.recordOutcome(spec, nil)
	if _, backing := b.backoffRemaining(spec); backing {
		t.Error("a successful build did not clear the backoff")
	}
}

var errTest = errStr("build failed")

type errStr string

func (e errStr) Error() string { return string(e) }

func TestMaxConvWorkspaceIsOnByDefault(t *testing.T) {
	// ONNX Runtime's own documentation calls this "an important flag to use while using an fp16
	// model as this allows CuDNN to pick tensor core algorithms". On a T4 -- Turing, where the
	// tensor cores ARE the gap between FP32 and FP16 -- that is the largest lever this service
	// has left, and clamping the workspace to 32MB quietly closes it.
	c := Config{}.withDefaults()
	if c.ConvUseMaxWorkspace != nil && !*c.ConvUseMaxWorkspace {
		t.Error("the convolution workspace is clamped by default; FP16 would not reach the tensor cores")
	}
}

func TestConvAlgoSearchDefaultsToExhaustive(t *testing.T) {
	// EXHAUSTIVE gives the fastest steady state and is ONNX Runtime's own default. It is also a
	// large share of the eight to fifteen seconds a session takes to build, which is why the
	// knob exists at all: cold start is what a fleet feels during a redeploy, and HEURISTIC
	// trades a little throughput for a lot of it.
	if got := (Config{}).withDefaults().ConvAlgoSearch; got != "" && got != "EXHAUSTIVE" {
		t.Errorf("ConvAlgoSearch default = %q", got)
	}
}

func TestNHWCIsOffByDefault(t *testing.T) {
	// NVIDIA's tensor cores prefer NHWC, but reaching it costs transposes inserted into the
	// graph, and whether that trade pays depends on how much of this particular model has an
	// NHWC kernel. It is a thing to measure on the sweep, never a thing to assume.
	if (Config{}).withDefaults().PreferNHWC {
		t.Error("prefer_nhwc is on by default; whether it helps is model-specific and unmeasured")
	}
}

func TestGPUMemoryIsUnlimitedByDefault(t *testing.T) {
	// A limit is worth setting once the session count per card is known. Guessing one before
	// then produces a build that fails on a card with free memory, which reads as a hardware
	// fault rather than as a setting.
	if got := (Config{}).withDefaults().GPUMemLimit; got != 0 {
		t.Errorf("GPUMemLimit default = %d, want 0 (unlimited)", got)
	}
}

func TestORTFormatOnCUDAIsFlagged(t *testing.T) {
	// The .ort format exists to shrink a mobile runtime. Both models shipped here carry
	// CPUExecutionProvider baked in, and the only converter in this workspace runs with
	// `--target_platform arm --optimization_style Runtime --enable_type_reduction`.
	//
	// On a server the .onnx loads directly and optimizes against the provider that will run it.
	// Worth saying out loud at boot, because nothing about a working .ort looks wrong.
	if w := checkFormat("/models/detector.ort", 0); w == "" {
		t.Error("a .ort on CUDA was not flagged")
	}
	if w := checkFormat("/models/detector.onnx", 0); w != "" {
		t.Errorf("a .onnx on CUDA was flagged: %s", w)
	}
	// On CPU the format is not the interesting question, and a warning there is noise.
	if w := checkFormat("/models/detector.ort", -1); w != "" {
		t.Errorf("a .ort on CPU was flagged: %s", w)
	}
}
