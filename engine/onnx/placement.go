package onnx

import (
	"bufio"
	"bytes"
	"io"
	"os"
	"regexp"
	"strconv"
	"strings"
	"sync"
)

// Node placement: which execution provider actually got each node of the graph.
//
// This answers a question the sibling C++ engine only appears to answer. There, the "active
// provider" is recorded BEFORE the session is created, so it reports the provider that was asked
// for. It catches the case where appending the provider fails outright; it cannot catch the
// model quietly falling back to CPU node by node -- which is the failure that matters, because
// it does not error, it just gets slow, and slow gets diagnosed as a camera or network problem.
//
// The Go binding exposes no placement query: there is no binding for GetAvailableProviders and
// none for SessionEndProfiling, and GetEpDevices lists plugin providers, which CUDA is not
// (it is compiled into the library).
//
// What does exist is the log. At verbose severity, ONNX Runtime prints a "Node placements"
// section while creating a session, ending either in "All nodes have been placed on
// [CUDAExecutionProvider]" or in a per-provider breakdown. So the first session of each
// (role, device) is built at verbose severity with stderr captured.
//
// The capture is process-global for the duration of one build. That is acceptable only because
// it happens at boot, before the server is serving, and because everything captured is written
// back to the real stderr afterwards -- nothing is swallowed.

// The lines ONNX Runtime prints (core/framework/session_state.cc, VerifyEachNodeIsAssignedToAnEp,
// unchanged from 1.17 to 1.27), each behind the logger's own "[V:onnxruntime:...]" prefix:
//
//	Node placements
//	 All nodes placed on [CUDAExecutionProvider]. Number of nodes: 846
//
// or, when more than one provider got nodes, one block per provider listing "OpType (name)":
//
//	Node placements
//	 Node(s) placed on [CPUExecutionProvider]. Number of nodes: 2
//	  Shape (/Shape)
//	  Gather (/Gather)
//	 Node(s) placed on [CUDAExecutionProvider]. Number of nodes: 498
//	  Conv (/conv1/Conv)
var (
	allNodesRE  = regexp.MustCompile(`All nodes placed on \[([A-Za-z]+)\]\. Number of nodes: (\d+)`)
	someNodesRE = regexp.MustCompile(`Node\(s\) placed on \[([A-Za-z]+)\]\. Number of nodes: (\d+)`)
	nodeLineRE  = regexp.MustCompile(`\s{2}([A-Za-z][A-Za-z0-9_]*) \(.*\)\s*$`)
)

const (
	cpuProvider = "CPUExecutionProvider"
	trtProvider = "TensorrtExecutionProvider"
)

// shapeOps are the operators the CUDA provider moves to the CPU on purpose.
//
// GetCpuPreferredNodes walks forward from Shape and Size and keeps on the CPU the small int64
// arithmetic that turns a shape into another shape, because a round trip to the GPU for four
// numbers costs more than the arithmetic. ONNX Runtime's own warning says so: "ORT explicitly
// assigns shape related ops to CPU to improve perf". Those nodes are not a fallback, and an OCR
// graph with attention always has some.
//
// Every operator that does real work on a tensor -- Conv, MatMul, Softmax, Resize, and every
// QLinear*, QGemm, Nhwc* and DequantizeLinear a CPU-quantized model brings -- is absent on
// purpose: on a GPU session, one of those on the CPU is the failure this package exists to see.
var shapeOps = map[string]bool{
	"Shape": true, "Size": true, "Gather": true, "Unsqueeze": true, "Squeeze": true,
	"Concat": true, "Slice": true, "Cast": true, "ConstantOfShape": true, "Range": true,
	"Reshape": true, "Expand": true, "Tile": true, "Equal": true, "Where": true, "Not": true,
	"Less": true, "Greater": true, "Add": true, "Sub": true, "Mul": true, "Div": true,
	"Floor": true, "Ceil": true, "Min": true, "Max": true, "ReduceProd": true,
	"Identity": true, "Constant": true,
}

// placement is what one verbose session build reported.
type placement struct {
	nodes  map[string]int // nodes per execution provider
	cpuOps map[string]int // operators of the nodes on the CPU, when ONNX Runtime listed them
}

// fallback returns the CPU nodes that are not shape arithmetic, by operator.
//
// A graph placed entirely on the CPU is reported in one line with no operator list, so it comes
// back as a single entry: every one of those nodes is a fallback on a GPU session.
func (p placement) fallback() map[string]int {
	out := map[string]int{}
	if p.nodes[cpuProvider] > 0 && len(p.cpuOps) == 0 {
		out["every node"] = p.nodes[cpuProvider]
		return out
	}
	for op, n := range p.cpuOps {
		if !shapeOps[op] {
			out[op] += n
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// captureMu serialises stderr redirection. The builder loop is already serial, so this only
// guards against a caller reaching in from elsewhere.
var captureMu sync.Mutex

// capturePlacement runs fn with stderr captured, and parses node placement out of what ONNX
// Runtime logged. When verbose is false it runs fn directly and reports nothing.
func capturePlacement(verbose bool, out *placement, fn func() error) error {
	if !verbose {
		return fn()
	}

	captureMu.Lock()
	defer captureMu.Unlock()

	saved, err := dupStderr()
	if err != nil {
		// Losing the placement report is worth strictly less than losing the session. Build
		// anyway and say nothing rather than fail the boot over diagnostics.
		return fn()
	}
	defer closeFD(saved)

	r, w, err := os.Pipe()
	if err != nil {
		return fn()
	}

	if err := redirectStderr(w); err != nil {
		_ = r.Close()
		_ = w.Close()
		return fn()
	}

	// The pipe has a finite buffer, and a chatty verbose build can fill it. Draining
	// concurrently is what keeps a full pipe from blocking the session creation itself.
	var buf bytes.Buffer
	done := make(chan struct{})
	go func() {
		_, _ = io.Copy(&buf, r)
		close(done)
	}()

	fnErr := fn()

	_ = restoreStderr(saved)
	_ = w.Close()
	<-done
	_ = r.Close()

	// Everything ONNX Runtime said still belongs in the log. Writing it back is what makes this
	// a capture rather than a gag.
	_, _ = os.Stderr.Write(buf.Bytes())

	if p := parsePlacement(buf.String()); len(p.nodes) > 0 {
		*out = p
	}
	return fnErr
}

// parsePlacement reads the "Node placements" section ONNX Runtime logs at verbose severity.
//
// The section ends at the first line that is neither a provider header nor a node of the current
// provider, so nothing logged after it -- allocator and memory-pattern lines name providers too --
// is counted as a placement.
func parsePlacement(log string) placement {
	p := placement{nodes: map[string]int{}, cpuOps: map[string]int{}}

	inSection, current := false, ""
	sc := bufio.NewScanner(strings.NewReader(log))
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		line := sc.Text()
		if strings.Contains(line, "Node placements") {
			inSection, current = true, ""
			continue
		}
		if !inSection {
			continue
		}
		if m := allNodesRE.FindStringSubmatch(line); m != nil {
			n, _ := strconv.Atoi(m[2])
			p.nodes[m[1]] += n
			inSection = false
			continue
		}
		if m := someNodesRE.FindStringSubmatch(line); m != nil {
			n, _ := strconv.Atoi(m[2])
			p.nodes[m[1]] += n
			current = m[1]
			continue
		}
		if m := nodeLineRE.FindStringSubmatch(line); m != nil && current != "" {
			if current == cpuProvider {
				p.cpuOps[m[1]]++
			}
			continue
		}
		inSection, current = false, ""
	}
	return p
}

// allCUDA reports whether every node that does real work landed on CUDA.
//
// An unmeasured placement is NOT treated as success. Nothing was measured, and answering "yes" to
// a question that was never asked is exactly the mistake the C++ version makes.
func (p placement) allCUDA() bool {
	if len(p.nodes) == 0 {
		return false
	}
	return p.onlyCUDAAndCPU() && len(p.fallback()) == 0
}

// onlyCUDAAndCPU reports whether no node went to a provider this service did not ask for.
// TensorRT counts as the GPU: it is the provider asked for first when ALPR_TENSORRT is on.
func (p placement) onlyCUDAAndCPU() bool {
	for provider, n := range p.nodes {
		if n > 0 && provider != "CUDAExecutionProvider" && provider != trtProvider && provider != cpuProvider {
			return false
		}
	}
	return true
}

// cpuNodes counts the nodes that fell back to the CPU, leaving out shape arithmetic.
func (p placement) cpuNodes() int {
	n := 0
	for _, c := range p.fallback() {
		n += c
	}
	return n
}
