package report

import (
	"fmt"
	"reflect"
	"sort"
	"strings"
)

// Severity says how a finding should be read.
type Severity string

const (
	// Blocking means the two runs cannot be compared at all. It is not a regression; it is a
	// refusal, and it is the most valuable thing this tool does.
	Blocking Severity = "incomparable"
	// Regression means the number moved the wrong way by more than the tolerance.
	Regression Severity = "regression"
	// Improvement is reported too, because a compare that only ever prints bad news gets read
	// as noise and then not read.
	Improvement Severity = "improvement"
)

// Finding is one difference worth printing.
type Finding struct {
	Severity Severity
	Field    string
	Baseline string
	Current  string
	// Why explains what the number means, not what it is. A reader who already knows p95 went
	// up needs to be told what that costs.
	Why string
}

// Tolerance is how much movement is noise.
//
// The defaults are deliberately loose. A benchmark that flags a 2% move flags something on
// every run, and a tool that cries wolf is a tool nobody runs before merging.
type Tolerance struct {
	// ThroughputDropRatio is how far throughput may fall before it is a regression (0.10 = 10%).
	ThroughputDropRatio float64
	// LatencyRiseRatio is how far a quantile may rise.
	LatencyRiseRatio float64
	// DropRateRise is an absolute rise in dropped-frame ratio (0.02 = two more frames in a
	// hundred), because a ratio-of-a-ratio is unreadable when the baseline is near zero.
	DropRateRise float64
	// VRAMRiseRatio is how far peak VRAM may grow.
	VRAMRiseRatio float64
	// RSSRiseRatio is how far peak RSS may grow.
	RSSRiseRatio float64
}

// DefaultTolerance is what compare uses when nothing is said.
func DefaultTolerance() Tolerance {
	return Tolerance{
		ThroughputDropRatio: 0.10,
		LatencyRiseRatio:    0.15,
		DropRateRise:        0.02,
		VRAMRiseRatio:       0.10,
		RSSRiseRatio:        0.20,
	}
}

// Compare diffs two runs, baseline first.
//
// It checks comparability BEFORE it checks numbers, and it stops there when the answer is no.
// Two runs on different models, different configs or different schemas produce a diff full of
// perfectly real numbers that mean nothing together, and printing that diff is worse than
// printing nothing: somebody acts on it.
//
// force skips only the comparability gate. It exists because "same config, one knob moved on
// purpose" is a legitimate comparison a human can make and a machine cannot.
func Compare(baseline, current Run, tol Tolerance, force bool) []Finding {
	var out []Finding

	blocking := comparability(baseline, current)
	if len(blocking) > 0 && !force {
		return blocking
	}
	out = append(out, blocking...)

	out = append(out, compareThroughput(baseline, current, tol)...)
	out = append(out, compareLatency(baseline, current, tol)...)
	out = append(out, compareDrops(baseline, current, tol)...)
	out = append(out, compareResources(baseline, current, tol)...)
	out = append(out, compareSLO(baseline, current)...)
	return out
}

// comparability is the gate: what must be identical for the numbers to mean the same thing.
func comparability(baseline, current Run) []Finding {
	var out []Finding
	add := func(field, a, b, why string) {
		out = append(out, Finding{Severity: Blocking, Field: field, Baseline: a, Current: b, Why: why})
	}

	if baseline.Schema != current.Schema {
		add("schema", baseline.Schema, current.Schema,
			"the two lines were produced by different versions of the harness, so the fields do not describe the same measurement")
	}
	if baseline.Tier != current.Tier {
		add("tier", baseline.Tier, current.Tier,
			"tier A measures the pipeline against a fake engine and tier B measures a real GPU fleet; they share field names and nothing else")
	}

	if diff := modelDiff(baseline.Models, current.Models); diff != "" {
		add("models", modelSummary(baseline.Models), modelSummary(current.Models),
			"different model bytes: "+diff+". Accuracy and cost both move with the artefact, so a throughput diff across two models attributes the change to the code that did not cause it")
	}

	// The config is compared as a whole rather than field by field, because the fields that
	// matter are exactly the ones nobody thought to list.
	if !reflect.DeepEqual(baseline.Config, current.Config) {
		add("config", configSummary(baseline.Config), configSummary(current.Config),
			"the runs were configured differently: "+configDiff(baseline.Config, current.Config))
	}
	return out
}

func compareThroughput(baseline, current Run, tol Tolerance) []Finding {
	var out []Finding

	if f, ok := ratioFinding(
		"throughput.detections_per_minute",
		baseline.Throughput.DetectionsPerMin, current.Throughput.DetectionsPerMin,
		-tol.ThroughputDropRatio,
		"plates read per minute is the product; a fall here is the fleet doing less work for the same hardware",
	); ok {
		out = append(out, f)
	}
	if f, ok := ratioFinding(
		"throughput.frames_per_second",
		baseline.Throughput.FramesPerSecond, current.Throughput.FramesPerSecond,
		-tol.ThroughputDropRatio,
		"frames reaching the loop; a fall with unchanged detections means the cameras are delivering less, not that the pipeline got slower",
	); ok {
		out = append(out, f)
	}
	return out
}

// latencyOrder pins the order the stages are printed in, which is the order they happen in.
// A reader scanning the output should be able to see where the time went without reordering it
// in their head.
var latencyOrder = []string{"pipeline", "detector", "ocr", "session_wait"}

func compareLatency(baseline, current Run, tol Tolerance) []Finding {
	var out []Finding
	for _, stage := range latencyNames(baseline.Latency, current.Latency) {
		a, b := baseline.Latency[stage], current.Latency[stage]
		for _, q := range []struct {
			name string
			a, b float64
			why  string
		}{
			{"p50", a.P50, b.P50, "the typical frame"},
			{"p95", a.P95, b.P95, "the bar the SLO is written against"},
			{"p99", a.P99, b.P99, "the tail that a saturated GPU shows up in first"},
		} {
			if f, ok := ratioFinding(
				"latency."+stage+"."+q.name,
				q.a, q.b, tol.LatencyRiseRatio, q.why,
			); ok {
				// A latency rise is a regression; a fall is an improvement. ratioFinding
				// already assigns those, but the sign convention is worth reading twice: a
				// bigger number is worse here and better in throughput.
				out = append(out, f)
			}
		}
		if b.Count > 0 && b.Count*20 < a.Count {
			out = append(out, Finding{
				Severity: Blocking,
				Field:    "latency." + stage + ".count",
				Baseline: fmt.Sprintf("%d", a.Count),
				Current:  fmt.Sprintf("%d", b.Count),
				Why:      "the current run has an order of magnitude fewer observations, so its quantiles rest on a sample the baseline's does not",
			})
		}
	}
	return out
}

func latencyNames(a, b map[string]Quantiles) []string {
	seen := map[string]bool{}
	var out []string
	for _, name := range latencyOrder {
		if _, ok := a[name]; ok {
			seen[name] = true
			out = append(out, name)
			continue
		}
		if _, ok := b[name]; ok {
			seen[name] = true
			out = append(out, name)
		}
	}
	var rest []string
	for name := range a {
		if !seen[name] {
			seen[name] = true
			rest = append(rest, name)
		}
	}
	for name := range b {
		if !seen[name] {
			seen[name] = true
			rest = append(rest, name)
		}
	}
	sort.Strings(rest)
	return append(out, rest...)
}

func compareDrops(baseline, current Run, tol Tolerance) []Finding {
	var out []Finding

	reasons := map[string]bool{}
	for r := range baseline.Drops {
		reasons[r] = true
	}
	for r := range current.Drops {
		reasons[r] = true
	}
	names := make([]string, 0, len(reasons))
	for r := range reasons {
		names = append(names, r)
	}
	sort.Strings(names)

	for _, reason := range names {
		before := share(baseline.Drops[reason], baseline.Throughput.FramesReceived)
		after := share(current.Drops[reason], current.Throughput.FramesReceived)
		delta := after - before
		if delta <= tol.DropRateRise {
			continue
		}
		out = append(out, Finding{
			Severity: Regression,
			Field:    "drops." + reason,
			Baseline: fmt.Sprintf("%.2f%% of frames", before*100),
			Current:  fmt.Sprintf("%.2f%% of frames", after*100),
			Why:      dropMeaning(reason),
		})
	}
	return out
}

// dropMeaning is the label's whole reason for existing: each reason points at a different
// remedy, and an operator who only knows the total cannot pick one.
func dropMeaning(reason string) string {
	switch reason {
	case "inference_busy":
		return "no session was free before the deadline: the GPU is the ceiling, and more cameras will not fit without more sessions or more devices"
	case "stale":
		return "frames arrived older than the ingest lag limit: the cameras or the network are behind, not the GPU"
	case "skip":
		return "frames landed inside the detection interval, which is the budget working as configured rather than a fault"
	case "blank":
		return "the camera is emitting a solid picture, which is signal loss rather than load"
	case "producer_replaced":
		return "a newer frame arrived before this one was taken, which is newest-wins working"
	default:
		return "dropped frames rose for this reason"
	}
}

func compareResources(baseline, current Run, tol Tolerance) []Finding {
	var out []Finding

	for i, gpu := range current.GPUs {
		if i >= len(baseline.GPUs) {
			break
		}
		if f, ok := ratioFinding(
			fmt.Sprintf("gpu[%d].vram_peak_mib", gpu.Index),
			baseline.GPUs[i].PeakMiB, gpu.PeakMiB, tol.VRAMRiseRatio,
			"peak VRAM decides whether the box survives its worst second; a rise here shrinks how many sessions fit",
		); ok {
			out = append(out, f)
		}
	}

	if f, ok := ratioFinding(
		"process.rss_peak_mib",
		baseline.Process.PeakRSSMiB, current.Process.PeakRSSMiB, tol.RSSRiseRatio,
		"host memory, which is where the frame buffer rings live: three buffers per camera, so a rise usually means more cameras or a bigger geometry rather than a leak",
	); ok {
		out = append(out, f)
	}
	return out
}

func compareSLO(baseline, current Run) []Finding {
	if baseline.SLO.Crossed == current.SLO.Crossed {
		return nil
	}
	sev := Regression
	why := "this configuration was inside the SLO and is now outside it: the cameras-per-GPU answer moved down"
	if baseline.SLO.Crossed && !current.SLO.Crossed {
		sev = Improvement
		why = "this configuration was outside the SLO and is now inside it: the cameras-per-GPU answer moved up"
	}
	return []Finding{{
		Severity: sev,
		Field:    "slo.crossed",
		Baseline: fmt.Sprintf("%t", baseline.SLO.Crossed),
		Current:  fmt.Sprintf("%t", current.SLO.Crossed),
		Why:      why,
	}}
}

// ratioFinding reports a relative move past a tolerance.
//
// A positive tolerance means "bigger is worse" (latency, memory); a negative one means "smaller
// is worse" (throughput). One function for both so the sign convention is written down once
// instead of being re-derived at every call site.
func ratioFinding(field string, before, after, tol float64, why string) (Finding, bool) {
	if before == 0 && after == 0 {
		return Finding{}, false
	}
	if before == 0 {
		// Nothing to divide by. A number appearing where there was none is worth saying, and
		// calling it a regression when it is latency and an improvement when it is throughput
		// would be guessing.
		return Finding{
			Severity: Improvement,
			Field:    field,
			Baseline: "0",
			Current:  format(after),
			Why:      why + " (the baseline recorded nothing here, so there is no ratio to judge)",
		}, true
	}

	delta := (after - before) / before
	worseWhenBigger := tol > 0
	limit := tol
	if !worseWhenBigger {
		limit = -tol
	}

	switch {
	case worseWhenBigger && delta > limit:
		return finding(Regression, field, before, after, delta, why), true
	case !worseWhenBigger && -delta > limit:
		return finding(Regression, field, before, after, delta, why), true
	case worseWhenBigger && -delta > limit:
		return finding(Improvement, field, before, after, delta, why), true
	case !worseWhenBigger && delta > limit:
		return finding(Improvement, field, before, after, delta, why), true
	}
	return Finding{}, false
}

func finding(sev Severity, field string, before, after, delta float64, why string) Finding {
	return Finding{
		Severity: sev,
		Field:    field,
		Baseline: format(before),
		Current:  fmt.Sprintf("%s (%+.1f%%)", format(after), delta*100),
		Why:      why,
	}
}

func format(v float64) string {
	switch {
	case v == 0:
		return "0"
	case v < 0.001:
		return fmt.Sprintf("%.3gs", v)
	case v < 1:
		return fmt.Sprintf("%.1fms", v*1000)
	default:
		return fmt.Sprintf("%.4g", v)
	}
}

func share(part, whole float64) float64 {
	if whole <= 0 {
		return 0
	}
	return part / whole
}

func modelDiff(a, b []Model) string {
	index := func(models []Model) map[string]Model {
		m := map[string]Model{}
		for _, mod := range models {
			m[mod.Role] = mod
		}
		return m
	}
	ia, ib := index(a), index(b)
	var parts []string
	roles := map[string]bool{}
	for r := range ia {
		roles[r] = true
	}
	for r := range ib {
		roles[r] = true
	}
	names := make([]string, 0, len(roles))
	for r := range roles {
		names = append(names, r)
	}
	sort.Strings(names)
	for _, role := range names {
		if ia[role].SHA256 != ib[role].SHA256 {
			parts = append(parts, fmt.Sprintf("%s %s -> %s", role, short(ia[role].SHA256), short(ib[role].SHA256)))
		}
	}
	return strings.Join(parts, ", ")
}

func modelSummary(models []Model) string {
	parts := make([]string, 0, len(models))
	for _, m := range models {
		parts = append(parts, m.Role+"="+short(m.SHA256))
	}
	sort.Strings(parts)
	return strings.Join(parts, " ")
}

func short(sha string) string {
	if len(sha) <= 12 {
		if sha == "" {
			return "(none)"
		}
		return sha
	}
	return sha[:12]
}

func configSummary(c Config) string {
	return fmt.Sprintf("%d cameras, devices %v, %d+%d sessions/gpu, detect every %.0fms",
		c.Cameras, c.Devices, c.SessionsPerDevice, c.OCRSessions, c.MinDetectIntervalMS)
}

// configDiff names the fields that differ, using reflection so a knob added tomorrow is covered
// the day it is added rather than the day somebody remembers this function exists.
func configDiff(a, b Config) string {
	va, vb := reflect.ValueOf(a), reflect.ValueOf(b)
	t := va.Type()

	var parts []string
	for i := 0; i < t.NumField(); i++ {
		fa, fb := va.Field(i), vb.Field(i)
		if reflect.DeepEqual(fa.Interface(), fb.Interface()) {
			continue
		}
		name := t.Field(i).Tag.Get("json")
		if comma := strings.IndexByte(name, ','); comma >= 0 {
			name = name[:comma]
		}
		if name == "" {
			name = t.Field(i).Name
		}
		parts = append(parts, fmt.Sprintf("%s %v -> %v", name, fa.Interface(), fb.Interface()))
	}
	if len(parts) == 0 {
		return "(no field differs, which means a nested struct compares unequal for a reason worth looking at)"
	}
	return strings.Join(parts, "; ")
}

// HasRegression reports whether any finding should fail a build.
func HasRegression(findings []Finding) bool {
	for _, f := range findings {
		if f.Severity == Regression || f.Severity == Blocking {
			return true
		}
	}
	return false
}
