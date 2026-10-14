package report

import (
	"strings"
	"testing"
)

// baselineRun is a plausible tier B line, used as the thing every case below moves one knob of.
func baselineRun() Run {
	return Run{
		Schema: Schema,
		Tier:   "B",
		Config: Config{
			Cameras:           40,
			Devices:           []int{0},
			SessionsPerDevice: 2,
			OCRSessions:       3,
			Realtime:          true,
		},
		Models: []Model{
			{Role: "detector", SHA256: "aaaaaaaaaaaaaaaaaaaaaaaa"},
			{Role: "ocr", SHA256: "bbbbbbbbbbbbbbbbbbbbbbbb"},
		},
		Throughput: Throughput{
			FramesReceived:   120000,
			FramesPerSecond:  200,
			DetectionsTotal:  4900,
			DetectionsPerMin: 490,
		},
		Latency: map[string]Quantiles{
			"pipeline": {Count: 4900, Sum: 1470, P50: 0.18, P95: 0.42, P99: 0.71},
			"detector": {Count: 30000, Sum: 300, P50: 0.009, P95: 0.014, P99: 0.02},
		},
		Drops: map[string]float64{"inference_busy": 1200, "skip": 80000},
		GPUs:  []GPU{{Index: 0, PeakMiB: 6100, SteadyMiB: 5800, TotalMiB: 24000}},
		SLO:   SLO{Metric: "alpr_pipeline_duration_seconds", P95BudgetSeconds: 1, P95ObservedSeconds: 0.42},
	}
}

// TestADifferentModelIsIncomparable is the most important thing this tool does.
//
// A throughput diff across two model files attributes the change to the code, which did not
// cause it. The refusal has to come before any number is printed, because a printed number gets
// acted on.
func TestADifferentModelIsIncomparable(t *testing.T) {
	before := baselineRun()
	after := baselineRun()
	after.Models[1].SHA256 = "cccccccccccccccccccccccc"
	after.Throughput.DetectionsPerMin = 200 // a catastrophic-looking fall, and not the point

	findings := Compare(before, after, DefaultTolerance(), false)
	if len(findings) != 1 || findings[0].Severity != Blocking {
		t.Fatalf("expected one blocking finding, got %+v", findings)
	}
	if !strings.Contains(findings[0].Field, "models") {
		t.Errorf("the blocking finding was about %q, not the models", findings[0].Field)
	}
	for _, f := range findings {
		if f.Severity == Regression {
			t.Errorf("a regression was reported across two different models: %+v", f)
		}
	}
}

// TestForceComparesAcrossADeliberateConfigChange.
//
// "Same code, one knob moved" is a real comparison a human can make. The gate exists so it is
// made on purpose rather than by accident.
func TestForceComparesAcrossADeliberateConfigChange(t *testing.T) {
	before := baselineRun()
	after := baselineRun()
	after.Config.SessionsPerDevice = 4
	after.Throughput.DetectionsPerMin = 300

	if findings := Compare(before, after, DefaultTolerance(), false); len(findings) != 1 {
		t.Fatalf("without force the compare should stop at the config, got %+v", findings)
	}

	findings := Compare(before, after, DefaultTolerance(), true)
	var sawThroughput bool
	for _, f := range findings {
		if f.Field == "throughput.detections_per_minute" && f.Severity == Regression {
			sawThroughput = true
		}
	}
	if !sawThroughput {
		t.Errorf("with force the throughput fall should be reported, got %+v", findings)
	}
}

func TestLatencyRiseIsARegressionAndAFallIsNot(t *testing.T) {
	before := baselineRun()
	after := baselineRun()
	after.Latency["pipeline"] = Quantiles{Count: 4900, Sum: 2900, P50: 0.18, P95: 0.90, P99: 1.4}

	findings := Compare(before, after, DefaultTolerance(), false)
	var p95 *Finding
	for i := range findings {
		if findings[i].Field == "latency.pipeline.p95" {
			p95 = &findings[i]
		}
	}
	if p95 == nil || p95.Severity != Regression {
		t.Fatalf("a p95 that more than doubled was not reported as a regression: %+v", findings)
	}

	faster := baselineRun()
	faster.Latency["pipeline"] = Quantiles{Count: 4900, Sum: 700, P50: 0.09, P95: 0.2, P99: 0.3}
	for _, f := range Compare(before, faster, DefaultTolerance(), false) {
		if strings.HasPrefix(f.Field, "latency.") && f.Severity == Regression {
			t.Errorf("a faster run was reported as a regression: %+v", f)
		}
	}
}

// TestBusyDropRiseIsReportedWithItsRemedy.
//
// The reason label is the difference between "we are losing frames" and "the GPU is the
// ceiling", and the second one tells an operator what to buy. A finding that only names the
// number leaves the useful half in somebody's head.
func TestBusyDropRiseIsReportedWithItsRemedy(t *testing.T) {
	before := baselineRun()
	after := baselineRun()
	after.Drops["inference_busy"] = 40000

	var found *Finding
	for _, f := range Compare(before, after, DefaultTolerance(), false) {
		if f.Field == "drops.inference_busy" {
			found = &f
		}
	}
	if found == nil || found.Severity != Regression {
		t.Fatal("a busy-drop rate that went from 1% to 33% was not reported")
	}
	if !strings.Contains(found.Why, "GPU") {
		t.Errorf("the finding does not say what the reason means: %q", found.Why)
	}
}

func TestQuantilesRestingOnAlmostNoObservationsBlockTheCompare(t *testing.T) {
	before := baselineRun()
	after := baselineRun()
	after.Latency["pipeline"] = Quantiles{Count: 11, Sum: 2, P50: 0.18, P95: 0.42, P99: 0.7}

	var blocked bool
	for _, f := range Compare(before, after, DefaultTolerance(), false) {
		if f.Field == "latency.pipeline.count" && f.Severity == Blocking {
			blocked = true
		}
	}
	if !blocked {
		t.Error("a p95 over eleven samples was compared to one over 4900 without complaint")
	}
}

func TestIdenticalRunsProduceNoFindings(t *testing.T) {
	if findings := Compare(baselineRun(), baselineRun(), DefaultTolerance(), false); len(findings) != 0 {
		t.Errorf("a run compared to itself produced %+v", findings)
	}
	if HasRegression(nil) {
		t.Error("no findings is not a regression")
	}
}

func TestSLOCrossingIsReportedInBothDirections(t *testing.T) {
	before := baselineRun()
	after := baselineRun()
	after.SLO.Crossed = true

	var found bool
	for _, f := range Compare(before, after, DefaultTolerance(), false) {
		if f.Field == "slo.crossed" && f.Severity == Regression {
			found = true
		}
	}
	if !found {
		t.Fatal("a configuration that fell outside the SLO was not reported")
	}

	for _, f := range Compare(after, before, DefaultTolerance(), false) {
		if f.Field == "slo.crossed" && f.Severity != Improvement {
			t.Errorf("recovering into the SLO was reported as %q", f.Severity)
		}
	}
}
