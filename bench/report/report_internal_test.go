package report

import (
	"bytes"
	"strings"
	"testing"

	"github.com/hyz-is/arandu-alpr/metrics"
)

func TestWriteLineIsOneLine(t *testing.T) {
	var buf bytes.Buffer
	for i := 0; i < 3; i++ {
		if err := WriteLine(&buf, Run{Tier: "B", Config: Config{Cameras: i}}); err != nil {
			t.Fatal(err)
		}
	}

	lines := strings.Split(strings.TrimRight(buf.String(), "\n"), "\n")
	if len(lines) != 3 {
		t.Fatalf("three runs produced %d lines\n\n"+
			"A run has to be one line or the file cannot be tailed, grepped or fed to compare "+
			"one run at a time, which is the only reason it is JSON lines and not JSON.",
			len(lines))
	}

	runs, err := ReadLines(strings.NewReader(buf.String()))
	if err != nil {
		t.Fatal(err)
	}
	if len(runs) != 3 || runs[2].Config.Cameras != 2 {
		t.Fatalf("round trip lost runs: %+v", runs)
	}
	if runs[0].Schema != Schema {
		t.Errorf("WriteLine did not stamp the schema: %q", runs[0].Schema)
	}
}

// TestATruncatedLineIsAnErrorNotAnOmission.
//
// A killed sweep leaves a half-written line. Skipping it quietly would make an interrupted run
// look like a shorter one that finished, which is the kind of missing data that gets averaged
// into a conclusion.
func TestATruncatedLineIsAnErrorNotAnOmission(t *testing.T) {
	input := `{"schema":"alpr.bench/v1","tier":"B"}` + "\n" + `{"schema":"alpr.ben`
	if _, err := ReadLines(strings.NewReader(input)); err == nil {
		t.Error("a truncated line parsed as if the file were complete")
	}
}

func TestFromSnapshotWeightsQuantilesByCount(t *testing.T) {
	reg := metrics.NewSnapshotRegistry()
	hist := reg.Histogram(metrics.PipelineDuration, metrics.PipelineBuckets(), "camera")

	// One busy camera and one quiet one. A plain mean of the two p95s would give the quiet
	// camera the same say as the busy one, and the fleet's p95 would follow the camera nobody
	// is waiting on.
	busy := hist.With("busy")
	for i := 0; i < 1000; i++ {
		busy.Observe(0.02)
	}
	quiet := hist.With("quiet")
	quiet.Observe(8)

	q := FromSnapshot(reg.Snapshot(), metrics.PipelineDuration)
	if q.Count != 1001 {
		t.Fatalf("count is %d, want 1001", q.Count)
	}
	if q.P95 > 0.5 {
		t.Errorf("fleet p95 is %v, dragged up by a single observation on an idle camera\n\n"+
			"Quantiles from different series must be weighted by how many observations each "+
			"rests on, or one quiet camera outvotes a thousand frames.", q.P95)
	}
}

func TestDropsByReasonKeepsTheLabel(t *testing.T) {
	reg := metrics.NewSnapshotRegistry()
	dropped := reg.Counter(metrics.FramesDropped, "camera", "reason")
	dropped.With("a", metrics.DropInferenceBusy).Add(10)
	dropped.With("b", metrics.DropInferenceBusy).Add(5)
	dropped.With("a", metrics.DropStale).Add(2)

	drops := DropsByReason(reg.Snapshot())
	if drops[metrics.DropInferenceBusy] != 15 || drops[metrics.DropStale] != 2 {
		t.Fatalf("drops by reason came out as %v", drops)
	}
	if total := CounterTotal(reg.Snapshot(), metrics.FramesDropped); total != 17 {
		t.Fatalf("total is %v, want 17", total)
	}
}
