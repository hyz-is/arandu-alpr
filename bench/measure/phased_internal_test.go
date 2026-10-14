package measure

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/hyz-is/arandu-alpr/metrics"
)

// TestArmSplitsWarmupFromWindow.
//
// The whole reason this type exists: a run's p95 must describe the measurement window and not
// the sixty seconds of settling before it. Subtracting two snapshots recovers counters and
// cannot recover a quantile, so the split has to happen at write time.
func TestArmSplitsWarmupFromWindow(t *testing.T) {
	p := NewPhased()
	hist := p.Histogram(metrics.PipelineDuration, metrics.PipelineBuckets(), "camera").With("cam-1")
	count := p.Counter(metrics.FramesReceived, "camera").With("cam-1")

	// A cold start: slow, and exactly what must not reach the reported number.
	for i := 0; i < 100; i++ {
		hist.Observe(8)
		count.Inc()
	}

	p.Arm()

	for i := 0; i < 100; i++ {
		hist.Observe(0.02)
		count.Inc()
	}

	warm := find(t, p.Warmup(), metrics.PipelineDuration)
	window := find(t, p.Window(), metrics.PipelineDuration)

	if warm.Count != 100 || window.Count != 100 {
		t.Fatalf("observations split %d/%d, want 100/100", warm.Count, window.Count)
	}
	if window.Quantiles["p95"] > 0.5 {
		t.Errorf("the window p95 is %v: the warmup leaked into the measurement, which is the "+
			"one thing arming exists to prevent", window.Quantiles["p95"])
	}
	if warm.Quantiles["p95"] < 1 {
		t.Errorf("the warmup p95 is %v, so the slow phase went somewhere else", warm.Quantiles["p95"])
	}

	warmCount := find(t, p.Warmup(), metrics.FramesReceived)
	if warmCount.Value != 100 {
		t.Errorf("counters split wrongly: warmup holds %v", warmCount.Value)
	}
}

// TestGaugesAreWrittenToBothPhases.
//
// A gauge is a level, not a tally. A window that had never seen a Set would read zero, and a
// zero in `alpr_session_pool_in_use` says "no sessions were busy" -- which is the opposite of
// what a saturated fleet is doing.
func TestGaugesAreWrittenToBothPhases(t *testing.T) {
	p := NewPhased()
	gauge := p.Gauge(metrics.SessionPoolInUse, "gpu").With("0")
	gauge.Set(3)

	p.Arm()

	if got := find(t, p.Window(), metrics.SessionPoolInUse).Value; got != 3 {
		t.Errorf("the window sees %v sessions in use, not the 3 that were set before arming\n\n"+
			"A level set before the window opened is still the level during it.", got)
	}
}

// TestArmingIsSafeWhileTheFleetIsWriting.
//
// Arming happens with cameras running, because stopping them to switch registries would put a
// cold start inside the measurement. That makes it a data race unless the switch is atomic, and
// a race here corrupts the number the whole sweep rests on.
func TestArmingIsSafeWhileTheFleetIsWriting(t *testing.T) {
	p := NewPhased()
	hist := p.Histogram(metrics.OCRDuration, metrics.DefaultBuckets(), "camera").With("cam-1")
	counter := p.Counter(metrics.DetectedPlates, "camera").With("cam-1")

	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()

	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for ctx.Err() == nil {
				hist.Observe(0.01)
				counter.Inc()
			}
		}()
	}

	time.Sleep(50 * time.Millisecond)
	p.Arm()
	wg.Wait()

	if !p.Armed() {
		t.Fatal("Armed reports false after Arm")
	}
	total := find(t, p.Warmup(), metrics.DetectedPlates).Value + find(t, p.Window(), metrics.DetectedPlates).Value
	if total == 0 {
		t.Fatal("nothing was recorded in either phase")
	}
	if find(t, p.Window(), metrics.DetectedPlates).Value == 0 {
		t.Error("nothing landed in the window after arming")
	}
}

func find(t *testing.T, samples []metrics.Sample, name string) metrics.Sample {
	t.Helper()
	for _, s := range samples {
		if s.Name == name {
			return s
		}
	}
	t.Fatalf("no series named %q in %d samples", name, len(samples))
	return metrics.Sample{}
}
