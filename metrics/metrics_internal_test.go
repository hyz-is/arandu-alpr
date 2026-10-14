package metrics

import (
	"math"
	"sync"
	"testing"
	"time"
)

func find(t *testing.T, samples []Sample, name string, labels map[string]string) Sample {
	t.Helper()
outer:
	for _, s := range samples {
		if s.Name != name {
			continue
		}
		for k, v := range labels {
			if s.Labels[k] != v {
				continue outer
			}
		}
		return s
	}
	t.Fatalf("no sample %q with labels %v in %v", name, labels, samples)
	return Sample{}
}

func TestDiscardIsUsableWithoutAnyNilCheck(t *testing.T) {
	// The pipeline calls these on every frame and never tests for nil. If Discard ever handed
	// back a nil handle, the service would run fine with metrics configured and panic without
	// -- which is the configuration nobody runs in development.
	r := Discard()
	r.Counter(FramesReceived, "camera").With("cam1").Inc()
	r.Counter(FramesDropped, "camera", "reason").With("cam1", DropStale).Add(3)
	r.Gauge(SessionPoolInUse, "gpu").With("0").Set(2)
	r.Gauge(SessionPoolInUse, "gpu").With("0").Add(-1)
	h := r.Histogram(OCRDuration, DefaultBuckets(), "gpu").With("0")
	h.Observe(0.01)
	h.ObserveSince(time.Now())
}

func TestSameNameReturnsTheSameFamily(t *testing.T) {
	// Two cameras bind their own handles from the same registry. If each call created a fresh
	// family, every camera would keep a private copy and Snapshot would report the last one
	// registered -- a number that is real, just not the total.
	r := NewSnapshotRegistry()
	r.Counter(FramesReceived, "camera").With("a").Add(2)
	r.Counter(FramesReceived, "camera").With("b").Add(5)

	got := r.Snapshot()
	if v := find(t, got, FramesReceived, map[string]string{"camera": "a"}).Value; v != 2 {
		t.Errorf("camera a = %v, want 2", v)
	}
	if v := find(t, got, FramesReceived, map[string]string{"camera": "b"}).Value; v != 5 {
		t.Errorf("camera b = %v, want 5", v)
	}
}

func TestLabelValuesDoNotCollideAcrossBoundaries(t *testing.T) {
	// {"a","bc"} and {"ab","c"} must be different series. Joining label values without a
	// separator merges them, and the symptom is two cameras' counts added together -- with
	// nothing anywhere looking wrong.
	r := NewSnapshotRegistry()
	r.Counter("x", "l", "m").With("a", "bc").Add(1)
	r.Counter("x", "l", "m").With("ab", "c").Add(10)

	got := r.Snapshot()
	if v := find(t, got, "x", map[string]string{"l": "a", "m": "bc"}).Value; v != 1 {
		t.Errorf("{a,bc} = %v, want 1", v)
	}
	if v := find(t, got, "x", map[string]string{"l": "ab", "m": "c"}).Value; v != 10 {
		t.Errorf("{ab,c} = %v, want 10", v)
	}
}

func TestBoundHandleIsStableAcrossCalls(t *testing.T) {
	// The whole point of binding once at camera start is that the handle stays the same
	// object. If With allocated a new series each time, the counts would scatter.
	r := NewSnapshotRegistry()
	v := r.Counter(FramesReceived, "camera")
	if v.With("a") != v.With("a") {
		t.Fatal("With returned a different handle for the same labels")
	}
}

func TestConcurrentAddLosesNothing(t *testing.T) {
	// Every camera goroutine writes to its own handle, but the pool gauge and the per-GPU
	// histograms are written by many. A lost update here is a metric that reads low forever
	// and can never be reconciled against anything.
	r := NewSnapshotRegistry()
	c := r.Counter(FramesReceived, "camera").With("a")

	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 200; j++ {
				c.Inc()
			}
		}()
	}
	wg.Wait()

	if v := find(t, r.Snapshot(), FramesReceived, map[string]string{"camera": "a"}).Value; v != 10000 {
		t.Errorf("counter = %v, want 10000", v)
	}
}

func TestHistogramReportsQuantilesAndTotals(t *testing.T) {
	r := NewSnapshotRegistry()
	h := r.Histogram(OCRDuration, []float64{1, 2, 3, 4, 5}, "gpu").With("0")
	// 100 observations spread evenly over the buckets.
	for i := 1; i <= 100; i++ {
		h.Observe(float64(i) / 20.0) // 0.05 .. 5.0
	}
	s := find(t, r.Snapshot(), OCRDuration, map[string]string{"gpu": "0"})
	if s.Count != 100 {
		t.Errorf("count = %d, want 100", s.Count)
	}
	if math.Abs(s.Sum-252.5) > 1e-6 {
		t.Errorf("sum = %v, want 252.5", s.Sum)
	}
	// Bucket resolution is coarse by construction; assert the ordering and the range rather
	// than an exact value, because the exact value is a property of the bucket boundaries.
	if !(s.Quantiles["p50"] < s.Quantiles["p95"] && s.Quantiles["p95"] <= s.Quantiles["p99"]) {
		t.Errorf("quantiles are not ordered: %v", s.Quantiles)
	}
	if s.Quantiles["p50"] < 2 || s.Quantiles["p50"] > 3 {
		t.Errorf("p50 = %v, want it in the middle bucket", s.Quantiles["p50"])
	}
}

func TestNaNDoesNotInflateQuantiles(t *testing.T) {
	// A NaN duration means a clock went backwards or a timer was never started. Bucketing it
	// as +Inf would push every upper quantile to the last bound, which reads as "the service
	// is slow" and sends someone looking at the GPU.
	r := NewSnapshotRegistry()
	h := r.Histogram(OCRDuration, []float64{1, 2, 3}, "gpu").With("0")
	for i := 0; i < 99; i++ {
		h.Observe(0.5)
	}
	h.Observe(math.NaN())

	s := find(t, r.Snapshot(), OCRDuration, map[string]string{"gpu": "0"})
	if s.Quantiles["p95"] > 1 {
		t.Errorf("p95 = %v; one NaN pushed the quantile past the real data", s.Quantiles["p95"])
	}
}

func TestEmptyHistogramReportsZeroRatherThanPanicking(t *testing.T) {
	// Snapshot runs against a service that just booted, where most series have no
	// observations yet. It must not divide by zero on the way to the operator.
	r := NewSnapshotRegistry()
	r.Histogram(SessionWait, DefaultBuckets(), "gpu").With("0")
	s := find(t, r.Snapshot(), SessionWait, map[string]string{"gpu": "0"})
	if s.Count != 0 || s.Quantiles["p99"] != 0 {
		t.Errorf("empty histogram reported count=%d p99=%v", s.Count, s.Quantiles["p99"])
	}
}

func TestPipelineBucketsReachPastTheStalenessLimit(t *testing.T) {
	// The camera loop discards frames older than 8s. A histogram whose last bound is 8s can
	// only ever say "8 or more", so the one question worth asking about a lagging fleet --
	// how far past the limit are we -- becomes unanswerable.
	b := PipelineBuckets()
	if b[len(b)-1] <= 8 {
		t.Fatalf("last bucket is %v; it must reach past the 8s staleness limit", b[len(b)-1])
	}
}

func TestGaugeGoesBothWays(t *testing.T) {
	// SessionPoolInUse is incremented on borrow and decremented on release. A gauge that only
	// climbs would show a permanently exhausted pool.
	r := NewSnapshotRegistry()
	g := r.Gauge(SessionPoolInUse, "gpu").With("1")
	g.Add(3)
	g.Add(-2)
	if v := find(t, r.Snapshot(), SessionPoolInUse, map[string]string{"gpu": "1"}).Value; v != 1 {
		t.Errorf("gauge = %v, want 1", v)
	}
}
