// Package measure holds the instruments a benchmark run needs and the service does not.
//
// Nothing here is imported by the pipeline. It is deliberately outside app/Services so that a
// benchmark's conveniences -- shelling out to nvidia-smi, reading /proc, running git -- never
// become something the service links.
package measure

import (
	"sync/atomic"
	"time"

	"github.com/hyz-is/arandu-alpr/metrics"
)

// Phased is a Registry with two backing snapshots and a switch between them.
//
// It exists for one number: the p95 of the MEASUREMENT window, with warmup excluded.
//
// Subtracting a warmup snapshot from a final one works for counters and does not work for
// quantiles -- a histogram's percentile cannot be recovered from two readings of it, because
// the bucket counts are not exposed and would not subtract into a percentile anyway. Running
// warmup and window against two separate registries is the only way to get a clean answer, and
// handles are bound once at camera start, so the switch has to live inside the handle.
//
// The cost is one atomic load per observation, on a path that already does an atomic add.
//
// Arming is one-way per run. Two-way would let a caller interleave phases, and a quantile over
// an interleaved set is a quantile over something nobody can name afterwards.
type Phased struct {
	warmup *metrics.SnapshotRegistry
	window *metrics.SnapshotRegistry
	armed  *atomic.Bool
}

var _ metrics.Registry = (*Phased)(nil)

// NewPhased returns a registry writing to its warmup snapshot until Arm is called.
func NewPhased() *Phased {
	return &Phased{
		warmup: metrics.NewSnapshotRegistry(),
		window: metrics.NewSnapshotRegistry(),
		armed:  new(atomic.Bool),
	}
}

// Arm sends every subsequent observation to the window snapshot.
//
// It is safe to call while the fleet is running -- that is the whole point, since stopping the
// cameras to switch registries would put a cold start inside the measurement.
func (p *Phased) Arm() { p.armed.Store(true) }

// Armed reports whether the window has started.
func (p *Phased) Armed() bool { return p.armed.Load() }

// Warmup returns the discarded phase. It is exposed because a warmup that looks nothing like
// the window is itself a finding: it usually means the fleet had not settled when the window
// opened.
func (p *Phased) Warmup() []metrics.Sample { return p.warmup.Snapshot() }

// Window returns the measured phase.
func (p *Phased) Window() []metrics.Sample { return p.window.Snapshot() }

func (p *Phased) Counter(name string, labelNames ...string) metrics.CounterVec {
	return phasedCounterVec{
		warmup: p.warmup.Counter(name, labelNames...),
		window: p.window.Counter(name, labelNames...),
		armed:  p.armed,
	}
}

func (p *Phased) Gauge(name string, labelNames ...string) metrics.GaugeVec {
	return phasedGaugeVec{
		warmup: p.warmup.Gauge(name, labelNames...),
		window: p.window.Gauge(name, labelNames...),
		armed:  p.armed,
	}
}

func (p *Phased) Histogram(name string, buckets []float64, labelNames ...string) metrics.HistogramVec {
	return phasedHistVec{
		warmup: p.warmup.Histogram(name, buckets, labelNames...),
		window: p.window.Histogram(name, buckets, labelNames...),
		armed:  p.armed,
	}
}

// Both handles are bound eagerly, at With time, rather than resolved per observation. Binding
// is what the metrics package's whole hot-path design is about; deferring it here would undo
// that for every camera in the fleet being measured.
type phasedCounterVec struct {
	warmup, window metrics.CounterVec
	armed          *atomic.Bool
}

func (v phasedCounterVec) With(labelValues ...string) metrics.Counter {
	return phasedCounter{warmup: v.warmup.With(labelValues...), window: v.window.With(labelValues...), armed: v.armed}
}

type phasedCounter struct {
	warmup, window metrics.Counter
	armed          *atomic.Bool
}

func (c phasedCounter) Inc() { c.target().Inc() }

func (c phasedCounter) Add(delta float64) { c.target().Add(delta) }

func (c phasedCounter) target() metrics.Counter {
	if c.armed.Load() {
		return c.window
	}
	return c.warmup
}

type phasedGaugeVec struct {
	warmup, window metrics.GaugeVec
	armed          *atomic.Bool
}

func (v phasedGaugeVec) With(labelValues ...string) metrics.Gauge {
	return phasedGauge{warmup: v.warmup.With(labelValues...), window: v.window.With(labelValues...), armed: v.armed}
}

type phasedGauge struct {
	warmup, window metrics.Gauge
	armed          *atomic.Bool
}

// Set writes to BOTH phases, unlike Inc.
//
// A gauge is a level, not a tally: sessions in use, the GPU a camera sits on. A window snapshot
// that had never seen a Set would read zero and be understood as "no sessions in use", which is
// a lie of exactly the kind this whole package exists to avoid.
func (g phasedGauge) Set(v float64) {
	g.warmup.Set(v)
	g.window.Set(v)
}

func (g phasedGauge) Add(delta float64) {
	g.warmup.Add(delta)
	g.window.Add(delta)
}

type phasedHistVec struct {
	warmup, window metrics.HistogramVec
	armed          *atomic.Bool
}

func (v phasedHistVec) With(labelValues ...string) metrics.Histogram {
	return phasedHist{warmup: v.warmup.With(labelValues...), window: v.window.With(labelValues...), armed: v.armed}
}

type phasedHist struct {
	warmup, window metrics.Histogram
	armed          *atomic.Bool
}

func (h phasedHist) Observe(v float64) { h.target().Observe(v) }

func (h phasedHist) ObserveSince(t time.Time) { h.target().ObserveSince(t) }

func (h phasedHist) target() metrics.Histogram {
	if h.armed.Load() {
		return h.window
	}
	return h.warmup
}
