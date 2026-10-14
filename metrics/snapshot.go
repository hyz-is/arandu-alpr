package metrics

import (
	"math"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// SnapshotRegistry is a Registry that keeps the numbers in memory and hands them back.
//
// It is what the benchmark harness reads and what the optional HTTP route serves, and it exists
// so that neither of those needs Prometheus. A Prometheus exporter is a separate concern and
// lives outside the core, where its dependency does not land in the go.sum of anyone importing
// only the recognition pipeline.
//
// Everything on the write path is atomic and lock-free after the handle is bound. The mutex is
// taken when a new label combination appears -- once per camera at start, not per frame.
type SnapshotRegistry struct {
	mu       sync.RWMutex
	counters map[string]*counterFamily
	gauges   map[string]*gaugeFamily
	hists    map[string]*histFamily
}

func NewSnapshotRegistry() *SnapshotRegistry {
	return &SnapshotRegistry{
		counters: map[string]*counterFamily{},
		gauges:   map[string]*gaugeFamily{},
		hists:    map[string]*histFamily{},
	}
}

// labelKey joins label values with a separator that cannot appear in an identifier we emit.
//
// A plain concatenation would make {"a","bc"} and {"ab","c"} the same series, which is a
// collision that shows up as two cameras' numbers added together and nothing looking wrong.
func labelKey(values []string) string {
	return strings.Join(values, "\x00")
}

type counterFamily struct {
	labelNames []string
	mu         sync.RWMutex
	series     map[string]*counterSeries
}

type counterSeries struct {
	labels []string
	bits   atomic.Uint64 // float64 bits
}

func (c *counterSeries) Inc() { c.Add(1) }

func (c *counterSeries) Add(delta float64) {
	for {
		old := c.bits.Load()
		next := math.Float64bits(math.Float64frombits(old) + delta)
		if c.bits.CompareAndSwap(old, next) {
			return
		}
	}
}

func (c *counterSeries) value() float64 { return math.Float64frombits(c.bits.Load()) }

func (f *counterFamily) With(values ...string) Counter {
	key := labelKey(values)
	f.mu.RLock()
	s := f.series[key]
	f.mu.RUnlock()
	if s != nil {
		return s
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if s = f.series[key]; s != nil {
		return s
	}
	s = &counterSeries{labels: append([]string(nil), values...)}
	f.series[key] = s
	return s
}

type gaugeFamily struct {
	labelNames []string
	mu         sync.RWMutex
	series     map[string]*gaugeSeries
}

type gaugeSeries struct {
	labels []string
	bits   atomic.Uint64
}

func (g *gaugeSeries) Set(v float64) { g.bits.Store(math.Float64bits(v)) }

func (g *gaugeSeries) Add(delta float64) {
	for {
		old := g.bits.Load()
		next := math.Float64bits(math.Float64frombits(old) + delta)
		if g.bits.CompareAndSwap(old, next) {
			return
		}
	}
}

func (g *gaugeSeries) value() float64 { return math.Float64frombits(g.bits.Load()) }

func (f *gaugeFamily) With(values ...string) Gauge {
	key := labelKey(values)
	f.mu.RLock()
	s := f.series[key]
	f.mu.RUnlock()
	if s != nil {
		return s
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if s = f.series[key]; s != nil {
		return s
	}
	s = &gaugeSeries{labels: append([]string(nil), values...)}
	f.series[key] = s
	return s
}

type histFamily struct {
	labelNames []string
	buckets    []float64
	mu         sync.RWMutex
	series     map[string]*histSeries
}

// histSeries counts observations per bucket, plus sum and count.
//
// Bucket counts are NOT cumulative in storage -- each observation lands in exactly one slot.
// Cumulating happens when reading. Storing cumulative counts would mean touching every bucket
// above the observed value on every write, which turns one atomic add into a dozen.
type histSeries struct {
	labels  []string
	buckets []float64       // upper bounds, ascending; +Inf is implicit as the last slot
	counts  []atomic.Uint64 // len(buckets)+1
	sumBits atomic.Uint64
	count   atomic.Uint64
}

func (h *histSeries) Observe(v float64) {
	// A NaN observation belongs in no bucket. Counting it as +Inf would quietly inflate every
	// upper quantile, so it goes to the count and sum only -- where a NaN sum is visible.
	i := sort.SearchFloat64s(h.buckets, v)
	if v != v { // NaN
		i = len(h.buckets)
	}
	h.counts[i].Add(1)
	h.count.Add(1)
	for {
		old := h.sumBits.Load()
		next := math.Float64bits(math.Float64frombits(old) + v)
		if h.sumBits.CompareAndSwap(old, next) {
			return
		}
	}
}

func (h *histSeries) ObserveSince(t time.Time) { h.Observe(time.Since(t).Seconds()) }

func (f *histFamily) With(values ...string) Histogram {
	key := labelKey(values)
	f.mu.RLock()
	s := f.series[key]
	f.mu.RUnlock()
	if s != nil {
		return s
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if s = f.series[key]; s != nil {
		return s
	}
	s = &histSeries{
		labels:  append([]string(nil), values...),
		buckets: f.buckets,
		counts:  make([]atomic.Uint64, len(f.buckets)+1),
	}
	f.series[key] = s
	return s
}

func (r *SnapshotRegistry) Counter(name string, labelNames ...string) CounterVec {
	r.mu.Lock()
	defer r.mu.Unlock()
	if f := r.counters[name]; f != nil {
		return f
	}
	f := &counterFamily{labelNames: labelNames, series: map[string]*counterSeries{}}
	r.counters[name] = f
	return f
}

func (r *SnapshotRegistry) Gauge(name string, labelNames ...string) GaugeVec {
	r.mu.Lock()
	defer r.mu.Unlock()
	if f := r.gauges[name]; f != nil {
		return f
	}
	f := &gaugeFamily{labelNames: labelNames, series: map[string]*gaugeSeries{}}
	r.gauges[name] = f
	return f
}

func (r *SnapshotRegistry) Histogram(name string, buckets []float64, labelNames ...string) HistogramVec {
	r.mu.Lock()
	defer r.mu.Unlock()
	if f := r.hists[name]; f != nil {
		return f
	}
	f := &histFamily{
		labelNames: labelNames,
		buckets:    append([]float64(nil), buckets...),
		series:     map[string]*histSeries{},
	}
	r.hists[name] = f
	return f
}

// Sample is one series of one metric, as read out.
type Sample struct {
	Name   string
	Labels map[string]string
	Value  float64 // counters and gauges

	// Histograms only.
	Count     uint64
	Sum       float64
	Quantiles map[string]float64 // "p50", "p95", "p99"
}

// Snapshot reads every series. It is a point-in-time read, not a consistent one: series are
// read one at a time and the pipeline keeps writing. For diagnosis and benchmarking that is
// what you want -- a globally consistent read would need a stop-the-world the hot path cannot
// afford.
func (r *SnapshotRegistry) Snapshot() []Sample {
	r.mu.RLock()
	counters := make(map[string]*counterFamily, len(r.counters))
	for k, v := range r.counters {
		counters[k] = v
	}
	gauges := make(map[string]*gaugeFamily, len(r.gauges))
	for k, v := range r.gauges {
		gauges[k] = v
	}
	hists := make(map[string]*histFamily, len(r.hists))
	for k, v := range r.hists {
		hists[k] = v
	}
	r.mu.RUnlock()

	var out []Sample
	for name, f := range counters {
		f.mu.RLock()
		for _, s := range f.series {
			out = append(out, Sample{Name: name, Labels: zip(f.labelNames, s.labels), Value: s.value()})
		}
		f.mu.RUnlock()
	}
	for name, f := range gauges {
		f.mu.RLock()
		for _, s := range f.series {
			out = append(out, Sample{Name: name, Labels: zip(f.labelNames, s.labels), Value: s.value()})
		}
		f.mu.RUnlock()
	}
	for name, f := range hists {
		f.mu.RLock()
		for _, s := range f.series {
			out = append(out, Sample{
				Name:   name,
				Labels: zip(f.labelNames, s.labels),
				Count:  s.count.Load(),
				Sum:    math.Float64frombits(s.sumBits.Load()),
				Quantiles: map[string]float64{
					"p50": s.quantile(0.50),
					"p95": s.quantile(0.95),
					"p99": s.quantile(0.99),
				},
			})
		}
		f.mu.RUnlock()
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Name != out[j].Name {
			return out[i].Name < out[j].Name
		}
		return labelKey(sortedValues(out[i].Labels)) < labelKey(sortedValues(out[j].Labels))
	})
	return out
}

// quantile interpolates linearly inside the bucket the quantile falls in.
//
// The result is only as precise as the bucket boundaries, and an observation past the last
// bound can only be reported as that bound -- which is why PipelineBuckets reaches past the
// staleness limit rather than stopping at it.
func (h *histSeries) quantile(q float64) float64 {
	total := h.count.Load()
	if total == 0 {
		return 0
	}
	want := q * float64(total)
	var cum uint64
	for i := range h.counts {
		c := h.counts[i].Load()
		if float64(cum+c) < want {
			cum += c
			continue
		}
		if i >= len(h.buckets) {
			// Beyond the last bound: the honest answer is the bound itself.
			if len(h.buckets) == 0 {
				return 0
			}
			return h.buckets[len(h.buckets)-1]
		}
		lower := 0.0
		if i > 0 {
			lower = h.buckets[i-1]
		}
		if c == 0 {
			return h.buckets[i]
		}
		frac := (want - float64(cum)) / float64(c)
		return lower + frac*(h.buckets[i]-lower)
	}
	return h.buckets[len(h.buckets)-1]
}

func zip(names, values []string) map[string]string {
	m := make(map[string]string, len(names))
	for i, n := range names {
		if i < len(values) {
			m[n] = values[i]
		}
	}
	return m
}

func sortedValues(m map[string]string) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := make([]string, 0, len(keys))
	for _, k := range keys {
		out = append(out, k+"="+m[k])
	}
	return out
}
