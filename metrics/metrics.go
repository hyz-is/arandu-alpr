// Package metrics is the counting surface the pipeline writes to.
//
// It exists because the framework has nothing that fits. The observability package there is a
// deprecation bridge to hesape/log, whose Gauges is a map[GaugeName]int64 with Set and Read --
// no counters, no histograms, and no label beyond tenant. Five of the thirteen numbers this
// service needs are histograms, so the core defines its own neutral interface and lets whoever
// embeds it decide where the numbers go.
//
// Two decisions here are about the hot path, and both matter more than they look.
//
// Labels are bound ONCE, when a camera starts, not formatted per frame. The variadic form
// allocates and escapes to the heap; a bound handle is a pointer dereference. At a few thousand
// frames a second across a fleet that is the difference between a few nanoseconds and a few
// hundred plus garbage.
//
// Discard is the zero value and every method on it is a no-op. The pipeline never checks for
// nil, and a service running without a metrics backend runs the same code path as one with.
package metrics

import "time"

// Registry hands out metric families. Implementations must be safe for concurrent use.
type Registry interface {
	// Counter returns a family that only goes up.
	Counter(name string, labelNames ...string) CounterVec
	// Gauge returns a family that goes up and down.
	Gauge(name string, labelNames ...string) GaugeVec
	// Histogram returns a family bucketed by the given upper bounds, which must be sorted
	// ascending and must not include +Inf -- implementations add it.
	Histogram(name string, buckets []float64, labelNames ...string) HistogramVec
}

// The Vec types bind label values to get a concrete handle. Bind at start, use in the loop.
type (
	CounterVec interface {
		With(labelValues ...string) Counter
	}
	GaugeVec interface {
		With(labelValues ...string) Gauge
	}
	HistogramVec interface {
		With(labelValues ...string) Histogram
	}
)

type Counter interface {
	Inc()
	Add(delta float64)
}

type Gauge interface {
	Set(v float64)
	Add(delta float64)
}

type Histogram interface {
	Observe(v float64)
	// ObserveSince records seconds elapsed since t. It is the form the pipeline actually
	// uses, and having it here keeps `time.Since(x).Seconds()` out of every call site.
	ObserveSince(t time.Time)
}

// Metric names. They are constants because the pipeline writes them and the tests, the
// exporter and the benchmark harness read them, and a typo in any one of those is a number
// that silently never appears.
const (
	FramesReceived   = "alpr_frames_received_total"
	FramesDropped    = "alpr_frames_dropped_total"
	DetectorDuration = "alpr_detector_duration_seconds"
	OCRDuration      = "alpr_ocr_duration_seconds"
	PipelineDuration = "alpr_pipeline_duration_seconds"
	SessionWait      = "alpr_session_wait_seconds"
	SessionPoolInUse = "alpr_session_pool_in_use"
	DetectedPlates   = "alpr_detected_plates_total"
	Deduplicated     = "alpr_deduplicated_total"
	StreamReconnects = "alpr_stream_reconnects_total"
	InferenceErrors  = "alpr_inference_errors_total"
	CameraGPU        = "alpr_gpu_id"
	ModelVersion     = "alpr_model_version"
)

// Reasons for FramesDropped.
//
// This label is the difference between knowing "we are losing frames" and knowing "we are
// losing frames because the GPU is full". The Python service had neither, and the incident
// where throughput fell from 490 to 12 detections a minute took a code reading to diagnose --
// with this label it is one query.
const (
	DropStale            = "stale"             // older than the ingest lag limit
	DropBlank            = "blank"             // empty or single-colour buffer
	DropSkip             = "skip"              // inside the detection interval
	DropProducerReplaced = "producer_replaced" // a newer frame arrived first
	DropInferenceBusy    = "inference_busy"    // no session free before the deadline
)

// Stages for InferenceErrors.
const (
	StageDetect       = "detect"
	StageOCR          = "ocr"
	StageSessionBuild = "session_build"
)

// DefaultBuckets covers a single inference or pool wait: sub-millisecond to ten seconds.
func DefaultBuckets() []float64 {
	return []float64{.001, .0025, .005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10}
}

// PipelineBuckets extends to 30s because the staleness limit is 8s and an operator has to be
// able to see what lands beyond it. A histogram whose last bucket is below the threshold you
// alert on cannot answer the question you are asking it.
func PipelineBuckets() []float64 {
	return []float64{.01, .05, .1, .25, .5, 1, 2, 4, 8, 15, 30}
}

// Discard is a Registry that counts nothing. It is the default.
type discard struct{}

// Discard returns the no-op Registry.
func Discard() Registry { return discard{} }

type nopHandle struct{}

func (nopHandle) Inc()                   {}
func (nopHandle) Add(float64)            {}
func (nopHandle) Set(float64)            {}
func (nopHandle) Observe(float64)        {}
func (nopHandle) ObserveSince(time.Time) {}

func (discard) Counter(string, ...string) CounterVec                { return nopCounterVec{} }
func (discard) Gauge(string, ...string) GaugeVec                    { return nopGaugeVec{} }
func (discard) Histogram(string, []float64, ...string) HistogramVec { return nopHistVec{} }

type nopCounterVec struct{}

func (nopCounterVec) With(...string) Counter { return nopHandle{} }

type nopGaugeVec struct{}

func (nopGaugeVec) With(...string) Gauge { return nopHandle{} }

type nopHistVec struct{}

func (nopHistVec) With(...string) Histogram { return nopHandle{} }
