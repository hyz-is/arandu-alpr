// Package report is the line a benchmark run leaves behind.
//
// One run is ONE line of JSON, and everything needed to argue about it is inside that line.
// That is not tidiness: a throughput number without the configuration that produced it, the
// commit that produced it and the model files that produced it cannot be compared to anything.
// The Python service was benchmarked that way -- "we ran 40 cameras and it looked fine" -- and
// the number could never be reproduced, because nobody could say what 40 cameras meant.
//
// So the config travels as the SERVICE'S OWN structs rather than as a hand-written mirror of
// them. A mirror drifts the first time a knob is added and nobody notices, and the failure is
// silent in the worst possible way: two lines that look comparable and are not.
package report

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
	"time"

	"github.com/hyz-is/arandu-alpr"
	"github.com/hyz-is/arandu-alpr/engine"
	"github.com/hyz-is/arandu-alpr/metrics"
)

// Schema is stamped on every line.
//
// It is compared before anything else: a compare across a schema change is a compare of two
// different measurements, and the honest answer is to refuse rather than to diff whatever
// fields happen to still line up.
const Schema = "alpr.bench/v1"

// Run is one measurement window, start to finish.
type Run struct {
	Schema string `json:"schema"`

	// Tier is "A" for the pipeline suite that runs anywhere, "B" for the fleet sweep that
	// needs a GPU. They measure different things and must never be diffed against each other.
	Tier string `json:"tier"`

	StartedAt time.Time `json:"started_at"`
	// Elapsed is the MEASUREMENT window only. Warmup is not in it, by construction.
	Elapsed float64 `json:"elapsed_seconds"`

	// GitSHA and GitDirty identify the code. A dirty tree is recorded rather than refused:
	// measuring an uncommitted fix is the normal case, and the flag is what stops the number
	// from later being attributed to the commit it was not.
	GitSHA   string `json:"git_sha"`
	GitDirty bool   `json:"git_dirty"`

	Host Host `json:"host"`

	Config Config  `json:"config"`
	Models []Model `json:"models"`

	Throughput Throughput           `json:"throughput"`
	Latency    map[string]Quantiles `json:"latency_seconds"`

	// Drops is keyed by the metrics.Drop* reason. An unlabelled total cannot distinguish a
	// fleet that is out of GPU from one whose cameras are late, and those two need opposite
	// remedies.
	Drops map[string]float64 `json:"frames_dropped_by_reason"`

	GPUs    []GPU   `json:"gpus"`
	Process Process `json:"process"`

	SLO SLO `json:"slo"`

	// Notes carries anything that could not be measured, in the words of whatever failed to
	// measure it. An absent nvidia-smi must show up as a sentence, never as a zero that reads
	// like "no VRAM used".
	Notes []string `json:"notes,omitempty"`
}

// Host is what the run happened on.
type Host struct {
	Hostname string `json:"hostname"`
	OS       string `json:"os"`
	Arch     string `json:"arch"`
	NumCPU   int    `json:"num_cpu"`
	GoVer    string `json:"go_version"`
}

// Config is the whole configuration, carried as the service's own types.
//
// See the package comment: a restated copy of these fields is a copy that goes stale, and a
// stale copy makes two incomparable runs look comparable.
type Config struct {
	Cameras           int   `json:"cameras"`
	Devices           []int `json:"devices"`
	SessionsPerDevice int   `json:"detector_sessions_per_device"`
	OCRSessions       int   `json:"ocr_sessions_per_device"`

	BorrowTimeoutMS     float64 `json:"borrow_timeout_ms"`
	MinDetectIntervalMS float64 `json:"min_detect_interval_ms"`
	MaxIngestLagMS      float64 `json:"max_ingest_lag_ms"`
	DetectTimeoutMS     float64 `json:"detect_timeout_ms"`
	RecognizeTimeoutMS  float64 `json:"recognize_timeout_ms"`

	// HWAccel is the hardware decoder, and it belongs in the record because a run with NVDEC and
	// one without are not the same experiment. Decoding a 1080p30 stream on the CPU costs 5.6% of
	// a core -- thirteen times what the whole Go pipeline costs for the same frames -- so a
	// cameras-per-GPU number measured without it can be a fact about the host rather than about
	// the card.
	HWAccel string `json:"hwaccel"`

	IncludeFullImage bool `json:"include_full_image"`
	IncludeCropImage bool `json:"include_crop_image"`
	JPEGQuality      int  `json:"jpeg_quality"`

	Gate        alpr.GateConfig          `json:"gate"`
	Postprocess engine.PostprocessConfig `json:"postprocess"`
	Dedup       Dedup                    `json:"dedup"`

	WarmupSeconds float64 `json:"warmup_seconds"`
	WindowSeconds float64 `json:"window_seconds"`

	// Clips and RTSPBase describe the INPUT, which is half of what a throughput number means.
	// Two runs over different footage are two different questions.
	Clips    []string `json:"clips"`
	RTSPBase string   `json:"rtsp_base"`

	// Realtime records whether the publisher paced the input with ffmpeg's -re.
	//
	// Without it a clip arrives as one burst and every rate derived from it is meaningless --
	// which is the single easiest way to produce a benchmark that says nothing. It is recorded
	// so a line that was produced without pacing can be thrown away rather than believed.
	Realtime bool `json:"realtime_paced"`
}

// Dedup is the cooldown the run used. It decides how many readings reach the sink, so a run
// with a different cooldown answers a different question.
type Dedup struct {
	CooldownSeconds float64 `json:"cooldown_seconds"`
	MaxDistance     int     `json:"max_distance"`
}

// Model identifies one artefact by content, not by name.
//
// The SHA256 is the whole point. A file called `detector.ort` is not a model; a specific 47 MB
// of bytes is, and re-exporting one under the same name is the ordinary way two runs stop
// meaning the same thing without anything looking different.
type Model struct {
	Role   string `json:"role"`
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
	Bytes  int64  `json:"bytes"`
}

// Throughput is what came out.
type Throughput struct {
	FramesReceived     float64 `json:"frames_received"`
	FramesPerSecond    float64 `json:"frames_per_second"`
	DetectionsTotal    float64 `json:"detections_total"`
	DetectionsPerMin   float64 `json:"detections_per_minute"`
	DeduplicatedTotal  float64 `json:"deduplicated_total"`
	FramesDroppedTotal float64 `json:"frames_dropped_total"`
}

// Quantiles is one latency histogram, read out.
//
// Count and Sum travel beside the quantiles because a p95 over eleven samples is not a p95, and
// nothing else in the line would show that.
type Quantiles struct {
	Count uint64  `json:"count"`
	Sum   float64 `json:"sum"`
	P50   float64 `json:"p50"`
	P95   float64 `json:"p95"`
	P99   float64 `json:"p99"`
}

// Mean is the only exact statistic a bucketed histogram has. The quantiles are interpolated
// inside their buckets and are no more precise than the bounds.
func (q Quantiles) Mean() float64 {
	if q.Count == 0 {
		return 0
	}
	return q.Sum / float64(q.Count)
}

// GPU is one device's memory over the window.
//
// Peak and steady are both here because they answer different questions: peak decides whether
// the box OOMs on a bad second, steady decides how many more sessions fit.
type GPU struct {
	Index       int     `json:"index"`
	Name        string  `json:"name"`
	PeakMiB     float64 `json:"vram_peak_mib"`
	SteadyMiB   float64 `json:"vram_steady_mib"`
	TotalMiB    float64 `json:"vram_total_mib"`
	Samples     int     `json:"samples"`
	UtilPercent float64 `json:"util_mean_percent"`
	UtilPeakPct float64 `json:"util_peak_percent"`
}

// Process is what the service itself cost the host.
type Process struct {
	PeakRSSMiB   float64 `json:"rss_peak_mib"`
	SteadyRSSMiB float64 `json:"rss_steady_mib"`
	CPUPercent   float64 `json:"cpu_mean_percent"`
	Goroutines   int     `json:"goroutines_end"`
	HeapMiB      float64 `json:"go_heap_mib_end"`
}

// SLO is the declared bar and whether this run cleared it.
//
// The sweep's answer -- "cameras per GPU" -- is exactly the last Cameras value whose Crossed is
// false. Writing the bar into the line is what makes that answer checkable later by somebody
// who was not there.
type SLO struct {
	Metric             string  `json:"metric"`
	P95BudgetSeconds   float64 `json:"p95_budget_seconds"`
	P95ObservedSeconds float64 `json:"p95_observed_seconds"`
	Crossed            bool    `json:"crossed"`
	// MaxBusyDropRatio is the second half of the bar. A fleet can hold its p95 by throwing
	// away most of its frames, and a latency-only SLO calls that a pass.
	MaxBusyDropRatio      float64 `json:"max_busy_drop_ratio"`
	ObservedBusyDropRatio float64 `json:"observed_busy_drop_ratio"`
}

// Quantiles reads one histogram family out of a snapshot, summing across label sets.
//
// Summing counts and sums across cameras is exact. Summing QUANTILES is not, and this returns
// the count-weighted mean of the per-series quantiles rather than pretending otherwise -- with
// Count carried alongside so a reader can see how much it rests on.
func FromSnapshot(samples []metrics.Sample, name string) Quantiles {
	var q Quantiles
	var wp50, wp95, wp99 float64
	for _, s := range samples {
		if s.Name != name || s.Count == 0 {
			continue
		}
		w := float64(s.Count)
		q.Count += s.Count
		q.Sum += s.Sum
		wp50 += s.Quantiles["p50"] * w
		wp95 += s.Quantiles["p95"] * w
		wp99 += s.Quantiles["p99"] * w
	}
	if q.Count == 0 {
		return q
	}
	total := float64(q.Count)
	q.P50, q.P95, q.P99 = wp50/total, wp95/total, wp99/total
	return q
}

// CounterTotal sums one counter family across every label set.
func CounterTotal(samples []metrics.Sample, name string) float64 {
	var total float64
	for _, s := range samples {
		if s.Name == name {
			total += s.Value
		}
	}
	return total
}

// DropsByReason groups the dropped-frame counter by its reason label.
func DropsByReason(samples []metrics.Sample) map[string]float64 {
	out := map[string]float64{}
	for _, s := range samples {
		if s.Name != metrics.FramesDropped {
			continue
		}
		out[s.Labels["reason"]] += s.Value
	}
	return out
}

// WriteLine appends one run as a single line of JSON.
//
// One line, never indented, because the file is meant to be `tail`ed, `grep`ed and fed to
// compare one line at a time. A pretty-printed run is a run that cannot be diffed with the
// tools already on the box.
func WriteLine(w io.Writer, r Run) error {
	r.Schema = Schema
	b, err := json.Marshal(r)
	if err != nil {
		return err
	}
	b = append(b, '\n')
	_, err = w.Write(b)
	return err
}

// AppendFile writes a run to a path, creating it if needed. An empty path writes to stdout.
func AppendFile(path string, r Run) error {
	if path == "" {
		return WriteLine(os.Stdout, r)
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	return WriteLine(f, r)
}

// ErrNoRuns is returned when a file holds no parseable run.
var ErrNoRuns = errors.New("report: no runs in input")

// ReadLines parses every run in a file.
//
// A blank line is skipped and a malformed one is an error rather than a silent omission: a
// truncated line is exactly what a killed sweep leaves behind, and quietly dropping it would
// make an interrupted run look like a shorter one.
func ReadLines(r io.Reader) ([]Run, error) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)

	var runs []Run
	for line := 1; scanner.Scan(); line++ {
		raw := scanner.Bytes()
		if len(trimSpace(raw)) == 0 {
			continue
		}
		var run Run
		if err := json.Unmarshal(raw, &run); err != nil {
			return nil, fmt.Errorf("report: line %d: %w", line, err)
		}
		runs = append(runs, run)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(runs) == 0 {
		return nil, ErrNoRuns
	}
	return runs, nil
}

func trimSpace(b []byte) []byte {
	start := 0
	for start < len(b) && (b[start] == ' ' || b[start] == '\t' || b[start] == '\r') {
		start++
	}
	end := len(b)
	for end > start && (b[end-1] == ' ' || b[end-1] == '\t' || b[end-1] == '\r') {
		end--
	}
	return b[start:end]
}

// SortedDropReasons returns the reasons present, in a stable order, so two printings of the
// same run cannot differ.
func SortedDropReasons(drops map[string]float64) []string {
	out := make([]string, 0, len(drops))
	for k := range drops {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
