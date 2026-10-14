package measure

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"os/exec"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// FileSHA256 hashes a file and reports its size.
//
// A model is its bytes, not its filename. Two runs against `detector.ort` are two runs against
// whatever was in that path at the time, and a re-export under the same name is the ordinary
// way a comparison quietly stops being one.
func FileSHA256(path string) (sum string, size int64, err error) {
	f, err := os.Open(path)
	if err != nil {
		return "", 0, err
	}
	defer f.Close()

	info, err := f.Stat()
	if err != nil {
		return "", 0, err
	}

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", 0, err
	}
	return hex.EncodeToString(h.Sum(nil)), info.Size(), nil
}

// GitSHA reports the commit and whether the tree has uncommitted changes.
//
// A missing git, or a directory that is not a checkout, is not an error: the sweep still has a
// number worth writing down. It returns an empty SHA, and the run records that rather than
// pretending to an identity it does not have.
func GitSHA(ctx context.Context, dir string) (sha string, dirty bool) {
	out, err := runQuiet(ctx, dir, "git", "rev-parse", "HEAD")
	if err != nil {
		return "", false
	}
	sha = strings.TrimSpace(out)

	status, err := runQuiet(ctx, dir, "git", "status", "--porcelain")
	if err != nil {
		return sha, false
	}
	return sha, strings.TrimSpace(status) != ""
}

func runQuiet(ctx context.Context, dir, name string, args ...string) (string, error) {
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Dir = dir
	out, err := cmd.Output()
	return string(out), err
}

// GPUSample is one reading of one device.
type GPUSample struct {
	Index       int
	Name        string
	UsedMiB     float64
	TotalMiB    float64
	UtilPercent float64
}

// GPUSampler polls nvidia-smi once a second for the life of a run.
//
// One second, not once at the end: peak VRAM and steady VRAM are different numbers and only the
// second one is visible in a single reading. A fleet that fits in steady state and spikes past
// the card on a bad second is a fleet that OOMs in production and benchmarks clean.
//
// It shells out rather than binding NVML. The dependency is the point: this is a benchmark tool
// and it must not put a CUDA library in the service's link line.
type GPUSampler struct {
	mu      sync.Mutex
	samples map[int][]GPUSample
	names   map[int]string
	note    string

	stop chan struct{}
	done chan struct{}
}

// NewGPUSampler starts sampling. Stop must be called.
//
// An absent nvidia-smi produces a NOTE, never a zero. A zero in the VRAM field of a report reads
// as "used no memory", which is the opposite of "was not measured".
func NewGPUSampler(ctx context.Context, every time.Duration) *GPUSampler {
	if every <= 0 {
		every = time.Second
	}
	s := &GPUSampler{
		samples: map[int][]GPUSample{},
		names:   map[int]string{},
		stop:    make(chan struct{}),
		done:    make(chan struct{}),
	}

	if _, err := exec.LookPath("nvidia-smi"); err != nil {
		s.note = "nvidia-smi is not on PATH, so no VRAM figure was measured; the gpus field is empty rather than zero"
		close(s.done)
		return s
	}

	go func() {
		defer close(s.done)
		ticker := time.NewTicker(every)
		defer ticker.Stop()
		for {
			s.sample(ctx)
			select {
			case <-ticker.C:
			case <-s.stop:
				return
			case <-ctx.Done():
				return
			}
		}
	}()
	return s
}

func (s *GPUSampler) sample(ctx context.Context) {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "nvidia-smi",
		"--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
		"--format=csv,noheader,nounits")
	out, err := cmd.Output()
	if err != nil {
		s.mu.Lock()
		if s.note == "" {
			s.note = "nvidia-smi failed while sampling: " + err.Error()
		}
		s.mu.Unlock()
		return
	}

	reader := csv.NewReader(strings.NewReader(string(out)))
	reader.TrimLeadingSpace = true
	reader.FieldsPerRecord = -1
	records, err := reader.ReadAll()
	if err != nil {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	for _, rec := range records {
		if len(rec) < 5 {
			continue
		}
		idx, err := strconv.Atoi(strings.TrimSpace(rec[0]))
		if err != nil {
			continue
		}
		sample := GPUSample{
			Index:       idx,
			Name:        strings.TrimSpace(rec[1]),
			UsedMiB:     parseFloat(rec[2]),
			TotalMiB:    parseFloat(rec[3]),
			UtilPercent: parseFloat(rec[4]),
		}
		s.names[idx] = sample.Name
		s.samples[idx] = append(s.samples[idx], sample)
	}
}

// Stop ends sampling and waits for the poller to finish.
func (s *GPUSampler) Stop() {
	select {
	case <-s.done:
	default:
		close(s.stop)
		<-s.done
	}
}

// Note reports why there are no samples, if there are none.
func (s *GPUSampler) Note() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.note
}

// GPUStats is the summary of one device over the sampled period.
type GPUStats struct {
	Index     int
	Name      string
	PeakMiB   float64
	SteadyMiB float64
	TotalMiB  float64
	Samples   int
	UtilMean  float64
	UtilPeak  float64
}

// Stats summarises what was sampled.
//
// "Steady" is the MEDIAN of the used-memory samples rather than the last one. The last reading
// lands wherever the run happened to stop -- often mid-teardown, with sessions already
// destroyed -- and a median over a ten-minute window is the number that describes the fleet
// while it was working.
func (s *GPUSampler) Stats() []GPUStats {
	s.mu.Lock()
	defer s.mu.Unlock()

	indices := make([]int, 0, len(s.samples))
	for idx := range s.samples {
		indices = append(indices, idx)
	}
	sort.Ints(indices)

	out := make([]GPUStats, 0, len(indices))
	for _, idx := range indices {
		samples := s.samples[idx]
		used := make([]float64, len(samples))
		st := GPUStats{Index: idx, Name: s.names[idx], Samples: len(samples)}
		var utilSum float64
		for i, sample := range samples {
			used[i] = sample.UsedMiB
			if sample.UsedMiB > st.PeakMiB {
				st.PeakMiB = sample.UsedMiB
			}
			if sample.UtilPercent > st.UtilPeak {
				st.UtilPeak = sample.UtilPercent
			}
			utilSum += sample.UtilPercent
			st.TotalMiB = sample.TotalMiB
		}
		if len(samples) > 0 {
			st.UtilMean = utilSum / float64(len(samples))
			st.SteadyMiB = median(used)
		}
		out = append(out, st)
	}
	return out
}

func median(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sorted := append([]float64(nil), values...)
	sort.Float64s(sorted)
	mid := len(sorted) / 2
	if len(sorted)%2 == 1 {
		return sorted[mid]
	}
	return (sorted[mid-1] + sorted[mid]) / 2
}

// ProcSampler watches this process's own footprint.
//
// RSS rather than the Go heap, because the frame rings are the biggest allocation in the
// service and a fleet's memory is dominated by them: three buffers of width*height*3 per
// camera. The Go heap number is carried too, so a divergence between the two points at the
// cgo side rather than at ours.
type ProcSampler struct {
	mu      sync.Mutex
	rss     []float64
	cpu     []float64
	note    string
	lastCPU time.Duration
	lastAt  time.Time

	stop chan struct{}
	done chan struct{}
}

// NewProcSampler starts sampling this process. Stop must be called.
func NewProcSampler(ctx context.Context, every time.Duration) *ProcSampler {
	if every <= 0 {
		every = time.Second
	}
	s := &ProcSampler{stop: make(chan struct{}), done: make(chan struct{})}

	go func() {
		defer close(s.done)
		ticker := time.NewTicker(every)
		defer ticker.Stop()
		for {
			s.sample()
			select {
			case <-ticker.C:
			case <-s.stop:
				return
			case <-ctx.Done():
				return
			}
		}
	}()
	return s
}

func (s *ProcSampler) sample() {
	rss, err := selfRSSMiB()

	s.mu.Lock()
	defer s.mu.Unlock()
	if err != nil {
		if s.note == "" {
			s.note = "resident set size could not be read on this platform: " + err.Error()
		}
	} else {
		s.rss = append(s.rss, rss)
	}

	cpu, err := selfCPUTime()
	if err != nil {
		return
	}
	now := time.Now()
	if !s.lastAt.IsZero() {
		if elapsed := now.Sub(s.lastAt); elapsed > 0 {
			s.cpu = append(s.cpu, float64(cpu-s.lastCPU)/float64(elapsed)*100)
		}
	}
	s.lastCPU, s.lastAt = cpu, now
}

// Stop ends sampling.
func (s *ProcSampler) Stop() {
	select {
	case <-s.done:
	default:
		close(s.stop)
		<-s.done
	}
}

// ProcStats summarises the process over the sampled period.
type ProcStats struct {
	PeakRSSMiB   float64
	SteadyRSSMiB float64
	CPUPercent   float64
	Goroutines   int
	HeapMiB      float64
	Note         string
}

// Stats summarises what was sampled, reading the Go runtime for the two figures that need no
// sampling.
func (s *ProcSampler) Stats() ProcStats {
	s.mu.Lock()
	defer s.mu.Unlock()

	var mem runtime.MemStats
	runtime.ReadMemStats(&mem)

	st := ProcStats{
		Goroutines: runtime.NumGoroutine(),
		HeapMiB:    float64(mem.HeapAlloc) / (1 << 20),
		Note:       s.note,
	}
	for _, v := range s.rss {
		if v > st.PeakRSSMiB {
			st.PeakRSSMiB = v
		}
	}
	st.SteadyRSSMiB = median(s.rss)
	if len(s.cpu) > 0 {
		var sum float64
		for _, v := range s.cpu {
			sum += v
		}
		st.CPUPercent = sum / float64(len(s.cpu))
	}
	return st
}

// selfRSSMiB reads this process's resident memory.
//
// /proc first because the sweep runs on Linux, with `ps` as the fallback that works everywhere
// else -- including the Mac this harness is written on, where the fleet binary does not run but
// the samplers still have to compile and be testable.
func selfRSSMiB() (float64, error) {
	if kb, err := procStatusRSSkB(); err == nil {
		return kb / 1024, nil
	}
	out, err := exec.Command("ps", "-o", "rss=", "-p", strconv.Itoa(os.Getpid())).Output()
	if err != nil {
		return 0, err
	}
	kb, err := strconv.ParseFloat(strings.TrimSpace(string(out)), 64)
	if err != nil {
		return 0, err
	}
	return kb / 1024, nil
}

func procStatusRSSkB() (float64, error) {
	f, err := os.Open("/proc/self/status")
	if err != nil {
		return 0, err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "VmRSS:") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			break
		}
		return strconv.ParseFloat(fields[1], 64)
	}
	return 0, fmt.Errorf("measure: no VmRSS in /proc/self/status")
}

func parseFloat(s string) float64 {
	v, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
	if err != nil {
		return 0
	}
	return v
}
