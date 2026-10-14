package measure

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestFileSHA256IdentifiesContentNotName(t *testing.T) {
	dir := t.TempDir()
	a := filepath.Join(dir, "model.ort")
	if err := os.WriteFile(a, []byte("weights"), 0o600); err != nil {
		t.Fatal(err)
	}
	sumA, size, err := FileSHA256(a)
	if err != nil {
		t.Fatal(err)
	}
	if size != 7 {
		t.Fatalf("size is %d, want 7", size)
	}

	// A re-export under the same name. This is the whole reason the hash is in the report:
	// nothing else about the run would look different.
	if err := os.WriteFile(a, []byte("weights2"), 0o600); err != nil {
		t.Fatal(err)
	}
	sumB, _, err := FileSHA256(a)
	if err != nil {
		t.Fatal(err)
	}
	if sumA == sumB {
		t.Error("two different files at the same path hashed the same")
	}
}

// TestAnAbsentGPUProducesANoteNotAZero.
//
// This machine has no nvidia-smi, which makes it the right place to pin the behaviour: a VRAM
// figure of zero in a report reads as "used no memory", and that is a different claim from
// "nobody measured".
func TestAnAbsentGPUProducesANoteNotAZero(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	s := NewGPUSampler(ctx, 10*time.Millisecond)
	time.Sleep(30 * time.Millisecond)
	s.Stop()

	stats := s.Stats()
	if len(stats) > 0 {
		// A machine that does have a GPU: the sampler must then have said something about it.
		if stats[0].Samples == 0 {
			t.Errorf("a device was reported with no samples: %+v", stats[0])
		}
		return
	}
	if s.Note() == "" {
		t.Error("no GPU stats and no note explaining why: an empty gpus field with no note is " +
			"indistinguishable from a run that used no VRAM")
	}
	if !strings.Contains(s.Note(), "nvidia-smi") {
		t.Errorf("the note does not name what was missing: %q", s.Note())
	}
}

func TestProcSamplerReportsSomething(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	s := NewProcSampler(ctx, 10*time.Millisecond)
	// Two intervals, so the CPU figure has a delta to divide by. The first sample only
	// establishes the baseline.
	time.Sleep(60 * time.Millisecond)
	s.Stop()

	st := s.Stats()
	if st.PeakRSSMiB <= 0 && st.Note == "" {
		t.Error("no RSS and no note explaining why")
	}
	if st.Goroutines <= 0 {
		t.Errorf("goroutine count is %d", st.Goroutines)
	}
	if st.HeapMiB <= 0 {
		t.Errorf("heap is %v MiB", st.HeapMiB)
	}
}

// TestStopIsIdempotent, because a sweep that fails mid-step stops the samplers on the way out of
// two different defers, and a second close would panic on the very path that is already failing.
func TestStopIsIdempotent(t *testing.T) {
	ctx := context.Background()

	gpu := NewGPUSampler(ctx, time.Hour)
	gpu.Stop()
	gpu.Stop()

	proc := NewProcSampler(ctx, time.Hour)
	proc.Stop()
	proc.Stop()
}

func TestGitSHAOnANonCheckoutIsEmptyRatherThanFatal(t *testing.T) {
	sha, dirty := GitSHA(context.Background(), t.TempDir())
	if sha != "" {
		t.Errorf("a temp dir reported a commit: %q", sha)
	}
	if dirty {
		t.Error("a temp dir reported a dirty tree")
	}
}

func TestMedianIsTheSteadyFigure(t *testing.T) {
	// A teardown reading of zero at the end must not become the reported steady state, which is
	// exactly what taking the last sample would do.
	if got := median([]float64{5800, 5810, 5790, 5805, 0}); got < 5000 {
		t.Errorf("median is %v: a single teardown reading dragged the steady figure down", got)
	}
	if got := median(nil); got != 0 {
		t.Errorf("median of nothing is %v", got)
	}
}
