package engine

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/hyz-is/arandu-alpr/contract"
)

// recorder stands in for *testing.T where the assertion is that the fake reported something. A
// test cannot assert on a failure it also suffers.
type recorder struct {
	mu       sync.Mutex
	messages []string
}

func (r *recorder) Helper() {}

func (r *recorder) Errorf(format string, args ...any) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.messages = append(r.messages, fmt.Sprintf(format, args...))
}

func (r *recorder) count() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.messages)
}

func testFrame(w, h int) Frame {
	return Frame{Pix: make([]byte, w*h*3), Width: w, Height: h, CapturedAt: time.Now()}
}

func TestFakeIsNotReadyUntilMarked(t *testing.T) {
	f := NewFake(FakeConfig{Detections: []Candidate{{Score: 0.9}}})

	if err := f.Ready(); !errors.Is(err, ErrNotReady) {
		t.Errorf("Ready before MarkReady = %v, want ErrNotReady", err)
	}
	if _, err := f.Detect(context.Background(), testFrame(8, 8), nil); !errors.Is(err, ErrNotReady) {
		t.Errorf("Detect during cold start = %v, want ErrNotReady", err)
	}
	if _, err := f.Recognize(context.Background(), testFrame(8, 8), contract.BBox{}); !errors.Is(err, ErrNotReady) {
		t.Errorf("Recognize during cold start = %v, want ErrNotReady", err)
	}

	f.MarkReady()
	if err := f.Ready(); err != nil {
		t.Errorf("Ready after MarkReady = %v, want nil", err)
	}
	got, err := f.Detect(context.Background(), testFrame(8, 8), nil)
	if err != nil || len(got) != 1 {
		t.Fatalf("Detect after MarkReady = %d candidates, %v", len(got), err)
	}

	if calls := f.Calls(); calls.NotReady != 2 || calls.Detect != 1 {
		t.Errorf("calls = %+v, want 2 refusals and 1 detect", calls)
	}
}

// TestFakeReturnsBusyAboveConcurrencyLimit.
//
// ErrBusy is the whole backpressure policy at the GPU boundary: under saturation the service drops
// a frame, which is recoverable, rather than queueing it, which turns into latency that is not.
// Testing that policy needs an engine that can actually be saturated.
func TestFakeReturnsBusyAboveConcurrencyLimit(t *testing.T) {
	const limit = 2
	f := NewFake(FakeConfig{
		ReadyAtStart:  true,
		MaxConcurrent: limit,
		DetectDelay:   2 * time.Second, // long enough that the occupancy is not a race
		Detections:    []Candidate{{Score: 0.9}},
	})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var wg sync.WaitGroup
	for i := 0; i < limit; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			// These are cancelled by the deferred cancel; the error is the cancellation.
			_, _ = f.Detect(ctx, testFrame(8, 8), nil)
		}()
	}

	// Wait for the engine to actually be full instead of sleeping and hoping.
	deadline := time.Now().Add(5 * time.Second)
	for f.InFlight() < limit {
		if time.Now().After(deadline) {
			t.Fatal("the fake never reached its concurrency limit")
		}
		time.Sleep(time.Millisecond)
	}

	if _, err := f.Detect(context.Background(), testFrame(8, 8), nil); !errors.Is(err, ErrBusy) {
		t.Errorf("Detect above the limit = %v, want ErrBusy", err)
	}
	// Recognize shares the same budget: sessions do not care which call is holding them.
	if _, err := f.Recognize(context.Background(), testFrame(8, 8), contract.BBox{}); !errors.Is(err, ErrBusy) {
		t.Errorf("Recognize above the limit = %v, want ErrBusy", err)
	}

	cancel()
	wg.Wait()

	if calls := f.Calls(); calls.Busy != 2 {
		t.Errorf("busy count = %d, want 2 (%+v)", calls.Busy, calls)
	}
	if f.InFlight() != 0 {
		t.Errorf("in flight = %d after every call returned, want 0: a slot leaked", f.InFlight())
	}

	// The limit is a ceiling, not a latch: once the slots are free the engine admits calls again.
	// The call is given a short deadline so the test does not wait out the synthetic delay -- what
	// matters is that it got in at all, which is anything other than ErrBusy.
	short, cancelShort := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancelShort()
	if _, err := f.Detect(short, testFrame(8, 8), nil); errors.Is(err, ErrBusy) {
		t.Error("Detect is still refused after the saturation cleared: the slots did not come back")
	}
}

// TestFakeIsSafeUnderConcurrentUse. Run with -race.
//
// The fake is shared by every camera goroutine in a supervisor test. A data race in it would show
// up as a flake in whatever test happened to be running, and be blamed on that test.
func TestFakeIsSafeUnderConcurrentUse(t *testing.T) {
	box := contract.BBox{X: 10, Y: 10, Width: 90, Height: 30}
	f := NewFake(FakeConfig{
		ReadyAtStart:   true,
		Detections:     []Candidate{{Box: box, Score: 0.9}},
		Readings:       map[contract.BBox]Reading{box: {Plate: "AAA1234", Confidence: 0.95}},
		DefaultReading: Reading{Plate: "ZZZ0000"},
	})

	const goroutines, iterations = 16, 50
	var wg sync.WaitGroup
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			dst := make([]Candidate, 0, 4)
			for i := 0; i < iterations; i++ {
				out, err := f.Detect(context.Background(), testFrame(16, 16), dst[:0])
				if err != nil {
					t.Errorf("Detect: %v", err)
					return
				}
				for _, c := range out {
					if _, err := f.Recognize(context.Background(), testFrame(16, 16), c.Box); err != nil {
						t.Errorf("Recognize: %v", err)
						return
					}
				}
				// Readers run alongside the writers on purpose: the camera loop polls these.
				_ = f.Ready()
				_ = f.InFlight()
				_ = f.Calls()
				_ = f.Providers()
				_ = f.ModelInfo()
			}
		}(g)
	}
	wg.Wait()

	calls := f.Calls()
	if calls.Detect != goroutines*iterations {
		t.Errorf("detect count = %d, want %d", calls.Detect, goroutines*iterations)
	}
	if calls.Recognize != goroutines*iterations {
		t.Errorf("recognize count = %d, want %d", calls.Recognize, goroutines*iterations)
	}
}

// TestFakeFailsTheTestWhenAGatedBoxReachesRecognize.
//
// This is the proof that the pipeline spends its GPU time in the right order. The OCR pass is the
// expensive one; score, edge and blur exist to run before it. A Recognize on a box the gate should
// have dropped means every frame is paying for work it was supposed to throw away, and nothing at
// run time looks wrong.
func TestFakeFailsTheTestWhenAGatedBoxReachesRecognize(t *testing.T) {
	gated := contract.BBox{X: 0, Y: 0, Width: 20, Height: 8} // touching the frame edge
	allowed := contract.BBox{X: 300, Y: 200, Width: 120, Height: 40}

	rec := &recorder{}
	f := NewFake(FakeConfig{
		ReadyAtStart:   true,
		TB:             rec,
		ForbiddenBoxes: []contract.BBox{gated},
	})

	if _, err := f.Recognize(context.Background(), testFrame(8, 8), allowed); err != nil {
		t.Fatalf("Recognize on an allowed box: %v", err)
	}
	if rec.count() != 0 {
		t.Errorf("an allowed box was reported: %v", rec.messages)
	}

	if _, err := f.Recognize(context.Background(), testFrame(8, 8), gated); err != nil {
		t.Fatalf("Recognize on a gated box: %v", err)
	}
	if rec.count() != 1 {
		t.Errorf("a gated box reached Recognize and %d failures were reported, want 1", rec.count())
	}

	calls := f.Calls()
	if len(calls.ForbiddenRecognized) != 1 || calls.ForbiddenRecognized[0] != gated {
		t.Errorf("ForbiddenRecognized = %+v, want exactly the gated box", calls.ForbiddenRecognized)
	}
	if len(calls.RecognizedBoxes) != 2 {
		t.Errorf("RecognizedBoxes = %+v, want both calls in order", calls.RecognizedBoxes)
	}
}

// TestFakeReportsAFrameThatDoesNotMatchItsDimensions.
//
// Against a real engine a mismatched buffer is not an error, it is a crop of somebody else's
// frame.
func TestFakeReportsAFrameThatDoesNotMatchItsDimensions(t *testing.T) {
	rec := &recorder{}
	f := NewFake(FakeConfig{ReadyAtStart: true, TB: rec})

	if _, err := f.Detect(context.Background(), Frame{Pix: make([]byte, 10), Width: 16, Height: 16}, nil); err != nil {
		t.Fatalf("Detect: %v", err)
	}
	if rec.count() != 1 {
		t.Errorf("a short pixel buffer produced %d reports, want 1", rec.count())
	}
}

func TestFakeInjectsErrors(t *testing.T) {
	boom := errors.New("cuda failure 700")
	f := NewFake(FakeConfig{
		ReadyAtStart: true,
		Detections:   []Candidate{{Score: 0.9}},
		DetectErr:    FailAt(2, boom),
		RecognizeErr: func(call int, _ contract.BBox) error {
			if call == 1 {
				return boom
			}
			return nil
		},
	})

	if _, err := f.Detect(context.Background(), testFrame(8, 8), nil); err != nil {
		t.Errorf("first Detect = %v, want nil", err)
	}
	if _, err := f.Detect(context.Background(), testFrame(8, 8), nil); !errors.Is(err, boom) {
		t.Errorf("second Detect = %v, want the injected error", err)
	}
	// Transient, not permanent: the recovery path needs a call that succeeds afterwards.
	if _, err := f.Detect(context.Background(), testFrame(8, 8), nil); err != nil {
		t.Errorf("third Detect = %v, want nil", err)
	}
	if _, err := f.Recognize(context.Background(), testFrame(8, 8), contract.BBox{}); !errors.Is(err, boom) {
		t.Errorf("first Recognize = %v, want the injected error", err)
	}

	if calls := f.Calls(); calls.Errors != 2 {
		t.Errorf("error count = %d, want 2 (%+v)", calls.Errors, calls)
	}

	always := NewFake(FakeConfig{ReadyAtStart: true, DetectErr: FailAlways(boom)})
	for i := 0; i < 3; i++ {
		if _, err := always.Detect(context.Background(), testFrame(8, 8), nil); !errors.Is(err, boom) {
			t.Fatalf("FailAlways call %d = %v", i, err)
		}
	}
}

func TestFakeReturnsFixedReadingsPerBox(t *testing.T) {
	first := contract.BBox{X: 1, Y: 2, Width: 3, Height: 4}
	second := contract.BBox{X: 10, Y: 20, Width: 30, Height: 40}
	f := NewFake(FakeConfig{
		ReadyAtStart: true,
		Detections: []Candidate{
			{Box: first, Score: 0.91},
			{Box: second, Score: 0.42},
		},
		Readings: map[contract.BBox]Reading{
			first:  {Plate: "ABC1234", Confidence: 0.99, Region: "Brazil"},
			second: {Plate: "XYZ9876", Confidence: 0.61, Region: "Portugal"},
		},
		DefaultReading: Reading{Plate: "UNKNOWN"},
	})

	got, err := f.Detect(context.Background(), testFrame(8, 8), nil)
	if err != nil {
		t.Fatalf("Detect: %v", err)
	}
	if len(got) != 2 || got[0].Box != first || got[1].Score != 0.42 {
		t.Fatalf("Detect returned %+v", got)
	}

	for _, c := range got {
		r, err := f.Recognize(context.Background(), testFrame(8, 8), c.Box)
		if err != nil {
			t.Fatalf("Recognize: %v", err)
		}
		if r.Plate != f.cfg.Readings[c.Box].Plate {
			t.Errorf("Recognize(%+v) = %q, want %q", c.Box, r.Plate, f.cfg.Readings[c.Box].Plate)
		}
	}

	unknown, err := f.Recognize(context.Background(), testFrame(8, 8), contract.BBox{X: -1})
	if err != nil || unknown.Plate != "UNKNOWN" {
		t.Errorf("an unmapped box gave %+v, %v; want the default reading", unknown, err)
	}
}

// TestFakeAllocatesNothingOnTheHotPath.
//
// The fake is what the per-frame allocation budgets are measured against. If it allocates, every
// budget measured through it is measuring the fake.
func TestFakeAllocatesNothingOnTheHotPath(t *testing.T) {
	box := contract.BBox{X: 10, Y: 10, Width: 90, Height: 30}
	f := NewFake(FakeConfig{
		ReadyAtStart: true,
		Detections:   []Candidate{{Box: box, Score: 0.9}},
		Readings:     map[contract.BBox]Reading{box: {Plate: "AAA1234"}},
		// Recording is bounded, so a long run must not grow the sample slice.
		RecordedBoxLimit: 8,
	})

	ctx := context.Background()
	frame := testFrame(16, 16)
	dst := make([]Candidate, 0, 4)

	if got := testing.AllocsPerRun(200, func() {
		out, err := f.Detect(ctx, frame, dst[:0])
		if err != nil {
			t.Fatalf("Detect: %v", err)
		}
		if _, err := f.Recognize(ctx, frame, out[0].Box); err != nil {
			t.Fatalf("Recognize: %v", err)
		}
	}); got != 0 {
		t.Errorf("Detect+Recognize allocate %v times, want 0", got)
	}
}

func TestFakeStopsServingAfterClose(t *testing.T) {
	f := NewFake(FakeConfig{ReadyAtStart: true, Detections: []Candidate{{Score: 0.9}}})

	if err := f.Close(context.Background()); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if _, err := f.Detect(context.Background(), testFrame(8, 8), nil); !errors.Is(err, ErrNotReady) {
		t.Errorf("Detect after Close = %v, want ErrNotReady", err)
	}
	if err := f.Close(context.Background()); err != nil {
		t.Errorf("second Close = %v, want nil: shutdown must be idempotent", err)
	}
	if calls := f.Calls(); calls.Close != 2 {
		t.Errorf("close count = %d, want 2", calls.Close)
	}
}

func TestFakeHonoursCancellationWhileDelaying(t *testing.T) {
	f := NewFake(FakeConfig{ReadyAtStart: true, DetectDelay: time.Minute})

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()

	start := time.Now()
	if _, err := f.Detect(ctx, testFrame(8, 8), nil); !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("Detect = %v, want the deadline", err)
	}
	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Errorf("Detect took %v: the synthetic delay ignored cancellation", elapsed)
	}
	if f.InFlight() != 0 {
		t.Errorf("in flight = %d after a cancelled call, want 0", f.InFlight())
	}
}

func TestFakeProvidersNeverClaimCUDA(t *testing.T) {
	report := NewFake(FakeConfig{}).Providers()
	if report.AllCUDA {
		t.Error("the fake claims every node ran on CUDA; the one check that catches a silent CPU " +
			"fallback would then pass everywhere")
	}
	if len(report.Models) != 2 {
		t.Errorf("Providers reported %d models, want detector and ocr", len(report.Models))
	}
}
