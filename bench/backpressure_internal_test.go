package bench

import (
	"context"
	"runtime"
	"testing"
	"time"

	"github.com/hyz-is/arandu-alpr/bench/measure"
	"github.com/hyz-is/arandu-alpr/bench/report"
	"github.com/hyz-is/arandu-alpr/engine"
	"github.com/hyz-is/arandu-alpr/metrics"
)

// TestSaturatedEngineDropsRatherThanQueues is the whole backpressure policy, measured.
//
// The claim the service makes is that under GPU saturation it LOSES FRAMES, which a fleet
// recovers from, instead of GROWING LATENCY, which it does not. Both halves of that sentence
// are checkable and neither is checkable from a code reading:
//
//   - the loss is labelled, so an operator can tell "the GPU is full" from "the cameras are
//     late" in one query. That label is what turned a four-hour diagnosis in the Python service
//     into a graph.
//   - the latency is FLAT over time. A queue would show the opposite: fewer refusals as the
//     backlog absorbs the pressure, and end-to-end age climbing without bound. That is why this
//     measures two consecutive windows instead of one.
func TestSaturatedEngineDropsRatherThanQueues(t *testing.T) {
	const (
		cameras = 8
		// One session for eight cameras. The ratio has to be lopsided or the run never
		// saturates and the test passes by not testing anything.
		sessions = 1
		// Long enough that a refusal is a real decision and short enough that the test runs in
		// seconds. It also sets the capacity this test checks the served work against.
		callDelay = 15 * time.Millisecond
		window    = 1500 * time.Millisecond
	)

	phased := measure.NewPhased()
	fake := readyFake(engine.FakeConfig{
		DetectDelay:    callDelay,
		RecognizeDelay: callDelay,
		MaxConcurrent:  sessions,
	})

	goroutinesBefore := runtime.NumGoroutine()

	f, err := startFleet(context.Background(), fleetConfig{
		cameras: cameras,
		engine:  fake,
		metrics: phased,
		sink:    &countingSink{},
		// Every frame is offered to the engine, so the refusals are the engine's and not the
		// detection budget's.
		detectEvery: 0,
		// Without this the end-to-end histogram gets one observation per camera and a p95 over
		// eight samples cannot distinguish flat from climbing.
		disableDedup: true,
		dir:          t.TempDir(),
	})
	if err != nil {
		t.Fatal(err)
	}

	time.Sleep(window)
	phased.Arm()
	time.Sleep(window)

	// Snapshot before the stop: cancelling the fleet cancels a context a camera is inside
	// Detect with, and those refusals belong to the shutdown rather than to the window.
	first := totals(phased.Warmup())
	second := totals(phased.Window())
	firstP95 := report.FromSnapshot(phased.Warmup(), metrics.PipelineDuration).P95
	secondP95 := report.FromSnapshot(phased.Window(), metrics.PipelineDuration).P95
	calls := fake.Calls()
	elapsed := f.stop()

	if first.framesReceived == 0 || second.framesReceived == 0 {
		t.Fatalf("no frames reached the loop (%.0f then %.0f): the harness did not run",
			first.framesReceived, second.framesReceived)
	}

	// 1. The refusal is labelled, and the label is the one an operator queries.
	busy := second.dropsByReason[metrics.DropInferenceBusy]
	if busy == 0 {
		t.Errorf("no frame was dropped with reason %q under a %d-camera fleet sharing %d session(s)\n\n"+
			"Either the engine never refused, in which case the test is not saturating anything, or "+
			"the refusal was counted under another reason -- and an unlabelled drop cannot tell "+
			"\"the GPU is full\" from \"the cameras are late\", which need opposite remedies.",
			metrics.DropInferenceBusy, cameras, sessions)
	}
	if calls.Busy == 0 {
		t.Errorf("the engine reported no ErrBusy at all, so nothing was under pressure")
	}

	// 2. Refusal dominates. A fleet eight times its capacity should be spending most of its
	// frames on the floor; a small share would mean something else is absorbing them.
	if share := busy / second.framesReceived; share < 0.25 {
		t.Errorf("only %.0f%% of frames were dropped as %q, with %d cameras on %d session(s)\n\n"+
			"The frames have to go somewhere. A low refusal rate here means they are being held "+
			"rather than discarded, which is the queue this policy exists to not have.",
			share*100, metrics.DropInferenceBusy, cameras, sessions)
	}

	// 3. Latency is flat between the two windows. This is the queue test.
	if firstP95 == 0 || secondP95 == 0 {
		t.Fatalf("no end-to-end durations were recorded (%v then %v): nothing completed a pass",
			firstP95, secondP95)
	}
	if secondP95 > firstP95*2 {
		t.Errorf("end-to-end p95 grew from %s to %s across two consecutive %s windows\n\n"+
			"Under saturation the age of a delivered reading must stay flat: the service drops "+
			"what it cannot serve. Age that climbs with elapsed time is the signature of a "+
			"backlog, and a backlog answers questions after they stopped being worth asking.",
			time.Duration(firstP95*float64(time.Second)).Round(time.Millisecond),
			time.Duration(secondP95*float64(time.Second)).Round(time.Millisecond),
			window)
	}

	// 4. The served work never exceeded what one session could do. A backlog served later shows
	//    up here and nowhere else.
	capacity := float64(sessions) * elapsed.Seconds() / callDelay.Seconds()
	served := float64(calls.Detect + calls.Recognize - calls.Busy - calls.NotReady)
	if served > capacity*1.5 {
		t.Errorf("the engine completed %.0f calls in %s, more than the %.0f one session can do at %s each\n\n"+
			"Either the concurrency limit is not being honoured or work is being served from a "+
			"backlog built while the engine was busy.",
			served, elapsed.Round(time.Millisecond), capacity, callDelay)
	}

	// 5. Nothing grew a goroutine per frame. A pipeline that hands refused frames to a worker
	//    would look fine on every number above and leak here.
	runtime.GC()
	if after := runtime.NumGoroutine(); after > goroutinesBefore+cameras*4 {
		t.Errorf("goroutines went from %d to %d for %d cameras\n\n"+
			"Nothing in the loop may start a goroutine per frame: a frame worth a GPU pass is "+
			"worth waiting for, and one that is not is dropped rather than deferred.",
			goroutinesBefore, after, cameras)
	}

	t.Logf("%d cameras, %d session(s), %s per call: %.0f frames, %.0f dropped busy (%.0f%%), "+
		"end-to-end p95 %s then %s",
		cameras, sessions, callDelay,
		second.framesReceived, busy, busy/second.framesReceived*100,
		time.Duration(firstP95*float64(time.Second)).Round(time.Millisecond),
		time.Duration(secondP95*float64(time.Second)).Round(time.Millisecond))
}

// TestBusyIsNotCountedAsAnInferenceError.
//
// ErrBusy is a capacity decision and an inference error is a fault, and they page different
// people. Counting saturation as an error means every busy hour raises an alert that says the
// models are broken.
func TestBusyIsNotCountedAsAnInferenceError(t *testing.T) {
	reg := metrics.NewSnapshotRegistry()
	fake := readyFake(engine.FakeConfig{
		DetectDelay:    10 * time.Millisecond,
		RecognizeDelay: 10 * time.Millisecond,
		MaxConcurrent:  1,
	})

	f, err := startFleet(context.Background(), fleetConfig{
		cameras: 6,
		engine:  fake,
		metrics: reg,
		sink:    &countingSink{},
		dir:     t.TempDir(),
	})
	if err != nil {
		t.Fatal(err)
	}
	time.Sleep(time.Second)

	// Read before stopping, and the order is load-bearing.
	//
	// Cancelling the fleet cancels the context a camera is inside Detect with, so the engine
	// returns context.Canceled -- which is neither ErrBusy nor ErrNotReady, so the loop counts
	// it as an inference error. That is shutdown noise rather than the behaviour under test,
	// and reading after the stop would make this assertion fail for a reason it is not asking
	// about.
	tot := totals(reg.Snapshot())
	f.stop()
	if tot.dropsByReason[metrics.DropInferenceBusy] == 0 {
		t.Fatal("the fleet never saturated, so this test proved nothing")
	}
	if tot.inferenceErrs != 0 {
		t.Errorf("%.0f inference errors were counted while the only failure was ErrBusy\n\n"+
			"Saturation is a capacity decision, not a fault. Counting it as one makes every busy "+
			"hour look like a broken model.", tot.inferenceErrs)
	}
}
