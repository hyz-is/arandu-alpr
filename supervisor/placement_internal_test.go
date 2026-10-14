package supervisor

import "testing"

// TestPlacementRebalancesAfterStop.
//
// This is the Python service's `gpu_id = counter % gpu_count` with the counter that never comes
// back down. It is balanced exactly once -- on a cold start with nothing having stopped yet --
// and every restart after that pushes it further out, because the counter describes the order
// cameras were started rather than what is running.
//
// The counter's own behaviour is simulated here rather than quoted, so the comparison stays true
// if anyone ever reads this test wondering whether the old way was really that bad.
func TestPlacementRebalancesAfterStop(t *testing.T) {
	const devices = 2
	ids := []string{"c0", "c1", "c2", "c3", "c4", "c5"}

	p := newPlacer([]int{0, 1})
	for _, id := range ids {
		p.Place(id, "")
	}
	if load := p.Load(); load[0] != 3 || load[1] != 3 {
		t.Fatalf("initial placement = %v, want 3 on each device", load)
	}

	// Stop everything on GPU 0.
	var stopped []string
	for _, id := range ids {
		if p.where[id] == 0 {
			stopped = append(stopped, id)
		}
	}
	for _, id := range stopped {
		p.Release(id)
	}
	if load := p.Load(); load[0] != 0 || load[1] != 3 {
		t.Fatalf("load after stopping GPU 0 = %v, want 0 and 3", load)
	}

	// Three new cameras arrive.
	for _, id := range []string{"c6", "c7", "c8"} {
		p.Place(id, "")
	}
	if load := p.Load(); load[0] != 3 || load[1] != 3 {
		t.Errorf("load after the restart = %v, want 3 on each device", load)
	}

	// The counter, for comparison: it has no idea three cameras left.
	counter := len(ids)
	counterLoad := map[int]int{0: 0, 1: 3} // what survived the stop
	for i := 0; i < 3; i++ {
		counterLoad[counter%devices]++
		counter++
	}
	if counterLoad[0] == counterLoad[1] {
		t.Fatalf("the counter simulation came out balanced (%v); it cannot, and if it does this "+
			"test is no longer demonstrating anything", counterLoad)
	}
	t.Logf("live map: %v -- counter: %v", p.Load(), counterLoad)
}

// TestPlacementIsDeterministicOnTies.
//
// "Least loaded" is ambiguous the moment two devices are level, which on a balanced fleet is most
// of the time. Ambiguous placement cannot be asserted, so it cannot be tested, so it drifts
// unnoticed.
func TestPlacementIsDeterministicOnTies(t *testing.T) {
	for run := 0; run < 20; run++ {
		p := newPlacer([]int{2, 0, 1})
		if got := p.Place("first", ""); got != 0 {
			t.Fatalf("first placement went to device %d, want the lowest id (0)", got)
		}
		if got := p.Place("second", ""); got != 1 {
			t.Fatalf("second placement went to device %d, want 1", got)
		}
		if got := p.Place("third", ""); got != 2 {
			t.Fatalf("third placement went to device %d, want 2", got)
		}
		if got := p.Place("fourth", ""); got != 0 {
			t.Fatalf("fourth placement went to device %d, want 0 again", got)
		}
	}
}

// TestPlacementHintWins.
//
// Pinning a heavy camera to a device is the reason CameraSpec.Placement exists. A balancer that
// overruled it would make the setting look broken rather than ignored, which is worse: the
// operator changes it, watches nothing happen, and stops trusting the panel.
func TestPlacementHintWins(t *testing.T) {
	p := newPlacer([]int{0, 1})

	if got := p.Place("pinned", "cuda:1"); got != 1 {
		t.Errorf("hinted placement went to device %d, want 1", got)
	}
	if got := p.Place("plain", "1"); got != 1 {
		t.Errorf("bare-number hint went to device %d, want 1", got)
	}

	// A hint naming a device this worker does not have falls back to balancing rather than
	// failing the camera: a camera that runs on the wrong GPU still reads plates.
	if got := p.Place("bad", "cuda:7"); got != 0 {
		t.Errorf("unknown hint went to device %d, want the least loaded (0)", got)
	}
	if got := p.Place("garbage", "the-fast-one"); got != 0 {
		t.Errorf("unparseable hint went to device %d, want the least loaded (0)", got)
	}
}

// TestPlacingTheSameCameraTwiceDoesNotDoubleCount.
//
// A restart places the camera again. Without releasing the old slot first, every restart makes
// its device look one camera busier than it is -- and the error accumulates silently.
func TestPlacingTheSameCameraTwiceDoesNotDoubleCount(t *testing.T) {
	p := newPlacer([]int{0, 1})
	p.Place("a", "")
	p.Place("a", "")
	p.Place("a", "")

	total := 0
	for _, n := range p.Load() {
		total += n
	}
	if total != 1 {
		t.Errorf("one camera placed three times counts as %d, want 1", total)
	}
}
