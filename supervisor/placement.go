package supervisor

import (
	"sort"
	"strconv"
	"strings"
	"sync"
)

// placer decides which device a camera runs on.
//
// # Why this is not a counter
//
// The Python service does `gpu_id = counter % gpu_count; counter += 1`, and **the counter never
// decrements when a camera stops**. It describes the order cameras were ever started, not what is
// running now, so the split drifts apart with every restart: stop three cameras off GPU 0 and
// start three more, and the counter -- which knows nothing about the three that left -- sends
// them wherever its arithmetic lands. After a day of ordinary churn one GPU is saturated while
// another idles, and nothing in the configuration explains why.
//
// A live map cannot drift, because it is not a model of the fleet -- it is the fleet. The
// tie-break on the lowest device id is not cosmetic either: without it "least loaded" is
// ambiguous whenever two devices are level, which is most of the time on a balanced fleet, and an
// ambiguous placement is one that cannot be asserted in a test.
type placer struct {
	mu      sync.Mutex
	devices []int
	load    map[int]map[string]struct{}
	where   map[string]int
}

func newPlacer(devices []int) *placer {
	if len(devices) == 0 {
		// One device, id 0. A worker with no GPU still has to put its cameras somewhere, and a
		// placer that returns "nowhere" would push that branch into every caller.
		devices = []int{0}
	}
	sorted := append([]int(nil), devices...)
	sort.Ints(sorted)

	p := &placer{devices: sorted, load: make(map[int]map[string]struct{}, len(sorted)), where: map[string]int{}}
	for _, d := range sorted {
		p.load[d] = map[string]struct{}{}
	}
	return p
}

// Place assigns a camera to a device and returns it.
//
// A hint on the spec wins when it names a device this worker has. It is how an operator pins a
// heavy camera away from the rest, and a balancer that overruled it would make the setting look
// broken rather than ignored.
func (p *placer) Place(cameraID, hint string) int {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.release(cameraID)

	device, ok := p.parseHint(hint)
	if !ok {
		device = p.leastLoaded()
	}
	p.load[device][cameraID] = struct{}{}
	p.where[cameraID] = device
	return device
}

// Release gives a camera's slot back. Every stop must call it, and that is the whole difference
// from the counter: a placement that is not released is a device that looks busier than it is,
// forever.
func (p *placer) Release(cameraID string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.release(cameraID)
}

func (p *placer) release(cameraID string) {
	device, ok := p.where[cameraID]
	if !ok {
		return
	}
	delete(p.load[device], cameraID)
	delete(p.where, cameraID)
}

// Load reports how many cameras each device is carrying. For diagnostics and tests.
func (p *placer) Load() map[int]int {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make(map[int]int, len(p.devices))
	for _, d := range p.devices {
		out[d] = len(p.load[d])
	}
	return out
}

// leastLoaded returns the emptiest device, lowest id first. Callers hold the lock.
func (p *placer) leastLoaded() int {
	best := p.devices[0]
	bestLoad := len(p.load[best])
	for _, d := range p.devices[1:] {
		if n := len(p.load[d]); n < bestLoad {
			best, bestLoad = d, n
		}
	}
	return best
}

// parseHint reads "cuda:1", "gpu:1" or "1". An unknown device is not an error worth failing a
// camera over: it falls back to balancing, which is what the operator would have got anyway.
func (p *placer) parseHint(hint string) (int, bool) {
	hint = strings.TrimSpace(hint)
	if hint == "" {
		return 0, false
	}
	if _, after, found := strings.Cut(hint, ":"); found {
		hint = after
	}
	n, err := strconv.Atoi(strings.TrimSpace(hint))
	if err != nil {
		return 0, false
	}
	if _, ok := p.load[n]; !ok {
		return 0, false
	}
	return n, true
}
