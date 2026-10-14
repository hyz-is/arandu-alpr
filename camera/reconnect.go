package camera

import (
	"context"
	"math/rand/v2"
	"time"
)

// backoff is the wait before attempt n of reopening a stream.
//
// The exponential part is the Python service's `min(2 ** restart_count, 30)`. **The jitter is
// not, and its absence is a fleet-wide hazard rather than a rough edge.** Every camera's backoff
// is driven by the same event -- a switch reboot, a VLAN flap, the recorder restarting -- so
// without jitter two hundred cameras count to the same power of two and reconnect in the same
// millisecond. The network hiccup is then followed by a self-inflicted one, and because every
// camera fails together they also retry together, which is a loop that can outlive its cause.
//
// The shape is "half fixed, half random": never longer than the ceiling, never zero, and the
// fleet's attempts spread across the window instead of stacking on its edge.
func (c *Camera) backoff(attempt int) time.Duration {
	if attempt < 1 {
		attempt = 1
	}

	d := c.cfg.MaxReconnectBackoff
	// Shifting by more than 62 overflows into a negative duration, which as a timer means "fire
	// immediately" -- a backoff that stops backing off exactly when things are at their worst.
	if attempt < 62 {
		if grown := time.Second << uint(attempt); grown > 0 && grown < d {
			d = grown
		}
	}

	half := d / 2
	return half + time.Duration(rand.Int64N(int64(half)+1))
}

// wait sleeps unless the context ends first. It reports whether the wait completed.
//
// This is the ONLY sleep in the package, and it is between connection attempts, never between
// frames. See serve for why pacing the frame loop would only add latency.
func (c *Camera) wait(ctx context.Context, d time.Duration) bool {
	if d <= 0 {
		return ctx.Err() == nil
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-t.C:
		return true
	}
}
