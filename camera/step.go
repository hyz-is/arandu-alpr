package camera

import (
	"context"
	"errors"
	"time"

	"github.com/hyz-is/arandu-alpr"
	"github.com/hyz-is/arandu-alpr/contract"
	"github.com/hyz-is/arandu-alpr/engine"
)

// stepResult says what the loop should do next.
type stepResult uint8

const (
	stepContinue stepResult = iota
	stepReconnect
)

// step is one frame, start to finish, on the camera's own goroutine.
//
// Nothing here starts a goroutine. A frame that is worth a GPU pass is worth waiting for, and a
// frame that is not is dropped, not deferred -- handing frames to workers is how a queue of
// pictures of the past gets built, and every entry in it costs GPU time to answer a question
// nobody is asking any more.
func (c *Camera) step(ctx context.Context, f engine.Frame) stepResult {
	now := c.now()
	c.m.framesReceived.Inc()

	if c.dropStale(now, f) {
		if c.staleRun >= c.cfg.StaleFramesBeforeReconnect {
			c.staleRun = 0
			return stepReconnect
		}
		return stepContinue
	}

	c.expireDetections(now)

	// The detection budget, and the reason it is a duration.
	//
	// The Python service gates on `frame_count % FRAME_SKIP`, which only means something when the
	// consumer sees every frame. Under latest-frame-wins it does not: the producer overwrites the
	// pending frame, so the counter no longer counts pictures the camera sent, it counts pictures
	// this loop happened to pick up. The same FRAME_SKIP=5 then means 6fps on a 30fps camera and
	// 0.4fps on a 2fps one -- and under GPU pressure, where fewer frames get picked up, it
	// silently relaxes exactly when it should be tightening. Time is the unit that survives all
	// three.
	if now.Sub(c.lastDetectAt) < c.detectInterval(now) {
		c.m.dropSkip.Inc()
		return stepContinue
	}
	c.lastDetectAt = now

	// Untouched costs nothing: at 100/100 the adjustment is not merely a no-op, it is not called.
	// A full-frame pass over 6MB is a real cost to pay for multiplying by one.
	if c.cfg.Adjust != nil && (c.cfg.Spec.Brightness != 100 || c.cfg.Spec.Contrast != 100) {
		c.cfg.Adjust(f.Pix, c.cfg.Spec.Brightness, c.cfg.Spec.Contrast)
	}

	candidates, ok := c.detect(ctx, f)
	if !ok {
		return stepContinue
	}

	for i := range candidates {
		cand := candidates[i]
		box := toGateBox(cand.Box)

		// Cheap before expensive, and this order is the whole reason detection and reading are
		// separate calls. Score is a comparison, the edge test is four comparisons, focus is a
		// pass over one crop -- and Recognize is a GPU pass. The Python service could not do
		// this: fast_alpr detects and reads in one call, so every blurred, half-cropped plate
		// was paid for in full before anything got to reject it.
		if !c.cfg.Gate.AcceptDetection(cand.Score, box, f.Width, f.Height) {
			continue
		}
		if c.cfg.Focus != nil && !c.cfg.Gate.AcceptFocus(c.cfg.Focus.FocusVariance(f, cand.Box), box) {
			continue
		}

		c.recognize(ctx, f, cand, now)
	}

	return stepContinue
}

// dropStale is the ingest lag watchdog, and the counter in it is load-bearing.
//
// A late frame is discarded ALWAYS: it cannot notify anyone in time, and a plate read from eight
// seconds ago is a plate that already left. But a reconnect happens only when the lag PERSISTS.
//
// Reconnecting on the first late frame is what produced the worst incident in this service's
// history. Reopening a stream suspends reading for seconds, so under a saturated GPU the loop fed
// itself: late frame -> reconnect -> more lateness -> reconnect. Throughput fell from 490
// detections a minute to 12 and stayed there, with every camera reporting healthy.
//
// The other half is the reset below, and it is the half that is easy to lose. See
// TestOneFreshFrameResetsStaleRun.
func (c *Camera) dropStale(now time.Time, f engine.Frame) bool {
	if f.CapturedAt.IsZero() {
		// No capture time means no lag to judge. Treating unknown as stale would drop every
		// frame from a producer that does not stamp them.
		c.staleRun = 0
		return false
	}
	if now.Sub(f.CapturedAt) <= c.cfg.MaxIngestLag {
		c.staleRun = 0
		return false
	}
	c.m.dropStale.Inc()
	c.staleRun++
	return true
}

// expireDetections drops overlay boxes older than the TTL, in place.
//
// `c.active[:0]` reuses the array the last frame used. Rebuilding the slice would allocate on
// every frame of every camera in the fleet, for a list that is almost always empty.
func (c *Camera) expireDetections(now time.Time) {
	kept := c.active[:0]
	for _, d := range c.active {
		if now.Sub(d.at) < c.cfg.DetectionTTL {
			kept = append(kept, d)
		}
	}
	c.active = kept
}

// detect runs the detector under a deadline and reports whether the result is usable.
func (c *Camera) detect(ctx context.Context, f engine.Frame) ([]engine.Candidate, bool) {
	detectCtx, cancel := c.withTimeout(ctx, c.cfg.DetectTimeout)
	start := c.now()
	candidates, err := c.cfg.Engine.Detect(detectCtx, f, c.candidates[:0])
	cancel()
	c.m.detectorDuration.Observe(c.now().Sub(start).Seconds())

	if err != nil {
		switch {
		case errors.Is(err, engine.ErrBusy):
			// Dropped, never queued. Under saturation this service loses frames, which the next
			// frame recovers from, rather than growing latency, which nothing recovers from.
			c.m.dropBusy.Inc()
		case errors.Is(err, engine.ErrNotReady):
			// Cold start. Not an error worth counting: the first seconds of every process would
			// otherwise raise an inference-error alert on every camera at once.
		default:
			c.m.errDetect.Inc()
			c.cfg.Logger.Warn("detector failed", LabelCamera, c.cfg.Spec.ID, "error", err)
			c.emit(ctx, contract.EventInferenceError, "detection failed")
		}
		return nil, false
	}

	// Keep whatever capacity the detector grew, so the next frame appends into it.
	c.candidates = candidates
	return candidates, true
}

// recognize reads one surviving box and, if it holds up, delivers it.
func (c *Camera) recognize(ctx context.Context, f engine.Frame, cand engine.Candidate, now time.Time) {
	ocrCtx, cancel := c.withTimeout(ctx, c.cfg.RecognizeTimeout)
	start := c.now()
	reading, err := c.cfg.Engine.Recognize(ocrCtx, f, cand.Box)
	cancel()
	c.m.ocrDuration.Observe(c.now().Sub(start).Seconds())

	if err != nil {
		switch {
		case errors.Is(err, engine.ErrBusy):
			c.m.dropBusy.Inc()
		case errors.Is(err, engine.ErrNotReady):
		default:
			c.m.errOCR.Inc()
			c.cfg.Logger.Warn("ocr failed", LabelCamera, c.cfg.Spec.ID, "error", err)
			c.emit(ctx, contract.EventInferenceError, "plate reading failed")
		}
		return
	}

	confidence := reading.Confidence
	if confidence == 0 && len(reading.CharConfidences) > 0 {
		// A backend that fills only the per-character values still gets judged by the same rule
		// as every other: the mean, never the minimum. Taking the minimum would re-tune the gate
		// without changing a single threshold.
		confidence = alpr.MeanConfidence(reading.CharConfidences)
	}
	if !c.cfg.Gate.AcceptReading(reading.Plate, confidence) {
		return
	}
	if c.cfg.Dedup != nil && !c.cfg.Dedup.Accept(reading.Plate, now) {
		c.m.deduplicated.Inc()
		return
	}

	c.m.detected.Inc()
	c.lastDetectionAt = now
	c.noDetectionSent = false
	c.remember(cand.Box, now)

	// Measured from CapturedAt, not from the top of the loop.
	//
	// End-to-end age is what an operator experiences and what the staleness limit is expressed
	// in. Timing from the loop's entry would report how fast this code is while a queue upstream
	// made the reading useless, which is a graph that stays green through the outage.
	c.m.pipelineDuration.Observe(c.now().Sub(f.CapturedAt).Seconds())

	c.deliver(ctx, f, cand, reading, confidence, now)
}

// deliver encodes what was asked for, publishes the overlay and hands the Detection over.
//
// The full frame is encoded at most once even when both the sink and the preview want it. Two
// encodes of the same 6MB picture is the kind of waste that only shows up as "the GPU is fine but
// the box is at 100% CPU".
func (c *Camera) deliver(
	ctx context.Context,
	f engine.Frame,
	cand engine.Candidate,
	reading engine.Reading,
	confidence float64,
	now time.Time,
) {
	var fullJPEG []byte
	if c.cfg.Encoder != nil && (c.cfg.IncludeFullImage || c.cfg.Preview != nil) {
		if b, err := c.cfg.Encoder.EncodeFrame(f); err == nil {
			fullJPEG = b
		} else {
			c.cfg.Logger.Warn("encoding the frame failed", LabelCamera, c.cfg.Spec.ID, "error", err)
		}
	}
	c.publishPreview(fullJPEG)

	if c.cfg.Sink == nil {
		return
	}
	if !c.cfg.IncludeFullImage {
		fullJPEG = nil
	}

	var cropJPEG []byte
	if c.cfg.IncludeCropImage && c.cfg.Encoder != nil {
		if b, err := c.cfg.Encoder.EncodeCrop(f, cand.Box); err == nil {
			cropJPEG = b
		} else {
			c.cfg.Logger.Warn("encoding the crop failed", LabelCamera, c.cfg.Spec.ID, "error", err)
		}
	}

	err := c.cfg.Sink.Detect(ctx, contract.Detection{
		CameraID: c.cfg.Spec.ID,
		Plate:    reading.Plate,
		// Copied: the Reading may point into a buffer the engine reuses on the next crop, and a
		// Detection is owned by whoever receives it. Sharing here would rewrite a delivered
		// reading's confidences with the next plate's, which looks like data, not like a bug.
		CharConfidences:  clone(reading.CharConfidences),
		Confidence:       confidence,
		Region:           reading.Region,
		RegionConfidence: reading.RegionConfidence,
		Box:              cand.Box,
		DetectorScore:    cand.Score,
		CapturedAt:       f.CapturedAt,
		DetectedAt:       now,
		Model:            c.cfg.Engine.ModelInfo(),
		FullJPEG:         fullJPEG,
		CropJPEG:         cropJPEG,
	})
	if err == nil {
		return
	}

	// A Sink failure is the application's problem, not the camera's.
	//
	// Killing the loop here would take a working camera off the air because something downstream
	// -- a database, a queue, a webhook -- had a bad minute. The camera counts it, says so, and
	// reads the next frame.
	if errors.Is(err, contract.ErrBackpressure) {
		c.applyBackpressure(now)
		c.cfg.Logger.Info("sink asked for backpressure",
			LabelCamera, c.cfg.Spec.ID, "detect_interval", c.interval)
		return
	}
	c.m.errSink.Inc()
	c.cfg.Logger.Warn("sink rejected detection", LabelCamera, c.cfg.Spec.ID, "error", err)
}

// remember puts a box on the overlay list.
func (c *Camera) remember(box contract.BBox, now time.Time) {
	if len(c.active) >= maxActiveDetections {
		// Drop the oldest rather than the newest: the overlay is about what is in frame now.
		copy(c.active, c.active[1:])
		c.active = c.active[:len(c.active)-1]
	}
	c.active = append(c.active, activeDetection{box: box, at: now})
}

// publishPreview offers the live overlay a frame, and only when a plate was just accepted.
//
// Encoding a JPEG per frame for a viewer who may not exist would cost more than the detection
// does. The contract says a preview is best-effort; this is what best-effort means in practice.
//
// The JPEG is the same buffer the Detection carries. That is safe and deliberate: an Encoder
// returns freshly allocated bytes that nothing ever rewrites, so two receivers reading it cannot
// see each other. The BOXES are different -- that slice is reused every frame and is borrowed for
// the duration of the call, which is what "best-effort preview" is allowed to cost.
func (c *Camera) publishPreview(jpeg []byte) {
	if c.cfg.Preview == nil || len(jpeg) == 0 {
		return
	}
	c.boxes = c.boxes[:0]
	for _, d := range c.active {
		c.boxes = append(c.boxes, d.box)
	}
	c.cfg.Preview.Frame(c.cfg.Spec.ID, jpeg, c.boxes)
}

// detectInterval is the current budget between two detection passes, and the place backpressure
// decays back to normal.
func (c *Camera) detectInterval(now time.Time) time.Duration {
	if !c.relaxAt.IsZero() && !now.Before(c.relaxAt) {
		c.interval = c.cfg.Spec.MinDetectInterval
		c.relaxAt = time.Time{}
	}
	return c.interval
}

// applyBackpressure raises the budget after the application said it cannot keep up.
//
// This is what makes contract.ErrBackpressure worth having: the pressure arrives as a decision to
// do less work, at the only place that can decide it, instead of as a queue growing somewhere
// nobody is looking.
func (c *Camera) applyBackpressure(now time.Time) {
	next := c.interval * 2
	if next < backpressureFloor {
		next = backpressureFloor
	}
	if next > c.cfg.MaxDetectInterval {
		next = c.cfg.MaxDetectInterval
	}
	c.interval = next
	c.relaxAt = now.Add(c.cfg.BackpressureRelax)
}

// withTimeout applies a deadline when one is configured.
//
// Zero means no deadline, and the returned cancel is always safe to call. The branch exists
// because context.WithTimeout allocates a context and a timer: worth it once per detection pass,
// not worth forcing on a caller that does not want a deadline at all.
func (c *Camera) withTimeout(ctx context.Context, d time.Duration) (context.Context, context.CancelFunc) {
	if d <= 0 {
		return ctx, noopCancel
	}
	return context.WithTimeout(ctx, d)
}

func noopCancel() {}

// toGateBox converts to the gate's rectangle. The two types are deliberately separate: the gate
// is pure arithmetic with no knowledge of the contract, and this is what that costs.
func toGateBox(b contract.BBox) alpr.BBox {
	return alpr.BBox{X: b.X, Y: b.Y, Width: b.Width, Height: b.Height}
}

// clone hands out an owned copy, or nil for nothing.
func clone[T any](src []T) []T {
	if len(src) == 0 {
		return nil
	}
	out := make([]T, len(src))
	copy(out, src)
	return out
}
