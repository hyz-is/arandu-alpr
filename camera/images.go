package camera

import (
	"context"
	"sync"
	"time"

	"github.com/hyz-is/arandu-alpr/contract"
	"github.com/hyz-is/arandu-alpr/engine"
)

// The encoder goroutine of a camera.
//
// A camera holds the raw frame of at most maxKeptFrames undecided tracks. A track past that bound
// does not lose its pictures and does not make the frame loop encode them either: its raw frame is
// handed as it is, with no copy, to one goroutine per camera that makes the JPEGs the verdict will
// carry. The loop never waits for it, and everything that touches the Sink, the Preview, the
// metrics or the detection budget stays on the camera's own goroutine, as contract.Sink promises.

// maxEncodeQueue is how many frames wait for the encoder goroutine; the one it is encoding is one
// more. Together they are the raw frames in transit, on top of maxKeptFrames. They are the
// buffers of tracks that were evicted, not copies made for the queue, and they go back to the
// spares once encoded.
const maxEncodeQueue = 2

// maxInFlight is the raw frames handed to the encoder and not yet taken back.
const maxInFlight = maxEncodeQueue + 1

// The encoded pictures of the tracks that have not decided. A full frame and a crop are a few
// hundred KB, so these hold the whole of a consensus (16 tracks) in the memory of about one raw
// frame; past them the oldest track loses its pictures, and that is counted when its verdict goes
// out without.
const (
	maxEncodedFrames = 16
	maxEncodedBytes  = 16 << 20
)

// encodeJob is one frame on its way to JPEG.
//
// The fields before done belong to the camera's loop until the job is sent, and to the encoder
// goroutine until done is closed; the fields after belong to the encoder goroutine until then and
// to the loop afterwards. Closing done is what hands them over, so nothing here needs a lock.
type encodeJob struct {
	pix                []byte
	w, h               int
	at                 time.Time
	box                contract.BBox
	wantFull, wantCrop bool

	full, cropJPEG []byte
	failed         bool
	done           chan struct{}
}

// finished says whether the encoder is done with the job. The loop may then read its pictures and
// take its buffer back.
func (j *encodeJob) finished() bool {
	select {
	case <-j.done:
		return true
	default:
		return false
	}
}

// imagesOf is what a finished job made. A job that failed made nothing, and that is counted: the
// reading goes out without the pictures that were asked for.
func (c *Camera) imagesOf(j *encodeJob) encodedImages {
	if j.failed {
		c.m.errImages.Inc()
		return encodedImages{}
	}
	return encodedImages{full: j.full, crop: j.cropJPEG}
}

// parkedVerdict is a decision whose pictures the encoder has not finished. Everything that is the
// camera's to decide about it was decided; the Sink has not been told.
type parkedVerdict struct {
	job        *encodeJob
	frame      engine.Frame
	cand       engine.Candidate
	reading    engine.Reading
	confidence float64
	at         time.Time
}

// startEncoder starts the camera's encoder goroutine and returns the function that ends it. Run
// calls it; a camera that does not decide with a consensus, or does not ask for pictures, has no
// queue and starts nothing.
func (c *Camera) startEncoder(ctx context.Context) (stop func()) {
	if c.jobs == nil || c.encoding {
		return func() {}
	}
	ctx, cancel := context.WithCancel(ctx)
	var wg sync.WaitGroup
	wg.Add(1)
	c.encoding = true
	go func() {
		defer wg.Done()
		for {
			select {
			case <-ctx.Done():
				return
			case j := <-c.jobs:
				c.encode(j)
			}
		}
	}()
	return func() {
		cancel()
		wg.Wait()
		c.encoding = false
		// What was still queued will never be encoded: finish it as failed, so nothing waits on it.
		for {
			select {
			case j := <-c.jobs:
				j.failed = true
				close(j.done)
			default:
				return
			}
		}
	}
}

// encode makes the pictures of one job, as deliver would have made them from the frame.
func (c *Camera) encode(j *encodeJob) {
	defer close(j.done)
	f := engine.Frame{Pix: j.pix, Width: j.w, Height: j.h, CapturedAt: j.at}
	if j.wantFull {
		b, err := c.cfg.Encoder.EncodeFrame(f)
		if err != nil {
			c.cfg.Logger.Warn("encoding the frame failed", LabelCamera, c.cfg.Spec.ID, "error", err)
			j.failed = true
			return
		}
		j.full = b
	}
	if j.wantCrop {
		b, err := c.cfg.Encoder.EncodeCrop(f, j.box)
		if err != nil {
			c.cfg.Logger.Warn("encoding the crop failed", LabelCamera, c.cfg.Spec.ID, "error", err)
			j.failed = true
			return
		}
		j.cropJPEG = b
	}
}

// encodeOldestKept hands the raw frame of the track whose best frame is the oldest to the encoder
// goroutine, so a new track can have a raw frame of its own. It never waits: when the encoder is
// behind, the track loses its pictures, which is what happened to every such track before the
// encoder existed, and now it is logged and counted.
func (c *Camera) encodeOldestKept() {
	c.reap()
	var oldest uint64
	var at time.Time
	for id, k := range c.kept {
		if len(k.pix) > 0 && (oldest == 0 || k.at.Before(at)) {
			oldest, at = id, k.at
		}
	}
	if oldest == 0 {
		return
	}
	k := c.kept[oldest]
	if !c.encoding || len(c.flight) >= maxInFlight {
		c.cfg.Logger.Warn("encoder is behind, dropping the pictures of an undecided track",
			LabelCamera, c.cfg.Spec.ID, "track", oldest, "in_flight", len(c.flight))
		c.drop(oldest)
		return
	}
	j := &encodeJob{
		pix: k.pix, w: k.w, h: k.h, at: k.at, box: k.box,
		wantFull: c.cfg.IncludeFullImage || c.cfg.Preview != nil,
		wantCrop: c.cfg.IncludeCropImage,
		done:     make(chan struct{}),
	}
	select {
	case c.jobs <- j:
	default:
		c.cfg.Logger.Warn("encoder is behind, dropping the pictures of an undecided track",
			LabelCamera, c.cfg.Spec.ID, "track", oldest, "in_flight", len(c.flight))
		c.drop(oldest)
		return
	}
	c.flight = append(c.flight, j)
	k.job = j
	k.pix = nil
	c.raw--
}

// collect is what the loop does about the encoder at the top of a frame: take back the buffers of
// finished jobs and deliver the verdicts that were waiting for them. With nothing in flight it is
// two length checks.
func (c *Camera) collect(ctx context.Context) {
	if len(c.flight) == 0 && len(c.parked) == 0 {
		return
	}
	c.reap()
	c.deliverParked(ctx)
}

// reap takes back the raw buffers of the jobs the encoder has finished, and bounds the pictures
// they left with the tracks.
func (c *Camera) reap() {
	reaped := false
	kept := c.flight[:0]
	for _, j := range c.flight {
		if !j.finished() {
			kept = append(kept, j)
			continue
		}
		c.recycle(j.pix)
		j.pix = nil
		reaped = true
	}
	for i := len(kept); i < len(c.flight); i++ {
		c.flight[i] = nil
	}
	c.flight = kept
	if reaped {
		c.trimEncoded()
	}
}

// deliverParked hands the Sink the verdicts whose pictures are ready, in the order they were
// decided.
func (c *Camera) deliverParked(ctx context.Context) {
	if len(c.parked) == 0 {
		return
	}
	waiting := c.parked[:0]
	for _, p := range c.parked {
		if !p.job.finished() {
			waiting = append(waiting, p)
			continue
		}
		if !p.frame.CapturedAt.IsZero() {
			c.m.pipelineDuration.Observe(c.now().Sub(p.frame.CapturedAt).Seconds())
		}
		c.deliver(ctx, p.frame, p.cand, p.reading, p.confidence, p.at, c.imagesOf(p.job))
	}
	for i := len(waiting); i < len(c.parked); i++ {
		c.parked[i] = parkedVerdict{}
	}
	c.parked = waiting
}

// trimEncoded lets go of the oldest encoded pictures while they are more than the bounds allow.
func (c *Camera) trimEncoded() {
	for {
		n, bytes := 0, 0
		var oldest uint64
		var at time.Time
		for id, k := range c.kept {
			if len(k.pix) > 0 || k.job == nil || !k.job.finished() {
				continue
			}
			n++
			bytes += len(k.job.full) + len(k.job.cropJPEG)
			if oldest == 0 || k.at.Before(at) {
				oldest, at = id, k.at
			}
		}
		if n <= maxEncodedFrames && bytes <= maxEncodedBytes {
			return
		}
		c.cfg.Logger.Warn("dropping the pictures of an undecided track", LabelCamera, c.cfg.Spec.ID,
			"track", oldest, "encoded_tracks", n, "encoded_bytes", bytes)
		c.drop(oldest)
	}
}
