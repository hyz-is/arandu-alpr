// Package runner builds one camera and starts it, which is the one thing supervisor.Runner
// leaves to whoever embeds the service.
//
// The supervisor decides WHICH cameras exist and WHERE they go; it deliberately does not know how
// to build one, because building one needs the dialer, the gate, the encoder and the JPEG policy
// -- four deployment choices a service does not get to make. What it does need is somewhere for
// the twenty lines of assembly to live, and without this package every embedder writes them
// again, including the three that are easy to get wrong:
//
//   - the Scaler is per camera (it caches weight tables for one geometry and is not safe for
//     concurrent use), while the Encoder is shared
//   - the Deduper is per camera (it is the camera's own memory of what it just read)
//   - the placement has to be written back onto the spec, or NVDEC decodes every camera on
//     device 0 -- see Start
//
// Nothing here decides anything. Every collaborator arrives in Config, and this package only
// puts them together in the arrangement each one requires.
package runner

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"time"

	"github.com/hyz-is/arandu-alpr"
	"github.com/hyz-is/arandu-alpr/camera"
	"github.com/hyz-is/arandu-alpr/contract"
	"github.com/hyz-is/arandu-alpr/supervisor"
)

// Config is everything a camera is built from.
//
// Camera carries the collaborators the application owns -- dialer, engine, gate, sink, observer,
// metrics, logger, encoder, adjust -- and every tuning knob camera.Config already names. Restating
// those here would be a second vocabulary for the same settings, and a second place for one of
// them to go missing.
type Config struct {
	// Camera is the template every camera is built from. It is copied, never shared, so two
	// cameras cannot influence each other through it.
	//
	// Three of its fields are owned by this package and must be left zero: Spec, Dedup and Focus.
	// They are per camera by nature, and a template that carried them would hand every camera in
	// the fleet the same identity, the same dedup memory and the same weight tables. New refuses
	// rather than overwriting, because silently ignoring a field somebody set is how a setting
	// comes to be believed in and absent.
	Camera camera.Config

	// NewFocus builds this camera's blur measurement. It is a constructor rather than a value
	// because *imaging.Scaler caches the bilinear weight tables for one source geometry and is
	// not safe for concurrent use: one shared between cameras makes every focus number wrong,
	// and wrong in the direction that quietly drops readable plates.
	//
	// Nil disables the blur gate. That fails OPEN on purpose -- a deployment without the imaging
	// helper should read more plates than it should, never fewer, because an empty result set is
	// the failure nobody notices.
	NewFocus func() (camera.Focus, error)

	// DedupCooldown is how long the same plate is suppressed on one camera, and DedupDistance
	// how many characters two readings may differ by and still count as the same plate.
	//
	// One Deduper per camera, never shared: a car in front of one camera has nothing to say
	// about the same plate arriving at another, and a shared memory would drop the second
	// reading -- which is the crossing an operator is usually most interested in.
	DedupCooldown time.Duration
	DedupDistance int
}

// Defaults mirror the Python service's production values.
const (
	DefaultDedupCooldown = 30 * time.Second
	DefaultDedupDistance = 1
)

// Runner builds and starts cameras. It is safe for concurrent use: the supervisor starts up to
// StartConcurrency of them at once.
type Runner struct {
	cfg Config
}

var _ supervisor.Runner = (*Runner)(nil)

// New validates the template and takes a snapshot of it.
//
// It fails at boot for what would otherwise fail once per camera, at reconcile time, inside a
// supervisor that treats a failed start as a flaky camera and retries it with backoff -- so a
// missing dialer would present as the whole fleet being unreachable rather than as one line of
// wiring that was never filled in.
func New(cfg Config) (*Runner, error) {
	var problems []error

	if cfg.Camera.Dialer == nil {
		problems = append(problems, errors.New("runner: Config.Camera.Dialer is required; the service does not know how to reach a camera"))
	}
	if cfg.Camera.Engine == nil {
		problems = append(problems, errors.New("runner: Config.Camera.Engine is required"))
	}
	if cfg.Camera.Spec.ID != "" {
		problems = append(problems, errors.New("runner: Config.Camera.Spec is per camera and comes from the Source; leave it zero"))
	}
	if cfg.Camera.Dedup != nil {
		problems = append(problems, errors.New("runner: Config.Camera.Dedup is per camera; set DedupCooldown and DedupDistance instead"))
	}
	if cfg.Camera.Focus != nil {
		problems = append(problems, errors.New("runner: Config.Camera.Focus is per camera; set NewFocus instead"))
	}
	if err := errors.Join(problems...); err != nil {
		return nil, err
	}

	if cfg.DedupCooldown <= 0 {
		cfg.DedupCooldown = DefaultDedupCooldown
	}
	if cfg.DedupDistance < 0 {
		cfg.DedupDistance = DefaultDedupDistance
	}
	return &Runner{cfg: cfg}, nil
}

// Start builds one camera and runs it on its own goroutine.
//
// # The placement write-back
//
// device is where the supervisor put this camera, and it is written back onto the spec as
// "cuda:N" before the camera is built. That is not bookkeeping: source.deviceFor parses exactly
// this string to decide which GPU's NVDEC block decodes the stream, and with no hint ffmpeg
// always chooses device 0. Skip it on a two-card box and every camera decodes on one card while
// the other's decoder silicon sits idle -- with both cards reporting healthy and the loaded one
// simply slower.
//
// It overwrites an operator's pin rather than preserving it, and that is correct: the placer has
// already honoured the pin when it named a device this worker has, so device is where the camera
// actually is. Keeping a hint that was not granted would point the decoder at a card the
// inference is not on.
//
// # Why it returns before the stream is open
//
// The supervisor bounds Start calls by StartConcurrency, and an implementation that returns early
// moves the expensive part outside that bound. Here the expensive part -- the dial -- is bounded
// by the dialer's own semaphore (source.Config.MaxConcurrentOpens), which is what that semaphore
// is for. Blocking here until the first frame instead would put one unreachable camera's connect
// timeout in front of every other camera's start, which is the failure the Python service had.
func (r *Runner) Start(ctx context.Context, spec contract.CameraSpec, device int) (supervisor.Handle, error) {
	cfg := r.cfg.Camera // a copy: the template is never mutated
	cfg.Spec = spec

	if device >= 0 {
		cfg.Spec.Placement = "cuda:" + strconv.Itoa(device)
	}

	if r.cfg.NewFocus != nil {
		focus, err := r.cfg.NewFocus()
		if err != nil {
			return nil, fmt.Errorf("runner: camera %s: focus: %w", spec.ID, err)
		}
		cfg.Focus = focus
	}
	cfg.Dedup = alpr.NewDeduper(r.cfg.DedupCooldown, r.cfg.DedupDistance)

	cam, err := camera.New(cfg)
	if err != nil {
		return nil, err
	}

	// Spawn's Stop cancels and WAITS, which is what keeps a replaced camera from holding a
	// session while its replacement asks for one.
	return supervisor.Spawn(ctx, cam.Run), nil
}
