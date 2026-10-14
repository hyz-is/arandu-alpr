package unit_test

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/hyz-is/arandu-alpr"
	"github.com/hyz-is/arandu-alpr/camera"
	"github.com/hyz-is/arandu-alpr/contract"
	"github.com/hyz-is/arandu-alpr/engine"
	"github.com/hyz-is/arandu-alpr/runner"
)

// stubStream is a feed that never delivers a frame. The camera loop parks in its select until the
// context ends, which is exactly the state these tests want: started, and doing nothing else.
type stubStream struct {
	frames chan engine.Frame
	once   sync.Once
}

func (s *stubStream) Frames() <-chan engine.Frame { return s.frames }
func (s *stubStream) Close()                      { s.once.Do(func() { close(s.frames) }) }

// stubDialer records the spec it was handed. That spec is the whole subject of these tests: it is
// what source.FFmpegDialer reads the decode device out of.
type stubDialer struct {
	mu     sync.Mutex
	specs  []contract.CameraSpec
	dialed chan struct{}
}

func newStubDialer() *stubDialer { return &stubDialer{dialed: make(chan struct{}, 8)} }

func (d *stubDialer) Dial(_ context.Context, spec contract.CameraSpec) (camera.Stream, error) {
	d.mu.Lock()
	d.specs = append(d.specs, spec)
	d.mu.Unlock()
	select {
	case d.dialed <- struct{}{}:
	default:
	}
	return &stubStream{frames: make(chan engine.Frame)}, nil
}

func (d *stubDialer) first(t *testing.T) contract.CameraSpec {
	t.Helper()
	select {
	case <-d.dialed:
	case <-time.After(2 * time.Second):
		t.Fatal("the camera never dialled")
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.specs[0]
}

func fakeEngine() engine.Engine {
	return engine.NewFake(engine.FakeConfig{ReadyAtStart: true})
}

func spec(id string) contract.CameraSpec {
	return contract.CameraSpec{ID: id, URL: "rtsp://camera/stream", Enabled: true, Brightness: 100, Contrast: 100}
}

// TestStartWritesThePlacementBackOntoTheSpec.
//
// This is the reason this package exists rather than being three lines in a main. The supervisor
// balances cameras across cards and tells the runner which one; nothing else does. If that number
// is not written onto the spec as "cuda:N", source.deviceFor sees an empty placement, ffmpeg
// picks device 0, and every camera in the fleet decodes on one card -- with both cards reporting
// healthy and nothing anywhere reporting an error.
func TestStartWritesThePlacementBackOntoTheSpec(t *testing.T) {
	dialer := newStubDialer()
	r, err := runner.New(runner.Config{
		Camera: camera.Config{Dialer: dialer, Engine: fakeEngine(), Gate: alpr.DefaultGate()},
	})
	if err != nil {
		t.Fatal(err)
	}

	handle, err := r.Start(context.Background(), spec("gate-1"), 1)
	if err != nil {
		t.Fatal(err)
	}
	defer handle.Stop()

	if got := dialer.first(t).Placement; got != "cuda:1" {
		t.Errorf("placement handed to the dialer = %q, want %q\n\n"+
			"Without it NVDEC decodes every camera on device 0.", got, "cuda:1")
	}
}

// TestStartOverwritesAPinThePlacerDidNotGrant.
//
// The spec may carry an operator's pin, and the placer has already had its say about it: what
// comes back as `device` is where the camera really is. Preserving a hint that was not granted
// would point the decoder at a card the inference is not on, which is worse than no hint at all.
func TestStartOverwritesAPinThePlacerDidNotGrant(t *testing.T) {
	dialer := newStubDialer()
	r, err := runner.New(runner.Config{
		Camera: camera.Config{Dialer: dialer, Engine: fakeEngine(), Gate: alpr.DefaultGate()},
	})
	if err != nil {
		t.Fatal(err)
	}

	pinned := spec("gate-1")
	pinned.Placement = "cuda:7" // a card this worker does not have
	handle, err := r.Start(context.Background(), pinned, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer handle.Stop()

	if got := dialer.first(t).Placement; got != "cuda:0" {
		t.Errorf("placement = %q, want %q: the placer's answer is where the camera actually is", got, "cuda:0")
	}
}

// TestStartLeavesThePlacementAloneWithoutADevice covers the CPU deployment: a negative device is
// "no card", and claiming cuda:-1 would be a hint that parses into nonsense.
func TestStartLeavesThePlacementAloneWithoutADevice(t *testing.T) {
	dialer := newStubDialer()
	r, err := runner.New(runner.Config{
		Camera: camera.Config{Dialer: dialer, Engine: fakeEngine(), Gate: alpr.DefaultGate()},
	})
	if err != nil {
		t.Fatal(err)
	}

	handle, err := r.Start(context.Background(), spec("gate-1"), -1)
	if err != nil {
		t.Fatal(err)
	}
	defer handle.Stop()

	if got := dialer.first(t).Placement; got != "" {
		t.Errorf("placement = %q, want empty: there is no card to name", got)
	}
}

// TestEachCameraGetsItsOwnFocus.
//
// *imaging.Scaler caches the weight tables for one source geometry and is not safe for concurrent
// use. One shared between cameras is not a crash: it is every focus measurement being computed
// against another camera's geometry, which drops readable plates and reports nothing.
func TestEachCameraGetsItsOwnFocus(t *testing.T) {
	var built int
	var mu sync.Mutex

	r, err := runner.New(runner.Config{
		Camera: camera.Config{Dialer: newStubDialer(), Engine: fakeEngine(), Gate: alpr.DefaultGate()},
		NewFocus: func() (camera.Focus, error) {
			mu.Lock()
			built++
			mu.Unlock()
			return stubFocus{}, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	for _, id := range []string{"a", "b", "c"} {
		handle, err := r.Start(context.Background(), spec(id), 0)
		if err != nil {
			t.Fatal(err)
		}
		defer handle.Stop()
	}

	mu.Lock()
	defer mu.Unlock()
	if built != 3 {
		t.Errorf("NewFocus called %d times for 3 cameras, want 3", built)
	}
}

type stubFocus struct{}

func (stubFocus) FocusVariance(engine.Frame, contract.BBox) float64 { return 1000 }

// TestNewRefusesPerCameraFieldsInTheTemplate.
//
// Overwriting them silently would be worse than refusing: a Dedup set in the template looks like
// a fleet-wide cooldown, and one that is ignored looks like a cooldown that does not work.
func TestNewRefusesPerCameraFieldsInTheTemplate(t *testing.T) {
	base := func() camera.Config {
		return camera.Config{Dialer: newStubDialer(), Engine: fakeEngine()}
	}

	cases := map[string]func(*camera.Config){
		"Spec":  func(c *camera.Config) { c.Spec = spec("templated") },
		"Dedup": func(c *camera.Config) { c.Dedup = alpr.NewDeduper(time.Second, 1) },
		"Focus": func(c *camera.Config) { c.Focus = stubFocus{} },
	}

	for name, set := range cases {
		t.Run(name, func(t *testing.T) {
			cam := base()
			set(&cam)
			if _, err := runner.New(runner.Config{Camera: cam}); err == nil {
				t.Fatalf("a template carrying %s was accepted", name)
			} else if !strings.Contains(err.Error(), name) {
				t.Errorf("error does not name the field that is wrong: %v", err)
			}
		})
	}
}

// TestNewRefusesAnIncompleteTemplate: the dialer and the engine are what a camera cannot be built
// without, and a supervisor would otherwise report their absence as a fleet of flaky cameras.
func TestNewRefusesAnIncompleteTemplate(t *testing.T) {
	_, err := runner.New(runner.Config{})
	if err == nil {
		t.Fatal("an empty template was accepted")
	}
	for _, want := range []string{"Dialer", "Engine"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error does not mention %s: %v", want, err)
		}
	}
}

// TestStopEndsTheCamera: Stop must return only once the loop has finished, or a replaced camera
// still holds its session while the replacement asks for one.
func TestStopEndsTheCamera(t *testing.T) {
	dialer := newStubDialer()
	r, err := runner.New(runner.Config{
		Camera: camera.Config{Dialer: dialer, Engine: fakeEngine(), Gate: alpr.DefaultGate()},
	})
	if err != nil {
		t.Fatal(err)
	}

	handle, err := r.Start(context.Background(), spec("gate-1"), 0)
	if err != nil {
		t.Fatal(err)
	}
	dialer.first(t)

	handle.Stop()
	select {
	case <-handle.Done():
	default:
		t.Fatal("Stop returned while the camera was still running")
	}
	if err := handle.Err(); !errors.Is(err, context.Canceled) {
		t.Errorf("Err() = %v, want context.Canceled", err)
	}
}
