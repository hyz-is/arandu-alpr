package supervisor

import (
	"context"
	"sync"
)

// Spawn runs fn in its own goroutine and returns a Handle for it.
//
// It is the adapter almost every Runner wants and the reason Runner is not defined in terms of a
// blocking call: a camera loop is `func(ctx) error`, and what the supervisor needs is something it
// can watch and stop. Without this each application would write the same twenty lines of
// context-cancel-plus-waitgroup, and the version that forgets to wait in Stop leaves the old
// camera holding a GPU session while its replacement asks for one.
//
// Stop cancels and WAITS. Returning before fn has finished would let a redeploy's new camera open
// a session while the old one still holds it -- the failure the engine's explicit Close exists to
// prevent.
func Spawn(ctx context.Context, fn func(context.Context) error) Handle {
	runCtx, cancel := context.WithCancel(ctx)
	h := &spawned{cancel: cancel, done: make(chan struct{})}

	go func() {
		defer close(h.done)
		err := fn(runCtx)
		h.mu.Lock()
		h.err = err
		h.mu.Unlock()
	}()

	return h
}

type spawned struct {
	cancel context.CancelFunc
	done   chan struct{}

	mu  sync.Mutex
	err error
}

func (h *spawned) Done() <-chan struct{} { return h.done }

func (h *spawned) Err() error {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.err
}

func (h *spawned) Stop() {
	h.cancel()
	<-h.done
}
