package alpr

import (
	"context"
	"errors"
	"fmt"
	"sync"
)

// SessionPool hands out exclusive use of an inference session.
//
// # Why exclusive, when ONNX Runtime's Run() is thread-safe
//
// It is, and that is exactly what makes this trap easy to fall into. The thread-safety belongs to
// the runtime; it does not extend to the wrapper holding the input and output tensors. Two
// cameras calling the same wrapper concurrently overwrite each other's buffers, and the GPU then
// reads memory that has already been reused.
//
// The Python service tried a shared per-GPU instance on 2026-08-04 and reverted it the same day:
//
//	CUDA failure 700: an illegal memory access was encountered ; GPU=1
//
// A corrupted CUDA context does not recover. Every subsequent inference on that device fails
// until the process restarts, so the blast radius is the whole worker, not one frame.
//
// The same shape of hazard exists here: an onnxruntime_go session owns its bound tensors. This
// pool is the thing that keeps two goroutines out of one session -- it is not an optimisation,
// and shrinking it to "just share, Run is thread-safe" reintroduces the outage.
type SessionPool[T any] struct {
	mu    sync.Mutex
	build func() (T, error)
	idle  chan T
	built []T

	// closed is set by Drain. It is what stops a shutdown from being held open by a camera
	// that asks for one more session on its way out.
	closed bool
}

// NewSessionPool builds a pool of at most size sessions.
//
// Sessions are created lazily, on first checkout, rather than all at startup. A worker holding
// eight GPUs would otherwise pay for every session it might need before serving one frame, and
// each session is hundreds of megabytes of VRAM.
func NewSessionPool[T any](size int, build func() (T, error)) (*SessionPool[T], error) {
	if size < 1 {
		return nil, fmt.Errorf("alpr: session pool size must be at least 1, got %d", size)
	}
	p := &SessionPool[T]{build: build, idle: make(chan T, size)}
	var zero T
	for i := 0; i < size; i++ {
		p.idle <- zero // a lazy slot: the zero value means "not built yet"
	}
	return p, nil
}

// Borrow takes a session out of the pool and returns it with the release function.
//
// It blocks when every session is busy, and blocking is the correct behaviour: waiting some
// milliseconds is unconditionally better than two concurrent Runs on one session, which costs the
// GPU context and every camera on that device.
//
// The returned release must be called even when the caller's work fails -- otherwise a single
// error permanently shrinks the pool, and enough of them deadlock the worker.
func (p *SessionPool[T]) Borrow(ctx context.Context) (session T, release func(), err error) {
	var zero T
	p.mu.Lock()
	closed := p.closed
	p.mu.Unlock()
	if closed {
		return zero, nil, errors.New("alpr: session pool is draining")
	}
	select {
	case s := <-p.idle:
		session = s
	case <-ctx.Done():
		return zero, nil, ctx.Err()
	}

	if p.isZero(session) {
		built, buildErr := p.build()
		if buildErr != nil {
			// Give the slot back as unbuilt so the next caller may retry. Swallowing it here
			// would leak a slot on every transient build failure.
			p.idle <- zero
			return zero, nil, fmt.Errorf("alpr: build session: %w", buildErr)
		}
		session = built
		p.mu.Lock()
		p.built = append(p.built, built)
		p.mu.Unlock()
	}

	var once sync.Once
	return session, func() { once.Do(func() { p.idle <- session }) }, nil
}

// Built returns the sessions that were actually created, for shutdown.
func (p *SessionPool[T]) Built() []T {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make([]T, len(p.built))
	copy(out, p.built)
	return out
}

// isZero reports whether a slot still holds the lazy placeholder.
//
// Sessions are pointer types in practice, so comparing against the zero value is enough. The
// helper exists to keep that assumption in one named place rather than spread across Borrow.
func (p *SessionPool[T]) isZero(v T) bool {
	return any(v) == any(*new(T))
}

// Drain closes the pool and waits for every session to come back.
//
// It exists for shutdown, and it must not go through Borrow. Borrow builds a session when it
// finds an empty slot, so draining that way would start a fresh multi-second session build in
// the middle of tearing the process down -- and then destroy it.
//
// After Drain, Borrow fails fast rather than blocking: a camera still running during shutdown
// should stop, not wait for a session that is never coming.
func (p *SessionPool[T]) Drain(ctx context.Context) ([]T, error) {
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return nil, nil
	}
	p.closed = true
	size := cap(p.idle)
	p.mu.Unlock()

	var out []T
	for i := 0; i < size; i++ {
		select {
		case s := <-p.idle:
			if !p.isZero(s) {
				out = append(out, s)
			}
		case <-ctx.Done():
			return out, fmt.Errorf("alpr: %d of %d sessions did not come back: %w",
				size-i, size, ctx.Err())
		}
	}
	return out, nil
}
