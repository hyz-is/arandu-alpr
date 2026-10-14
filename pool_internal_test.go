package alpr

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type fakeSession struct{ id int }

func TestPoolNeverHandsOutOneSessionTwice(t *testing.T) {
	// This is the whole point of the type. Two goroutines holding the same session overwrite each
	// other's bound tensors, and the GPU reads memory that has been reused -- CUDA 700, and a
	// context that stays corrupted until the process restarts.
	const size = 4
	var built int32
	p, err := NewSessionPool(size, func() (*fakeSession, error) {
		return &fakeSession{id: int(atomic.AddInt32(&built, 1))}, nil
	})
	if err != nil {
		t.Fatal(err)
	}

	var mu sync.Mutex
	inUse := map[*fakeSession]bool{}

	var wg sync.WaitGroup
	for i := 0; i < 200; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			s, release, err := p.Borrow(context.Background())
			if err != nil {
				t.Error(err)
				return
			}
			defer release()

			mu.Lock()
			if inUse[s] {
				mu.Unlock()
				t.Errorf("session %d handed out twice -- this is the CUDA 700 path", s.id)
				return
			}
			inUse[s] = true
			mu.Unlock()

			time.Sleep(time.Millisecond)

			mu.Lock()
			delete(inUse, s)
			mu.Unlock()
		}()
	}
	wg.Wait()

	if got := atomic.LoadInt32(&built); got > size {
		t.Errorf("built %d sessions for a pool of %d", got, size)
	}
}

func TestPoolBuildsLazily(t *testing.T) {
	// A worker with several GPUs would otherwise pay hundreds of megabytes of VRAM per session
	// before serving a single frame.
	var built int32
	p, err := NewSessionPool(8, func() (*fakeSession, error) {
		atomic.AddInt32(&built, 1)
		return &fakeSession{}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := atomic.LoadInt32(&built); got != 0 {
		t.Fatalf("%d sessions built before any checkout", got)
	}

	_, release, err := p.Borrow(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	release()

	if got := atomic.LoadInt32(&built); got != 1 {
		t.Errorf("built %d sessions for one checkout, want 1", got)
	}
}

func TestPoolReturnsSlotWhenBuildFails(t *testing.T) {
	// Swallowing the slot here would shrink the pool on every transient failure, and enough of
	// them deadlock the worker with no error to point at.
	var attempts int32
	p, err := NewSessionPool(1, func() (*fakeSession, error) {
		if atomic.AddInt32(&attempts, 1) == 1 {
			return nil, errors.New("cuda unavailable")
		}
		return &fakeSession{}, nil
	})
	if err != nil {
		t.Fatal(err)
	}

	if _, _, err := p.Borrow(context.Background()); err == nil {
		t.Fatal("first borrow should have failed")
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	s, release, err := p.Borrow(ctx)
	if err != nil {
		t.Fatalf("slot was lost after a failed build: %v", err)
	}
	release()
	if s == nil {
		t.Error("retry returned a nil session")
	}
}

func TestPoolBlocksRatherThanOversubscribing(t *testing.T) {
	// Waiting some milliseconds is unconditionally better than two Runs on one session. If this
	// ever returns instead of blocking, the pool has stopped being a mutual-exclusion device.
	p, err := NewSessionPool(1, func() (*fakeSession, error) { return &fakeSession{}, nil })
	if err != nil {
		t.Fatal(err)
	}

	_, release, err := p.Borrow(context.Background())
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	if _, _, err := p.Borrow(ctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("borrow returned %v while the only session was held; expected to block", err)
	}
	release()
}

func TestReleaseIsIdempotent(t *testing.T) {
	// A double release would put the same session into the pool twice, and from then on two
	// goroutines could hold it at once -- the exact failure the pool exists to prevent, arriving
	// through the door marked "cleanup".
	p, err := NewSessionPool(1, func() (*fakeSession, error) { return &fakeSession{}, nil })
	if err != nil {
		t.Fatal(err)
	}
	_, release, err := p.Borrow(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	release()
	release()

	_, r2, err := p.Borrow(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer r2()

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	if _, _, err := p.Borrow(ctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Error("pool grew past its size after a double release")
	}
}

func TestPoolRejectsZeroSize(t *testing.T) {
	if _, err := NewSessionPool(0, func() (*fakeSession, error) { return nil, nil }); err == nil {
		t.Error("a pool of zero sessions would deadlock on first use")
	}
}
