// Package bench measures the ALPR pipeline, in two tiers that answer two different questions.
//
// # Tier A -- here, on a laptop, with no GPU
//
// Everything in this package. It measures the PIPELINE, not the model: the camera loop, the
// buffer ring, the backpressure policy and the per-frame allocation budget, against
// engine.Fake with synthetic latency. That is not a compromise for want of hardware -- these
// are the parts whose behaviour under saturation decides whether a fleet fits, and they are
// exactly the parts a GPU benchmark cannot isolate because the model's variance swamps them.
//
//	go test -run '^$' -bench . -benchmem ./bench/
//	go test ./bench/...
//
// Tier A numbers are ratios and shapes -- allocations per frame, whether a busy engine drops or
// queues, whether the ring starves a slow consumer. They are not fleet capacity, and nothing
// here should ever be quoted as such.
//
// # Tier B -- bench/fleet, on the CUDA box
//
// A separate binary behind `//go:build cuda`. It answers the one question Tier A cannot: how
// many cameras one GPU carries. See that package for the protocol; the two rules that make its
// numbers mean anything are that the input is paced with ffmpeg's `-re` and that every line
// carries the full config, the git SHA and the SHA256 of each model.
//
// # bench/compare
//
// Diffs two JSON lines and prints what moved. It refuses to compare runs whose config or models
// differ, which is the most useful thing it does.
package bench
