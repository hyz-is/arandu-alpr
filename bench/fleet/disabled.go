//go:build !cuda

// Command fleet is the tier B sweep. It runs only on a machine with CUDA and ONNX Runtime, and
// it is behind a build tag so that an accidental `go run ./bench/fleet` on a laptop says so
// instead of failing somewhere inside dlopen.
//
// The tag guards the RUN, not the compile. Everything under -tags cuda still has to type-check
// on a machine with no GPU:
//
//	GOOS=linux go vet -tags cuda ./...
//
// That is deliberate. A benchmark harness that only compiles on the box it runs on rots between
// visits to that box, and the rot is discovered at the moment somebody needs a number.
package main

import (
	"fmt"
	"os"
)

func main() {
	fmt.Fprintln(os.Stderr, `fleet: built without -tags cuda, so there is nothing to run here.

This is the tier B sweep: it needs a CUDA GPU, ONNX Runtime, ffmpeg, and an RTSP server to
replay clips through. Build and run it on the GPU box:

    go build -tags cuda -o fleet ./bench/fleet
    ./fleet -help

On a machine without a GPU, the suite that does run is tier A:

    go test ./bench/...
    go test -run '^$' -bench . -benchmem ./bench/`)
	os.Exit(2)
}
