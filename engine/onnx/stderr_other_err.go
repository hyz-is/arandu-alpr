//go:build !unix

package onnx

import "errors"

var errNoCapture = errors.New("alpr/onnx: stderr capture is not supported on this platform")
