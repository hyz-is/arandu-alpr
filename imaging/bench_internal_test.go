package imaging

import (
	"testing"
	"time"

	"github.com/hyz-is/arandu-alpr/contract"
	"github.com/hyz-is/arandu-alpr/engine"
)

// These measure the CPU work that happens on BOTH machines.
//
// Everything else in the pipeline either waits on the GPU or is arithmetic on a handful of
// numbers. This package is where the per-frame cost of megabytes of pixels actually lands, and it
// is the part of "how many cameras per server" that a laptop can answer honestly -- the same code
// runs on the CUDA box, on similar cores, doing the same work.
//
// A benchmark that only exists in a report cannot be re-run against a change. These are committed
// for that reason.

func benchFrame(w, h int) engine.Frame {
	pix := make([]byte, w*h*3)
	// Textured rather than flat: a solid buffer lets a resize hit best-case cache behaviour and
	// makes the number optimistic in a way that never reproduces on a real camera.
	for i := range pix {
		pix[i] = byte(i * 7 % 251)
	}
	return engine.Frame{Pix: pix, Width: w, Height: h, CapturedAt: time.Now()}
}

func BenchmarkLetterbox(b *testing.B) {
	for _, size := range []struct {
		name string
		w, h int
	}{
		{"640x480", 640, 480},
		{"1280x720", 1280, 720},
		{"1920x1080", 1920, 1080},
		{"3840x2160", 3840, 2160},
	} {
		b.Run(size.name, func(b *testing.B) {
			s, err := NewScaler(engine.DetectorInputSize)
			if err != nil {
				b.Fatal(err)
			}
			f := benchFrame(size.w, size.h)
			dst := make([]float32, s.TensorLen())

			b.ReportAllocs()
			b.SetBytes(int64(len(f.Pix)))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := s.Letterbox(f, dst); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkCrop(b *testing.B) {
	s, err := NewScaler(engine.DetectorInputSize)
	if err != nil {
		b.Fatal(err)
	}
	f := benchFrame(1920, 1080)
	box := contract.BBox{X: 800, Y: 500, Width: 240, Height: 70}
	dst := make([]float32, CropOCRTensorLen)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := s.Crop(f, box, dst); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFocusVariance(b *testing.B) {
	s, err := NewScaler(engine.DetectorInputSize)
	if err != nil {
		b.Fatal(err)
	}
	f := benchFrame(1920, 1080)
	box := contract.BBox{X: 800, Y: 500, Width: 240, Height: 70}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = s.FocusVariance(f, box)
	}
}

// BenchmarkAdjustBrightnessContrast measures both paths, because the interesting number is the
// gap between them: the 100/100 case is skipped entirely by the camera loop, and this is what it
// would otherwise cost on every frame.
func BenchmarkAdjustBrightnessContrast(b *testing.B) {
	f := benchFrame(1920, 1080)

	b.Run("untouched-100/100", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(len(f.Pix)))
		for i := 0; i < b.N; i++ {
			AdjustBrightnessContrast(f.Pix, 100, 100)
		}
	})
	b.Run("adjusted-110/120", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(len(f.Pix)))
		for i := 0; i < b.N; i++ {
			AdjustBrightnessContrast(f.Pix, 110, 120)
		}
	})
}

func BenchmarkEncodeCrop(b *testing.B) {
	e, err := NewEncoder(70)
	if err != nil {
		b.Fatal(err)
	}
	f := benchFrame(1920, 1080)
	box := contract.BBox{X: 800, Y: 500, Width: 240, Height: 70}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := e.EncodeCrop(f, box); err != nil {
			b.Fatal(err)
		}
	}
}
