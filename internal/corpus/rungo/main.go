// Command rungo produces the Go side of the accuracy corpus.
//
// It reads the manifest, runs each frame through the real pipeline, and writes one JSON file per
// frame. The Python side writes the same shape from the same manifest, and compare reads both.
//
// Without ONNX Runtime and a model this cannot run, and that is correct: a corpus produced by a
// fake engine would compare a fake against Python and pass.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	_ "image/png"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/hyz-is/arandu-alpr"
	"github.com/hyz-is/arandu-alpr/engine"
	"github.com/hyz-is/arandu-alpr/engine/onnx"
	"github.com/hyz-is/arandu-alpr/internal/corpus"
)

func main() {
	manifestPath := flag.String("manifest", "internal/corpus/manifest.json", "frame manifest")
	outDir := flag.String("out", "internal/corpus/results/go", "where to write results")
	detector := flag.String("detector", os.Getenv("DETECTOR_MODEL_PATH"), "detector .ort")
	ocr := flag.String("ocr", os.Getenv("OCR_MODEL_PATH"), "ocr .ort")
	device := flag.Int("device", -1, "CUDA device id, or -1 for CPU")
	gates := flag.Bool("gates", true, "apply the quality gates before the OCR")
	flag.Parse()

	if *detector == "" || *ocr == "" {
		fmt.Fprintln(os.Stderr, "set -detector and -ocr (or DETECTOR_MODEL_PATH / OCR_MODEL_PATH)")
		os.Exit(2)
	}

	if err := run(*manifestPath, *outDir, *detector, *ocr, *device, *gates); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(manifestPath, outDir, detectorPath, ocrPath string, device int, gates bool) error {
	body, err := os.ReadFile(manifestPath)
	if err != nil {
		return err
	}
	var m corpus.Manifest
	if err := json.Unmarshal(body, &m); err != nil {
		return err
	}

	// The alphabet is hashed rather than compared, because the failure it guards against is a
	// reordering: same characters, same length, different indices, and every plate comes back
	// looking like a plate.
	alphabetSHA := sha(alpr.Alphabet)
	if m.Alphabet != "" && m.Alphabet != alpr.Alphabet {
		return fmt.Errorf("the manifest was built against a different alphabet:\n  manifest %q\n  binary   %q\n"+
			"Every plate would decode with the wrong letters at ordinary confidence",
			m.Alphabet, alpr.Alphabet)
	}

	devices := []int{device}
	builder := onnx.NewBuilder(onnx.Config{Devices: devices, LibraryPath: os.Getenv("ONNXRUNTIME_LIB_PATH")})
	if err := builder.Init(); err != nil {
		return err
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Serve has to own this goroutine's thread, and this command's main goroutine is the only
	// one that is on the process's main thread. See the package comment in engine/onnx.
	errc := make(chan error, 1)
	go func() { errc <- builder.Serve(ctx) }()

	eng, err := onnx.New(ctx, builder, onnx.EngineConfig{
		DetectorPath: detectorPath,
		OCRPath:      ocrPath,
	})
	if err != nil {
		return err
	}
	defer eng.Close(context.Background())

	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return err
	}
	gate := alpr.DefaultGate()
	base := filepath.Dir(manifestPath)

	for _, f := range m.Frames {
		res, err := processFrame(ctx, eng, gate, gates, base, f, alphabetSHA)
		if err != nil {
			return fmt.Errorf("%s: %w", f.ID, err)
		}
		out, err := json.MarshalIndent(res, "", "  ")
		if err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(outDir, f.ID+".json"), out, 0o644); err != nil {
			return err
		}
		fmt.Printf("%-24s %d reading(s)\n", f.ID, len(res.Readings))
	}
	cancel()
	<-errc
	return nil
}

func processFrame(
	ctx context.Context, eng *onnx.Engine, gate alpr.GateConfig, applyGates bool,
	base string, f corpus.Frame, alphabetSHA string,
) (corpus.Result, error) {
	frame, err := loadBGR(filepath.Join(base, f.File))
	if err != nil {
		return corpus.Result{}, err
	}
	if f.SHA256 != "" {
		sum, err := fileSHA(filepath.Join(base, f.File))
		if err != nil {
			return corpus.Result{}, err
		}
		if sum != f.SHA256 {
			return corpus.Result{}, fmt.Errorf("frame changed since the manifest was written "+
				"(sha256 %s, manifest says %s); a corpus that drifts silently proves nothing", sum, f.SHA256)
		}
	}

	info := eng.ModelInfo()
	res := corpus.Result{
		FrameID: f.ID, Impl: "go",
		DetectorSHA: info.Detector, OCRSHA: info.OCR, AlphabetSHA: alphabetSHA,
		GatesApplied: applyGates,
	}

	cands, err := eng.Detect(ctx, frame, nil)
	if err != nil {
		return res, err
	}

	for _, c := range cands {
		box := alpr.BBox{X: c.Box.X, Y: c.Box.Y, Width: c.Box.Width, Height: c.Box.Height}
		if applyGates && !gate.AcceptDetection(c.Score, box, frame.Width, frame.Height) {
			continue
		}
		r, err := eng.Recognize(ctx, frame, c.Box)
		if err != nil {
			return res, err
		}
		if applyGates && !gate.AcceptReading(r.Plate, r.Confidence) {
			continue
		}
		res.Readings = append(res.Readings, corpus.Reading{
			Plate: r.Plate, CharConfidences: r.CharConfidences, Confidence: r.Confidence,
			Region: r.Region, RegionConfidence: r.RegionConfidence,
			Box:   [4]float64{c.Box.X, c.Box.Y, c.Box.Width, c.Box.Height},
			Score: c.Score,
		})
	}

	// Deterministic order, so a diff between two runs is a real difference and never an
	// artefact of which candidate the NMS happened to emit first.
	sort.Slice(res.Readings, func(i, j int) bool {
		a, b := res.Readings[i], res.Readings[j]
		if a.Score != b.Score {
			return a.Score > b.Score
		}
		if a.Box[0] != b.Box[0] {
			return a.Box[0] < b.Box[0]
		}
		return a.Box[1] < b.Box[1]
	})
	return res, nil
}

// loadBGR decodes a PNG into the packed BGR24 the pipeline expects.
//
// PNG, never JPEG: a lossy codec makes the two implementations disagree at the decoder, before
// either model has seen anything, and that difference is indistinguishable from a preprocessing
// bug in the report.
func loadBGR(path string) (engine.Frame, error) {
	f, err := os.Open(path)
	if err != nil {
		return engine.Frame{}, err
	}
	defer f.Close()

	img, format, err := image.Decode(f)
	if err != nil {
		return engine.Frame{}, err
	}
	if format != "png" {
		return engine.Frame{}, fmt.Errorf("%s is %s; the corpus is PNG so the decode cannot "+
			"itself be a source of disagreement", path, format)
	}

	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	pix := make([]byte, w*h*3)
	i := 0
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			r, g, bl, _ := img.At(x, y).RGBA()
			pix[i+0] = byte(bl >> 8)
			pix[i+1] = byte(g >> 8)
			pix[i+2] = byte(r >> 8)
			i += 3
		}
	}
	return engine.Frame{Pix: pix, Width: w, Height: h, CapturedAt: time.Now()}, nil
}

func fileSHA(path string) (string, error) {
	body, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(body)
	return hex.EncodeToString(sum[:]), nil
}

func sha(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])[:16]
}
