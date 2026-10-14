package onnx

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"

	ort "github.com/yalue/onnxruntime_go"
)

// TensorRTConfig is implicit INT8: a float model plus a calibration table.
//
// It is the configuration the Python worker runs in production on the T4s, with the same model
// files and the same two tables. The QDQ exports were tried first and TensorRT refused them on
// Turing (a DequantizeLinear on the bias, chooseHigherPrecision on the detector, the Myelin weight
// conversion on the OCR). A float model with a table is what TensorRT quantizes itself, and the
// tensors left out of the table -- the detector head, the stem, the OCR attention -- run in FP16.
// Boxes and scores share one scale in that head, which is why INT8 there found no plate at all.
//
// The CUDA provider stays behind it for any node TensorRT does not take.
type TensorRTConfig struct {
	Enabled bool

	// FP16 lets TensorRT run in half precision whatever the table leaves out. Without it those
	// tensors run in FP32, which is correct and slower.
	FP16 bool

	// CacheDir holds the compiled engines, the timing cache and a copy of each table, one
	// subdirectory per role. It belongs on a volume: compiling the two engines on a T4 takes the
	// better part of half an hour, and a cache that dies with the container is paid on every
	// deploy, with no camera read in the meantime.
	CacheDir string

	// DetectorTable and OCRTable are ONNX Runtime calibration tables (flatbuffers). TensorRT
	// reads the table from the engine cache directory, so each is copied there before the
	// session is built.
	DetectorTable string
	OCRTable      string
}

// table returns the calibration table configured for a role.
func (t TensorRTConfig) table(role string) string {
	if role == RoleDetector {
		return t.DetectorTable
	}
	return t.OCRTable
}

// calibrationName is the file name the table is copied to inside each role's cache directory.
const calibrationName = "calibration.flatbuffers"

// appendTensorRT adds the TensorRT provider for one session, in implicit INT8.
func (b *Builder) appendTensorRT(opts *ort.SessionOptions, spec SessionSpec) error {
	t := b.cfg.TensorRT
	dir := filepath.Join(t.CacheDir, spec.Role)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("alpr/onnx: TensorRT cache %s: %w", dir, err)
	}
	if err := copyIfNewer(t.table(spec.Role), filepath.Join(dir, calibrationName)); err != nil {
		return fmt.Errorf("alpr/onnx: %s calibration table: %w", spec.Role, err)
	}

	trt, err := ort.NewTensorRTProviderOptions()
	if err != nil {
		return fmt.Errorf("alpr/onnx: TensorRT provider options: %w", err)
	}
	defer trt.Destroy()

	options := map[string]string{
		"device_id":       strconv.Itoa(spec.Device),
		"trt_int8_enable": "1",
		"trt_fp16_enable": boolFlag(t.FP16),
		// The table is ONNX Runtime's own format, written by its calibrator, not TensorRT's.
		"trt_int8_calibration_table_name":       calibrationName,
		"trt_int8_use_native_calibration_table": "0",
		// The engine is keyed by the model and the GPU, so one cache serves every session of a
		// role on cards of the same kind, and a different card compiles its own.
		"trt_engine_cache_enable": "1",
		"trt_engine_cache_path":   dir,
		"trt_timing_cache_enable": "1",
		"trt_timing_cache_path":   dir,
	}
	if err := trt.Update(options); err != nil {
		return fmt.Errorf("alpr/onnx: TensorRT options: %w", err)
	}
	if err := opts.AppendExecutionProviderTensorRT(trt); err != nil {
		return fmt.Errorf("alpr/onnx: append TensorRT provider for device %d: %w", spec.Device, err)
	}
	return nil
}

// copyIfNewer copies src to dst unless dst is already at least as recent.
func copyIfNewer(src, dst string) error {
	si, err := os.Stat(src)
	if err != nil {
		return err
	}
	if di, err := os.Stat(dst); err == nil && !di.ModTime().Before(si.ModTime()) {
		return nil
	}
	data, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	tmp := dst + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, dst)
}

func boolFlag(v bool) string {
	if v {
		return "1"
	}
	return "0"
}
