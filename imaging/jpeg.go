package imaging

import (
	"bytes"
	"errors"
	"fmt"
	"image"
	"image/jpeg"
	"sync"

	"github.com/hyz-is/arandu-alpr/contract"
	"github.com/hyz-is/arandu-alpr/engine"
)

// Encoder turns frames and crops into JPEG bytes.
//
// It is the one thing in this package that is safe to share between cameras, and it is shared on
// purpose. Unlike the Scaler's buffers, whose size is fixed by one camera's geometry, the scratch
// an encode needs is proportional to whatever it was handed and is held only for the duration of
// a call. A pool is the right shape for that: the number of encodes in flight tracks the number
// of detections, which is bursty and unpredictable, so a per-camera buffer would be sized for the
// worst case and idle the rest of the time.
//
// The returned bytes are always a fresh copy. contract.Detection promises that FullJPEG and
// CropJPEG are owned by the receiver and may be kept forever, and handing back a pooled buffer
// would make that a lie in the way that corrupts an image rather than failing.
type Encoder struct {
	opts jpeg.Options
	bufs sync.Pool
	imgs sync.Pool
}

// NewEncoder returns an Encoder writing JPEGs at the given quality (1-100).
func NewEncoder(quality int) (*Encoder, error) {
	if quality < 1 || quality > 100 {
		return nil, fmt.Errorf("imaging: jpeg quality must be 1-100, got %d", quality)
	}
	return &Encoder{
		opts: jpeg.Options{Quality: quality},
		bufs: sync.Pool{New: func() any { return new(bytes.Buffer) }},
		imgs: sync.Pool{New: func() any { return new(image.RGBA) }},
	}, nil
}

// EncodeFrame encodes the whole frame.
func (e *Encoder) EncodeFrame(f engine.Frame) ([]byte, error) {
	if f.Width <= 0 || f.Height <= 0 || len(f.Pix) != f.Width*f.Height*3 {
		return nil, ErrFrameSize
	}
	return e.encodeRegion(f, 0, 0, f.Width, f.Height)
}

// EncodeCrop encodes the plate region of the frame.
//
// The box is expanded by the same CropExpand* fractions the OCR's crop uses, so the stored
// picture is the picture the model read. A tighter evidence crop than the model's input makes
// every disagreement about a reading unarguable in the wrong direction: the human sees less than
// the model did.
func (e *Encoder) EncodeCrop(f engine.Frame, box contract.BBox) ([]byte, error) {
	if f.Width <= 0 || f.Height <= 0 || len(f.Pix) != f.Width*f.Height*3 {
		return nil, ErrFrameSize
	}
	x0, y0, w, h := clampedRegion(ExpandBox(box), f.Width, f.Height)
	return e.encodeRegion(f, x0, y0, w, h)
}

// encodeRegion converts a packed BGR24 region to RGBA and encodes it.
//
// image/jpeg has no BGR path, so the conversion is unavoidable; doing it into a pooled RGBA
// keeps it to one pass and no garbage beyond the result.
func (e *Encoder) encodeRegion(f engine.Frame, x0, y0, w, h int) ([]byte, error) {
	if w <= 0 || h <= 0 {
		return nil, errors.New("imaging: cannot encode an empty region")
	}

	img, _ := e.imgs.Get().(*image.RGBA)
	buf, _ := e.bufs.Get().(*bytes.Buffer)
	defer func() {
		e.imgs.Put(img)
		e.bufs.Put(buf)
	}()

	n := w * h * 4
	if cap(img.Pix) < n {
		img.Pix = make([]uint8, n)
	}
	img.Pix = img.Pix[:n]
	img.Stride = w * 4
	img.Rect = image.Rect(0, 0, w, h)

	src := f.Pix
	for y := 0; y < h; y++ {
		row := ((y0+y)*f.Width + x0) * 3
		out := y * img.Stride
		for x := 0; x < w; x++ {
			i := row + x*3
			o := out + x*4
			img.Pix[o] = src[i+2]
			img.Pix[o+1] = src[i+1]
			img.Pix[o+2] = src[i]
			img.Pix[o+3] = 0xff
		}
	}

	buf.Reset()
	if err := jpeg.Encode(buf, img, &e.opts); err != nil {
		return nil, fmt.Errorf("imaging: encode jpeg: %w", err)
	}

	out := make([]byte, buf.Len())
	copy(out, buf.Bytes())
	return out, nil
}
