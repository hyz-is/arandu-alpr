package camera

import (
	"github.com/hyz-is/arandu-alpr/engine"
)

// roiFrame returns the part of f the camera's ROI covers, copied into a buffer the camera owns,
// and the offset of its top-left corner in f. Without a ROI, or with one that does not fit, it
// returns f itself and no offset.
//
// The copy is rows of packed BGR24, one per row of the region, into a buffer that grows once to
// the region's size and is reused for every frame after: the detector borrows it for the length
// of one call, as it borrows f.
func (c *Camera) roiFrame(f engine.Frame) (engine.Frame, int, int) {
	x0, y0, x1, y1, ok := c.cfg.Spec.ROI.Pixels(f.Width, f.Height)
	if !ok || len(f.Pix) < f.Width*f.Height*3 {
		return f, 0, 0
	}
	w, h := x1-x0, y1-y0
	need := w * h * 3
	if cap(c.roiPix) < need {
		c.roiPix = make([]byte, need)
	}
	buf := c.roiPix[:need]
	for y := range h {
		src := ((y0+y)*f.Width + x0) * 3
		copy(buf[y*w*3:(y+1)*w*3], f.Pix[src:src+w*3])
	}
	return engine.Frame{Pix: buf, Width: w, Height: h, CapturedAt: f.CapturedAt}, x0, y0
}
