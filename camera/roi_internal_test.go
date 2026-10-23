package camera

import (
	"testing"

	"github.com/hyz-is/arandu-alpr/contract"
	"github.com/hyz-is/arandu-alpr/engine"
)

func TestTheROIIsCutOutOfTheFrameRowByRow(t *testing.T) {
	const w, h = 100, 50
	pix := make([]byte, w*h*3)
	for y := range h {
		for x := range w {
			pix[(y*w+x)*3] = byte(x)
			pix[(y*w+x)*3+1] = byte(y)
		}
	}
	c := &Camera{cfg: Config{Spec: contract.CameraSpec{ROI: &contract.Region{X: 0.2, Y: 0.4, Width: 0.5, Height: 0.5}}}}
	region, dx, dy := c.roiFrame(engine.Frame{Pix: pix, Width: w, Height: h})
	if dx != 20 || dy != 20 || region.Width != 50 || region.Height != 25 {
		t.Fatalf("region %dx%d at %d,%d", region.Width, region.Height, dx, dy)
	}
	// The region's top-left pixel is the frame's pixel at the offset.
	if region.Pix[0] != 20 || region.Pix[1] != 20 {
		t.Fatalf("top-left pixel is x=%d y=%d, want 20,20", region.Pix[0], region.Pix[1])
	}
	last := (24*50 + 49) * 3
	if region.Pix[last] != 69 || region.Pix[last+1] != 44 {
		t.Fatalf("bottom-right pixel is x=%d y=%d, want 69,44", region.Pix[last], region.Pix[last+1])
	}
}

func TestARegionThatDoesNotFitIsTheWholeFrame(t *testing.T) {
	f := engine.Frame{Pix: make([]byte, 640*360*3), Width: 640, Height: 360}
	for _, r := range []*contract.Region{
		nil,
		{X: 0, Y: 0, Width: 1, Height: 1},       // the whole frame
		{X: 1.2, Y: 0, Width: 0.5, Height: 0.5}, // out of range
		{X: 0.5, Y: 0.5, Width: 0.01, Height: 0.5}, // too narrow to hold a plate
		{X: 0.1, Y: 0.1, Width: -1, Height: 0.5},
	} {
		c := &Camera{cfg: Config{Spec: contract.CameraSpec{ROI: r}}}
		if got, dx, dy := c.roiFrame(f); got.Width != 640 || dx != 0 || dy != 0 {
			t.Errorf("region %+v cropped to %dx%d at %d,%d", r, got.Width, got.Height, dx, dy)
		}
	}
}
