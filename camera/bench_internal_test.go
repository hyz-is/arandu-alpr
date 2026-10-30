package camera

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/hyz-is/arandu-alpr"
	"github.com/hyz-is/arandu-alpr/contract"
	"github.com/hyz-is/arandu-alpr/engine"
	"github.com/hyz-is/arandu-alpr/imaging"
)

// sceneEngine reads the plate of the current scene at each vehicle's place.
type sceneEngine struct {
	*fakeEngine
	scene    *int
	readings [][]engine.Reading
	at       map[float64]int
}

func (e sceneEngine) Recognize(_ context.Context, _ engine.Frame, box contract.BBox) (engine.Reading, error) {
	return e.readings[*e.scene%len(e.readings)][e.at[box.X*10000+box.Y]], nil
}

// BenchmarkStepConsensus is one frame of a camera with a consensus, an encoder and both pictures
// asked for, in scenes of ten frames: vehicles arrive, are read, decide, and leave.
func BenchmarkStepConsensus(b *testing.B) {
	for _, vehicles := range []int{1, 3, 4, 6} {
		b.Run(fmt.Sprintf("vehicles=%d", vehicles), func(b *testing.B) {
			cfg := alpr.DefaultConsensus()
			enc, err := imaging.NewEncoder(80)
			if err != nil {
				b.Fatal(err)
			}
			r := newRig(b, func(c *Config) {
				c.Consensus = &cfg
				c.Encoder = enc
				c.IncludeFullImage = true
				c.IncludeCropImage = true
				c.Focus = fakeFocus{variance: 300}
			})
			var cands []engine.Candidate
			at := map[float64]int{}
			for i := 0; i < vehicles; i++ {
				box := contract.BBox{X: float64(100 + 700*(i%3)), Y: float64(100 + 700*(i/3)), Width: 100, Height: 30}
				cands = append(cands, engine.Candidate{Box: box, Score: 0.9})
				at[box.X*10000+box.Y] = i
			}
			const scenes = 32
			readings := make([][]engine.Reading, scenes)
			for s := range readings {
				for i := 0; i < vehicles; i++ {
					text := fmt.Sprintf("%c%c%c%d%c%d%d", 'A'+s%26, 'A'+i, 'K', s%10, 'B'+i, i, (s/10)%10)
					readings[s] = append(readings[s], readingOf(probs(text, 0.95, -1, 0)))
				}
			}
			scene := 0
			r.cam.cfg.Engine = sceneEngine{fakeEngine: r.engine, scene: &scene, readings: readings, at: at}
			r.engine.setCandidates(cands)
			pix := make([]byte, frameW*frameH*3)
			for i := range pix {
				pix[i] = byte(i)
			}
			ctx := context.Background()
			stop := r.cam.startEncoder(ctx)
			defer stop()
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				r.cam.step(ctx, engine.Frame{Pix: pix, Width: frameW, Height: frameH, CapturedAt: r.clock.Now()})
				r.clock.Advance(125 * time.Millisecond)
				if i%10 == 9 {
					r.clock.Advance(5 * time.Second)
					scene++
				}
			}
		})
	}
}
