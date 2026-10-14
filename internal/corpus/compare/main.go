// Command compare reads two result sets and reports where they disagree.
//
// Bit-for-bit equality is the wrong ruler. Both implementations run the same ONNX weights, but
// they get there through different resize code, so a confidence differing in the fifth decimal
// is arithmetic and a confidence differing in the third is a preprocessing bug. The tolerances
// here encode that distinction, and each one says what a violation means.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"

	"github.com/hyz-is/arandu-alpr/internal/corpus"
)

const (
	// iouMatch is how much two boxes must overlap to be the same detection. High, because the
	// question is "did both find this plate", not "did both draw the same rectangle".
	iouMatch = 0.90

	// confidenceTolerance separates arithmetic from a bug. Same weights, same input, different
	// resize path: the last few bits move. Anything larger is not rounding.
	confidenceTolerance = 1e-3
)

func main() {
	goDir := flag.String("go", "results/go", "directory of Go results")
	pyDir := flag.String("py", "results/py", "directory of Python results")
	gates := flag.String("gates", "on", "on: measure what the gates discard; off: compare model against model")
	exact := flag.Float64("min-exact", 0.90, "minimum exact-plate agreement to pass")
	minIoU := flag.Float64("min-iou", 0.95, "minimum mean IoU over matched boxes to pass")
	flag.Parse()

	if *gates != "on" && *gates != "off" {
		fmt.Fprintln(os.Stderr, "-gates must be on or off")
		os.Exit(2)
	}

	goRes, err := load(*goDir)
	if err != nil {
		fatal(err)
	}
	pyRes, err := load(*pyDir)
	if err != nil {
		fatal(err)
	}

	r := compare(goRes, pyRes, *gates == "off")
	r.print(*gates)

	if fail := r.verdict(*exact, *minIoU); fail != "" {
		fmt.Fprintf(os.Stderr, "\nFAIL: %s\n", fail)
		os.Exit(1)
	}
	fmt.Println("\nPASS")
}

func fatal(err error) { fmt.Fprintln(os.Stderr, err); os.Exit(1) }

func load(dir string) (map[string]corpus.Result, error) {
	out := map[string]corpus.Result{}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("reading %s: %w", dir, err)
	}
	for _, e := range entries {
		if e.IsDir() || filepath.Ext(e.Name()) != ".json" {
			continue
		}
		body, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			return nil, err
		}
		var r corpus.Result
		if err := json.Unmarshal(body, &r); err != nil {
			return nil, fmt.Errorf("%s: %w", e.Name(), err)
		}
		out[r.FrameID] = r
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no results in %s", dir)
	}
	return out, nil
}

type report struct {
	frames int

	plateExact  int
	plateTotal  int
	charErrors  int
	charTotal   int
	confusions  map[string]int
	mismatches  []string
	iouSum      float64
	iouMatched  int
	onlyGo      int
	onlyPy      int
	confDeltas  []float64
	worstConf   float64
	worstFrame  string
	regionWrong []string
	shaWarnings []string
}

func compare(goRes, pyRes map[string]corpus.Result, ignoreGates bool) *report {
	r := &report{confusions: map[string]int{}}

	ids := make([]string, 0, len(goRes))
	for id := range goRes {
		ids = append(ids, id)
	}
	sort.Strings(ids)

	for _, id := range ids {
		g, ok := goRes[id]
		if !ok {
			continue
		}
		p, ok := pyRes[id]
		if !ok {
			r.mismatches = append(r.mismatches, id+": no Python result")
			continue
		}
		r.frames++

		// A result set produced against different weights is not stale, it is meaningless.
		// Saying so by hash beats letting it surface as a mysterious accuracy drop.
		if g.DetectorSHA != "" && p.DetectorSHA != "" && g.DetectorSHA != p.DetectorSHA {
			r.shaWarnings = append(r.shaWarnings,
				fmt.Sprintf("%s: detector weights differ (go %s, py %s)", id, g.DetectorSHA, p.DetectorSHA))
		}
		if g.OCRSHA != "" && p.OCRSHA != "" && g.OCRSHA != p.OCRSHA {
			r.shaWarnings = append(r.shaWarnings,
				fmt.Sprintf("%s: OCR weights differ (go %s, py %s)", id, g.OCRSHA, p.OCRSHA))
		}

		// In gates-off mode the Go side is expected to carry everything the model produced, so
		// a shorter list is a real disagreement. In gates-on mode it is the measurement.
		if ignoreGates && g.GatesApplied {
			r.mismatches = append(r.mismatches,
				id+": asked to compare model-against-model but the Go result had gates applied")
		}

		r.matchReadings(id, g.Readings, p.Readings)
	}
	return r
}

// matchReadings pairs readings by overlap, then compares the pairs field by field.
func (r *report) matchReadings(frameID string, gs, ps []corpus.Reading) {
	usedPy := make([]bool, len(ps))

	for _, g := range gs {
		best, bestIoU := -1, 0.0
		for i, p := range ps {
			if usedPy[i] {
				continue
			}
			if v := iou(g.Box, p.Box); v > bestIoU {
				best, bestIoU = i, v
			}
		}
		if best < 0 || bestIoU < iouMatch {
			r.onlyGo++
			continue
		}
		usedPy[best] = true
		p := ps[best]

		r.iouSum += bestIoU
		r.iouMatched++
		r.comparePlate(frameID, g, p)

		if d := math.Abs(g.Confidence - p.Confidence); d > confidenceTolerance {
			r.confDeltas = append(r.confDeltas, d)
			if d > r.worstConf {
				r.worstConf, r.worstFrame = d, frameID
			}
		}
		if g.Region != p.Region {
			r.regionWrong = append(r.regionWrong,
				fmt.Sprintf("%s: go %q, py %q", frameID, g.Region, p.Region))
		}
	}
	for i, used := range usedPy {
		if !used {
			r.onlyPy++
			_ = i
		}
	}
}

// comparePlate is the one field with no tolerance at all, and it records HOW it differed.
//
// The confusion pairs are the diagnostic. A scatter of unrelated substitutions is a
// preprocessing difference; a systematic 8-for-B or 0-for-O is the alphabet indexed at an
// offset, which is a one-character bug producing perfectly plausible plates.
func (r *report) comparePlate(frameID string, g, p corpus.Reading) {
	r.plateTotal++
	if g.Plate == p.Plate {
		r.plateExact++
	} else {
		r.mismatches = append(r.mismatches,
			fmt.Sprintf("%s: go %q, py %q", frameID, g.Plate, p.Plate))
	}

	d, subs := levenshtein(g.Plate, p.Plate)
	r.charErrors += d
	r.charTotal += max(len(p.Plate), 1)
	for _, s := range subs {
		r.confusions[s]++
	}
}

func (r *report) verdict(minExact, minIoU float64) string {
	if r.frames == 0 {
		return "no frames were compared"
	}
	if r.plateTotal > 0 {
		got := float64(r.plateExact) / float64(r.plateTotal)
		if got < minExact {
			return fmt.Sprintf("exact plate agreement %.1f%% is below the %.1f%% floor",
				got*100, minExact*100)
		}
	}
	if r.iouMatched > 0 {
		if mean := r.iouSum / float64(r.iouMatched); mean < minIoU {
			return fmt.Sprintf("mean IoU %.3f is below the %.3f floor", mean, minIoU)
		}
	}
	if len(r.regionWrong) > 0 {
		return fmt.Sprintf("%d region disagreements; the region table is POSITIONAL, so a "+
			"reordering relabels every reading and nothing at runtime looks wrong", len(r.regionWrong))
	}
	return ""
}

func (r *report) print(gates string) {
	fmt.Printf("frames compared      %d\n", r.frames)

	if r.plateTotal > 0 {
		fmt.Printf("plate exact          %d/%d  (%.1f%%)\n",
			r.plateExact, r.plateTotal, 100*float64(r.plateExact)/float64(r.plateTotal))
	}
	if r.charTotal > 0 {
		fmt.Printf("character error rate %.3f\n", float64(r.charErrors)/float64(r.charTotal))
	}
	if r.iouMatched > 0 {
		fmt.Printf("mean IoU             %.3f  over %d matched boxes\n",
			r.iouSum/float64(r.iouMatched), r.iouMatched)
	}

	if gates == "on" {
		fmt.Printf("\ndropped by the Go gates   %d readings\n", r.onlyPy)
		fmt.Printf("found only by Go          %d readings\n", r.onlyGo)
		fmt.Println("\nThe Python service reads every detection and filters afterwards; the Go\n" +
			"service drops a candidate before the OCR sees it. Fewer readings on the Go side is\n" +
			"the design working. Run with -gates=off to compare model against model.")
	} else {
		if r.onlyGo > 0 || r.onlyPy > 0 {
			fmt.Printf("\nunmatched            go-only %d, py-only %d\n", r.onlyGo, r.onlyPy)
		}
	}

	if len(r.confusions) > 0 {
		fmt.Println("\ncharacter confusions (go -> py)")
		type kv struct {
			k string
			n int
		}
		var pairs []kv
		for k, n := range r.confusions {
			pairs = append(pairs, kv{k, n})
		}
		sort.Slice(pairs, func(i, j int) bool { return pairs[i].n > pairs[j].n })
		for i, p := range pairs {
			if i == 8 {
				break
			}
			fmt.Printf("  %-6s %d\n", p.k, p.n)
		}
		fmt.Println("\nA systematic pair -- 8 for B, 0 for O -- is the alphabet indexed at an offset,\n" +
			"which produces perfectly plausible plates. A scatter is a preprocessing difference.")
	}

	if len(r.confDeltas) > 0 {
		fmt.Printf("\nconfidence deltas over %g: %d (worst %.4f on %s)\n",
			confidenceTolerance, len(r.confDeltas), r.worstConf, r.worstFrame)
		fmt.Println("Same weights, same input: a delta this size is preprocessing, not rounding.\n" +
			"Check, in this order:\n" +
			"  1. letterbox padding colour (must be 114, on the [0,255] scale)\n" +
			"  2. resize interpolation (bilinear, pixel centres)\n" +
			"  3. channel order (BGR from ffmpeg, RGB into both models)\n" +
			"  4. input scale -- the OCR takes [0,255] and must NOT be divided by 255")
	}

	if len(r.regionWrong) > 0 {
		fmt.Printf("\nregion disagreements  %d\n", len(r.regionWrong))
		for i, s := range r.regionWrong {
			if i == 5 {
				fmt.Printf("  ... and %d more\n", len(r.regionWrong)-5)
				break
			}
			fmt.Println("  " + s)
		}
		fmt.Println("The region table is positional: its index IS the class the model emits.\n" +
			"Sorting it looks like tidying and relabels every reading ever produced.")
	}

	if len(r.mismatches) > 0 {
		fmt.Printf("\nplate mismatches      %d\n", len(r.mismatches))
		for i, s := range r.mismatches {
			if i == 10 {
				fmt.Printf("  ... and %d more\n", len(r.mismatches)-10)
				break
			}
			fmt.Println("  " + s)
		}
	}

	if len(r.shaWarnings) > 0 {
		fmt.Printf("\nWEIGHTS DIFFER on %d frames -- this comparison is meaningless:\n", len(r.shaWarnings))
		for i, s := range r.shaWarnings {
			if i == 3 {
				break
			}
			fmt.Println("  " + s)
		}
	}
}

func iou(a, b [4]float64) float64 {
	ax2, ay2 := a[0]+a[2], a[1]+a[3]
	bx2, by2 := b[0]+b[2], b[1]+b[3]
	ix1, iy1 := math.Max(a[0], b[0]), math.Max(a[1], b[1])
	ix2, iy2 := math.Min(ax2, bx2), math.Min(ay2, by2)
	iw, ih := ix2-ix1, iy2-iy1
	if iw <= 0 || ih <= 0 {
		return 0
	}
	inter := iw * ih
	union := a[2]*a[3] + b[2]*b[3] - inter
	if union <= 0 {
		return 0
	}
	return inter / union
}

// levenshtein returns the edit distance and the substitutions it made, which are what turn a
// number into a diagnosis.
func levenshtein(a, b string) (int, []string) {
	if a == b {
		return 0, nil
	}
	n, m := len(a), len(b)
	prev := make([]int, m+1)
	cur := make([]int, m+1)
	for j := 0; j <= m; j++ {
		prev[j] = j
	}
	for i := 1; i <= n; i++ {
		cur[0] = i
		for j := 1; j <= m; j++ {
			cost := 1
			if a[i-1] == b[j-1] {
				cost = 0
			}
			cur[j] = min(min(cur[j-1]+1, prev[j]+1), prev[j-1]+cost)
		}
		prev, cur = cur, prev
	}

	// Position-wise substitutions, which is all the confusion table needs and is stable in a
	// way that reconstructing the edit path is not.
	var subs []string
	for i := 0; i < min(n, m); i++ {
		if a[i] != b[i] {
			subs = append(subs, fmt.Sprintf("%c->%c", a[i], b[i]))
		}
	}
	return prev[m], subs
}
