// Command compare diffs two benchmark runs and prints what moved.
//
// It takes two JSON lines -- the last line of two files, or two named lines of one file -- and
// says which numbers changed by more than a tolerance, and what each change costs. Its most
// useful behaviour is refusing: two runs on different models or different configurations are
// not comparable, and printing a diff of them is worse than printing nothing, because somebody
// acts on it.
//
//	go run ./bench/compare -baseline before.jsonl -current after.jsonl
//	go run ./bench/compare -baseline sweep.jsonl -current sweep.jsonl -baseline-line 1 -current-line 4
//
// The exit code is 1 when anything blocking or regressing was found, so it can gate a merge.
package main

import (
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/hyz-is/arandu-alpr/bench/report"
)

func main() {
	baselinePath := flag.String("baseline", "", "file of JSON-line runs to compare against")
	currentPath := flag.String("current", "", "file of JSON-line runs to compare")
	baselineLine := flag.Int("baseline-line", 0, "which run in -baseline (1-based; 0 means the last)")
	currentLine := flag.Int("current-line", 0, "which run in -current (1-based; 0 means the last)")

	force := flag.Bool("force", false,
		"compare even when the config or the models differ. Only for a deliberate one-knob change")

	throughput := flag.Float64("tol-throughput", report.DefaultTolerance().ThroughputDropRatio,
		"how far throughput may fall before it is a regression (0.10 = 10%)")
	latency := flag.Float64("tol-latency", report.DefaultTolerance().LatencyRiseRatio,
		"how far a latency quantile may rise")
	drops := flag.Float64("tol-drops", report.DefaultTolerance().DropRateRise,
		"absolute rise in dropped-frame share that is a regression (0.02 = two frames in a hundred)")
	vram := flag.Float64("tol-vram", report.DefaultTolerance().VRAMRiseRatio, "how far peak VRAM may grow")
	rss := flag.Float64("tol-rss", report.DefaultTolerance().RSSRiseRatio, "how far peak RSS may grow")

	flag.Parse()

	if *baselinePath == "" || *currentPath == "" {
		fmt.Fprintln(os.Stderr, "usage: compare -baseline before.jsonl -current after.jsonl")
		flag.PrintDefaults()
		os.Exit(2)
	}

	before, err := pick(*baselinePath, *baselineLine)
	if err != nil {
		fail(err)
	}
	after, err := pick(*currentPath, *currentLine)
	if err != nil {
		fail(err)
	}

	findings := report.Compare(before, after, report.Tolerance{
		ThroughputDropRatio: *throughput,
		LatencyRiseRatio:    *latency,
		DropRateRise:        *drops,
		VRAMRiseRatio:       *vram,
		RSSRiseRatio:        *rss,
	}, *force)

	printHeader(before, after)
	printFindings(findings)

	if report.HasRegression(findings) {
		os.Exit(1)
	}
}

// pick reads one run out of a file, defaulting to the LAST.
//
// The last rather than the first, because a sweep appends and the line worth comparing is the
// one the sweep just wrote. A default of "first" would silently compare against the smallest
// camera count in the file.
func pick(path string, line int) (report.Run, error) {
	f, err := os.Open(path)
	if err != nil {
		return report.Run{}, err
	}
	defer f.Close()

	runs, err := report.ReadLines(f)
	if err != nil {
		return report.Run{}, fmt.Errorf("%s: %w", path, err)
	}
	if line == 0 {
		return runs[len(runs)-1], nil
	}
	if line < 1 || line > len(runs) {
		return report.Run{}, fmt.Errorf("%s holds %d runs, so line %d does not exist", path, len(runs), line)
	}
	return runs[line-1], nil
}

func printHeader(before, after report.Run) {
	fmt.Printf("baseline  %s  tier %s  %d cameras  %s\n",
		commit(before), before.Tier, before.Config.Cameras, before.StartedAt.Format("2006-01-02 15:04"))
	fmt.Printf("current   %s  tier %s  %d cameras  %s\n\n",
		commit(after), after.Tier, after.Config.Cameras, after.StartedAt.Format("2006-01-02 15:04"))
}

func commit(r report.Run) string {
	sha := r.GitSHA
	if sha == "" {
		sha = "(no commit)"
	} else if len(sha) > 12 {
		sha = sha[:12]
	}
	if r.GitDirty {
		sha += "+dirty"
	}
	return sha
}

func printFindings(findings []report.Finding) {
	if len(findings) == 0 {
		fmt.Println("no change beyond the tolerances")
		return
	}

	// Blocking first, then regressions, then improvements. A reader who stops after the first
	// section must have read the one that changes what they do next.
	order := map[report.Severity]int{report.Blocking: 0, report.Regression: 1, report.Improvement: 2}
	sorted := append([]report.Finding(nil), findings...)
	sort.SliceStable(sorted, func(i, j int) bool { return order[sorted[i].Severity] < order[sorted[j].Severity] })

	var current report.Severity
	for _, f := range sorted {
		if f.Severity != current {
			current = f.Severity
			fmt.Printf("\n%s\n%s\n", strings.ToUpper(string(current)), strings.Repeat("-", len(current)))
		}
		fmt.Printf("  %-38s %s -> %s\n", f.Field, f.Baseline, f.Current)
		for _, line := range wrap(f.Why, 88) {
			fmt.Printf("      %s\n", line)
		}
	}

	if current == report.Blocking {
		fmt.Println("\nNothing was diffed. Two runs whose config or models differ produce real numbers")
		fmt.Println("that mean nothing together; pass -force only for a deliberate one-knob change.")
	}
}

// wrap breaks a sentence at word boundaries so the explanation stays readable in a terminal.
// The explanations are the point of the output, and an unwrapped one is skipped.
func wrap(s string, width int) []string {
	words := strings.Fields(s)
	if len(words) == 0 {
		return nil
	}
	var lines []string
	line := words[0]
	for _, w := range words[1:] {
		if len(line)+1+len(w) > width {
			lines = append(lines, line)
			line = w
			continue
		}
		line += " " + w
	}
	return append(lines, line)
}

func fail(err error) {
	fmt.Fprintf(os.Stderr, "compare: %v\n", err)
	os.Exit(2)
}
