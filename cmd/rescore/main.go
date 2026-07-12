// Command rescore reads the per-question prediction dumps a bench run writes
// (bench/results/<run>.jsonl) and reports scores with conversation-level
// bootstrap confidence intervals, plus paired deltas against a baseline run.
//
// It exists because the matrix reports point estimates from 10 conversations and
// then invites decisions off differences of 0.02-0.06. That is not enough to know
// whether a lever works. LoCoMo's unit of independence is the conversation, not
// the question: the ~200 questions about one conversation all share its facts, so
// resampling questions would understate the variance badly. This resamples
// conversations.
//
// It runs entirely offline against the dumps, so re-scoring costs nothing and any
// future scoring change can be applied to past runs without paying for them again.
//
// Usage:
//
//	go run ./cmd/rescore -baseline bench/results/additive_3small.jsonl \
//	  bench/results/extract_flash.jsonl bench/results/additive_boosters.jsonl
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// prediction mirrors bench.Prediction. Duplicated rather than imported so the
// tool can read dumps written by an older harness whose struct has since changed.
type prediction struct {
	SampleID     string `json:"sample_id"`
	Category     string `json:"category"`
	Gold         string `json:"gold"`
	Unanswerable bool   `json:"unanswerable"`
	Predicted    string `json:"predicted"`
	Judge        bool   `json:"judge"`
	EM           bool   `json:"em"`
}

type run struct {
	name  string
	preds []prediction
	// byConv groups predictions by conversation, the unit we resample.
	byConv map[string][]prediction
	convs  []string
}

func main() {
	if err := run_(); err != nil {
		fmt.Fprintln(os.Stderr, "rescore:", err)
		os.Exit(1)
	}
}

func run_() error {
	var (
		baseline = flag.String("baseline", "", "baseline dump to compare the others against (JSONL)")
		resamps  = flag.Int("resamples", 10000, "bootstrap resamples")
		seed     = flag.Int64("seed", 1, "RNG seed, so a reported interval is reproducible")
		category = flag.String("category", "", "restrict to one category (single_hop|multi_hop|temporal|open_domain). Beware: a category is a fraction of an already-small set, so its interval is correspondingly wider")
	)
	flag.Parse()
	categoryFilter = *category

	paths := flag.Args()
	if *baseline == "" && len(paths) == 0 {
		return fmt.Errorf("give -baseline and/or one or more .jsonl dumps")
	}

	var runs []*run
	if *baseline != "" {
		r, err := load(*baseline)
		if err != nil {
			return err
		}
		runs = append(runs, r)
	}
	for _, p := range paths {
		r, err := load(p)
		if err != nil {
			return err
		}
		runs = append(runs, r)
	}

	rng := rand.New(rand.NewSource(*seed))

	fmt.Printf("%d conversations per run, %d bootstrap resamples\n\n", len(runs[0].convs), *resamps)

	fmt.Printf("%-24s %-22s %-22s %s\n", "run", "answerable judge", "overall judge", "abstains on answerable")
	fmt.Printf("%s\n", strings.Repeat("-", 96))
	for _, r := range runs {
		ansMean, ansLo, ansHi := r.bootstrap(rng, *resamps, answerableJudge)
		allMean, allLo, allHi := r.bootstrap(rng, *resamps, overallJudge)
		fmt.Printf("%-24s %.3f [%.3f, %.3f]  %.3f [%.3f, %.3f]  %.1f%%\n",
			r.name, ansMean, ansLo, ansHi, allMean, allLo, allHi, 100*r.abstainRate())
	}

	if *baseline == "" || len(runs) < 2 {
		return nil
	}

	base := runs[0]
	fmt.Printf("\nPaired deltas vs %s (answerable judge). Same conversations resampled for both\n", base.name)
	fmt.Printf("runs in each draw, so per-conversation difficulty cancels out.\n\n")
	fmt.Printf("%-24s %-24s %s\n", "run", "delta [95% CI]", "P(better than baseline)")
	fmt.Printf("%s\n", strings.Repeat("-", 76))
	for _, r := range runs[1:] {
		d, lo, hi, pWin := pairedDelta(rng, *resamps, base, r)
		verdict := ""
		if lo > 0 {
			verdict = "  <- real (CI excludes 0)"
		} else if hi < 0 {
			verdict = "  <- real regression"
		} else {
			verdict = "  <- indistinguishable from noise"
		}
		fmt.Printf("%-24s %+.3f [%+.3f, %+.3f]   %.0f%%%s\n", r.name, d, lo, hi, 100*pWin, verdict)
	}
	return nil
}

// categoryFilter, when set by -category, restricts every metric to one category.
var categoryFilter string

// answerableJudge is the metric that matters: accuracy on questions that have an
// answer. The overall number folds in the adversarial category, which sits at
// ~0.95 for every configuration and washes out the differences between them.
func answerableJudge(preds []prediction) (hits, n float64) {
	for _, p := range preds {
		if p.Unanswerable {
			continue
		}
		if categoryFilter != "" && p.Category != categoryFilter {
			continue
		}
		n++
		if p.Judge {
			hits++
		}
	}
	return hits, n
}

func overallJudge(preds []prediction) (hits, n float64) {
	for _, p := range preds {
		n++
		if p.Judge {
			hits++
		}
	}
	return hits, n
}

// abstainRate is the share of ANSWERABLE questions the system declined. It is the
// clearest single diagnostic of extraction recall: the fact was not there to find.
func (r *run) abstainRate() float64 {
	var declined, n float64
	for _, p := range r.preds {
		if p.Unanswerable {
			continue
		}
		n++
		if abstained(p.Predicted) {
			declined++
		}
	}
	if n == 0 {
		return 0
	}
	return declined / n
}

var abstentionMarkers = []string{
	"i don t know", "don t know", "not mentioned", "no information",
	"not specified", "cannot be determined", "unanswerable",
}

func abstained(pred string) bool {
	n := normalize(pred)
	if n == "" {
		return true
	}
	for _, m := range abstentionMarkers {
		if strings.Contains(n, m) {
			return true
		}
	}
	return false
}

func normalize(s string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(s) {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
		} else {
			b.WriteRune(' ')
		}
	}
	return strings.Join(strings.Fields(b.String()), " ")
}

// bootstrap resamples conversations with replacement and returns the metric's
// mean and its 95% percentile interval.
func (r *run) bootstrap(rng *rand.Rand, resamples int, metric func([]prediction) (float64, float64)) (mean, lo, hi float64) {
	samples := make([]float64, 0, resamples)
	for i := 0; i < resamples; i++ {
		var hits, n float64
		for j := 0; j < len(r.convs); j++ {
			conv := r.convs[rng.Intn(len(r.convs))]
			h, c := metric(r.byConv[conv])
			hits += h
			n += c
		}
		if n > 0 {
			samples = append(samples, hits/n)
		}
	}
	return summarize(samples)
}

// pairedDelta resamples a set of conversations once per draw and scores BOTH runs
// on that same set. Pairing matters: conversations differ a lot in difficulty, and
// an unpaired comparison buries a small real effect under that between-conversation
// variance.
func pairedDelta(rng *rand.Rand, resamples int, a, b *run) (mean, lo, hi, pWin float64) {
	samples := make([]float64, 0, resamples)
	wins := 0
	for i := 0; i < resamples; i++ {
		var aHits, aN, bHits, bN float64
		for j := 0; j < len(a.convs); j++ {
			conv := a.convs[rng.Intn(len(a.convs))]
			h, n := answerableJudge(a.byConv[conv])
			aHits, aN = aHits+h, aN+n
			h, n = answerableJudge(b.byConv[conv])
			bHits, bN = bHits+h, bN+n
		}
		if aN == 0 || bN == 0 {
			continue
		}
		d := bHits/bN - aHits/aN
		samples = append(samples, d)
		if d > 0 {
			wins++
		}
	}
	mean, lo, hi = summarize(samples)
	return mean, lo, hi, float64(wins) / float64(len(samples))
}

func summarize(samples []float64) (mean, lo, hi float64) {
	if len(samples) == 0 {
		return 0, 0, 0
	}
	for _, s := range samples {
		mean += s
	}
	mean /= float64(len(samples))
	sorted := append([]float64(nil), samples...)
	sort.Float64s(sorted)
	lo = sorted[int(0.025*float64(len(sorted)))]
	hi = sorted[int(0.975*float64(len(sorted)))]
	return mean, lo, hi
}

func load(path string) (*run, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := &run{
		name:   strings.TrimSuffix(filepath.Base(path), ".jsonl"),
		byConv: map[string][]prediction{},
	}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<20), 1<<20)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var p prediction
		if err := json.Unmarshal([]byte(line), &p); err != nil {
			return nil, fmt.Errorf("%s: %w", path, err)
		}
		r.preds = append(r.preds, p)
		r.byConv[p.SampleID] = append(r.byConv[p.SampleID], p)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	for c := range r.byConv {
		r.convs = append(r.convs, c)
	}
	sort.Strings(r.convs) // deterministic resampling order for a given seed
	return r, nil
}
