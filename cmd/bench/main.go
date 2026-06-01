// Command bench runs mneme's benchmark harness against a public long-term-memory
// dataset (LoCoMo or LongMemEval) and prints a per-category + aggregate QA table,
// then writes it to bench/RESULTS.md. This is the v2 baseline (PLAN-v2.md §4.1):
// every later task is measured against the number this prints.
//
// Usage:
//
//	go run ./cmd/bench -dataset locomo -path bench/data/locomo10.json
//	go run ./cmd/bench -dataset longmemeval -path bench/data/longmemeval_s.json -k 10
//
// The dataset files are gitignored (download separately, point -path at them).
// Like cmd/eval it is outside `go test ./...` (which stays offline): a run needs
// network and an API key.
package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/AccursedGalaxy/mneme/bench"
	"github.com/AccursedGalaxy/mneme/provider"
	"github.com/AccursedGalaxy/mneme/provider/fake"
	"github.com/AccursedGalaxy/mneme/provider/openai"
	"github.com/AccursedGalaxy/mneme/store/sqlite"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "bench:", err)
		os.Exit(1)
	}
}

func run() error {
	loadDotenv(".env")

	var (
		dataset   = flag.String("dataset", "locomo", "dataset: locomo | longmemeval")
		path      = flag.String("path", "", "path to the dataset file (required; gitignored)")
		base      = flag.String("base", envOr("MNEME_LLM_BASE_URL", "https://openrouter.ai/api/v1"), "OpenAI-compatible base URL")
		model     = flag.String("model", envOr("MNEME_LLM_MODEL", "openai/gpt-4o-mini"), "extraction/answer/judge model")
		embedKind = flag.String("embedder", "openai", "embedder: openai | fake")
		k         = flag.Int("k", 5, "search top-k facts fed to the answer model")
		strategy  = flag.String("strategy", bench.StrategyAdditive, "write strategy: additive | consolidate")
		limit     = flag.Int("limit", 0, "if >0, run only the first N samples (smoke test)")
		maxQ      = flag.Int("maxq", 0, "if >0, cap questions per sample (smoke/cost control)")
		out       = flag.String("out", "bench/RESULTS.md", "results file to write (markdown); empty to skip")
		verbose   = flag.Bool("v", false, "print each question's gold/predicted answer")
	)
	flag.Parse()

	if *path == "" {
		return fmt.Errorf("-path is required (point it at the gitignored dataset file)")
	}

	samples, err := loadDataset(*dataset, *path)
	if err != nil {
		return err
	}
	if *limit > 0 && *limit < len(samples) {
		samples = samples[:*limit]
	}
	if *maxQ > 0 {
		for i := range samples {
			samples[i].Questions = capBalanced(samples[i].Questions, *maxQ)
		}
	}
	if len(samples) == 0 {
		return fmt.Errorf("no samples loaded from %s", *path)
	}

	key := firstNonEmpty(os.Getenv("MNEME_LLM_API_KEY"), os.Getenv("OPENROUTER_API_KEY"), os.Getenv("OPENAI_API_KEY"))
	if key == "" {
		return fmt.Errorf("no API key found (set MNEME_LLM_API_KEY or OPENROUTER_API_KEY)")
	}
	llm := &openai.LLM{BaseURL: *base, APIKey: key, Model: *model}

	var embedder provider.Embedder
	switch *embedKind {
	case "openai":
		embedder = &openai.Embedder{
			BaseURL: envOr("MNEME_EMBED_BASE_URL", *base),
			APIKey:  firstNonEmpty(os.Getenv("MNEME_EMBED_API_KEY"), key),
			Model:   envOr("MNEME_EMBED_MODEL", "text-embedding-3-small"),
		}
	case "fake":
		embedder = &fake.Embedder{D: 256}
		fmt.Fprintln(os.Stderr, "note: using deterministic fake (lexical) embedder — retrieval is lexical, not semantic")
	default:
		return fmt.Errorf("unknown -embedder %q (want openai|fake)", *embedKind)
	}

	st, err := sqlite.Open(":memory:")
	if err != nil {
		return err
	}
	defer st.Close()

	totalQ := 0
	for _, s := range samples {
		totalQ += len(s.Questions)
	}
	fmt.Fprintf(os.Stderr, "running %s: %d samples, %d questions, model=%s, embedder=%s, k=%d, strategy=%s\n",
		*dataset, len(samples), totalQ, *model, *embedKind, *k, *strategy)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Hour)
	defer cancel()

	report, err := bench.Run(ctx, samples, bench.Config{
		LLM:      llm,
		Embedder: embedder,
		Store:    st,
		K:        *k,
		Strategy: *strategy,
		Progress: func(done, total int) {
			fmt.Fprintf(os.Stderr, "\r  scored %d/%d samples", done, total)
			if done == total {
				fmt.Fprintln(os.Stderr)
			}
		},
	})
	if err != nil {
		return err
	}

	if *verbose {
		printAnswers(report)
	}

	md := render(report, *dataset, *path, *model, *embedKind)
	fmt.Print(md)

	if *out != "" {
		if err := os.WriteFile(*out, []byte(md), 0o644); err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "\nwrote results to %s\n", *out)
	}
	return nil
}

// capBalanced returns at most n questions chosen round-robin across categories,
// so a small per-sample cap still touches every category (a first-n slice would
// skew toward whichever category clusters first in the file). Preserves original
// order within each category.
func capBalanced(qs []bench.QAPair, n int) []bench.QAPair {
	if n <= 0 || len(qs) <= n {
		return qs
	}
	// Group by category, preserving first-seen category order and intra-category order.
	var order []string
	byCat := map[string][]bench.QAPair{}
	for _, q := range qs {
		if _, ok := byCat[q.Category]; !ok {
			order = append(order, q.Category)
		}
		byCat[q.Category] = append(byCat[q.Category], q)
	}
	out := make([]bench.QAPair, 0, n)
	for len(out) < n {
		progressed := false
		for _, c := range order {
			if len(byCat[c]) == 0 {
				continue
			}
			out = append(out, byCat[c][0])
			byCat[c] = byCat[c][1:]
			progressed = true
			if len(out) == n {
				break
			}
		}
		if !progressed {
			break
		}
	}
	return out
}

func loadDataset(dataset, path string) ([]bench.Sample, error) {
	switch strings.ToLower(dataset) {
	case "locomo":
		return bench.LoadLoCoMo(path)
	case "longmemeval", "lme":
		return bench.LoadLongMemEval(path)
	default:
		return nil, fmt.Errorf("unknown -dataset %q (want locomo|longmemeval)", dataset)
	}
}

// render builds the human + markdown report: a per-category table (the lever
// finder) followed by the bold overall row, plus run metadata.
func render(report bench.Report, dataset, path, model, embedder string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# mneme bench results — %s\n\n", dataset)
	fmt.Fprintf(&b, "dataset: `%s`  ·  model: `%s`  ·  embedder: `%s`  ·  k: %d  ·  strategy: `%s`\n\n",
		path, model, embedder, report.K, report.Strategy)
	fmt.Fprintf(&b, "Metrics: **EM** = normalized exact match, **F1** = token-overlap F1, **Judge** = LLM semantic match. n = question count.\n\n")

	fmt.Fprintf(&b, "| category | n | EM | F1 | judge |\n")
	fmt.Fprintf(&b, "|---|---|---|---|---|\n")
	for _, c := range report.Categories() {
		fmt.Fprintf(&b, "| %s | %d | %.2f | %.2f | %.2f |\n", c.Category, c.N, c.EM, c.F1, c.Judge)
	}
	o := report.Overall()
	fmt.Fprintf(&b, "| **overall** | **%d** | **%.2f** | **%.2f** | **%.2f** |\n", o.N, o.EM, o.F1, o.Judge)
	return b.String()
}

// printAnswers dumps each question's gold/predicted for eyeballing, grouped by
// sample, sorted by category then question.
func printAnswers(report bench.Report) {
	results := append([]bench.QAResult(nil), report.Results...)
	sort.Slice(results, func(i, j int) bool {
		if results[i].Category != results[j].Category {
			return results[i].Category < results[j].Category
		}
		return results[i].Question < results[j].Question
	})
	fmt.Fprintln(os.Stderr, "\n=== answers ===")
	for _, r := range results {
		mark := " "
		if r.Judge {
			mark = "✓"
		}
		fmt.Fprintf(os.Stderr, "[%s] %s\n   Q: %s\n   gold: %s\n   pred: %s\n", mark, r.Category, r.Question, r.Gold, r.Predicted)
	}
	fmt.Fprintln(os.Stderr)
}

// loadDotenv loads KEY=VALUE lines from path into the environment without
// overwriting existing variables. Missing file is fine.
func loadDotenv(path string) {
	f, err := os.Open(path)
	if err != nil {
		return
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		line = strings.TrimPrefix(line, "export ")
		key, val, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		val = strings.Trim(strings.TrimSpace(val), `"'`)
		if _, exists := os.LookupEnv(key); !exists {
			os.Setenv(key, val)
		}
	}
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}
