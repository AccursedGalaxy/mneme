// Command eval runs mneme's prompt-evaluation harness against a live LLM and
// prints a per-metric table plus an aggregate score per prompt version. It is
// the number we trust when we change the extraction prompt.
//
// Usage:
//
//	go run ./cmd/eval                 # uses .env / MNEME_* env, fake embedder
//	go run ./cmd/eval -model x -k 5   # override model and search depth
//
// It is intentionally outside `go test ./...` (which must stay offline): a run
// needs network and an API key.
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

	"github.com/AccursedGalaxy/mneme/eval"
	"github.com/AccursedGalaxy/mneme/provider"
	"github.com/AccursedGalaxy/mneme/provider/fake"
	"github.com/AccursedGalaxy/mneme/provider/openai"
	"github.com/AccursedGalaxy/mneme/store/sqlite"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "eval:", err)
		os.Exit(1)
	}
}

func run() error {
	loadDotenv(".env")

	var (
		fixturesDir = flag.String("fixtures", "eval/fixtures", "directory of *.json fixtures")
		base        = flag.String("base", envOr("MNEME_LLM_BASE_URL", "https://openrouter.ai/api/v1"), "OpenAI-compatible base URL")
		model       = flag.String("model", envOr("MNEME_LLM_MODEL", "openai/gpt-4o-mini"), "extraction/judge model")
		embedKind   = flag.String("embedder", "fake", "embedder: fake | openai")
		k           = flag.Int("k", 3, "search top-k for recall@k")
		out         = flag.String("out", "", "optional results file to write (markdown)")
		verbose     = flag.Bool("v", false, "print the facts extracted for each fixture")
	)
	flag.Parse()

	key := firstNonEmpty(os.Getenv("MNEME_LLM_API_KEY"), os.Getenv("OPENROUTER_API_KEY"), os.Getenv("OPENAI_API_KEY"))
	if key == "" {
		return fmt.Errorf("no API key found (set MNEME_LLM_API_KEY or OPENROUTER_API_KEY)")
	}

	llm := &openai.LLM{BaseURL: *base, APIKey: key, Model: *model}

	var embedder provider.Embedder
	switch *embedKind {
	case "fake":
		embedder = &fake.Embedder{D: 256}
		fmt.Println("note: using deterministic fake embedder (lexical) — extraction metrics are real, search recall is lexical")
	case "openai":
		embedder = &openai.Embedder{
			BaseURL: envOr("MNEME_EMBED_BASE_URL", *base),
			APIKey:  firstNonEmpty(os.Getenv("MNEME_EMBED_API_KEY"), key),
			Model:   envOr("MNEME_EMBED_MODEL", "text-embedding-3-small"),
		}
	default:
		return fmt.Errorf("unknown -embedder %q (want fake|openai)", *embedKind)
	}

	fixtures, err := eval.LoadFixtures(*fixturesDir)
	if err != nil {
		return err
	}
	if len(fixtures) == 0 {
		return fmt.Errorf("no fixtures found in %s", *fixturesDir)
	}

	st, err := sqlite.Open(":memory:")
	if err != nil {
		return err
	}
	defer st.Close()

	fmt.Printf("running %d fixtures, model=%s, base=%s, k=%d\n\n", len(fixtures), *model, *base, *k)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	reports, err := eval.Run(ctx, fixtures, eval.Config{
		LLM:      llm,
		Embedder: embedder,
		Store:    st,
		K:        *k,
	})
	if err != nil {
		return err
	}

	if *verbose && len(reports) > 0 {
		fmt.Println("=== extracted facts (version " + reports[0].Version + ") ===")
		for _, res := range reports[0].Results {
			fmt.Printf("\n[%s]\n", res.Name)
			if len(res.Extracted) == 0 {
				fmt.Println("  (none)")
			}
			for _, f := range res.Extracted {
				fmt.Printf("  - %s\n", f)
			}
		}
		fmt.Println()
	}

	report := render(reports, *model, *embedKind, *k)
	fmt.Print(report)

	if *out != "" {
		if err := os.WriteFile(*out, []byte(report), 0o644); err != nil {
			return err
		}
		fmt.Printf("\nwrote results to %s\n", *out)
	}
	return nil
}

// render builds the human + markdown report shown on stdout and optionally
// written to -out.
func render(reports []eval.VersionReport, model, embedder string, k int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# mneme eval results\n\nmodel: `%s`  ·  embedder: `%s`  ·  search k: %d\n\n", model, embedder, k)
	fmt.Fprintf(&b, "| version | recall | precision | specificity | search@k | dedup | aggregate |\n")
	fmt.Fprintf(&b, "|---|---|---|---|---|---|---|\n")
	for _, r := range reports {
		fmt.Fprintf(&b, "| %s | %.2f | %.2f | %.2f | %.2f | %.2f | **%.2f** |\n",
			r.Version, r.MeanRecall(), r.MeanPrecision(), r.MeanSpecificity(),
			r.MeanSearchRecall(), r.MeanDedup(), r.MeanAggregate())
	}

	// Per-fixture detail for the first (or only) version, sorted by name.
	if len(reports) > 0 {
		r := reports[0]
		fmt.Fprintf(&b, "\n## Per-fixture (version %s)\n\n", r.Version)
		fmt.Fprintf(&b, "| fixture | recall | precision | spec | search | new-on-redup | aggregate |\n")
		fmt.Fprintf(&b, "|---|---|---|---|---|---|---|\n")
		results := append([]eval.Result(nil), r.Results...)
		sort.Slice(results, func(i, j int) bool { return results[i].Name < results[j].Name })
		for _, res := range results {
			fmt.Fprintf(&b, "| %s | %.2f | %.2f | %s | %s | %d | %.2f |\n",
				res.Name, res.Recall, res.Precision,
				optFloat(res.HasSpecificity, res.Specificity),
				optFloat(res.HasSearch, res.SearchRecall),
				res.DedupNewFacts, res.Aggregate())
		}
	}
	return b.String()
}

func optFloat(has bool, v float64) string {
	if !has {
		return "—"
	}
	return fmt.Sprintf("%.2f", v)
}

// loadDotenv loads KEY=VALUE lines from path into the process environment,
// without overwriting variables already set. Missing file is fine.
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
