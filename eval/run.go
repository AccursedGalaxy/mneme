package eval

import (
	"context"
	"fmt"

	"github.com/AccursedGalaxy/mneme"
	"github.com/AccursedGalaxy/mneme/provider"
	"github.com/AccursedGalaxy/mneme/store"
)

// Config holds everything Run needs: the providers under test, a judge LLM for
// semantic scoring, the store to use, the search depth, and which prompt
// versions to score.
type Config struct {
	LLM      provider.LLM
	Embedder provider.Embedder
	Store    store.Store
	Judge    provider.LLM // semantic match oracle; defaults to LLM
	K        int          // search top-k; defaults to 3
	Versions []string     // prompt versions; defaults to all registered
}

// Result is one fixture's score under one prompt version.
type Result struct {
	Name      string
	Extracted []string

	Recall    float64
	Precision float64

	HasSpecificity bool
	Specificity    float64

	HasSearch    bool
	SearchRecall float64

	HasDedup      bool
	DedupNewFacts int // 0 is ideal: re-ingesting added nothing new
}

// dedupScore is 1.0 when re-ingestion added no new facts, else 0.
func (r Result) dedupScore() float64 {
	if r.DedupNewFacts == 0 {
		return 1
	}
	return 0
}

// Aggregate is the fixture's single [0,1] score: the mean of its applicable
// metrics (specificity, search and dedup are only counted when the fixture
// exercises them).
func (r Result) Aggregate() float64 {
	sum := r.Recall + r.Precision
	n := 2.0
	if r.HasSpecificity {
		sum += r.Specificity
		n++
	}
	if r.HasSearch {
		sum += r.SearchRecall
		n++
	}
	if r.HasDedup {
		sum += r.dedupScore()
		n++
	}
	return sum / n
}

// VersionReport collects all fixture results for one prompt version.
type VersionReport struct {
	Version string
	Results []Result
}

// Mean returns the average of f over the results.
func (vr VersionReport) mean(f func(Result) float64) float64 {
	if len(vr.Results) == 0 {
		return 0
	}
	var sum float64
	for _, r := range vr.Results {
		sum += f(r)
	}
	return sum / float64(len(vr.Results))
}

func (vr VersionReport) MeanRecall() float64 {
	return vr.mean(func(r Result) float64 { return r.Recall })
}
func (vr VersionReport) MeanPrecision() float64 {
	return vr.mean(func(r Result) float64 { return r.Precision })
}
func (vr VersionReport) MeanAggregate() float64 { return vr.mean(Result.Aggregate) }

func (vr VersionReport) MeanSpecificity() float64 {
	return condMean(vr.Results, func(r Result) (float64, bool) { return r.Specificity, r.HasSpecificity })
}
func (vr VersionReport) MeanSearchRecall() float64 {
	return condMean(vr.Results, func(r Result) (float64, bool) { return r.SearchRecall, r.HasSearch })
}
func (vr VersionReport) MeanDedup() float64 {
	return condMean(vr.Results, func(r Result) (float64, bool) { return r.dedupScore(), r.HasDedup })
}

// condMean averages only the results for which sel reports applicability.
func condMean(rs []Result, sel func(Result) (float64, bool)) float64 {
	var sum float64
	var n int
	for _, r := range rs {
		if v, ok := sel(r); ok {
			sum += v
			n++
		}
	}
	if n == 0 {
		return 0
	}
	return sum / float64(n)
}

// Run scores every fixture under every configured prompt version against the
// live providers, returning one VersionReport per version.
func Run(ctx context.Context, fixtures []Fixture, cfg Config) ([]VersionReport, error) {
	if cfg.K <= 0 {
		cfg.K = 3
	}
	judge := Judge{LLM: cfg.Judge}
	if cfg.Judge == nil {
		judge = Judge{LLM: cfg.LLM}
	}
	versions := cfg.Versions
	if len(versions) == 0 {
		versions = mneme.PromptVersions()
	}

	var reports []VersionReport
	for _, version := range versions {
		mem, err := mneme.New(
			mneme.WithStore(cfg.Store),
			mneme.WithLLM(cfg.LLM),
			mneme.WithEmbedder(cfg.Embedder),
			mneme.WithPromptVersion(version),
		)
		if err != nil {
			return nil, fmt.Errorf("build memory for version %q: %w", version, err)
		}
		rep := VersionReport{Version: version}
		for _, fx := range fixtures {
			res, err := scoreFixture(ctx, mem, judge, fx, version, cfg.K)
			if err != nil {
				return nil, fmt.Errorf("fixture %q (version %q): %w", fx.Name, version, err)
			}
			rep.Results = append(rep.Results, res)
		}
		reports = append(reports, rep)
	}
	return reports, nil
}

func scoreFixture(ctx context.Context, mem mneme.Memory, judge Judge, fx Fixture, version string, k int) (Result, error) {
	// Isolate every (version, fixture) pair in its own scope.
	scope := mneme.Scope{RunID: version + ":" + fx.Name}

	written, err := mem.Add(ctx, fx.Messages, scope)
	if err != nil {
		return Result{}, fmt.Errorf("add: %w", err)
	}
	extracted := factTexts(written)

	res := Result{
		Name:      fx.Name,
		Extracted: extracted,
		Recall:    recall(ctx, judge, fx.ExpectedFacts, extracted),
		Precision: precision(ctx, judge, fx.ExpectedFacts, extracted),
	}

	if len(fx.RequiredTokens) > 0 {
		res.HasSpecificity = true
		res.Specificity = specificity(fx.RequiredTokens, extracted)
	}

	if len(fx.Queries) > 0 {
		res.HasSearch = true
		res.SearchRecall = searchRecall(ctx, mem, judge, scope, fx.Queries, k)
	}

	// Dedup correctness: re-ingest the same conversation; a correct additive
	// pipeline (extractor sees existing memories + hash dedup) writes ~nothing.
	if !fx.SkipDedup && len(fx.ExpectedFacts) > 0 {
		res.HasDedup = true
		again, err := mem.Add(ctx, fx.Messages, scope)
		if err != nil {
			return Result{}, fmt.Errorf("dedup re-add: %w", err)
		}
		res.DedupNewFacts = len(again)
	}

	return res, nil
}

// searchRecall returns the fraction of queries whose top-k results contain a
// fact semantically matching one of the query's should_recall targets.
func searchRecall(ctx context.Context, mem mneme.Memory, judge Judge, scope mneme.Scope, queries []Query, k int) float64 {
	if len(queries) == 0 {
		return 1
	}
	hitCount := 0
	for _, q := range queries {
		hits, err := mem.Search(ctx, q.Q, scope, k)
		if err != nil {
			continue
		}
		got := factTexts(hits)
		recalled := false
		for _, target := range q.ShouldRecall {
			if judge.anyMatch(ctx, target, got) {
				recalled = true
				break
			}
		}
		if recalled {
			hitCount++
		}
	}
	return float64(hitCount) / float64(len(queries))
}

func factTexts(facts []mneme.Fact) []string {
	out := make([]string, len(facts))
	for i, f := range facts {
		out[i] = f.Text
	}
	return out
}
