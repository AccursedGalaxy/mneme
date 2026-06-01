package bench

import (
	"context"
	"fmt"

	"github.com/AccursedGalaxy/mneme"
	"github.com/AccursedGalaxy/mneme/eval"
	"github.com/AccursedGalaxy/mneme/provider"
	"github.com/AccursedGalaxy/mneme/store"
)

// Strategy names the memory-write strategy under test. Additive is v1's
// keep-everything pipeline; Consolidate (PLAN-v2.md §4.2) is built next and the
// harness already accepts its name so the runner's flag is stable.
const (
	StrategyAdditive    = "additive"
	StrategyConsolidate = "consolidate"
)

// Config holds everything Run needs: the providers under test, the store, the
// answer/judge models, search depth, prompt versions, and the write strategy.
// Mirrors eval.Config so the two harnesses feel the same.
type Config struct {
	LLM      provider.LLM // extraction LLM (the mneme pipeline)
	Embedder provider.Embedder
	Store    store.Store

	AnswerLLM provider.LLM // QA answerer; defaults to LLM
	Judge     provider.LLM // semantic-match oracle; defaults to LLM

	K             int    // search top-k; defaults to 5
	AnswerVersion string // answer prompt version; defaults to DefaultAnswerVersion
	Strategy      string // write strategy; defaults to StrategyAdditive

	// Progress, if set, is called after each sample is fully scored. Lets the
	// runner print live progress without the library writing to stdout.
	Progress func(doneSamples, totalSamples int)
}

// Run executes the benchmark protocol over every sample and returns a Report:
// for each sample, ingest its sessions into a per-sample scope, then for each
// question retrieve top-k facts, answer with the answer LLM, and score the
// answer with EM/F1 and the semantic judge. Samples are isolated by scope so
// they never cross-contaminate (PLAN-v2.md §4.1).
func Run(ctx context.Context, samples []Sample, cfg Config) (Report, error) {
	if cfg.K <= 0 {
		cfg.K = 5
	}
	if cfg.AnswerVersion == "" {
		cfg.AnswerVersion = DefaultAnswerVersion
	}
	if cfg.Strategy == "" {
		cfg.Strategy = StrategyAdditive
	}
	if cfg.Strategy != StrategyAdditive {
		// Consolidation is PLAN-v2.md §4.2 — the next task. The flag is accepted
		// so cmd/bench's surface is stable, but only additive runs today.
		return Report{}, fmt.Errorf("strategy %q not implemented yet (additive only; see PLAN-v2.md §4.2)", cfg.Strategy)
	}

	answerLLM := cfg.AnswerLLM
	if answerLLM == nil {
		answerLLM = cfg.LLM
	}
	judge := eval.Judge{LLM: cfg.Judge}
	if cfg.Judge == nil {
		judge = eval.Judge{LLM: cfg.LLM}
	}

	mem, err := mneme.New(
		mneme.WithStore(cfg.Store),
		mneme.WithLLM(cfg.LLM),
		mneme.WithEmbedder(cfg.Embedder),
	)
	if err != nil {
		return Report{}, fmt.Errorf("build memory: %w", err)
	}

	report := Report{K: cfg.K, Strategy: cfg.Strategy}
	for i, s := range samples {
		results, err := runSample(ctx, mem, answerLLM, judge, cfg, s)
		if err != nil {
			return Report{}, fmt.Errorf("sample %q: %w", s.ID, err)
		}
		report.Results = append(report.Results, results...)
		if cfg.Progress != nil {
			cfg.Progress(i+1, len(samples))
		}
	}
	return report, nil
}

// runSample ingests one sample's sessions then scores its questions.
func runSample(ctx context.Context, mem mneme.Memory, answerLLM provider.LLM, judge eval.Judge, cfg Config, s Sample) ([]QAResult, error) {
	scope := mneme.Scope{RunID: s.ID}

	for si, sess := range s.Sessions {
		if _, err := mem.Add(ctx, ingestMessages(sess), scope); err != nil {
			return nil, fmt.Errorf("ingest session %d: %w", si, err)
		}
	}

	results := make([]QAResult, 0, len(s.Questions))
	for _, q := range s.Questions {
		facts, err := mem.Search(ctx, q.Question, scope, cfg.K)
		if err != nil {
			return nil, fmt.Errorf("search %q: %w", q.Question, err)
		}
		pred, err := Answer(ctx, answerLLM, cfg.AnswerVersion, q.Question, facts)
		if err != nil {
			return nil, err
		}
		results = append(results, QAResult{
			SampleID:  s.ID,
			Question:  q.Question,
			Gold:      q.Answer,
			Predicted: pred,
			Category:  q.Category,
			EM:        exactMatch(pred, q.Answer),
			F1:        f1(pred, q.Answer),
			Judge:     judge.Same(ctx, pred, q.Answer),
		})
	}
	return results, nil
}

// ingestMessages prepends the session date (when present) as a single dated
// note so the extractor has an anchor for relative dates in the turns. The note
// uses role "user" because the pipeline ignores system messages. Structured
// temporal grounding replaces this in PLAN-v2.md §5.1; here it is the honest
// additive baseline.
func ingestMessages(sess Session) []mneme.Message {
	if sess.Date == "" {
		return sess.Messages
	}
	dated := make([]mneme.Message, 0, len(sess.Messages)+1)
	dated = append(dated, mneme.Message{Role: "user", Content: "(The following conversation took place on " + sess.Date + ".)"})
	dated = append(dated, sess.Messages...)
	return dated
}
