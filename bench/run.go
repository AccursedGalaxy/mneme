package bench

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/AccursedGalaxy/mneme"
	"github.com/AccursedGalaxy/mneme/eval"
	"github.com/AccursedGalaxy/mneme/provider"
	"github.com/AccursedGalaxy/mneme/store"
)

// Strategy names the memory-write strategy under test. Additive is v1's
// keep-everything pipeline; Consolidate (PLAN-v2.md §4.2) runs the second LLM
// call that UPDATEs/DELETEs stale facts.
const (
	StrategyAdditive    = "additive"
	StrategyConsolidate = "consolidate"
)

// mnemeStrategy maps the harness's strategy name to the library's Strategy enum.
func mnemeStrategy(name string) (mneme.Strategy, error) {
	switch name {
	case StrategyAdditive:
		return mneme.Additive, nil
	case StrategyConsolidate:
		return mneme.Consolidate, nil
	default:
		return 0, fmt.Errorf("unknown strategy %q (want %s|%s)", name, StrategyAdditive, StrategyConsolidate)
	}
}

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

	// ConsolidationVersion selects the consolidation prompt version, used only
	// when Strategy is consolidate. Empty means the library default. Lets the
	// harness A/B prompt versions (e.g. v1 vs the conservative v2).
	ConsolidationVersion string

	// Reranker, when set, enables a rerank pass in Search (PLAN-v2.md §4.3): the
	// harness over-retrieves and reorders candidates before truncating to K. Nil
	// leaves Search at plain top-k cosine.
	Reranker mneme.Reranker

	// MultiQuery, when > 1, makes Search expand each question into that many
	// phrasings and union the hits before reranking (PLAN-v2.md §4.3).
	MultiQuery int

	// RawTurnWindow sets how many consecutive messages go into one stored chunk
	// under StrategyRawTurns. 1 (the default) stores a turn per record; larger
	// windows keep a question and its reply in the same chunk. Ignored by every
	// other strategy.
	RawTurnWindow int

	// Oracle, when set (OracleSource), replaces part of the pipeline with ground
	// truth to isolate one stage. Empty runs the real pipeline end to end.
	Oracle string

	// Concurrency bounds how many questions within a sample are scored in
	// parallel. Ingestion stays sequential (writes must be ordered for
	// consolidation); only the read-only Search→Answer→Judge phase fans out, and
	// it is dominated by network LLM calls, so a worker pool cuts a full run from
	// hours to minutes. Defaults to 1 (sequential) when <= 0.
	Concurrency int

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
	if cfg.Concurrency <= 0 {
		cfg.Concurrency = 1
	}

	if _, ok := answerPrompts[cfg.AnswerVersion]; !ok {
		// Silently falling back to the default would make a typo'd version look
		// like a real result for the wrong prompt, which is the class of bug the
		// adversarial-scoring erratum already cost a paid re-run to catch.
		return Report{}, fmt.Errorf("unknown answer prompt version %q (have %v)", cfg.AnswerVersion, AnswerPrompts())
	}

	if err := validateOracle(cfg.Oracle); err != nil {
		return Report{}, err
	}

	// rawturns bypasses the library pipeline on the write side, so it has no
	// mneme.Strategy. It still builds a Memory: search/answer/judge must run on
	// exactly the same code path as every other row, or the comparison is not one.
	rawTurns := cfg.Strategy == StrategyRawTurns
	strategy := mneme.Additive
	if !rawTurns {
		s, err := mnemeStrategy(cfg.Strategy)
		if err != nil {
			return Report{}, err
		}
		strategy = s
	}

	answerLLM := cfg.AnswerLLM
	if answerLLM == nil {
		answerLLM = cfg.LLM
	}
	judge := eval.Judge{LLM: cfg.Judge}
	if cfg.Judge == nil {
		judge = eval.Judge{LLM: cfg.LLM}
	}

	opts := []mneme.Option{
		mneme.WithStore(cfg.Store),
		mneme.WithLLM(cfg.LLM),
		mneme.WithEmbedder(cfg.Embedder),
		mneme.WithStrategy(strategy),
	}
	if cfg.ConsolidationVersion != "" {
		opts = append(opts, mneme.WithConsolidationVersion(cfg.ConsolidationVersion))
	}
	if cfg.Reranker != nil {
		opts = append(opts, mneme.WithReranker(cfg.Reranker))
	}
	if cfg.MultiQuery > 1 {
		opts = append(opts, mneme.WithMultiQuery(cfg.MultiQuery))
	}
	mem, err := mneme.New(opts...)
	if err != nil {
		return Report{}, fmt.Errorf("build memory: %w", err)
	}

	report := Report{
		K:                    cfg.K,
		Strategy:             cfg.Strategy,
		ConsolidationVersion: cfg.ConsolidationVersion,
		Rerank:               cfg.Reranker != nil,
		MultiQuery:           cfg.MultiQuery,
	}
	for i, s := range samples {
		results, err := runSample(ctx, mem, answerLLM, judge, cfg, s, rawTurns)
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
func runSample(ctx context.Context, mem mneme.Memory, answerLLM provider.LLM, judge eval.Judge, cfg Config, s Sample, rawTurns bool) ([]QAResult, error) {
	scope := mneme.Scope{RunID: s.ID}

	// The source oracle answers from the dataset's evidence turns, so there is
	// nothing to ingest: no extraction, no embedding, no store.
	if cfg.Oracle == OracleSource {
		return scoreQuestions(ctx, cfg, s, func(ctx context.Context, q QAPair) (QAResult, error) {
			return scoreOracleQuestion(ctx, answerLLM, judge, cfg, s, q)
		})
	}

	if rawTurns {
		if err := ingestRawTurns(ctx, cfg.Embedder, cfg.Store, scope, s.Sessions, cfg.RawTurnWindow, time.Now()); err != nil {
			return nil, err
		}
	} else {
		for si, sess := range s.Sessions {
			if _, err := mem.Add(ctx, ingestMessages(sess), scope); err != nil {
				return nil, fmt.Errorf("ingest session %d: %w", si, err)
			}
		}
	}

	return scoreQuestions(ctx, cfg, s, func(ctx context.Context, q QAPair) (QAResult, error) {
		return scoreQuestion(ctx, mem, answerLLM, judge, cfg, s.ID, scope, q)
	})
}

// scoreQuestions fans a sample's questions across a bounded worker pool. Scoring
// is read-only and independent per question, and it is dominated by network LLM
// calls, so the pool cuts a full run from hours to minutes. Results are written
// to a pre-sized slice by index, so output order is deterministic regardless of
// completion order.
func scoreQuestions(ctx context.Context, cfg Config, s Sample, score func(context.Context, QAPair) (QAResult, error)) ([]QAResult, error) {
	results := make([]QAResult, len(s.Questions))
	sem := make(chan struct{}, cfg.Concurrency)
	var wg sync.WaitGroup
	var (
		mu       sync.Mutex
		firstErr error
	)
	for i, q := range s.Questions {
		mu.Lock()
		stop := firstErr != nil
		mu.Unlock()
		if stop {
			break
		}
		wg.Add(1)
		sem <- struct{}{}
		go func(i int, q QAPair) {
			defer wg.Done()
			defer func() { <-sem }()

			res, err := score(ctx, q)
			if err != nil {
				mu.Lock()
				if firstErr == nil {
					firstErr = err
				}
				mu.Unlock()
				return
			}
			results[i] = res
		}(i, q)
	}
	wg.Wait()
	if firstErr != nil {
		return nil, firstErr
	}
	return results, nil
}

// scoreQuestion runs the read-only scoring path for one question: retrieve
// top-k facts, answer, and score with EM/F1 and the semantic judge. Safe to
// call concurrently — it only reads from mem and the stateless answer/judge
// LLMs, never writes to the store.
func scoreQuestion(ctx context.Context, mem mneme.Memory, answerLLM provider.LLM, judge eval.Judge, cfg Config, sampleID string, scope mneme.Scope, q QAPair) (QAResult, error) {
	facts, err := mem.Search(ctx, q.Question, scope, cfg.K)
	if err != nil {
		return QAResult{}, fmt.Errorf("search %q: %w", q.Question, err)
	}
	pred, err := Answer(ctx, answerLLM, cfg.AnswerVersion, q.Question, facts)
	if err != nil {
		return QAResult{}, err
	}
	res := QAResult{
		SampleID:     sampleID,
		Question:     q.Question,
		Gold:         q.Answer,
		Predicted:    pred,
		Category:     q.Category,
		Unanswerable: q.Unanswerable,
		Retrieved:    factTexts(facts),
	}
	res.EM, res.F1, res.Judge = Score(ctx, judge, q.Unanswerable, pred, q.Answer)
	return res, nil
}

// factTexts renders the search hits exactly as the answer model saw them, for
// the prediction dump — so a wrong answer can be traced to what the model was
// actually given, and so cmd/replay can re-answer from the dump without losing
// anything the original prompt carried (the observation dates, in particular).
func factTexts(facts []mneme.Fact) []string {
	out := make([]string, len(facts))
	for i, f := range facts {
		out[i] = RenderFact(f)
	}
	return out
}

// ingestMessages stamps every message with the session's timestamp and prepends
// the date as a dated note.
//
// The Timestamp is the structured grounding (PLAN-v2.md §5.1): the pipeline reads
// it to date the extractor's OBSERVATION DATE and to set each fact's ObservedAt,
// so a relative date in a turn ("I went yesterday") resolves against when the
// conversation happened rather than against the day the benchmark was run. Before
// this, flash-lite extracted "Caroline attended an LGBTQ support group on
// 2026-07-11" — the ingestion date — for a May 2023 event (bench/RESULTS.md).
//
// The prose note stays. It costs a few tokens, it is what the pre-timestamp runs
// in the matrix were measured with, and a session whose date does not parse still
// gets *some* anchor from it.
// A date the layouts cannot parse leaves the timestamps alone rather than
// stamping the zero time over them: the session still gets the prose note, and a
// message that already carried its own Timestamp keeps it. Silently overwriting a
// real timestamp with "unknown" would be the same class of corruption this whole
// mechanism exists to prevent.
func ingestMessages(sess Session) []mneme.Message {
	if sess.Date == "" {
		return sess.Messages
	}
	at, ok := parseSessionDate(sess.Date)
	if !ok {
		unparsedSessionDates.Add(1)
	}

	dated := make([]mneme.Message, 0, len(sess.Messages)+1)
	dated = append(dated, mneme.Message{Role: "user", Timestamp: at, Content: "(The following conversation took place on " + sess.Date + ".)"})
	for _, msg := range sess.Messages {
		if ok {
			msg.Timestamp = at
		}
		dated = append(dated, msg)
	}
	return dated
}

// unparsedSessionDates counts sessions whose date matched no layout. A run where
// this is nonzero has silently lost temporal grounding — every fact in those
// sessions falls back to the ingestion date — and the headline would read "no
// temporal improvement" rather than "the dates never parsed". UnparsedSessionDates
// is what lets the harness say which.
var unparsedSessionDates atomic.Int64

// UnparsedSessionDates reports how many sessions have so far failed to yield a
// timestamp. A nonzero count means the dataset's date format is not in
// sessionDateLayouts and the run's temporal numbers are not trustworthy.
func UnparsedSessionDates() int64 { return unparsedSessionDates.Load() }

// sessionDateLayouts are the shapes a dataset writes a session date in. LoCoMo
// uses the prose form ("1:56 pm on 8 May, 2023"); LongMemEval's haystack_dates
// are ISO-ish, and its published form carries a weekday and time
// ("2023/05/20 (Sat) 02:21"), so both are accepted.
//
// Order matters only in that the first match wins; the shapes are disjoint.
var sessionDateLayouts = []string{
	"3:04 pm on 2 January, 2006",
	"15:04 on 2 January, 2006",
	"2 January, 2006",
	time.RFC3339,
	"2006/01/02 (Mon) 15:04",
	"2006-01-02 15:04",
	"2006-01-02",
}

// parseSessionDate turns a dataset's session date into a timestamp. It reports
// whether it parsed: an unparsed date must stay the zero time ("unknown") rather
// than silently becoming some default, because a wrong timestamp is worse than an
// absent one — it would be confidently propagated into every fact.
func parseSessionDate(s string) (time.Time, bool) {
	s = strings.TrimSpace(s)
	for _, layout := range sessionDateLayouts {
		if t, err := time.Parse(layout, s); err == nil {
			return t.UTC(), true
		}
	}
	return time.Time{}, false
}
