// Command replay re-runs the ANSWER step of a finished bench run against a
// different answer prompt, reading the run's prediction dump and writing a new
// dump that cmd/rescore can compare against the original.
//
// The dump records, per question, the exact facts the answer model saw. So a
// change confined to the answer stage — a new prompt version, a different answer
// model — can be evaluated without re-ingesting the corpus or re-running
// retrieval, which are the slow and expensive parts of a run. The answer prompt
// cannot influence what retrieval returned, so holding `retrieved` fixed is not
// an approximation here; it is exact.
//
// This is how answer prompt v2 was measured (bench/RESULTS.md, "The answer
// stage"): +0.097 answerable Judge at the source oracle, for the cost of one
// answer call and up to two judge calls per question, against a full matrix
// re-run.
//
// Usage:
//
//	go run ./cmd/replay -answer-version v2 \
//	  -in bench/results/rawturns_w1.jsonl -out bench/results/rawturns_w1_v2.jsonl
//	go run ./cmd/rescore -baseline bench/results/rawturns_w1.jsonl \
//	  bench/results/rawturns_w1_v2.jsonl
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"sync"
	"syscall"

	"github.com/AccursedGalaxy/mneme"
	"github.com/AccursedGalaxy/mneme/bench"
	"github.com/AccursedGalaxy/mneme/eval"
	"github.com/AccursedGalaxy/mneme/provider/openai"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "replay:", err)
		os.Exit(1)
	}
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func run() error {
	var (
		in        = flag.String("in", "", "prediction dump of a finished run (JSONL, required)")
		out       = flag.String("out", "", "dump to write with the new answers (JSONL, required)")
		version   = flag.String("answer-version", bench.DefaultAnswerVersion, "answer prompt version to replay with (v1 | v2)")
		base      = flag.String("base", envOr("MNEME_LLM_BASE_URL", "https://openrouter.ai/api/v1"), "OpenAI-compatible base URL")
		key       = flag.String("key", envOr("MNEME_LLM_API_KEY", os.Getenv("OPENROUTER_API_KEY")), "API key")
		answerMdl = flag.String("answer-model", "google/gemini-2.5-flash-lite", "model that answers; pin this to the model the original run used or the comparison is confounded")
		judgeMdl  = flag.String("judge-model", "google/gemini-2.5-flash-lite", "model for the semantic judge; pin this to the original run's judge")
		concur    = flag.Int("concurrency", 16, "questions replayed in parallel")
	)
	flag.Parse()

	if *in == "" || *out == "" {
		flag.Usage()
		return fmt.Errorf("-in and -out are required")
	}
	if *key == "" {
		return fmt.Errorf("no API key (set MNEME_LLM_API_KEY or OPENROUTER_API_KEY, or pass -key)")
	}

	preds, err := readDump(*in)
	if err != nil {
		return err
	}

	answerLLM := &openai.LLM{BaseURL: *base, APIKey: *key, Model: *answerMdl}
	judge := eval.Judge{LLM: &openai.LLM{BaseURL: *base, APIKey: *key, Model: *judgeMdl}}

	// Ctrl-C cancels in flight rather than orphaning paid calls.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	fmt.Fprintf(os.Stderr, "replaying %d questions from %s with answer prompt %s\n", len(preds), *in, *version)
	replayed, err := replayAll(ctx, answerLLM, judge, *version, preds, *concur)
	if err != nil {
		return err
	}

	if err := writeDump(*out, replayed); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "\nwrote %s — compare with:\n  go run ./cmd/rescore -baseline %s %s\n", *out, *in, *out)
	return nil
}

// replayAll re-answers and re-judges every question, preserving the original
// order so the two dumps line up question-for-question.
func replayAll(ctx context.Context, llm *openai.LLM, judge eval.Judge, version string, preds []bench.Prediction, concur int) ([]bench.Prediction, error) {
	if concur < 1 {
		concur = 1
	}
	out := make([]bench.Prediction, len(preds))

	var (
		wg   sync.WaitGroup
		sem  = make(chan struct{}, concur)
		mu   sync.Mutex
		done int
		errs []error
	)
	for i, p := range preds {
		wg.Add(1)
		sem <- struct{}{}
		go func(i int, p bench.Prediction) {
			defer wg.Done()
			defer func() { <-sem }()

			res, err := replayOne(ctx, llm, judge, version, p)
			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				errs = append(errs, err)
				// Keep the original prediction so a partial failure degrades to
				// "this question did not move" rather than silently scoring 0.
				out[i] = p
			} else {
				out[i] = res
			}
			done++
			fmt.Fprintf(os.Stderr, "\r  replayed %d/%d", done, len(preds))
		}(i, p)
	}
	wg.Wait()

	if len(errs) > 0 {
		// Loud, but not fatal: a handful of provider errors should not throw away
		// the other few thousand paid answers. The count is what matters — a run
		// with many failures is not comparable and the operator must see that.
		fmt.Fprintf(os.Stderr, "\n%d/%d questions failed and kept their original answer (first: %v)\n", len(errs), len(preds), errs[0])
	}
	return out, nil
}

// replayOne re-answers a single question from the facts the original run
// retrieved, then re-scores it exactly as the harness would.
func replayOne(ctx context.Context, llm *openai.LLM, judge eval.Judge, version string, p bench.Prediction) (bench.Prediction, error) {
	facts := make([]mneme.Fact, 0, len(p.Retrieved))
	for _, t := range p.Retrieved {
		facts = append(facts, mneme.Fact{Text: t})
	}

	pred, err := bench.Answer(ctx, llm, version, p.Question, facts)
	if err != nil {
		return bench.Prediction{}, err
	}

	p.Predicted = pred
	p.EM, p.F1, p.Judge = bench.Score(ctx, judge, p.Unanswerable, pred, p.Gold)
	return p, nil
}

func readDump(path string) ([]bench.Prediction, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var preds []bench.Prediction
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024) // retrieved turns can be long
	for sc.Scan() {
		if len(sc.Bytes()) == 0 {
			continue
		}
		var p bench.Prediction
		if err := json.Unmarshal(sc.Bytes(), &p); err != nil {
			return nil, fmt.Errorf("%s: %w", path, err)
		}
		preds = append(preds, p)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	if len(preds) == 0 {
		return nil, fmt.Errorf("%s: no predictions", path)
	}
	return preds, nil
}

func writeDump(path string, preds []bench.Prediction) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	enc := json.NewEncoder(w)
	for _, p := range preds {
		if err := enc.Encode(p); err != nil {
			return err
		}
	}
	return w.Flush()
}
