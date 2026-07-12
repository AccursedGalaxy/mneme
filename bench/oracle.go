package bench

import (
	"context"
	"fmt"

	"github.com/AccursedGalaxy/mneme"
	"github.com/AccursedGalaxy/mneme/eval"
	"github.com/AccursedGalaxy/mneme/provider"
)

// Oracle modes replace part of the pipeline with ground truth, so a run measures
// one stage in isolation instead of the end-to-end product of all of them.
//
// The end-to-end score confounds four failures: the fact was never extracted, it
// was extracted but not retrieved, it was retrieved but the answerer fumbled it,
// or the judge was wrong. A single number cannot tell those apart, and mneme's
// 40% abstention rate on answerable questions (RESULTS.md finding #2) is
// currently attributed to extraction on circumstantial evidence.
const (
	// OracleSource answers from the dataset's own evidence turns, skipping the
	// store completely. Nothing is ingested, nothing is retrieved. Whatever it
	// scores is the ceiling: no memory system can beat answering from the exact
	// turns the question was written against. The gap between it and a real run
	// is the total cost of the memory pipeline, and the gap between it and 1.0
	// is the answer model plus the judge, which no memory work can fix.
	OracleSource = "source"
)

// answerFromEvidence renders the turns a question cites as if they were retrieved
// facts, so the answer path is byte-for-byte the one a real run uses.
func answerFromEvidence(sample Sample, q QAPair) []mneme.Fact {
	facts := make([]mneme.Fact, 0, len(q.Evidence))
	for _, id := range q.Evidence {
		if text, ok := sample.Turns[id]; ok {
			facts = append(facts, mneme.Fact{Text: text})
		}
	}
	return facts
}

// scoreOracleQuestion answers one question from its gold evidence turns.
//
// Adversarial questions have no evidence by construction, so the oracle hands the
// answerer nothing and the correct behavior is still abstention. That keeps the
// unanswerable rule identical to a real run rather than special-casing it.
func scoreOracleQuestion(ctx context.Context, answerLLM provider.LLM, judge eval.Judge, cfg Config, sample Sample, q QAPair) (QAResult, error) {
	facts := answerFromEvidence(sample, q)

	pred, err := Answer(ctx, answerLLM, cfg.AnswerVersion, q.Question, facts)
	if err != nil {
		return QAResult{}, err
	}

	res := QAResult{
		SampleID:     sample.ID,
		Question:     q.Question,
		Gold:         q.Answer,
		Predicted:    pred,
		Category:     q.Category,
		Unanswerable: q.Unanswerable,
		Retrieved:    factTexts(facts),
	}
	if q.Unanswerable {
		ok := abstained(pred)
		res.EM = ok
		res.Judge = ok
		if ok {
			res.F1 = 1
		}
		return res, nil
	}
	res.EM = exactMatch(pred, q.Answer)
	res.F1 = f1(pred, q.Answer)
	res.Judge = judge.Same(ctx, pred, q.Answer)
	return res, nil
}

// validateOracle rejects an unknown mode early, before a run spends money.
func validateOracle(mode string) error {
	switch mode {
	case "", OracleSource:
		return nil
	default:
		return fmt.Errorf("unknown oracle %q (want %s)", mode, OracleSource)
	}
}
