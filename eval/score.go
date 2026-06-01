package eval

import (
	"context"
	"strings"

	"github.com/AccursedGalaxy/mneme/provider"
)

// Judge decides whether two free-text facts assert the same thing, using an LLM
// for semantic matching. Facts are free text, so exact-string comparison is too
// brittle; the judge is the semantic-equality oracle the metrics build on.
type Judge struct {
	LLM provider.LLM
}

const judgeSystem = `You compare two statements about a person and decide whether they convey the
same core fact.

They MATCH if a reader would take away the same essential claim. Differences in
wording, and EXTRA detail present in only one statement — such as an exact date,
a number, or added precision — do NOT break a match; added specificity is fine.

They do NOT match if they contradict each other, or assert genuinely different
facts: a different subject, action, or object, or the opposite direction of a
claim (e.g. "acquired by" vs "acquired", "loves X" vs "used to love X but
stopped").

Answer with exactly one word: YES or NO.`

// Same reports whether a and b mean the same thing. Semantic equality is
// symmetric, but the LLM judge is mildly order-sensitive, so we ask both
// orderings and accept a match if either says yes — this removes spurious
// asymmetry between the recall and precision passes (which compare the same
// pair in opposite orders). On any LLM error or ambiguous answer a single ask
// returns false: a conservative miss rather than a score-inflating false match.
func (j Judge) Same(ctx context.Context, a, b string) bool {
	if strings.EqualFold(strings.TrimSpace(a), strings.TrimSpace(b)) {
		return true // cheap exact-match short-circuit, no LLM call
	}
	return j.ask(ctx, a, b) || j.ask(ctx, b, a)
}

func (j Judge) ask(ctx context.Context, a, b string) bool {
	user := "Statement A: " + a + "\nStatement B: " + b
	out, err := j.LLM.Complete(ctx, judgeSystem, user, false)
	if err != nil {
		return false
	}
	out = strings.ToLower(strings.TrimSpace(out))
	return strings.HasPrefix(out, "yes")
}

// anyMatch reports whether target semantically matches any candidate.
func (j Judge) anyMatch(ctx context.Context, target string, candidates []string) bool {
	for _, c := range candidates {
		if j.Same(ctx, target, c) {
			return true
		}
	}
	return false
}

// recall: fraction of expected facts matched by some extracted fact.
func recall(ctx context.Context, j Judge, expected, extracted []string) float64 {
	if len(expected) == 0 {
		return 1 // nothing required -> trivially satisfied
	}
	matched := 0
	for _, e := range expected {
		if j.anyMatch(ctx, e, extracted) {
			matched++
		}
	}
	return float64(matched) / float64(len(expected))
}

// precision: fraction of extracted facts that match some expected fact. With no
// expected facts (the nothing-to-extract case), precision is 1 only when the
// extractor also produced nothing.
func precision(ctx context.Context, j Judge, expected, extracted []string) float64 {
	if len(extracted) == 0 {
		return 1 // produced nothing -> no false positives
	}
	matched := 0
	for _, x := range extracted {
		if j.anyMatch(ctx, x, expected) {
			matched++
		}
	}
	return float64(matched) / float64(len(extracted))
}

// specificity: fraction of required tokens that survive (case-insensitively)
// into some extracted fact. Deterministic — no LLM needed.
func specificity(required, extracted []string) float64 {
	if len(required) == 0 {
		return 1
	}
	joined := strings.ToLower(strings.Join(extracted, "\n"))
	kept := 0
	for _, tok := range required {
		if strings.Contains(joined, strings.ToLower(tok)) {
			kept++
		}
	}
	return float64(kept) / float64(len(required))
}
