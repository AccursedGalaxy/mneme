package bench

import (
	"context"
	"fmt"
	"strings"

	"github.com/AccursedGalaxy/mneme"
	"github.com/AccursedGalaxy/mneme/provider"
)

// answerPromptV1 is the QA system prompt: answer strictly from the retrieved
// memories, abstain when they do not contain the answer, and stay terse. The
// benchmarks score the generated answer, so this prompt is a real score lever —
// it is versioned (like the extraction prompt) so a change is a measured
// decision, not a vibe. "Closed-book" discipline (only the given memories,
// explicit "I don't know") is what makes adversarial/unanswerable questions
// scorable instead of hallucinated.
const answerPromptV1 = `You answer a QUESTION using ONLY the MEMORIES provided below.

Rules:
- Use only the MEMORIES. Do not use outside knowledge or guess.
- If the MEMORIES do not contain the answer, reply exactly: I don't know.
- Be terse: answer with just the fact (a name, date, number, short phrase) — no
  preamble, no explanation, no restating the question.
- If the question asks "when", answer with the most specific date or time the
  MEMORIES support.`

// answerPromptV2 fixes the two answer-stage failures the source oracle exposed
// (RESULTS.md, "The answer stage"). v1 lost more score than retrieval did.
//
// First, v1's abstention rule is too strong: flash-lite replies "I don't know."
// to questions whose answer sits verbatim in the memory, whenever the memory
// states it in passing or as something the speaker realized rather than as a
// flat assertion. v2 says a memory still answers the question in that case, and
// reserves the abstention for memories that do not bear on the question at all.
//
// Second, v1 never tells the answerer that a memory is timestamped, so when the
// memory dates something relatively ("I went yesterday") the model echoes the
// relative phrase back instead of resolving it. 85 of 144 temporal misses at the
// oracle were this. v2 requires an absolute date.
//
// This is worth +0.097 [+0.081, +0.112] answerable Judge at the oracle and
// +0.063 on rawturns, against -0.03 on the adversarial category: loosening the
// abstention rule buys real answers at the cost of a few extra bites on the
// traps, and the answerable set is 3.5x the adversarial one.
//
// It is worth ~nothing on the fact-store runs, and the reason is the finding
// that matters: facts carry no source timestamp, so there is nothing for the
// date rule to resolve against. Raw turns keep the timestamp; extraction throws
// it away. See RESULTS.md.
const answerPromptV2 = `You answer a QUESTION using ONLY the MEMORIES provided below.

Rules:
- Use only the MEMORIES. Do not use outside knowledge.
- A memory answers the question even when it says so in passing, in the speaker's
  own words, or as something they realized, felt or planned. Draw the answer out
  of what the memories say. Reply exactly "I don't know." only when no memory
  bears on the question at all.
- Be terse: reply with the fact itself — a name, a date, a number, a short
  phrase. Never quote a memory back, never restate the question, no preamble.
- Each memory is prefixed with the date and time it was said. If a memory dates
  something relatively ("yesterday", "last week", "next month", "two years ago"),
  resolve it against that prefix and answer with the absolute date. Never answer
  a "when" question with a relative phrase.`

// DefaultAnswerVersion is the answer prompt version used unless overridden.
const DefaultAnswerVersion = "v2"

// answerPrompts maps a version name to its QA system prompt. Mirrors the
// extraction prompt registry in the top-level package: a change is a new entry,
// never an in-place edit of a shipped one.
var answerPrompts = map[string]string{
	"v1": answerPromptV1,
	"v2": answerPromptV2,
}

// AnswerPrompts returns the registered answer prompt version names (unordered).
func AnswerPrompts() []string {
	out := make([]string, 0, len(answerPrompts))
	for k := range answerPrompts {
		out = append(out, k)
	}
	return out
}

// answerSystem returns the QA system prompt for a version, falling back to the
// default when the version is unknown.
func answerSystem(version string) string {
	if p, ok := answerPrompts[version]; ok {
		return p
	}
	return answerPrompts[DefaultAnswerVersion]
}

// Answer is the retrieve→answer step's second half: it feeds the retrieved
// facts and the question to the answer LLM under the versioned QA prompt and
// returns the model's answer. With no facts it still asks (the prompt makes the
// model abstain), so unanswerable/adversarial questions score correctly.
func Answer(ctx context.Context, llm provider.LLM, version, question string, facts []mneme.Fact) (string, error) {
	user := buildAnswerUser(question, facts)
	out, err := llm.Complete(ctx, answerSystem(version), user, false)
	if err != nil {
		return "", fmt.Errorf("answer LLM call: %w", err)
	}
	return strings.TrimSpace(out), nil
}

// buildAnswerUser renders the MEMORIES block and the QUESTION. Facts are listed
// highest-scored first (Search already returns them in that order).
func buildAnswerUser(question string, facts []mneme.Fact) string {
	var b strings.Builder
	b.WriteString("MEMORIES:\n")
	if len(facts) == 0 {
		b.WriteString("(none)\n")
	} else {
		for _, f := range facts {
			fmt.Fprintf(&b, "- %s\n", f.Text)
		}
	}
	b.WriteString("\nQUESTION: ")
	b.WriteString(question)
	b.WriteString("\n")
	return b.String()
}
