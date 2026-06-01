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

// DefaultAnswerVersion is the answer prompt version used unless overridden.
const DefaultAnswerVersion = "v1"

// answerPrompts maps a version name to its QA system prompt. Mirrors the
// extraction prompt registry in the top-level package: a change is a new entry,
// never an in-place edit of a shipped one.
var answerPrompts = map[string]string{
	"v1": answerPromptV1,
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
