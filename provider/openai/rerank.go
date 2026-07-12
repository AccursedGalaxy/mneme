package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/AccursedGalaxy/mneme/provider"
	"github.com/AccursedGalaxy/mneme/types"
)

// LLMReranker reorders retrieved candidates by asking an LLM to score each one's
// relevance to the query. It satisfies mneme.Reranker structurally (mneme.Fact
// is an alias for types.Fact), so wire it in with mneme.WithReranker.
//
// It depends only on provider.LLM, not on the OpenAI wire format specifically —
// it lives here because provider/openai is where mneme ships its concrete
// providers, and a cross-encoder or hosted rerank API can implement the same
// mneme.Reranker interface elsewhere later.
//
// It is best-effort by design, like mneme's other boosters: a malformed or
// empty model response — and a failed LLM call — leaves the candidate order
// unchanged rather than erroring, so enabling reranking never makes Search
// availability depend on the rerank model being up.
type LLMReranker struct {
	// LLM scores the candidates. Reuse the same client as extraction, or point
	// it at a cheaper/faster model dedicated to reranking.
	LLM provider.LLM
}

// Rerank scores each candidate against the query and returns them ordered most
// relevant first. The model's relevance scores determine position only —
// Fact.Score is left untouched (it remains the retrieval similarity Search
// populated), so callers never see rerank-internal values in a public field.
// Candidates the model omits keep their incoming (cosine) order behind the
// scored ones.
func (r *LLMReranker) Rerank(ctx context.Context, query string, candidates []types.Fact) ([]types.Fact, error) {
	if len(candidates) <= 1 {
		return candidates, nil
	}
	raw, err := r.LLM.Complete(ctx, rerankSystemPrompt, buildRerankUser(query, candidates), true)
	if err != nil {
		// Best-effort: a rerank that cannot run degrades to the cosine order
		// instead of failing the caller's Search.
		return candidates, nil
	}
	scores := parseRerankScores(raw)
	if len(scores) == 0 {
		// Nothing usable came back — preserve the cosine order rather than
		// shuffle on garbage.
		return candidates, nil
	}

	// Order by relevance desc without mutating the facts: scored candidates
	// first (higher relevance wins), unscored ones after in their incoming
	// relative order (stable sort).
	out := make([]types.Fact, len(candidates))
	copy(out, candidates)
	relevance := func(i int) (float32, bool) {
		s, ok := scores[i]
		return s, ok
	}
	idx := make([]int, len(out))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool {
		sa, oka := relevance(idx[a])
		sb, okb := relevance(idx[b])
		if oka != okb {
			return oka // scored candidates come first
		}
		return sa > sb
	})
	ordered := make([]types.Fact, len(out))
	for i, j := range idx {
		ordered[i] = out[j]
	}
	return ordered, nil
}

const rerankSystemPrompt = `You score how relevant each candidate memory is to a search query, so the most useful ones can be surfaced first.

You are given a QUERY and a numbered list of CANDIDATE memories. For each candidate, output a relevance score from 0.0 (irrelevant) to 1.0 (directly answers or strongly supports answering the query). Judge relevance to the query only — not whether the memory is recent or independently interesting.

OUTPUT
Return ONLY a JSON object, no prose and no code fences, in exactly this shape:
{"scores":[{"id":0,"score":0.9},{"id":1,"score":0.2}]}
Include every candidate id exactly once, using the ids shown.`

// buildRerankUser lists the candidates under the integer ids the model scores
// against. The ids are positional indices into the candidate slice.
func buildRerankUser(query string, candidates []types.Fact) string {
	var b strings.Builder
	fmt.Fprintf(&b, "QUERY: %s\n\n", query)
	b.WriteString("CANDIDATES (id: memory):\n")
	for i, c := range candidates {
		fmt.Fprintf(&b, "%d: %s\n", i, c.Text)
	}
	return b.String()
}

// rerankScore is one scored candidate. id is the positional index from the
// prompt; flexInt tolerates the model quoting it as a string.
type rerankScore struct {
	ID    flexInt `json:"id"`
	Score float32 `json:"score"`
}

type rerankEnvelope struct {
	Scores []rerankScore `json:"scores"`
}

// parseRerankScores extracts an id->score map from a raw rerank response
// defensively (fences, embedded object, bare array), returning nil on any
// unrecoverable failure so Rerank can preserve the cosine order.
func parseRerankScores(raw string) map[int]float32 {
	s := stripFences(raw)

	if scores, ok := tryRerankEnvelope(s); ok {
		return scores
	}
	if obj := firstBalanced(s, '{', '}'); obj != "" {
		if scores, ok := tryRerankEnvelope(obj); ok {
			return scores
		}
	}
	if arr := firstBalanced(s, '[', ']'); arr != "" {
		var items []rerankScore
		if err := json.Unmarshal([]byte(arr), &items); err == nil {
			return scoreMap(items)
		}
	}
	return nil
}

func tryRerankEnvelope(s string) (map[int]float32, bool) {
	var env rerankEnvelope
	if err := json.Unmarshal([]byte(s), &env); err == nil && env.Scores != nil {
		return scoreMap(env.Scores), true
	}
	return nil, false
}

func scoreMap(items []rerankScore) map[int]float32 {
	// A wrong-schema response can decode "successfully" with every field
	// ignored (unknown JSON keys are dropped), yielding all-zero items. Treat
	// a multi-item batch that carries no signal at all as a parse failure so
	// Rerank preserves the cosine order instead of collapsing on it.
	signal := false
	for _, it := range items {
		if it.ID != 0 || it.Score != 0 {
			signal = true
			break
		}
	}
	if !signal && len(items) > 1 {
		return nil
	}
	out := make(map[int]float32, len(items))
	for _, it := range items {
		out[int(it.ID)] = it.Score
	}
	return out
}

// stripFences removes a leading/trailing markdown code fence if present. It is a
// local copy of the same helper in package mneme (which provider/openai cannot
// import without a cycle), kept tiny and in sync deliberately.
func stripFences(s string) string {
	s = strings.TrimSpace(s)
	if !strings.HasPrefix(s, "```") {
		return s
	}
	s = strings.TrimPrefix(s, "```")
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		if first := strings.TrimSpace(s[:i]); !strings.ContainsAny(first, "{}[]") {
			s = s[i+1:]
		}
	}
	if i := strings.LastIndex(s, "```"); i >= 0 {
		s = s[:i]
	}
	return strings.TrimSpace(s)
}

// firstBalanced returns the substring from the first open delimiter to its
// matching close, ignoring delimiters inside JSON strings. Local copy of the
// mneme helper (no-import-cycle); see parse.go for the annotated original.
func firstBalanced(s string, open, close byte) string {
	start := strings.IndexByte(s, open)
	if start < 0 {
		return ""
	}
	depth := 0
	inStr := false
	escaped := false
	for i := start; i < len(s); i++ {
		c := s[i]
		if inStr {
			switch {
			case escaped:
				escaped = false
			case c == '\\':
				escaped = true
			case c == '"':
				inStr = false
			}
			continue
		}
		switch c {
		case '"':
			inStr = true
		case open:
			depth++
		case close:
			depth--
			if depth == 0 {
				return s[start : i+1]
			}
		}
	}
	return ""
}

// flexInt is an int that unmarshals from either a JSON number or a quoted
// string, so a model that emits "id":"0" instead of "id":0 is still parsed.
type flexInt int

func (f *flexInt) UnmarshalJSON(b []byte) error {
	s := strings.Trim(strings.TrimSpace(string(b)), `"`)
	if s == "" {
		*f = 0
		return nil
	}
	n, err := strconv.Atoi(s)
	if err != nil {
		return err
	}
	*f = flexInt(n)
	return nil
}
