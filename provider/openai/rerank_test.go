package openai

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/AccursedGalaxy/mneme/provider/fake"
	"github.com/AccursedGalaxy/mneme/types"
)

func cands(texts ...string) []types.Fact {
	out := make([]types.Fact, len(texts))
	for i, t := range texts {
		out[i] = types.Fact{ID: string(rune('a' + i)), Text: t, Score: float32(len(texts)-i) / 10} // descending cosine
	}
	return out
}

func order(facts []types.Fact) string {
	ids := make([]string, len(facts))
	for i, f := range facts {
		ids[i] = f.ID
	}
	return strings.Join(ids, "")
}

func TestLLMRerankerReorders(t *testing.T) {
	// Cosine order is a,b,c; the model says c is most relevant, a least.
	llm := &fake.LLM{Default: `{"scores":[{"id":0,"score":0.1},{"id":1,"score":0.5},{"id":2,"score":0.9}]}`}
	r := &LLMReranker{LLM: llm}

	got, err := r.Rerank(context.Background(), "q", cands("fact a", "fact b", "fact c"))
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if order(got) != "cba" {
		t.Errorf("reorder by relevance: got %s want cba", order(got))
	}
	// Fact.Score is a public field documented as the retrieval similarity;
	// reranking reorders but must never overwrite it with model-internal values.
	if got[0].Score != 0.1 { // candidate c's original cosine score
		t.Errorf("Score must keep the retrieval similarity (0.1), got %v", got[0].Score)
	}
}

func TestLLMRerankerStringIDs(t *testing.T) {
	// A model that quotes the ids must still parse (flexInt).
	llm := &fake.LLM{Default: `{"scores":[{"id":"0","score":0.2},{"id":"1","score":0.8}]}`}
	r := &LLMReranker{LLM: llm}
	got, err := r.Rerank(context.Background(), "q", cands("a", "b"))
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if order(got) != "ba" {
		t.Errorf("string-id reorder: got %s want ba", order(got))
	}
}

func TestLLMRerankerMissingScoreSortsLast(t *testing.T) {
	// Only id 1 is scored; the unscored ones keep their incoming relative order
	// behind it.
	llm := &fake.LLM{Default: `{"scores":[{"id":1,"score":0.9}]}`}
	r := &LLMReranker{LLM: llm}
	got, err := r.Rerank(context.Background(), "q", cands("a", "b", "c"))
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if order(got) != "bac" {
		t.Errorf("scored-first, then stable cosine order: got %s want bac", order(got))
	}
}

func TestLLMRerankerJunkPreservesOrder(t *testing.T) {
	llm := &fake.LLM{Default: "I cannot rank these, sorry"}
	r := &LLMReranker{LLM: llm}
	in := cands("a", "b", "c")
	got, err := r.Rerank(context.Background(), "q", in)
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if order(got) != "abc" {
		t.Errorf("junk response must preserve cosine order: got %s", order(got))
	}
}

func TestLLMRerankerSingleCandidateSkipsCall(t *testing.T) {
	llm := &fake.LLM{Default: `{"scores":[]}`}
	r := &LLMReranker{LLM: llm}
	got, err := r.Rerank(context.Background(), "q", cands("only"))
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if len(got) != 1 || llm.CallCount() != 0 {
		t.Errorf("single candidate should pass through without an LLM call: calls=%d", llm.CallCount())
	}
}

func TestLLMRerankerTransportErrorDegradesToCosineOrder(t *testing.T) {
	// Reranking is a best-effort booster: a failed LLM call must not fail the
	// caller's Search, it degrades to the incoming cosine order.
	llm := &fake.LLM{Responder: func(string, string) (string, error) { return "", errors.New("503") }}
	r := &LLMReranker{LLM: llm}
	got, err := r.Rerank(context.Background(), "q", cands("a", "b"))
	if err != nil {
		t.Fatalf("LLM failure must not propagate out of Rerank: %v", err)
	}
	if order(got) != "ab" {
		t.Errorf("LLM failure must preserve cosine order: got %s", order(got))
	}
}

func TestLLMRerankerUnscoredKeepOriginalScores(t *testing.T) {
	// A partial response must not leak sentinel values into unscored
	// candidates' public Score field.
	llm := &fake.LLM{Default: `{"scores":[{"id":1,"score":0.9}]}`}
	r := &LLMReranker{LLM: llm}
	in := cands("a", "b", "c")
	got, err := r.Rerank(context.Background(), "q", in)
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	for _, f := range got {
		if f.Score < 0 {
			t.Errorf("candidate %s has corrupted Score %v", f.ID, f.Score)
		}
	}
}

func TestScoreMapRejectsSchemalessDecode(t *testing.T) {
	// A wrong-schema array (unknown fields ignored -> all-zero items) must be
	// treated as a parse failure, not as "everything scored 0".
	got := parseRerankScores(`[{"index":2,"relevance":0.9},{"index":0,"relevance":0.1}]`)
	if got != nil {
		t.Errorf("schema-mismatched response should yield nil scores, got %v", got)
	}
}

func TestBuildRerankUserListsCandidates(t *testing.T) {
	u := buildRerankUser("where does Bob live?", cands("Bob lives in Austin", "Bob has a dog"))
	if !strings.Contains(u, "QUERY: where does Bob live?") {
		t.Errorf("user message should embed the query: %q", u)
	}
	if !strings.Contains(u, "0: Bob lives in Austin") || !strings.Contains(u, "1: Bob has a dog") {
		t.Errorf("user message should number the candidates: %q", u)
	}
}

func TestParseRerankScores(t *testing.T) {
	cases := []struct {
		name string
		raw  string
		want map[int]float32
	}{
		{"envelope", `{"scores":[{"id":0,"score":0.5}]}`, map[int]float32{0: 0.5}},
		{"fenced", "```json\n{\"scores\":[{\"id\":2,\"score\":1}]}\n```", map[int]float32{2: 1}},
		{"bare array", `[{"id":1,"score":0.3}]`, map[int]float32{1: 0.3}},
		{"garbage", `no json`, nil},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := parseRerankScores(c.raw)
			if len(got) != len(c.want) {
				t.Fatalf("parseRerankScores(%q) len = %d, want %d (%v)", c.raw, len(got), len(c.want), got)
			}
			for k, v := range c.want {
				if got[k] != v {
					t.Errorf("score[%d] = %v, want %v", k, got[k], v)
				}
			}
		})
	}
}
