package mneme

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/AccursedGalaxy/mneme/provider/fake"
	"github.com/AccursedGalaxy/mneme/types"
)

// seedFacts inserts pre-embedded records straight into the store so Search tests
// don't depend on the extraction LLM. Returns the assigned ids in input order.
func seedFacts(t *testing.T, m *memory, scope Scope, texts ...string) []string {
	t.Helper()
	ctx := context.Background()
	vecs, err := m.embedder.Embed(ctx, texts)
	if err != nil {
		t.Fatalf("embed seed: %v", err)
	}
	base := time.Date(2026, 6, 1, 12, 0, 0, 0, time.UTC)
	recs := make([]types.Record, len(texts))
	ids := make([]string, len(texts))
	for i, txt := range texts {
		ids[i] = fmt.Sprintf("fact-%02d", i)
		recs[i] = types.Record{
			ID:        ids[i],
			Text:      txt,
			Hash:      ids[i],
			Embedding: vecs[i],
			Scope:     scope,
			CreatedAt: base.Add(time.Duration(i) * time.Second),
		}
	}
	if err := m.store.Insert(ctx, recs); err != nil {
		t.Fatalf("seed insert: %v", err)
	}
	return ids
}

func ids(facts []Fact) []string {
	out := make([]string, len(facts))
	for i, f := range facts {
		out[i] = f.ID
	}
	return out
}

func contains(ss []string, want string) bool {
	for _, s := range ss {
		if s == want {
			return true
		}
	}
	return false
}

// reverseReranker reverses the candidate order. Combined with over-retrieval it
// proves the rerank pass sees the whole pool and that its order — not raw cosine
// — decides the final top-k.
type reverseReranker struct{}

func (reverseReranker) Rerank(_ context.Context, _ string, c []Fact) ([]Fact, error) {
	out := make([]Fact, len(c))
	for i, f := range c {
		out[len(c)-1-i] = f
	}
	return out, nil
}

type errReranker struct{}

func (errReranker) Rerank(_ context.Context, _ string, _ []Fact) ([]Fact, error) {
	return nil, errors.New("boom")
}

// TestSearchRerankPullsDilutedFactIntoWindow checks the core claim of task #1:
// a fact ranked below the k cutoff by raw cosine is recoverable once a reranker
// reorders the wider pool. With the reverse reranker, the worst-cosine fact
// (rank N, normally dropped) becomes rank 1.
func TestSearchRerankPullsDilutedFactIntoWindow(t *testing.T) {
	m := newTestMemory(t, &fake.LLM{})
	ctx := context.Background()
	scope := Scope{UserID: "u"}
	seedFacts(t, m, scope,
		"alpha one", "alpha two", "alpha three", "beta four", "gamma five", "delta six")

	query := "alpha"
	k := 2

	// Baseline cosine order over the whole pool (poolN covers all 6 facts).
	full, err := m.gatherCandidates(ctx, []string{query}, scope, DefaultRerankPoolN)
	if err != nil {
		t.Fatalf("gather: %v", err)
	}
	cosineOrder := ids(full)

	plain, err := m.Search(ctx, query, scope, k)
	if err != nil {
		t.Fatalf("plain search: %v", err)
	}
	if got := ids(plain); !equalSlice(got, cosineOrder[:k]) {
		t.Fatalf("no-booster search should be plain top-k cosine: got %v want %v", got, cosineOrder[:k])
	}

	m.reranker = reverseReranker{}
	reranked, err := m.Search(ctx, query, scope, k)
	if err != nil {
		t.Fatalf("rerank search: %v", err)
	}
	gotRR := ids(reranked)
	if len(gotRR) != k {
		t.Fatalf("rerank should truncate to k=%d, got %d", k, len(gotRR))
	}
	// The worst-cosine fact (last in cosineOrder) is now first, and is one the
	// plain top-k dropped — the dilution-recovery the task is about.
	worst := cosineOrder[len(cosineOrder)-1]
	if gotRR[0] != worst {
		t.Errorf("reverse reranker should put worst-cosine fact first: got %v, worst=%s", gotRR, worst)
	}
	if contains(ids(plain), worst) {
		t.Fatalf("test precondition: worst-cosine fact must be outside plain top-k")
	}
}

func TestSearchRerankErrorPropagates(t *testing.T) {
	m := newTestMemory(t, &fake.LLM{})
	ctx := context.Background()
	scope := Scope{UserID: "u"}
	seedFacts(t, m, scope, "alpha one", "alpha two", "alpha three")

	m.reranker = errReranker{}
	_, err := m.Search(ctx, "alpha", scope, 2)
	if err == nil || !strings.Contains(err.Error(), "rerank") {
		t.Fatalf("rerank error should propagate as a rerank error, got %v", err)
	}
}

// TestSearchMultiQueryUnionsHits checks that a second phrasing surfaces a fact
// the original query would not, via the union+dedup path.
func TestSearchMultiQueryUnionsHits(t *testing.T) {
	// The expander returns a phrasing that matches factB exactly.
	llm := &fake.LLM{Responder: func(system, _ string) (string, error) {
		if strings.Contains(system, "rewrite a search query") {
			return `{"queries":["bravo banana"]}`, nil
		}
		return "", nil
	}}
	m := newTestMemory(t, llm)
	ctx := context.Background()
	scope := Scope{UserID: "u"}
	// factA matches the original query; factC partially matches it (shares
	// "alpha"); factB shares nothing with the original but is the expansion.
	seed := seedFacts(t, m, scope, "alpha apple", "bravo banana", "alpha orange")
	factA, factB, factC := seed[0], seed[1], seed[2]

	query := "alpha apple"
	k := 2

	plain, err := m.Search(ctx, query, scope, k)
	if err != nil {
		t.Fatalf("plain: %v", err)
	}
	plainIDs := ids(plain)
	if !contains(plainIDs, factA) || !contains(plainIDs, factC) || contains(plainIDs, factB) {
		t.Fatalf("plain top-2 should be {A,C} and exclude B, got %v (A=%s B=%s C=%s)", plainIDs, factA, factB, factC)
	}

	m.multiQueryN = 2
	multi, err := m.Search(ctx, query, scope, k)
	if err != nil {
		t.Fatalf("multiquery: %v", err)
	}
	multiIDs := ids(multi)
	if !contains(multiIDs, factA) || !contains(multiIDs, factB) {
		t.Errorf("multiquery top-2 should surface both A and the expansion-matched B, got %v", multiIDs)
	}
}

// TestSearchSingleQueryNoBoosterUnchanged guards that the booster plumbing does
// not perturb the plain path: result equals store.Search top-k exactly.
func TestSearchSingleQueryNoBoosterUnchanged(t *testing.T) {
	m := newTestMemory(t, &fake.LLM{})
	ctx := context.Background()
	scope := Scope{UserID: "u"}
	seedFacts(t, m, scope, "alpha apple", "alpha orange", "beta banana", "gamma grape")

	qvec, err := m.embedOne(ctx, "alpha")
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	hits, err := m.store.Search(ctx, scope, qvec, 3)
	if err != nil {
		t.Fatalf("store search: %v", err)
	}
	want := make([]string, len(hits))
	for i, h := range hits {
		want[i] = h.ID
	}

	got, err := m.Search(ctx, "alpha", scope, 3)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if !equalSlice(ids(got), want) {
		t.Errorf("no-booster Search diverged from store top-k: got %v want %v", ids(got), want)
	}
}

func TestMergeQueries(t *testing.T) {
	got := mergeQueries("Where does Bob live?", []string{
		"Bob residence city",
		"where does bob live?", // case-dup of original, dropped
		"Bob home location",
		"Bob city",
	}, 3)
	want := []string{"Where does Bob live?", "Bob residence city", "Bob home location"}
	if !equalSlice(got, want) {
		t.Errorf("mergeQueries = %v, want %v", got, want)
	}
}

func TestSearchQueriesFallback(t *testing.T) {
	scope := Scope{UserID: "u"}
	cases := map[string]*fake.LLM{
		"llm error":  {Responder: func(string, string) (string, error) { return "", errors.New("down") }},
		"junk":       {Default: "not json at all"},
		"empty list": {Default: `{"queries":[]}`},
	}
	for name, llm := range cases {
		t.Run(name, func(t *testing.T) {
			m := newTestMemory(t, llm)
			m.multiQueryN = 3
			_ = scope
			qs := m.searchQueries(context.Background(), "original q")
			if len(qs) != 1 || qs[0] != "original q" {
				t.Errorf("%s: should fall back to the original query alone, got %v", name, qs)
			}
		})
	}
}

func TestParseQueries(t *testing.T) {
	cases := []struct {
		name string
		raw  string
		want []string
	}{
		{"plain envelope", `{"queries":["a","b"]}`, []string{"a", "b"}},
		{"fenced", "```json\n{\"queries\":[\"a\",\" b \"]}\n```", []string{"a", "b"}},
		{"prose-wrapped", `sure: {"queries":["a"]} hope that helps`, []string{"a"}},
		{"bare array", `["a","b"]`, []string{"a", "b"}},
		{"drops empties", `{"queries":["a","","  "]}`, []string{"a"}},
		{"garbage", `nothing here`, nil},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := parseQueries(c.raw)
			if !equalSlice(got, c.want) {
				t.Errorf("parseQueries(%q) = %v, want %v", c.raw, got, c.want)
			}
		})
	}
}

func equalSlice(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
