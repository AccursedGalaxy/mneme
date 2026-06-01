package mneme

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/AccursedGalaxy/mneme/provider/fake"
	"github.com/AccursedGalaxy/mneme/store/sqlite"
	"github.com/AccursedGalaxy/mneme/types"
)

// newTestMemory wires the concrete pipeline with offline fakes and a temp DB.
func newTestMemory(t *testing.T, llm *fake.LLM) *memory {
	t.Helper()
	st, err := sqlite.Open(t.TempDir() + "/test.db")
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	t.Cleanup(func() { st.Close() })
	return &memory{
		store:          st,
		llm:            llm,
		embedder:       &fake.Embedder{D: 128},
		extractionTopK: DefaultExtractionTopK,
		promptVersion:  DefaultPromptVersion,
		clock:          func() time.Time { return time.Date(2026, 6, 1, 12, 0, 0, 0, time.UTC) },
	}
}

func TestAddMultiTopicYieldsMultipleFacts(t *testing.T) {
	llm := &fake.LLM{Responses: []string{fake.JSON(
		"Alice was promoted to Senior Engineer at Shopify",
		"Alice adopted a beagle named Max",
	)}}
	m := newTestMemory(t, llm)
	ctx := context.Background()

	facts, err := m.Add(ctx, []Message{{Role: "user", Content: "I got promoted to Senior Engineer at Shopify. Also adopted a beagle named Max."}}, Scope{UserID: "alice"})
	if err != nil {
		t.Fatalf("Add: %v", err)
	}
	if len(facts) != 2 {
		t.Fatalf("want 2 facts, got %d: %+v", len(facts), facts)
	}
	for _, f := range facts {
		if f.ID == "" || f.Hash == "" || f.CreatedAt.IsZero() {
			t.Errorf("fact not fully populated: %+v", f)
		}
		if f.Scope.UserID != "alice" {
			t.Errorf("scope not propagated: %+v", f.Scope)
		}
	}
}

func TestAddNothingToExtract(t *testing.T) {
	llm := &fake.LLM{Responses: []string{`{"memory":[]}`}}
	m := newTestMemory(t, llm)
	facts, err := m.Add(context.Background(), []Message{{Role: "user", Content: "hello there"}}, Scope{UserID: "x"})
	if err != nil {
		t.Fatalf("Add: %v", err)
	}
	if len(facts) != 0 {
		t.Errorf("want 0 facts, got %d", len(facts))
	}
}

func TestAddMalformedResponseIsClean(t *testing.T) {
	llm := &fake.LLM{Default: "I'm sorry, I cannot do that."}
	m := newTestMemory(t, llm)
	facts, err := m.Add(context.Background(), []Message{{Role: "user", Content: "tell me a fact"}}, Scope{UserID: "x"})
	if err != nil {
		t.Fatalf("malformed response should not error, got: %v", err)
	}
	if len(facts) != 0 {
		t.Errorf("malformed response should yield 0 facts, got %d", len(facts))
	}
}

func TestAddEmptyMessages(t *testing.T) {
	llm := &fake.LLM{}
	m := newTestMemory(t, llm)
	facts, err := m.Add(context.Background(), nil, Scope{UserID: "x"})
	if err != nil {
		t.Fatal(err)
	}
	if len(facts) != 0 {
		t.Errorf("empty messages should yield 0 facts")
	}
	if llm.CallCount() != 0 {
		t.Errorf("empty messages should not call the LLM, got %d calls", llm.CallCount())
	}
}

func TestAddInBatchDedup(t *testing.T) {
	// LLM emits the same fact twice in one batch -> only one stored.
	llm := &fake.LLM{Responses: []string{fake.JSON(
		"Alice drives a Ferrari 488 GTB",
		"Alice drives a Ferrari 488 GTB",
	)}}
	m := newTestMemory(t, llm)
	facts, err := m.Add(context.Background(), []Message{{Role: "user", Content: "I drive a Ferrari 488 GTB"}}, Scope{UserID: "alice"})
	if err != nil {
		t.Fatalf("Add: %v", err)
	}
	if len(facts) != 1 {
		t.Errorf("in-batch dup should collapse to 1, got %d", len(facts))
	}
}

func TestAddDedupAgainstExisting(t *testing.T) {
	// First Add stores a fact; second Add re-emits the same text -> deduped.
	llm := &fake.LLM{Responses: []string{
		fake.JSON("Alice works at Shopify"),
		fake.JSON("Alice works at Shopify"),
	}}
	m := newTestMemory(t, llm)
	ctx := context.Background()
	scope := Scope{UserID: "alice"}

	first, err := m.Add(ctx, []Message{{Role: "user", Content: "I work at Shopify"}}, scope)
	if err != nil || len(first) != 1 {
		t.Fatalf("first Add: facts=%d err=%v", len(first), err)
	}
	second, err := m.Add(ctx, []Message{{Role: "user", Content: "I work at Shopify"}}, scope)
	if err != nil {
		t.Fatalf("second Add: %v", err)
	}
	if len(second) != 0 {
		t.Errorf("re-emitting an existing fact should dedup to 0, got %d", len(second))
	}
}

func TestAddScopeIsolation(t *testing.T) {
	// Same fact text under two scopes is NOT deduped across scopes.
	llm := &fake.LLM{Responses: []string{
		fake.JSON("The user likes coffee"),
		fake.JSON("The user likes coffee"),
	}}
	m := newTestMemory(t, llm)
	ctx := context.Background()

	a, _ := m.Add(ctx, []Message{{Role: "user", Content: "I like coffee"}}, Scope{UserID: "alice"})
	b, _ := m.Add(ctx, []Message{{Role: "user", Content: "I like coffee"}}, Scope{UserID: "bob"})
	if len(a) != 1 || len(b) != 1 {
		t.Errorf("each scope should store its own copy: alice=%d bob=%d", len(a), len(b))
	}
}

func TestAddExistingMemoriesShownToExtractor(t *testing.T) {
	// Capture the user prompt on the second call and assert the first fact
	// appears as an integer-labelled existing memory.
	var secondPrompt string
	calls := 0
	llm := &fake.LLM{Responder: func(system, user string) (string, error) {
		calls++
		if calls == 1 {
			return fake.JSON("Alice works at Shopify"), nil
		}
		secondPrompt = user
		return `{"memory":[]}`, nil
	}}
	m := newTestMemory(t, llm)
	ctx := context.Background()
	scope := Scope{UserID: "alice"}

	m.Add(ctx, []Message{{Role: "user", Content: "I work at Shopify"}}, scope)
	m.Add(ctx, []Message{{Role: "user", Content: "Where do I work again?"}}, scope)

	if !strings.Contains(secondPrompt, "0: Alice works at Shopify") {
		t.Errorf("existing memory not shown to extractor with integer label:\n%s", secondPrompt)
	}
}

func TestSearchSurfacesRelevantFact(t *testing.T) {
	llm := &fake.LLM{Responses: []string{fake.JSON(
		"Alice works at Shopify as a senior engineer",
		"Alice adopted a beagle named Max",
	)}}
	m := newTestMemory(t, llm)
	ctx := context.Background()
	scope := Scope{UserID: "alice"}

	if _, err := m.Add(ctx, []Message{{Role: "user", Content: "I work at Shopify as a senior engineer and adopted a beagle named Max"}}, scope); err != nil {
		t.Fatalf("Add: %v", err)
	}

	hits, err := m.Search(ctx, "what is Alice's role working at Shopify?", scope, 1)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(hits) != 1 {
		t.Fatalf("want 1 hit, got %d", len(hits))
	}
	if !strings.Contains(hits[0].Text, "Shopify") {
		t.Errorf("top hit for work query should mention Shopify, got %q", hits[0].Text)
	}
	if hits[0].Score == 0 {
		t.Errorf("search hit should have a populated score")
	}

	petHits, _ := m.Search(ctx, "what beagle did Alice adopt?", scope, 1)
	if len(petHits) != 1 || !strings.Contains(petHits[0].Text, "beagle") {
		t.Errorf("pet query should surface the beagle fact, got %+v", petHits)
	}
}

func TestSearchEmptyQuery(t *testing.T) {
	m := newTestMemory(t, &fake.LLM{})
	hits, err := m.Search(context.Background(), "  ", Scope{UserID: "x"}, 5)
	if err != nil || hits != nil {
		t.Errorf("empty query should no-op: hits=%v err=%v", hits, err)
	}
}

func TestGetAndDelete(t *testing.T) {
	llm := &fake.LLM{Responses: []string{fake.JSON("Alice drives a Ferrari")}}
	m := newTestMemory(t, llm)
	ctx := context.Background()
	scope := Scope{UserID: "alice"}

	facts, _ := m.Add(ctx, []Message{{Role: "user", Content: "I drive a Ferrari"}}, scope)
	id := facts[0].ID

	got, err := m.Get(ctx, id)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got.Text != "Alice drives a Ferrari" {
		t.Errorf("Get text = %q", got.Text)
	}

	if err := m.Delete(ctx, id); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if _, err := m.Get(ctx, id); err == nil {
		t.Error("Get after Delete should fail")
	}
}

func TestRelabelExistingRoundTrips(t *testing.T) {
	hits := []types.Hit{
		{Record: types.Record{ID: "uuid-aaa", Text: "fact a"}},
		{Record: types.Record{ID: "uuid-bbb", Text: "fact b"}},
		{Record: types.Record{ID: "uuid-ccc", Text: "fact c"}},
	}
	labels, idMap := relabelExisting(hits)
	if len(labels) != 3 {
		t.Fatalf("want 3 labels, got %d", len(labels))
	}
	for i, h := range hits {
		intID := labels[i].ID
		if labels[i].Text != h.Text {
			t.Errorf("label %d text = %q, want %q", i, labels[i].Text, h.Text)
		}
		if idMap[intID] != h.ID {
			t.Errorf("idMap[%q] = %q, want %q", intID, idMap[intID], h.ID)
		}
	}
	if labels[0].ID != "0" || labels[2].ID != "2" {
		t.Errorf("integer labels should be sequential from 0: %v", labels)
	}
}

func TestAddSingleLLMCall(t *testing.T) {
	// Additive pipeline must use exactly one LLM call per Add.
	llm := &fake.LLM{Responses: []string{fake.JSON("a fact")}}
	m := newTestMemory(t, llm)
	m.Add(context.Background(), []Message{{Role: "user", Content: "here is a fact"}}, Scope{UserID: "x"})
	if llm.CallCount() != 1 {
		t.Errorf("Add should make exactly 1 LLM call, made %d", llm.CallCount())
	}
}
