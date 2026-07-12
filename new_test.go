package mneme

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/AccursedGalaxy/mneme/provider/fake"
	"github.com/AccursedGalaxy/mneme/store/sqlite"
)

func TestNewWithOptions(t *testing.T) {
	st, err := sqlite.Open(t.TempDir() + "/n.db")
	if err != nil {
		t.Fatal(err)
	}
	m, err := New(
		WithStore(st),
		WithLLM(&fake.LLM{Responses: []string{fake.JSON("a fact about coffee")}}),
		WithEmbedder(&fake.Embedder{D: 64}),
		WithExtractionTopK(5),
		WithPromptVersion("v1"),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer m.Close()

	mm := m.(*memory)
	if mm.extractionTopK != 5 {
		t.Errorf("WithExtractionTopK not applied: %d", mm.extractionTopK)
	}
	if mm.promptVersion != "v1" {
		t.Errorf("WithPromptVersion not applied: %q", mm.promptVersion)
	}

	facts, err := m.Add(context.Background(), []Message{{Role: "user", Content: "I drink coffee"}}, Scope{UserID: "x"})
	if err != nil || len(facts) != 1 {
		t.Fatalf("Add via New: facts=%d err=%v", len(facts), err)
	}
}

// stubReranker is a do-nothing Reranker for wiring assertions.
type stubReranker struct{}

func (stubReranker) Rerank(_ context.Context, _ string, c []Fact) ([]Fact, error) { return c, nil }

func TestOptionConstructorsApply(t *testing.T) {
	// Every public Option must actually set its field: elsewhere the tests
	// configure the internals directly, so a no-op constructor would otherwise
	// pass the whole suite while being broken for real users.
	r := stubReranker{}
	m, err := New(
		WithStore(mustStore(t)),
		WithLLM(&fake.LLM{}),
		WithEmbedder(&fake.Embedder{D: 64}),
		WithStrategy(Consolidate),
		WithConsolidationTopK(7),
		WithConsolidationVersion("v2"),
		WithReranker(r),
		WithMultiQuery(3),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer m.Close()
	mm := m.(*memory)
	if mm.strategy != Consolidate {
		t.Error("WithStrategy not applied")
	}
	if mm.consolidationTopK != 7 {
		t.Errorf("WithConsolidationTopK not applied: %d", mm.consolidationTopK)
	}
	if mm.consolidationVersion != "v2" {
		t.Errorf("WithConsolidationVersion not applied: %q", mm.consolidationVersion)
	}
	if mm.reranker != r {
		t.Error("WithReranker not applied")
	}
	if mm.multiQueryN != 3 {
		t.Errorf("WithMultiQuery not applied: %d", mm.multiQueryN)
	}
}

func TestNewMissingLLMEnvErrors(t *testing.T) {
	// No LLM option and no MNEME_LLM_* env -> clear error.
	t.Setenv(envLLMBaseURL, "")
	t.Setenv(envLLMModel, "")
	_, err := New(WithEmbedder(&fake.Embedder{}), WithStore(mustStore(t)))
	if err == nil || !strings.Contains(err.Error(), "LLM") {
		t.Errorf("expected LLM env error, got %v", err)
	}
}

func TestNewBuildsProvidersFromEnv(t *testing.T) {
	t.Setenv(envLLMBaseURL, "http://localhost:1234/v1")
	t.Setenv(envLLMModel, "test-model")
	t.Setenv(envLLMAPIKey, "sk-x")
	t.Setenv(envEmbedModel, "test-embed")
	t.Setenv(envDBPath, t.TempDir()+"/env.db")

	m, err := New()
	if err != nil {
		t.Fatalf("New from env: %v", err)
	}
	defer m.Close()
	// Providers built but unused (no network call here) — just assert wiring.
	mm := m.(*memory)
	if mm.llm == nil || mm.embedder == nil || mm.store == nil {
		t.Error("New did not populate all providers from env")
	}
}

func mustStore(t *testing.T) *sqlite.Store {
	t.Helper()
	s, err := sqlite.Open(t.TempDir() + "/m.db")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

// emptyVecEmbedder returns a zero-length vector for every input — the failure
// mode of a broken embedding backend that unit fakes cannot produce.
type emptyVecEmbedder struct{}

func (emptyVecEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range out {
		out[i] = []float32{}
	}
	return out, nil
}
func (emptyVecEmbedder) Dim() int { return 0 }

func TestAddRejectsEmptyEmbeddings(t *testing.T) {
	// An empty vector stored would score 0 against every query — a fact that
	// silently never surfaces. Add must fail loudly instead of persisting it.
	m := &memory{
		store:         mustStore(t),
		llm:           &fake.LLM{Default: fake.JSON("a fact")},
		embedder:      emptyVecEmbedder{},
		promptVersion: DefaultPromptVersion,
		clock:         time.Now,
	}
	_, err := m.Add(context.Background(), []Message{{Role: "user", Content: "hi"}}, Scope{UserID: "x"})
	if err == nil || !strings.Contains(err.Error(), "empty vector") {
		t.Errorf("Add must reject empty embeddings, got err=%v", err)
	}
}
