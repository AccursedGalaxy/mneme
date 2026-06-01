package mneme

import (
	"context"
	"strings"
	"testing"

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
