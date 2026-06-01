package fake

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/AccursedGalaxy/mneme/store"
)

func TestEmbedderDeterministicAndDim(t *testing.T) {
	e := &Embedder{D: 32}
	if e.Dim() != 32 {
		t.Fatalf("Dim = %d, want 32", e.Dim())
	}
	a, _ := e.Embed(context.Background(), []string{"the user drives a ferrari"})
	b, _ := e.Embed(context.Background(), []string{"the user drives a ferrari"})
	if len(a[0]) != 32 {
		t.Fatalf("vector len = %d, want 32", len(a[0]))
	}
	for i := range a[0] {
		if a[0][i] != b[0][i] {
			t.Fatalf("embedding not deterministic at %d", i)
		}
	}
}

func TestEmbedderDefaultDim(t *testing.T) {
	e := &Embedder{}
	if e.Dim() != 64 {
		t.Fatalf("default Dim = %d, want 64", e.Dim())
	}
}

func TestEmbedderSemanticOrdering(t *testing.T) {
	// Overlapping-word texts should be closer than disjoint ones.
	e := &Embedder{D: 128}
	ctx := context.Background()
	vecs, _ := e.Embed(ctx, []string{
		"the user works at shopify as a senior engineer",
		"the user is employed at shopify",
		"the user owns a beagle named max",
	})
	related := store.Cosine(vecs[0], vecs[1])
	unrelated := store.Cosine(vecs[0], vecs[2])
	if related <= unrelated {
		t.Errorf("expected related > unrelated, got %v vs %v", related, unrelated)
	}
}

func TestLLMResponsesQueue(t *testing.T) {
	l := &LLM{Responses: []string{"first", "second"}, Default: "fallback"}
	ctx := context.Background()
	for _, want := range []string{"first", "second", "fallback", "fallback"} {
		got, err := l.Complete(ctx, "sys", "usr", true)
		if err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Errorf("got %q, want %q", got, want)
		}
	}
	if l.CallCount() != 4 {
		t.Errorf("CallCount = %d, want 4", l.CallCount())
	}
	if !l.Calls[0].JSONObject {
		t.Error("expected JSONObject recorded true")
	}
}

func TestLLMResponder(t *testing.T) {
	l := &LLM{Responder: func(system, user string) (string, error) {
		return "echo:" + user, nil
	}}
	got, _ := l.Complete(context.Background(), "s", "hello", false)
	if got != "echo:hello" {
		t.Errorf("got %q", got)
	}
}

func TestJSONHelper(t *testing.T) {
	s := JSON("Alice drives a Ferrari 488 GTB", "Alice adopted a beagle named Max")
	var parsed struct {
		Memory []struct {
			ID           string `json:"id"`
			Text         string `json:"text"`
			AttributedTo string `json:"attributed_to"`
		} `json:"memory"`
	}
	if err := json.Unmarshal([]byte(s), &parsed); err != nil {
		t.Fatalf("JSON() produced invalid JSON: %v\n%s", err, s)
	}
	if len(parsed.Memory) != 2 {
		t.Fatalf("want 2 memories, got %d", len(parsed.Memory))
	}
	if parsed.Memory[1].Text != "Alice adopted a beagle named Max" {
		t.Errorf("text mismatch: %q", parsed.Memory[1].Text)
	}
	if parsed.Memory[0].ID != "0" {
		t.Errorf("id mismatch: %q", parsed.Memory[0].ID)
	}
}
