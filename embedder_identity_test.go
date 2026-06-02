package mneme

import (
	"context"
	"errors"
	"testing"

	"github.com/AccursedGalaxy/mneme/provider/fake"
	"github.com/AccursedGalaxy/mneme/store/sqlite"
)

// namedEmbedder wraps the deterministic fake with a model name (satisfying
// provider.Named) and an optional reportZeroDim, which makes Dim() return 0
// regardless of the real dimension — simulating clients like the OpenAI one
// whose Dim() is unknown until the first Embed call.
type namedEmbedder struct {
	fake.Embedder
	name          string
	reportZeroDim bool
}

func (e *namedEmbedder) Name() string { return e.name }

func (e *namedEmbedder) Dim() int {
	if e.reportZeroDim {
		return 0
	}
	return e.Embedder.Dim()
}

func factLLM() *fake.LLM {
	return &fake.LLM{Responses: []string{fake.JSON("a fact about coffee")}}
}

// llmReturning builds an LLM that extracts the given fact text, so a second
// session can write a fact that is not hash-deduped against the first.
func llmReturning(fact string) *fake.LLM {
	return &fake.LLM{Responses: []string{fake.JSON(fact)}}
}

func addOne(t *testing.T, m Memory) {
	t.Helper()
	facts, err := m.Add(context.Background(),
		[]Message{{Role: "user", Content: "I drink coffee"}}, Scope{UserID: "u"})
	if err != nil {
		t.Fatalf("Add: %v", err)
	}
	if len(facts) != 1 {
		t.Fatalf("Add wrote %d facts, want 1", len(facts))
	}
}

// The first insert pins the store's embedder identity (model name + dimension).
func TestAddRecordsEmbedderIdentity(t *testing.T) {
	st := mustStore(t)
	m, err := New(WithStore(st), WithLLM(factLLM()),
		WithEmbedder(&namedEmbedder{Embedder: fake.Embedder{D: 64}, name: "model-a"}))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer m.Close()

	// Nothing recorded before the first write.
	if _, ok, err := st.EmbedderMeta(context.Background()); err != nil || ok {
		t.Fatalf("EmbedderMeta before insert: ok=%v err=%v, want ok=false", ok, err)
	}

	addOne(t, m)

	info, ok, err := st.EmbedderMeta(context.Background())
	if err != nil || !ok {
		t.Fatalf("EmbedderMeta after insert: ok=%v err=%v, want ok=true", ok, err)
	}
	if info.Model != "model-a" || info.Dim != 64 {
		t.Errorf("recorded identity = %+v, want {model-a 64}", info)
	}
}

// Reopening a populated store with a different embedder model is rejected.
func TestNewRejectsModelSwap(t *testing.T) {
	path := t.TempDir() + "/swap.db"

	st1, err := sqlite.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	m1, err := New(WithStore(st1), WithLLM(factLLM()),
		WithEmbedder(&namedEmbedder{Embedder: fake.Embedder{D: 64}, name: "model-a"}))
	if err != nil {
		t.Fatalf("New (session 1): %v", err)
	}
	addOne(t, m1)
	m1.Close()

	st2, err := sqlite.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { st2.Close() })
	_, err = New(WithStore(st2), WithLLM(factLLM()),
		WithEmbedder(&namedEmbedder{Embedder: fake.Embedder{D: 64}, name: "model-b"}))

	var mismatch *EmbedderMismatchError
	if !errors.As(err, &mismatch) {
		t.Fatalf("New (session 2) error = %v, want *EmbedderMismatchError", err)
	}
	if mismatch.Reason != "model name" {
		t.Errorf("mismatch reason = %q, want \"model name\"", mismatch.Reason)
	}
}

// A dimension change is caught at New when the embedder reports its dimension.
func TestNewRejectsDimensionSwap(t *testing.T) {
	path := t.TempDir() + "/dim.db"

	st1, err := sqlite.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	m1, err := New(WithStore(st1), WithLLM(factLLM()), WithEmbedder(&fake.Embedder{D: 64}))
	if err != nil {
		t.Fatalf("New (session 1): %v", err)
	}
	addOne(t, m1)
	m1.Close()

	st2, err := sqlite.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { st2.Close() })
	_, err = New(WithStore(st2), WithLLM(factLLM()), WithEmbedder(&fake.Embedder{D: 128}))

	var mismatch *EmbedderMismatchError
	if !errors.As(err, &mismatch) {
		t.Fatalf("New (session 2) error = %v, want *EmbedderMismatchError", err)
	}
	if mismatch.Reason != "dimension" {
		t.Errorf("mismatch reason = %q, want \"dimension\"", mismatch.Reason)
	}
}

// Reopening with the same embedder identity is fine.
func TestNewAcceptsSameEmbedder(t *testing.T) {
	path := t.TempDir() + "/same.db"

	st1, err := sqlite.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	m1, err := New(WithStore(st1), WithLLM(factLLM()),
		WithEmbedder(&namedEmbedder{Embedder: fake.Embedder{D: 64}, name: "model-a"}))
	if err != nil {
		t.Fatalf("New (session 1): %v", err)
	}
	addOne(t, m1)
	m1.Close()

	st2, err := sqlite.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	m2, err := New(WithStore(st2), WithLLM(factLLM()),
		WithEmbedder(&namedEmbedder{Embedder: fake.Embedder{D: 64}, name: "model-a"}))
	if err != nil {
		t.Fatalf("New (session 2) with matching embedder: %v", err)
	}
	m2.Close()
}

// AllowEmbedderMismatch turns the guard off for users who knowingly swap.
func TestAllowEmbedderMismatchOverrides(t *testing.T) {
	path := t.TempDir() + "/allow.db"

	st1, err := sqlite.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	m1, err := New(WithStore(st1), WithLLM(factLLM()),
		WithEmbedder(&namedEmbedder{Embedder: fake.Embedder{D: 64}, name: "model-a"}))
	if err != nil {
		t.Fatalf("New (session 1): %v", err)
	}
	addOne(t, m1)
	m1.Close()

	st2, err := sqlite.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	m2, err := New(WithStore(st2), WithLLM(llmReturning("a fact about tea")),
		WithEmbedder(&namedEmbedder{Embedder: fake.Embedder{D: 64}, name: "model-b"}),
		AllowEmbedderMismatch())
	if err != nil {
		t.Fatalf("New with AllowEmbedderMismatch: %v", err)
	}
	defer m2.Close()
	// And a subsequent Add must not error on the recorded mismatch either.
	addOne(t, m2)
}

// When the embedder cannot report its dimension at New (Dim()==0) and is
// unnamed, the New-time check has nothing to compare — the insert-time guard is
// the authoritative one and catches the changed dimension.
func TestAddRejectsDimensionSwapAtInsert(t *testing.T) {
	path := t.TempDir() + "/insert.db"

	st1, err := sqlite.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	m1, err := New(WithStore(st1), WithLLM(factLLM()),
		WithEmbedder(&namedEmbedder{Embedder: fake.Embedder{D: 64}, reportZeroDim: true}))
	if err != nil {
		t.Fatalf("New (session 1): %v", err)
	}
	addOne(t, m1)
	m1.Close()

	st2, err := sqlite.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { st2.Close() })
	// Dim()==0 and no name, so New cannot detect the swap and must succeed.
	m2, err := New(WithStore(st2), WithLLM(llmReturning("a fact about tea")),
		WithEmbedder(&namedEmbedder{Embedder: fake.Embedder{D: 128}, reportZeroDim: true}))
	if err != nil {
		t.Fatalf("New (session 2) should pass the best-effort check: %v", err)
	}
	defer m2.Close()

	// But the insert, which knows the real dimension, must reject it.
	_, err = m2.Add(context.Background(),
		[]Message{{Role: "user", Content: "I drink tea"}}, Scope{UserID: "u"})
	var mismatch *EmbedderMismatchError
	if !errors.As(err, &mismatch) {
		t.Fatalf("Add error = %v, want *EmbedderMismatchError", err)
	}
	if mismatch.Reason != "dimension" {
		t.Errorf("mismatch reason = %q, want \"dimension\"", mismatch.Reason)
	}
}
