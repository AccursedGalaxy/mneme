package mneme

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/AccursedGalaxy/mneme/provider/fake"
)

// consolidateMemory wires the offline test memory in Consolidate mode.
func consolidateMemory(t *testing.T, llm *fake.LLM) *memory {
	m := newTestMemory(t, llm)
	m.strategy = Consolidate
	return m
}

// storeCount returns how many facts live in scope, by asking the store for up to
// 1000 hits (cosine returns every row in scope, ranked).
func storeCount(t *testing.T, m *memory, scope Scope) int {
	t.Helper()
	vecs, err := m.embedder.Embed(context.Background(), []string{"probe"})
	if err != nil {
		t.Fatal(err)
	}
	hits, err := m.store.Search(context.Background(), scope, vecs[0], 1000)
	if err != nil {
		t.Fatal(err)
	}
	return len(hits)
}

// isConsolidationCall reports whether an LLM call is the consolidation step (vs
// the extraction step) by a marker unique to the consolidation system prompt.
func isConsolidationCall(system string) bool {
	return strings.Contains(system, "maintain a person's long-term memory")
}

// newMsgs returns the NEW MESSAGES portion of an extraction prompt. Fake
// extractors must branch on this, not the whole prompt: existing memories are
// also in the prompt (for dedup), so keying off the whole thing would re-emit a
// prior fact on the next Add.
func newMsgs(user string) string {
	if i := strings.Index(user, "NEW MESSAGES"); i >= 0 {
		return user[i:]
	}
	return user
}

func TestConsolidateUpdatesChangedFact(t *testing.T) {
	// Seed "lives in Seattle", then a conversation that moves to Austin. The
	// changed fact must UPDATE in place (same id, no second fact piling up).
	llm := &fake.LLM{Responder: func(system, user string) (string, error) {
		if isConsolidationCall(system) {
			// CURRENT MEMORIES shows the Seattle fact as id 0; UPDATE it.
			return `{"memory":[{"id":"0","text":"Alice lives in Austin","event":"UPDATE"}]}`, nil
		}
		if strings.Contains(newMsgs(user), "Seattle") {
			return fake.JSON("Alice lives in Seattle"), nil
		}
		return fake.JSON("Alice lives in Austin"), nil
	}}
	m := consolidateMemory(t, llm)
	ctx := context.Background()
	scope := Scope{UserID: "alice"}

	first, err := m.Add(ctx, []Message{{Role: "user", Content: "I live in Seattle"}}, scope)
	if err != nil || len(first) != 1 {
		t.Fatalf("seed Add: facts=%d err=%v", len(first), err)
	}
	seededID := first[0].ID

	changed, err := m.Add(ctx, []Message{{Role: "user", Content: "I moved to Austin"}}, scope)
	if err != nil {
		t.Fatalf("update Add: %v", err)
	}
	if len(changed) != 1 {
		t.Fatalf("UPDATE should report 1 changed fact, got %d", len(changed))
	}
	if changed[0].ID != seededID {
		t.Errorf("UPDATE should reuse the existing id %q, got %q", seededID, changed[0].ID)
	}
	if n := storeCount(t, m, scope); n != 1 {
		t.Errorf("move should UPDATE in place, not pile up: scope has %d facts, want 1", n)
	}
	got, err := m.Get(ctx, seededID)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if !strings.Contains(got.Text, "Austin") {
		t.Errorf("stored fact not updated: %q", got.Text)
	}
	if got.Hash != hashText("Alice lives in Austin") {
		t.Errorf("hash not re-derived from new text: %q", got.Hash)
	}
}

func TestConsolidateUpdateOfDeletedTargetIsNotClaimed(t *testing.T) {
	// The consolidation window is built from a retrieval snapshot, the LLM emits
	// an UPDATE for a fact in it, but that fact is deleted (a concurrent Delete in
	// another goroutine) before the write lands. Store.Update no-ops on the now-
	// missing id, so the pipeline must NOT report a fact it never wrote — and the
	// store must stay empty rather than gain a phantom row.
	var m *memory
	var seededID string
	ctx := context.Background()
	scope := Scope{UserID: "alice"}
	llm := &fake.LLM{Responder: func(system, user string) (string, error) {
		if isConsolidationCall(system) {
			// Retrieval already captured id 0 -> seededID; now the target vanishes
			// before the UPDATE write. This is the TOCTOU the fix guards against.
			if err := m.store.Delete(ctx, seededID); err != nil {
				t.Fatalf("inject concurrent delete: %v", err)
			}
			return `{"memory":[{"id":"0","text":"Alice lives in Austin","event":"UPDATE"}]}`, nil
		}
		if strings.Contains(newMsgs(user), "Seattle") {
			return fake.JSON("Alice lives in Seattle"), nil
		}
		return fake.JSON("Alice lives in Austin"), nil
	}}
	m = consolidateMemory(t, llm)

	first, err := m.Add(ctx, []Message{{Role: "user", Content: "I live in Seattle"}}, scope)
	if err != nil || len(first) != 1 {
		t.Fatalf("seed Add: facts=%d err=%v", len(first), err)
	}
	seededID = first[0].ID

	out, err := m.Add(ctx, []Message{{Role: "user", Content: "I moved to Austin"}}, scope)
	if err != nil {
		t.Fatalf("update Add: %v", err)
	}
	if len(out) != 0 {
		t.Errorf("UPDATE that no-ops on a deleted target must not be claimed as written, got %+v", out)
	}
	if n := storeCount(t, m, scope); n != 0 {
		t.Errorf("no-op UPDATE must not resurrect the row: scope has %d facts, want 0", n)
	}
}

func TestConsolidateRetrievesByCandidateNotConversation(t *testing.T) {
	// extractionTopK = 0: the extractor is shown NO existing-memory context. The
	// old design reused that context for consolidation, so it could never
	// reconcile here. Candidate-keyed retrieval finds the fact the new fact
	// overturns on its own, so the stale fact still UPDATEs in place.
	llm := &fake.LLM{Responder: func(system, user string) (string, error) {
		if isConsolidationCall(system) {
			// The consolidation window holds the stale Munich fact as id 0.
			return `{"memory":[{"id":"0","text":"Alice lives in Berlin","event":"UPDATE"}]}`, nil
		}
		if strings.Contains(newMsgs(user), "Munich") {
			return fake.JSON("Alice lives in Munich"), nil
		}
		return fake.JSON("Alice lives in Berlin"), nil
	}}
	m := consolidateMemory(t, llm)
	m.extractionTopK = 0 // extractor gets no conversation-keyed context at all
	ctx := context.Background()
	scope := Scope{UserID: "alice"}

	first, err := m.Add(ctx, []Message{{Role: "user", Content: "I live in Munich"}}, scope)
	if err != nil || len(first) != 1 {
		t.Fatalf("seed Add: facts=%d err=%v", len(first), err)
	}
	seededID := first[0].ID

	changed, err := m.Add(ctx, []Message{{Role: "user", Content: "I moved to Berlin"}}, scope)
	if err != nil {
		t.Fatalf("update Add: %v", err)
	}
	if len(changed) != 1 || changed[0].ID != seededID {
		t.Fatalf("candidate-keyed retrieval should UPDATE the stale fact in place, got %+v", changed)
	}
	if n := storeCount(t, m, scope); n != 1 {
		t.Errorf("move should UPDATE in place, not pile up: scope has %d facts, want 1", n)
	}
}

func TestConsolidateAddsUnrelatedFact(t *testing.T) {
	llm := &fake.LLM{Responder: func(system, user string) (string, error) {
		if isConsolidationCall(system) {
			return `{"memory":[{"id":"new","text":"Alice adopted a dog named Max","event":"ADD"}]}`, nil
		}
		if strings.Contains(newMsgs(user), "Seattle") {
			return fake.JSON("Alice lives in Seattle"), nil
		}
		return fake.JSON("Alice adopted a dog named Max"), nil
	}}
	m := consolidateMemory(t, llm)
	ctx := context.Background()
	scope := Scope{UserID: "alice"}

	first, _ := m.Add(ctx, []Message{{Role: "user", Content: "I live in Seattle"}}, scope)
	added, err := m.Add(ctx, []Message{{Role: "user", Content: "I adopted a dog named Max"}}, scope)
	if err != nil {
		t.Fatalf("add Add: %v", err)
	}
	if len(added) != 1 {
		t.Fatalf("ADD should report 1 new fact, got %d", len(added))
	}
	if added[0].ID == first[0].ID {
		t.Errorf("ADD must mint a new id, not reuse %q", first[0].ID)
	}
	if n := storeCount(t, m, scope); n != 2 {
		t.Errorf("unrelated fact should ADD: scope has %d facts, want 2", n)
	}
}

func TestConsolidateRestatedFactIsNoOp(t *testing.T) {
	llm := &fake.LLM{Responder: func(system, user string) (string, error) {
		if isConsolidationCall(system) {
			return `{"memory":[{"id":"0","text":"Alice lives in Seattle","event":"NONE"}]}`, nil
		}
		return fake.JSON("Alice lives in Seattle"), nil
	}}
	m := consolidateMemory(t, llm)
	ctx := context.Background()
	scope := Scope{UserID: "alice"}

	m.Add(ctx, []Message{{Role: "user", Content: "I live in Seattle"}}, scope)
	again, err := m.Add(ctx, []Message{{Role: "user", Content: "I'm in Seattle"}}, scope)
	if err != nil {
		t.Fatalf("restate Add: %v", err)
	}
	if len(again) != 0 {
		t.Errorf("NONE should write nothing, got %d facts", len(again))
	}
	if n := storeCount(t, m, scope); n != 1 {
		t.Errorf("restated fact should not duplicate: scope has %d facts, want 1", n)
	}
}

func TestConsolidateDeletesObsoleteFact(t *testing.T) {
	llm := &fake.LLM{Responder: func(system, user string) (string, error) {
		if isConsolidationCall(system) {
			return `{"memory":[{"id":"0","event":"DELETE"}]}`, nil
		}
		if strings.Contains(newMsgs(user), "own") {
			return fake.JSON("Alice owns a cat"), nil
		}
		return fake.JSON("Alice no longer owns a cat"), nil
	}}
	m := consolidateMemory(t, llm)
	ctx := context.Background()
	scope := Scope{UserID: "alice"}

	first, _ := m.Add(ctx, []Message{{Role: "user", Content: "I own a cat"}}, scope)
	seededID := first[0].ID

	out, err := m.Add(ctx, []Message{{Role: "user", Content: "I gave my cat away"}}, scope)
	if err != nil {
		t.Fatalf("delete Add: %v", err)
	}
	if len(out) != 0 {
		t.Errorf("a pure DELETE writes no facts, got %d", len(out))
	}
	if n := storeCount(t, m, scope); n != 0 {
		t.Errorf("DELETE should remove the obsolete fact: scope has %d facts, want 0", n)
	}
	if _, err := m.Get(ctx, seededID); err == nil {
		t.Error("deleted fact should not be retrievable")
	}
}

func TestConsolidateDuplicateAddIsHashDeduped(t *testing.T) {
	// The prompt asks for NONE on already-known facts, but a sloppy model ADDs a
	// fact whose text exactly matches one already stored. The hash-dedup safety
	// net must drop it rather than create a redundant row.
	llm := &fake.LLM{Responder: func(system, user string) (string, error) {
		if isConsolidationCall(system) {
			return `{"memory":[{"id":"new","text":"Alice lives in Seattle","event":"ADD"}]}`, nil
		}
		return fake.JSON("Alice lives in Seattle"), nil
	}}
	m := consolidateMemory(t, llm)
	ctx := context.Background()
	scope := Scope{UserID: "alice"}

	m.Add(ctx, []Message{{Role: "user", Content: "I live in Seattle"}}, scope)
	out, err := m.Add(ctx, []Message{{Role: "user", Content: "I'm in Seattle"}}, scope)
	if err != nil {
		t.Fatalf("restate Add: %v", err)
	}
	if len(out) != 0 {
		t.Errorf("a duplicate ADD should write nothing, got %d facts", len(out))
	}
	if n := storeCount(t, m, scope); n != 1 {
		t.Errorf("duplicate ADD should be deduped: scope has %d facts, want 1", n)
	}
}

func TestConsolidateMalformedResponseFallsBackToAdditive(t *testing.T) {
	// The consolidation call returns garbage. The pipeline must not corrupt the
	// store — it falls back to inserting the candidate additively.
	llm := &fake.LLM{Responder: func(system, user string) (string, error) {
		if isConsolidationCall(system) {
			return "I'm sorry, I can't do that.", nil
		}
		if strings.Contains(newMsgs(user), "Seattle") {
			return fake.JSON("Alice lives in Seattle"), nil
		}
		return fake.JSON("Alice adopted a dog named Max"), nil
	}}
	m := consolidateMemory(t, llm)
	ctx := context.Background()
	scope := Scope{UserID: "alice"}

	m.Add(ctx, []Message{{Role: "user", Content: "I live in Seattle"}}, scope)
	out, err := m.Add(ctx, []Message{{Role: "user", Content: "I adopted a dog named Max"}}, scope)
	if err != nil {
		t.Fatalf("fallback Add should not error: %v", err)
	}
	if len(out) != 1 || !strings.Contains(out[0].Text, "Max") {
		t.Errorf("malformed consolidation should fall back to additive insert, got %+v", out)
	}
	if n := storeCount(t, m, scope); n != 2 {
		t.Errorf("fallback should have inserted the candidate: scope has %d facts, want 2", n)
	}
}

func TestConsolidateCallErrorFallsBackToAdditive(t *testing.T) {
	// A transport error on the consolidation call (not just bad content) must
	// degrade to additive insert, never failing the Add or corrupting the store.
	llm := &fake.LLM{Responder: func(system, user string) (string, error) {
		if isConsolidationCall(system) {
			return "", errors.New("transient: empty response body")
		}
		if strings.Contains(newMsgs(user), "Seattle") {
			return fake.JSON("Alice lives in Seattle"), nil
		}
		return fake.JSON("Alice adopted a dog named Max"), nil
	}}
	m := consolidateMemory(t, llm)
	ctx := context.Background()
	scope := Scope{UserID: "alice"}

	m.Add(ctx, []Message{{Role: "user", Content: "I live in Seattle"}}, scope)
	out, err := m.Add(ctx, []Message{{Role: "user", Content: "I adopted a dog named Max"}}, scope)
	if err != nil {
		t.Fatalf("consolidation call error should not fail Add: %v", err)
	}
	if len(out) != 1 || !strings.Contains(out[0].Text, "Max") {
		t.Errorf("should fall back to additive insert of the candidate, got %+v", out)
	}
	if n := storeCount(t, m, scope); n != 2 {
		t.Errorf("fallback should have inserted the candidate: scope has %d facts, want 2", n)
	}
}

func TestConsolidateLLMCallCounts(t *testing.T) {
	// First Add (empty scope) skips consolidation -> 1 call. Second Add (a fact
	// exists) runs extraction + consolidation -> 2 calls.
	llm := &fake.LLM{Responder: func(system, user string) (string, error) {
		if isConsolidationCall(system) {
			return `{"memory":[]}`, nil
		}
		return fake.JSON("Alice lives in Seattle"), nil
	}}
	m := consolidateMemory(t, llm)
	ctx := context.Background()
	scope := Scope{UserID: "alice"}

	m.Add(ctx, []Message{{Role: "user", Content: "I live in Seattle"}}, scope)
	if llm.CallCount() != 1 {
		t.Fatalf("first Add (empty scope) should make 1 LLM call, made %d", llm.CallCount())
	}
	m.Add(ctx, []Message{{Role: "user", Content: "Still in Seattle"}}, scope)
	if llm.CallCount() != 3 { // 1 (first) + 2 (extraction + consolidation)
		t.Errorf("second Add should add 2 calls (extraction + consolidation), total now %d, want 3", llm.CallCount())
	}
}

func TestParseConsolidation(t *testing.T) {
	// Well-formed envelope.
	ops := parseConsolidation(`{"memory":[{"id":"0","text":"x","event":"UPDATE"},{"id":"new","text":"y","event":"add"}]}`)
	if len(ops) != 2 {
		t.Fatalf("want 2 ops, got %d", len(ops))
	}
	if ops[1].Event != "ADD" {
		t.Errorf("event should be upper-cased: %q", ops[1].Event)
	}

	// Fenced + prose-wrapped, with an invalid event and an empty-text ADD that
	// must be dropped; a DELETE with no text must be kept.
	raw := "Here you go:\n```json\n" +
		`{"memory":[{"id":"0","event":"DELETE"},{"id":"1","text":"","event":"ADD"},{"id":"2","text":"z","event":"BOGUS"}]}` +
		"\n```"
	ops = parseConsolidation(raw)
	if len(ops) != 1 || ops[0].Event != "DELETE" {
		t.Fatalf("expected only the DELETE to survive, got %+v", ops)
	}

	// Garbage -> nil (pipeline falls back to additive).
	if got := parseConsolidation("no json here"); got != nil {
		t.Errorf("garbage should parse to nil, got %+v", got)
	}
}

func TestConsolidateCapsMutationsAtCandidateCount(t *testing.T) {
	// Blast-radius containment: conversation text is untrusted and reaches the
	// consolidation LLM verbatim, so a hostile/confused response that DELETEs or
	// UPDATEs the whole reconciliation window must be capped at one mutation per
	// extracted candidate — a single Add can never mass-wipe stored memories.
	llm := &fake.LLM{Responder: func(system, user string) (string, error) {
		if isConsolidationCall(system) {
			// One candidate, but the model tries to delete all three memories.
			return `{"memory":[
				{"id":"0","event":"DELETE"},
				{"id":"1","event":"DELETE"},
				{"id":"2","event":"DELETE"},
				{"id":"new","text":"Alice moved to Austin","event":"ADD"}]}`, nil
		}
		msgs := newMsgs(user)
		switch {
		case strings.Contains(msgs, "seed"):
			return `{"memory":[
				{"id":"0","text":"Alice lives in Seattle","attributed_to":"user"},
				{"id":"1","text":"Alice works at Shopify","attributed_to":"user"},
				{"id":"2","text":"Alice has a cat named Pixel","attributed_to":"user"}]}`, nil
		default:
			return fake.JSON("Alice moved to Austin"), nil
		}
	}}
	m := consolidateMemory(t, llm)
	ctx := context.Background()
	scope := Scope{UserID: "alice"}

	if _, err := m.Add(ctx, []Message{{Role: "user", Content: "seed facts"}}, scope); err != nil {
		t.Fatalf("seed Add: %v", err)
	}
	if n := storeCount(t, m, scope); n != 3 {
		t.Fatalf("seed should store 3 facts, got %d", n)
	}

	if _, err := m.Add(ctx, []Message{{Role: "user", Content: "I moved to Austin"}}, scope); err != nil {
		t.Fatalf("hostile Add: %v", err)
	}
	// 1 candidate => at most 1 mutation applied: 3 seeded - 1 delete + 1 add = 3.
	if n := storeCount(t, m, scope); n != 3 {
		t.Errorf("mutations must be capped at the candidate count: scope has %d facts, want 3", n)
	}
}
