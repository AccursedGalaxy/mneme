package mneme

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/AccursedGalaxy/mneme/provider/fake"
	"github.com/AccursedGalaxy/mneme/store/sqlite"
)

// The extractor must resolve relative dates against when the conversation
// happened, not against when it was ingested. Grounded on ingestion time,
// "I went yesterday" (said in May 2023, ingested today) extracts as a fact dated
// today — a confidently wrong date nothing downstream can detect or repair. This
// is the bug that put "on 2026-07-11" into a fact about a May 2023 event.
func TestExtractionGroundsOnConversationTimeNotIngestTime(t *testing.T) {
	llm := &fake.LLM{Default: `{"memory":[{"text":"Caroline attended a support group on 2023-05-07."}]}`}
	st, _ := sqlite.Open(":memory:")
	defer st.Close()

	m, err := New(WithLLM(llm), WithEmbedder(&fake.Embedder{D: 8}), WithStore(st))
	if err != nil {
		t.Fatal(err)
	}
	// Ingest "today" is 2026, but the conversation happened in 2023.
	m.(*memory).clock = func() time.Time { return time.Date(2026, 7, 13, 0, 0, 0, 0, time.UTC) }

	said := time.Date(2023, 5, 8, 13, 56, 0, 0, time.UTC)
	facts, err := m.Add(context.Background(), []Message{
		{Role: "user", Content: "I went to a support group yesterday.", Timestamp: said},
	}, Scope{UserID: "u"})
	if err != nil {
		t.Fatal(err)
	}

	if !strings.Contains(llm.Calls[0].User, "OBSERVATION DATE: 2023-05-08") {
		t.Errorf("the extractor must be grounded on the conversation's date, got prompt:\n%s", llm.Calls[0].User)
	}
	if len(facts) != 1 {
		t.Fatalf("want 1 fact, got %d", len(facts))
	}
	if !facts[0].ObservedAt.Equal(said) {
		t.Errorf("ObservedAt = %v, want the source message's timestamp %v", facts[0].ObservedAt, said)
	}
	if facts[0].CreatedAt.Year() != 2026 {
		t.Errorf("CreatedAt must stay ingestion time, got %v", facts[0].CreatedAt)
	}
}

// Messages with no Timestamp are the documented default, and must keep working:
// grounding falls back to the clock and ObservedAt stays zero ("unknown").
func TestUntimestampedMessagesFallBackToClock(t *testing.T) {
	llm := &fake.LLM{Default: `{"memory":[{"text":"Caroline likes tea."}]}`}
	st, _ := sqlite.Open(":memory:")
	defer st.Close()

	m, _ := New(WithLLM(llm), WithEmbedder(&fake.Embedder{D: 8}), WithStore(st))
	m.(*memory).clock = func() time.Time { return time.Date(2026, 7, 13, 0, 0, 0, 0, time.UTC) }

	facts, err := m.Add(context.Background(), []Message{{Role: "user", Content: "I like tea."}}, Scope{UserID: "u"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(llm.Calls[0].User, "OBSERVATION DATE: 2026-07-13") {
		t.Errorf("with no timestamps the extractor falls back to the clock, got:\n%s", llm.Calls[0].User)
	}
	if len(facts) != 1 || !facts[0].ObservedAt.IsZero() {
		t.Errorf("ObservedAt must stay zero when the caller gave no timestamp, got %v", facts[0].ObservedAt)
	}
}

// The consolidation UPDATE path dropped ObservedAt entirely: the pipeline set it
// on the record, Store.Update's SQL never wrote the column, and the fact read
// back undated. Every consolidated fact silently lost its temporal grounding.
func TestConsolidationUpdatePersistsObservedAt(t *testing.T) {
	llm := &fake.LLM{Responder: func(system, user string) (string, error) {
		if isConsolidationCall(system) {
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

	seeded := time.Date(2023, 1, 4, 9, 0, 0, 0, time.UTC)
	if _, err := m.Add(ctx, []Message{
		{Role: "user", Content: "I live in Seattle", Timestamp: seeded},
	}, scope); err != nil {
		t.Fatalf("seed Add: %v", err)
	}

	moved := time.Date(2023, 6, 1, 9, 0, 0, 0, time.UTC)
	changed, err := m.Add(ctx, []Message{
		{Role: "user", Content: "I moved to Austin", Timestamp: moved},
	}, scope)
	if err != nil || len(changed) != 1 {
		t.Fatalf("update Add: facts=%d err=%v", len(changed), err)
	}
	if !changed[0].ObservedAt.Equal(moved) {
		t.Errorf("the returned Fact must carry the updating conversation's time: got %v want %v",
			changed[0].ObservedAt, moved)
	}

	// And it must actually be on disk — the returned Fact agreeing with a store
	// that kept the old value is exactly the bug.
	hits, err := m.Search(ctx, "where does Alice live", scope, 5)
	if err != nil || len(hits) != 1 {
		t.Fatalf("search: hits=%d err=%v", len(hits), err)
	}
	if !hits[0].ObservedAt.Equal(moved) {
		t.Errorf("the stored fact must read back with the updating conversation's time: got %v want %v",
			hits[0].ObservedAt, moved)
	}
}

// An UPDATE from an untimestamped conversation must keep the date the fact
// already had. Naively writing the zero time would convert the silent-drop bug
// into a silent-erase bug.
func TestConsolidationUpdateWithoutTimestampKeepsPriorObservedAt(t *testing.T) {
	llm := &fake.LLM{Responder: func(system, user string) (string, error) {
		if isConsolidationCall(system) {
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

	seeded := time.Date(2023, 1, 4, 9, 0, 0, 0, time.UTC)
	if _, err := m.Add(ctx, []Message{
		{Role: "user", Content: "I live in Seattle", Timestamp: seeded},
	}, scope); err != nil {
		t.Fatalf("seed Add: %v", err)
	}

	// No Timestamp on the reconciling turn.
	changed, err := m.Add(ctx, []Message{{Role: "user", Content: "I moved to Austin"}}, scope)
	if err != nil || len(changed) != 1 {
		t.Fatalf("update Add: facts=%d err=%v", len(changed), err)
	}
	if !changed[0].ObservedAt.Equal(seeded) {
		t.Errorf("an undated update must keep the fact's prior date: got %v want %v",
			changed[0].ObservedAt, seeded)
	}

	hits, err := m.Search(ctx, "where does Alice live", scope, 5)
	if err != nil || len(hits) != 1 {
		t.Fatalf("search: hits=%d err=%v", len(hits), err)
	}
	if !hits[0].ObservedAt.Equal(seeded) {
		t.Errorf("the stored date must survive an undated update: got %v want %v",
			hits[0].ObservedAt, seeded)
	}
}
