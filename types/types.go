// Package types holds the small value types shared across mneme's packages.
//
// It is a leaf package: it imports nothing from the rest of mneme. This breaks
// what would otherwise be an import cycle between the top-level mneme package
// (which defines the public API) and the store package (which persists these
// values). The top-level package re-exports the public ones as aliases, so a
// consumer still sees mneme.Fact, mneme.Scope and mneme.Message.
package types

import "time"

// Message is a single conversational turn fed into Add. The JSON tags let
// messages round-trip through eval fixtures and a future wire API.
type Message struct {
	Role    string `json:"role"`           // "user" | "assistant" | "system" (system is ignored for extraction)
	Content string `json:"content"`        // the message text
	Name    string `json:"name,omitempty"` // optional speaker name (multi-speaker conversations)

	// Timestamp is when the message was said. Optional: zero means "unknown",
	// and the pipeline falls back to its clock (ingestion time).
	//
	// Setting it is worth real accuracy. It grounds the extractor's OBSERVATION
	// DATE on when the conversation happened rather than on today, which is what
	// lets "I went yesterday" become a correct absolute date instead of one
	// anchored to the ingestion run. It also becomes the fact's ObservedAt, which
	// is what a question like "when did X happen" is ultimately answered from.
	Timestamp time.Time `json:"timestamp,omitempty"`
}

// Scope is the namespace a memory belongs to. Any subset of the fields may be
// set. Facts are isolated per-scope on both write and search: a fact written
// with UserID="a" is never returned to a search in UserID="b".
type Scope struct {
	UserID  string `json:"user_id,omitempty"`
	AgentID string `json:"agent_id,omitempty"`
	RunID   string `json:"run_id,omitempty"`
}

// Fact is a single durable statement extracted from a conversation. It is the
// unit a consumer reads back out of memory.
type Fact struct {
	ID    string
	Text  string  // the self-contained fact statement
	Hash  string  // md5(Text) — dedup key
	Score float32 // similarity score, only populated by Search

	// CreatedAt is when the fact was written to the store: ingestion time.
	CreatedAt time.Time

	// ObservedAt is when the fact was *said* — the timestamp of the source
	// messages it was extracted from, not when it was ingested. Zero when the
	// caller gave the messages no Timestamp.
	//
	// It is say-time, not event-time. For "I went to the group yesterday" said on
	// 8 May, ObservedAt is 8 May; the event itself was 7 May. Resolving the
	// utterance to the event's date is the extractor's job, and it lands in Text
	// (the pipeline grounds the extraction prompt on this same timestamp) — so do
	// not answer "when did X happen" from this field directly, or every relative
	// expression comes back skewed by its own offset.
	//
	// What it buys is the anchor. Without it a fact is a sentence floating free of
	// when it was said, and the only way to date anything in it is to hope the
	// extraction model resolved "yesterday" correctly at write time — which
	// flash-lite does not: it stamps the ingestion date into the fact text
	// instead (bench/RESULTS.md). Keeping the source timestamp on the record means
	// a consumer, a reranker, or an answer model can date the utterance without
	// trusting the extractor to have done it.
	ObservedAt time.Time

	Scope Scope
}

// Record is how a fact is stored. It is the same data as a Fact plus the
// embedding, which is never exported on the Fact returned to callers.
type Record struct {
	ID         string
	Text       string
	Hash       string
	Embedding  []float32
	Scope      Scope
	CreatedAt  time.Time // ingestion time
	ObservedAt time.Time // when the source conversation happened; zero if unknown
}

// Hit is a search result: a stored Record plus its similarity score.
type Hit struct {
	Record
	Score float32
}
