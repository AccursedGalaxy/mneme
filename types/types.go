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
	ID        string
	Text      string  // the self-contained fact statement
	Hash      string  // md5(Text) — dedup key
	Score     float32 // similarity score, only populated by Search
	CreatedAt time.Time
	Scope     Scope
}

// Record is how a fact is stored. It is the same data as a Fact plus the
// embedding, which is never exported on the Fact returned to callers.
type Record struct {
	ID        string
	Text      string
	Hash      string
	Embedding []float32
	Scope     Scope
	CreatedAt time.Time
}

// Hit is a search result: a stored Record plus its similarity score.
type Hit struct {
	Record
	Score float32
}
