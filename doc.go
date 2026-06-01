// Package mneme gives an AI agent persistent, searchable long-term memory.
//
// You feed it conversation messages with Add; it uses an LLM to extract durable,
// self-contained facts, embeds and deduplicates them, and stores them in a
// per-scope vector index. Later you call Search to retrieve the facts most
// relevant to a query within the same scope.
//
// mneme is library-first and self-contained: it carries its own
// OpenAI-compatible LLM/embedding client (provider/openai) and its own storage
// (store/sqlite, pure-Go, no cgo). Construct a Memory with New:
//
//	m, err := mneme.New() // builds providers + store from MNEME_* env vars
//	if err != nil { ... }
//	defer m.Close()
//
//	scope := mneme.Scope{UserID: "alice"}
//	m.Add(ctx, []mneme.Message{{Role: "user", Content: "I drive a Ferrari 488 GTB"}}, scope)
//	hits, _ := m.Search(ctx, "what car does the user own?", scope, 5)
//
// The pipeline is additive: facts accumulate and are deduped by hash and by the
// extractor's own awareness of existing memories; there is no update/delete LLM
// pass in v1. See PLAN.md for the full design.
package mneme
