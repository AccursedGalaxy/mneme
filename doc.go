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
// By default the pipeline is additive: facts accumulate and are deduped by hash
// and by the extractor's own awareness of existing memories. Pass
// WithStrategy(Consolidate) to instead run a second LLM call that reconciles new
// facts against existing ones (ADD/UPDATE/DELETE/NONE), so a changed fact
// ("moved from Seattle to Austin") updates the stale one in place rather than
// piling up beside it — at the cost of one extra LLM call per Add (plus a cheap
// per-candidate retrieval to find the facts each new fact might overturn; see
// WithConsolidationTopK). See PLAN.md (v1 additive design) and PLAN-v2.md §4.2
// (consolidation) for the full design.
package mneme
