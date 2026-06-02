# mneme

A small, self-contained **agent memory** library for Go — drop it into any agent
to give it persistent, searchable long-term memory.

You feed it conversation messages; it uses an LLM to extract durable,
self-contained facts, embeds and deduplicates them, and stores them in a
per-scope vector index. Later you search for the facts relevant to a query.

- **Library-first.** Import `github.com/AccursedGalaxy/mneme` directly. A thin
  HTTP + MCP server binary wrapping the same core lands later (see `PLAN.md`).
- **Self-contained.** No shared code with any consumer. Its own OpenAI-compatible
  LLM/embedding client, its own storage.
- **Single binary.** Default storage is pure-Go SQLite (no cgo). Pluggable for
  sqlite-vec / pgvector as you scale, behind the `Store` interface.
- **Our own prompts.** The extraction prompt is ours, scored by a built-in eval
  harness — not a vendored copy.

## Install

```sh
go get github.com/AccursedGalaxy/mneme
```

## Quick start

```go
package main

import (
	"context"
	"fmt"

	"github.com/AccursedGalaxy/mneme"
)

func main() {
	m, err := mneme.New() // builds LLM, embedder and store from MNEME_* env
	if err != nil {
		panic(err)
	}
	defer m.Close()

	ctx := context.Background()
	scope := mneme.Scope{UserID: "alice"}

	// Ingest a conversation — durable facts are extracted and stored.
	m.Add(ctx, []mneme.Message{
		{Role: "user", Content: "I moved to Berlin and adopted a tabby cat named Pixel."},
	}, scope)

	// Retrieve by meaning.
	hits, _ := m.Search(ctx, "what pets does the user have?", scope, 3)
	for _, h := range hits {
		fmt.Printf("(%.3f) %s\n", h.Score, h.Text)
	}
}
```

A runnable version is in [`examples/basic`](./examples/basic).

## Configuration (env)

`mneme.New()` builds providers from the environment (override any of them with
`WithLLM` / `WithEmbedder` / `WithStore`). The client speaks the
OpenAI-compatible wire format, so it works with OpenAI, OpenRouter, Ollama,
vLLM, LM Studio, etc.

| Env var | Meaning | Example |
|---|---|---|
| `MNEME_LLM_BASE_URL` | OpenAI-compatible base URL | `https://api.openai.com/v1` |
| `MNEME_LLM_API_KEY` | API key (may be empty for local) | `sk-…` |
| `MNEME_LLM_MODEL` | extraction model | `gpt-4o-mini` |
| `MNEME_EMBED_BASE_URL` | embeddings base URL (defaults to LLM base) | — |
| `MNEME_EMBED_API_KEY` | embeddings key (defaults to LLM key) | — |
| `MNEME_EMBED_MODEL` | embedding model | `text-embedding-3-small` |
| `MNEME_DB_PATH` | SQLite file path | `./mneme.db` |

## Public API

```go
type Memory interface {
	Add(ctx context.Context, msgs []Message, scope Scope) ([]Fact, error)
	Search(ctx context.Context, query string, scope Scope, k int) ([]Fact, error)
	Get(ctx context.Context, id string) (Fact, error)
	Delete(ctx context.Context, id string) error
	Close() error
}
```

Facts are isolated per `Scope` (`UserID` / `AgentID` / `RunID`) on both write and
search. By default the pipeline is **additive**: facts accumulate and are deduped
by hash and by the extractor's awareness of existing memories — one LLM call per
`Add`, no update/delete pass. Pass `WithStrategy(Consolidate)` to add a second
LLM call that reconciles new facts against existing ones (ADD/UPDATE/DELETE/NONE)
so changed facts update in place — see below.

### What the default (additive) strategy means for callers (read this)

By default mneme **never updates or deletes a fact on its own.** This has
consequences you must design around:

- **Garbage in, garbage forever.** A wrong fact you `Add` is recalled until you
  `Delete` it by id. Be deliberate about what you feed `Add` — don't blindly
  store unverified model output.
- **Stale facts coexist with new ones.** If the world changes ("works at Shopify"
  → "works at Stripe"), both facts can live in the store and both can surface in
  `Search`. There is no notion of "the current value." If you need that, dedupe
  or expire at the application layer, or `Delete` superseded facts yourself.
- **The embedding model is part of the store's identity.** All vectors in one
  store must come from the same embedding model — query and stored vectors share
  a space. The store records its embedder's model name and dimension on the first
  insert, and `New` returns an `*EmbedderMismatchError` if you later point a
  different embedder at it — so an accidental swap fails loudly instead of
  silently degrading search. Use a fresh store when you intentionally change
  embedders, or pass `mneme.AllowEmbedderMismatch()` to override the guard.

These are deliberate tradeoffs for the default strategy (simpler, one LLM call
per `Add`). When staleness matters, opt into consolidation:

```go
m, _ := mneme.New(mneme.WithStrategy(mneme.Consolidate))
```

Consolidate runs a second LLM call per `Add` that reconciles the new facts
against the existing ones — UPDATE a changed value in place, DELETE a
contradicted one, ADD genuinely new facts, leave the rest untouched. The facts it
reconciles against are retrieved _per extracted candidate_, not by the
conversation, so a new fact ("moved to Berlin") surfaces the stale one it
overturns ("lives in Munich") even when that fact is unlike the rest of the turn.
`WithConsolidationTopK` sets how wide that reconciliation window is (default 30) —
larger widens recall on stale facts at the cost of a longer prompt. It costs the
extra LLM call (only when there are facts in scope to reconcile against) plus a
cheap per-candidate retrieval, and a malformed/failed consolidation response
safely falls back to an additive insert. See `PLAN-v2.md` §4.2.

## Eval harness

The extraction prompt is versioned (`extractionPromptV1`, …) and scored by the
harness in [`eval/`](./eval), so prompt changes are decisions backed by numbers.

```sh
go run ./cmd/eval                    # offline-safe fake embedder, real LLM
go run ./cmd/eval -embedder openai   # real embeddings, true semantic search@k
```

Current baseline for `extractionPromptV1` (gpt-4o-mini, text-embedding-3-small,
k=3) — see [`eval/RESULTS.md`](./eval/RESULTS.md):

| recall | precision | specificity | search@k | dedup | **aggregate** |
|---|---|---|---|---|---|
| 0.94 | 0.97 | 1.00 | 0.94 | 1.00 | **0.97** |

Unit tests run fully offline with deterministic fakes — no network:

```sh
go test ./...
```

## License

MIT — see [`LICENSE`](./LICENSE).
