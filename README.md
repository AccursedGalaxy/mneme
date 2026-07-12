# mneme

[![CI](https://github.com/AccursedGalaxy/mneme/actions/workflows/ci.yml/badge.svg)](https://github.com/AccursedGalaxy/mneme/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/AccursedGalaxy/mneme?sort=semver)](https://github.com/AccursedGalaxy/mneme/releases)
[![Go Reference](https://pkg.go.dev/badge/github.com/AccursedGalaxy/mneme.svg)](https://pkg.go.dev/github.com/AccursedGalaxy/mneme)

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
| `MNEME_LLM_BASE_URL` | OpenAI-compatible base URL | `https://openrouter.ai/api/v1` |
| `MNEME_LLM_API_KEY` | API key (may be empty for local) | `sk-…` |
| `MNEME_LLM_MODEL` | extraction model | `google/gemini-2.5-flash` |
| `MNEME_EMBED_BASE_URL` | embeddings base URL (defaults to LLM base) | — |
| `MNEME_EMBED_API_KEY` | embeddings key (defaults to LLM key) | — |
| `MNEME_EMBED_MODEL` | embedding model | `text-embedding-3-small` |
| `MNEME_DB_PATH` | SQLite file path | `./mneme.db` |

**Recommended extraction model: `google/gemini-2.5-flash`.** Memory quality is
dominated by how well the extraction model captures facts, not by the embedder or
retrieval tricks. In our LoCoMo A/B (answer and judge models pinned, n=1986
questions), switching extraction from `gemini-2.5-flash-lite` to
`gemini-2.5-flash` lifted the end-to-end judge score 0.23 → 0.30, with temporal
questions 0.22 → 0.45; a stronger embedder, consolidation, and reranking each
measured flat on the same harness. Treat these as relative numbers from our
harness, not as comparable to published LoCoMo results — earlier harness
versions scored the adversarial category against its trap answer instead of
abstention, so the absolute scores will shift when the matrix is re-run under
the corrected scoring. The `gpt-5-mini` and `deepseek-v3.2` bench runs failed
before completing, so there is no end-to-end evidence about them either way;
the small fixture eval suggests `gpt-5-mini` extracts more facts at higher
cost, and nothing more. `text-embedding-3-small` is fine — a larger embedder
did not help. See [`bench/RESULTS.md`](bench/RESULTS.md) for the full matrix.

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
  Stores written before this guard existed have no recorded identity, so the
  first insert after upgrading pins whatever embedder is configured then —
  point the same embedder you originally used at a legacy store, or the guard
  will lock in the wrong one.

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
safely falls back to an additive insert. Conversation text is untrusted input
that reaches this LLM call, so the write path caps the blast radius in code: at
most one UPDATE/DELETE of an existing memory is applied per extracted candidate,
which keeps a hostile or confused turn from rewriting or wiping the whole
reconciliation window in a single `Add`. See `PLAN-v2.md` §4.2.

### Retrieval boosters: rerank + multi-query

Brute-force cosine over a single query phrasing leaves recall on the table: the
fact that answers a question often exists but sits just past the top-k cutoff
(retrieval dilution). Two opt-in `Search` boosters recover it, both leaving the
public `Search` signature unchanged:

```go
m, _ := mneme.New(
    mneme.WithReranker(&openai.LLMReranker{LLM: llm}), // over-retrieve, reorder, keep top-k
    mneme.WithMultiQuery(3),                            // expand into 3 phrasings, union the hits
)
```

- **Rerank** (`WithReranker`): `Search` retrieves a wider pool (`DefaultRerankPoolN`,
  ~20) instead of just `k`, hands it to the `Reranker` to reorder by relevance,
  then keeps the leading `k`. The shipped `openai.LLMReranker` scores each
  candidate with one LLM call and is best-effort — a malformed response or a
  failed LLM call leaves the cosine order untouched, so enabling it never makes
  `Search` depend on the rerank model being up. It reorders only; `Fact.Score`
  keeps the retrieval similarity. The `Reranker` interface is tiny, so a
  cross-encoder or hosted rerank API can drop in later.
- **Multi-query** (`WithMultiQuery(n)`): one LLM call rewrites the query into up
  to `n` diverse phrasings; their hits are unioned (deduped by id, best score
  wins) before reranking. Targets multi-hop questions where one phrasing won't
  surface every needed fact. Expansion is best-effort: if the call or its parse
  fails, `Search` falls back to the original query alone.

Each booster adds an LLM call per `Search` — opt in when retrieval accuracy is
worth the cost. The `bench` harness exposes both as `-rerank` and `-multiquery n`.
See `PLAN-v2.md` §4.3.

## Eval harness

The extraction prompt is versioned (`extractionPromptV1`, …) and scored by the
harness in [`eval/`](./eval), so prompt changes are decisions backed by numbers.

```sh
go run ./cmd/eval                       # offline-safe fake embedder, real LLM
go run ./cmd/eval -embedder openai      # real embeddings, true semantic search@k
go run ./cmd/eval -min-aggregate 0.95   # fail (exit 1) if any version regresses
```

Current baseline for `extractionPromptV1` (gpt-4o-mini, text-embedding-3-small,
k=3) — see [`eval/RESULTS.md`](./eval/RESULTS.md):

| recall | precision | specificity | search@k | dedup | **aggregate** |
|---|---|---|---|---|---|
| 0.94 | 0.97 | 1.00 | 0.94 | 1.00 | **0.97** |

The invariant — *every later prompt version must not regress the aggregate* — is
enforced by the `-min-aggregate` gate, run by the
[`Eval regression gate`](./.github/workflows/eval.yml) workflow (manual + weekly,
needs the `OPENROUTER_API_KEY` repo secret). The workflow pins the judge to a
fixed model (`openai/gpt-4o-mini`) so the scoring oracle stays constant when the
extraction default changes. The gate is a regression *floor* (default `0.95`)
rather than the exact baseline: with 18 fixtures, one fixture swinging moves the
aggregate by up to ~0.05, so the floor is set to tolerate a single noisy fixture
while still tripping on a regression that touches several. Note the gate runs on
demand and weekly, not on PRs — a prompt regression can merge and is caught by
the next scheduled run.

## Continuous integration

[`CI`](./.github/workflows/ci.yml) runs `gofmt`, `go vet`, `go build` and the
full race-enabled test suite on every push and PR — all offline, no secrets.
Mirror it locally with `make ci` (and `make eval` for the live prompt gate).

Unit tests run fully offline with deterministic fakes — no network:

```sh
go test ./...
```

## Releases

Versioned with [SemVer](https://semver.org); the git tag *is* the version
(`go get github.com/AccursedGalaxy/mneme@vX.Y.Z`). See [`CHANGELOG.md`](./CHANGELOG.md)
for what changed and [`RELEASING.md`](./RELEASING.md) for how a release is cut —
pushing a `v*` tag auto-publishes a GitHub release via
[`release.yml`](./.github/workflows/release.yml).

## License

MIT — see [`LICENSE`](./LICENSE).
