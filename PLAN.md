# mneme — v1 Build Plan

> Single source of truth for building mneme v1. Written so an agent with **no
> prior context** can build the whole thing end-to-end. Read it top to bottom
> once before writing code.

---

## 1. What mneme is

A **standalone Go library** that gives any AI agent persistent, searchable
long-term memory. You feed it conversation messages; it extracts durable facts,
deduplicates them, stores them, and lets the agent retrieve the relevant ones
later by semantic search.

It is **library-first**: the substance is a Go package you import. A thin
HTTP + MCP **server binary** wrapping the exact same core comes *after* the core
is proven (not in v1 — but the core must be designed so the server is a trivial
adapter, see §9).

**First consumer:** the `driver-os` project will depend on mneme. mneme itself
depends on nothing from driver-os — the dependency arrow points one way only.

### Non-goals for v1 (designed-for, not built)
- No HTTP/MCP server yet (core lib only).
- No temporal reasoning (no valid-from/valid-to, no "fact was true until March").
- No knowledge-graph layer (no entity/relationship graph).
- No Postgres/pgvector, no sqlite-vec (the `Store` interface allows them later).
- No memory-update/delete LLM pass (we use **additive** memory — see §4).

These are deliberately out of scope. Do not build them. Do leave the seams
(interfaces) that let them be added without a rewrite.

---

## 2. Decisions already made (do not relitigate)

| Decision | Choice | Rationale |
|---|---|---|
| Distribution | Core Go lib first; server/MCP later as thin wrapper | "Drop into any agent"; Go consumers import, others use the wire later |
| Memory strategy | **Approach A — additive (ADD-only)** | Cheaper (1 LLM call/add), simpler; matches current mem0 default |
| Temporal | None | Not needed for v1 |
| Code reuse | **None** with consumers | Avoid cross-project coupling; mneme is self-contained |
| Storage (v1) | Pure-Go SQLite, embeddings as BLOB, **cosine in Go** | Single static binary, **no cgo**; correctness over speed for v1 |
| Storage (later) | sqlite-vec (cgo) / pgvector — behind `Store` iface | Future-proof scale without rewriting the pipeline |
| LLM/embeddings | Own minimal **OpenAI-compatible** HTTP client | Works with OpenAI, Ollama, vLLM, LM Studio, etc.; no SDK dep |
| Prompts / IP | **Our own** extraction prompt, proven by an eval harness | We own the IP; mem0 is orientation only (see §5) |
| License | MIT | Maximally permissive; meant to be reused |
| Module path | `github.com/AccursedGalaxy/mneme` | — |

---

## 3. Public API (the surface a consumer sees)

Keep this small and stable. Everything else is internal.

```go
package mneme

// Memory is the top-level handle an agent holds.
type Memory interface {
    // Add ingests messages, extracts durable facts, dedups, and stores them.
    // Returns the facts that were newly written (after dedup).
    Add(ctx context.Context, msgs []Message, scope Scope) ([]Fact, error)

    // Search returns the top-k facts most relevant to query within scope.
    Search(ctx context.Context, query string, scope Scope, k int) ([]Fact, error)

    // Get returns one fact by id.
    Get(ctx context.Context, id string) (Fact, error)

    // Delete removes one fact by id.
    Delete(ctx context.Context, id string) error

    // Close releases resources (DB handle).
    Close() error
}

type Message struct {
    Role    string // "user" | "assistant" | "system" (system is ignored for extraction)
    Content string
    Name    string // optional speaker name (multi-speaker convos)
}

// Scope is the namespace a memory belongs to. Any subset may be set.
// Facts are isolated per-scope on both write and search.
type Scope struct {
    UserID  string
    AgentID string
    RunID   string
}

type Fact struct {
    ID        string
    Text      string    // the self-contained fact statement
    Hash      string    // md5(Text) — dedup key
    Score     float32   // similarity score, only populated by Search
    CreatedAt time.Time
    Scope     Scope
    // Embedding is stored but not exported on the struct returned to callers.
}
```

Construction (functional options so new knobs don't break callers):

```go
func New(opts ...Option) (Memory, error)

func WithStore(s store.Store) Option
func WithLLM(l provider.LLM) Option
func WithEmbedder(e provider.Embedder) Option
func WithExtractionTopK(n int) Option   // existing memories shown to extractor; default 10
// Sensible defaults: if no store/LLM/embedder given, build them from env (§8).
```

---

## 4. The pipeline (Approach A — additive)

This is the heart of mneme. It is a faithful re-implementation of the *current*
mem0 v3 "additive" pipeline, in idiomatic Go, with **our own prompt**. Reference
for orientation only: `mem0/memory/main.py::_add_to_vector_store` (see §5).

### `Add(msgs, scope)`
1. **Gather context.** Load the last ~10 stored messages for this scope (for
   pronoun/reference resolution in the extractor). v1 may store raw messages in
   a `messages` table; if that's deferred, pass only the incoming `msgs`.
2. **Retrieve existing memories.** Embed the incoming conversation, vector-search
   the top-K (default 10) existing facts *in this scope*. These are shown to the
   extractor so it can dedup/link rather than re-emit known facts.
3. **Anti-hallucination relabel.** Map the retrieved facts' real UUIDs to small
   integer strings `"0","1","2",…` before putting them in the prompt. Keep a
   `map[string]realUUID`. (LLMs invent/copy long UUIDs unreliably; integers are
   stable. This is a load-bearing trick — keep it.)
4. **Extract (1 LLM call).** Call the LLM with our extraction prompt (§5),
   passing: incoming messages, the integer-labelled existing memories, last-k
   context, and today's date. Require JSON output:
   `{"memory":[{"id","text","attributed_to"}]}`.
5. **Parse defensively.** Strip code fences; tolerate a JSON object embedded in
   prose; on parse failure, treat as "nothing extracted" and return cleanly.
   Never let a malformed LLM response panic or error the whole Add.
6. **Embed extracted facts** in a batch.
7. **Dedup by hash.** `hash = md5(text)`. Drop any fact whose hash matches an
   existing stored fact in scope, or another fact in the same batch.
8. **Persist** surviving facts (id=uuid, text, hash, embedding, scope,
   created_at) via `Store.Insert`.
9. Return the newly written facts.

### `Search(query, scope, k)`
1. Embed the query.
2. `Store.Search(scope, queryVec, k)` → cosine similarity, descending.
3. Return facts with `Score` populated. (Optional v1.1: drop results below a
   similarity threshold ~0.1.)

> **Why no update/delete pass?** Approach A keeps old facts and lets new ones
> accumulate; dedup is by hash + the extractor's own dedup instructions. This is
> 1 LLM call per Add instead of 2. If recall later gets noisy, a consolidation
> pass (mem0's classic ADD/UPDATE/DELETE/NONE prompt) can be added behind a flag
> — but **not in v1**.

---

## 5. Prompts — we own these

The extraction prompt is **our IP**. We write it ourselves. We may *orient* by
reading mem0's `ADDITIVE_EXTRACTION_PROMPT` (it lives in the driver-os repo at
`deps/mem0/mem0/configs/prompts.py`, which is a gitignored clone — it will NOT be
present in the mneme repo; if you need it, clone mem0 separately). mem0 is
**Apache-2.0**: do not paste its prompt verbatim. Reimplement the *ideas* in our
own words and prove ours is at least as good via the eval harness (§6).

### Prompt design principles to encode (distilled from what works)
Put these in `prompt.go` as a versioned constant (`extractionPromptV1`). The goal
is: extract durable, self-contained, specific facts; ignore chatter.

- **Role:** "extract durable facts worth remembering from this conversation."
- **Extract:** stable personal/world facts, preferences, plans, relationships,
  decisions, specific named entities, numbers, and dates.
- **Skip:** greetings, filler, acknowledgements, transient pleasantries, and
  anything already in the shown existing memories (dedup).
- **Self-contained:** every fact must stand alone — resolve pronouns to names;
  no "he/she/it/this". A fact a stranger could read cold.
- **Specific, not vague:** preserve proper nouns, exact quantities, titles, and
  qualifiers ("assistant manager", not "manager"; "Ferrari 488 GTB", not "a car";
  "416 pages", not "about 400"). Specificity is what makes a fact retrievable.
- **Meaning-preserving:** capture the *exact* claim; do not invert or soften it.
- **One fact per item; all topics covered:** a multi-topic message yields
  multiple facts. Do not stop after the first topic ("first-topic dominance" is a
  known failure — guard against it explicitly).
- **No fabrication:** every fact must trace to the input. No inferred
  attributes (gender/age/etc.) that weren't stated.
- **Date grounding:** convert "yesterday/last week" to absolute dates using the
  provided observation date.
- **Output:** strict JSON, `{"memory":[{"id":"0","text":"…","attributed_to":"user|assistant"}]}`,
  empty list when nothing is worth keeping. IDs are sequential strings from "0".

Prompt is **versioned** (`extractionPromptV1`, `…V2`, …). The eval harness scores
each version so prompt changes are decisions backed by numbers, not vibes. This
is the core of "owning the IP."

---

## 6. Eval / prompt-harness (first-class v1 deliverable)

Without this, "our own prompt" is just an untested string. The harness is how we
prove a prompt version is good and catch regressions when we change it.

### Package: `eval/`
- **Fixtures** (`eval/fixtures/*.json`): small, hand-authored, LoCoMo-style cases:
  ```json
  {
    "name": "promotion-and-pet",
    "messages": [{"role":"user","content":"I got promoted to Senior Engineer at Shopify. Also adopted a beagle named Max."}],
    "expected_facts": ["User was promoted to Senior Engineer at Shopify",
                       "User adopted a beagle named Max"],
    "queries": [
      {"q":"where does the user work?", "should_recall":["…Shopify…"]},
      {"q":"does the user have pets?",  "should_recall":["…beagle named Max…"]}
    ]
  }
  ```
  Author ~15–30 cases spanning: single fact, multi-topic, nothing-to-extract,
  dedup (a fact already in existing memories), specificity (must keep a proper
  noun), and meaning-preservation (must not invert "used to love X").

### Metrics (`eval/score.go`)
Facts are free text, so score with an LLM-judge for semantic match (an extracted
fact "matches" an expected fact if a judge LLM says they mean the same thing),
plus cheap deterministic checks:
- **Extraction recall** — fraction of `expected_facts` matched by some extracted fact.
- **Extraction precision** — fraction of extracted facts that match some expected (catches over-extraction).
- **Specificity check** — deterministic: for cases tagged with a required token
  (e.g. "Shopify"), assert it survives into a fact. No LLM needed.
- **Dedup correctness** — feed `expected_facts` as existing memories, re-run Add,
  assert ≈0 new facts.
- **Search recall@k** — after ingesting, each `query` must surface its
  `should_recall` fact within top-k.

### Runner (`eval/run.go` or `cmd/eval/`)
`go run ./cmd/eval` ingests every fixture through the real pipeline against a
configured LLM, prints a per-metric table and an aggregate score per prompt
version. **Baseline to beat:** establish the first number with `extractionPromptV1`;
every later prompt change must not regress aggregate score. (Optional stretch:
run mem0's own prompt through the same harness once to sanity-check our number is
in the same ballpark.)

> Eval needs a live LLM, so it is **not** part of `go test ./...` (unit tests
> must run offline with fakes — see §7). Gate eval behind its own command and/or
> a `MNEME_EVAL=1` env check.

---

## 7. Internal packages & interfaces

```
mneme/
  go.mod
  LICENSE                 MIT
  README.md
  PLAN.md                 (this file)
  doc.go                  package docs
  memory.go               Memory interface, Fact/Message/Scope, New(), options
  pipeline.go             Add/Search orchestration (§4)
  prompt.go               extractionPromptV1 + JSON-prompt builder (§5)
  parse.go                defensive JSON extraction from LLM output
  dedup.go                md5 hashing + batch/existing dedup
  memory_test.go          pipeline tests with fake LLM+embedder+store (offline)

  store/
    store.go              Store interface
    sqlite/               pure-Go SQLite impl (modernc.org/sqlite), cosine in Go
      sqlite.go
      sqlite_test.go
    cosine.go             shared float32 cosine helper (or in store.go)

  provider/
    provider.go           LLM + Embedder interfaces
    openai/               OpenAI-compatible HTTP client (net/http, no SDK)
      openai.go
      openai_test.go      httptest-based
    fake/                 in-memory fakes for tests (deterministic)
      fake.go

  eval/
    eval.go               fixture types + loader
    score.go              metrics (§6)
    fixtures/*.json
  cmd/
    eval/main.go          eval runner (needs live LLM)
```

### Interfaces

```go
// store/store.go
type Record struct {
    ID, Text, Hash string
    Embedding      []float32
    Scope          mneme.Scope   // or a local mirror to avoid import cycle
    CreatedAt      time.Time
}
type Hit struct { Record; Score float32 }

type Store interface {
    Insert(ctx context.Context, recs []Record) error
    Search(ctx context.Context, scope Scope, vec []float32, k int) ([]Hit, error)
    Get(ctx context.Context, id string) (Record, error)
    Delete(ctx context.Context, id string) error
    ExistingHashes(ctx context.Context, scope Scope) (map[string]struct{}, error) // for dedup
    Close() error
}
```
> Watch the import cycle: `Scope`/`Fact` are defined in package `mneme`, but
> `store` shouldn't import `mneme` if `mneme` imports `store`. Resolve by either
> (a) defining shared value types (`Scope`, `Record`) in a leaf package like
> `mneme/types`, or (b) keeping `store` self-contained with its own `Scope`
> mirror. Pick (a) — a small `types` package — it's cleaner.

```go
// provider/provider.go
type LLM interface {
    // Complete returns the assistant text. JSON mode requested when jsonObject=true.
    Complete(ctx context.Context, system, user string, jsonObject bool) (string, error)
}
type Embedder interface {
    Embed(ctx context.Context, texts []string) ([][]float32, error) // batch
    Dim() int
}
```

### SQLite schema (v1, pure-Go via `modernc.org/sqlite`)
```sql
CREATE TABLE IF NOT EXISTS facts (
  id          TEXT PRIMARY KEY,
  text        TEXT NOT NULL,
  hash        TEXT NOT NULL,
  embedding   BLOB NOT NULL,         -- float32 little-endian, len = dim*4
  user_id     TEXT NOT NULL DEFAULT '',
  agent_id    TEXT NOT NULL DEFAULT '',
  run_id      TEXT NOT NULL DEFAULT '',
  created_at  TEXT NOT NULL          -- RFC3339
);
CREATE INDEX IF NOT EXISTS idx_facts_scope ON facts(user_id, agent_id, run_id);
CREATE INDEX IF NOT EXISTS idx_facts_hash  ON facts(user_id, agent_id, run_id, hash);
```
Search = `SELECT … WHERE scope matches`, decode each `embedding` BLOB, compute
cosine in Go, partial-sort top-k. O(n) per scope — fine for thousands of facts.
When that hurts, add a `store/vec` (sqlite-vec) or `store/pg` (pgvector) impl;
the pipeline doesn't change.

---

## 8. Configuration (env, OpenAI-compatible)

`New()` with no providers builds them from env:

| Env var | Meaning | Example |
|---|---|---|
| `MNEME_LLM_BASE_URL` | OpenAI-compatible base URL | `https://api.openai.com/v1` or `http://localhost:11434/v1` (Ollama) |
| `MNEME_LLM_API_KEY` | API key (may be empty for local) | `sk-…` |
| `MNEME_LLM_MODEL` | extraction model | `gpt-4o-mini` / `llama3.1` |
| `MNEME_EMBED_BASE_URL` | embeddings base URL (defaults to LLM base) | — |
| `MNEME_EMBED_MODEL` | embedding model | `text-embedding-3-small` |
| `MNEME_DB_PATH` | sqlite file path | `./mneme.db` |

Keep secrets out of git (`.gitignore` already covers `.env`, `*.db`).

---

## 9. Server/MCP later (design constraint, not v1 work)

Do not build this now. Just don't make it hard: keep the `Memory` interface free
of HTTP/CLI concerns so a later `cmd/server` can expose `Add`/`Search`/`Get`/
`Delete` as REST + MCP tools by calling the same core. If a design choice would
make that wrapper awkward, prefer the choice that keeps it trivial.

---

## 10. Build sequence (TDD — do it in this order)

Each step is red→green→refactor. Unit tests use `provider/fake` + an in-memory or
temp-file store; **no network in `go test ./...`**.

1. **Scaffolding:** `types` package (`Scope`, `Record`, `Fact`), `Message`, the
   `Memory`/`Store`/`LLM`/`Embedder` interfaces, `doc.go`. Compiles, no logic.
2. **`store/sqlite`:** Insert/Get/Delete/ExistingHashes + Search (cosine in Go).
   Test against a temp `.db`: insert vectors, assert nearest-neighbour order.
3. **`provider/fake`:** deterministic Embedder (e.g. hash-seeded vectors) and a
   scriptable LLM (returns canned JSON per input). This unblocks pipeline tests.
4. **`provider/openai`:** real OpenAI-compatible client; test with `httptest`.
5. **`parse.go` + `dedup.go`:** defensive JSON parsing and hashing — pure, easy
   to unit-test with nasty inputs (fenced JSON, prose-wrapped JSON, garbage).
6. **`pipeline.go` `Add`:** wire extract→embed→dedup→store using fakes. Assert:
   multi-topic yields N facts; nothing-to-extract yields 0; a fact already in
   existing memories is deduped; the integer-relabel mapping round-trips.
7. **`pipeline.go` `Search`:** ingest, then assert queries surface expected facts.
8. **`prompt.go`:** `extractionPromptV1` + builder; unit-test the *builder*
   (correct sections/JSON), not the model output.
9. **`eval/`:** fixtures + scorer + `cmd/eval`. Run against a live model, record
   the **baseline aggregate score** for `extractionPromptV1` in a results file or
   README badge. This closes the loop on "owned, tested IP".
10. **`New()` + env wiring (§8)** and a top-level example test / `examples/`.

**Definition of done for v1:** `go test ./...` green and offline; `go run
./cmd/eval` produces a metrics table with a recorded baseline; a consumer can
`mneme.New()` from env and do `Add` then `Search` end-to-end against a real
OpenAI-compatible endpoint.

---

## 11. Dependencies (keep it lean)
- `modernc.org/sqlite` — pure-Go SQLite (no cgo). The one non-stdlib core dep.
- Everything else (HTTP client, JSON, md5, uuid) — stdlib (`crypto/rand` for
  uuid, or a tiny uuid lib if preferred).
- Do **not** pull an LLM SDK — the OpenAI-compatible surface we use is a couple
  of JSON endpoints; hand-roll it.

## 12. Orientation references
- mem0 additive pipeline: `mem0/memory/main.py::_add_to_vector_store` (Apache-2.0).
- mem0 extraction prompt: `mem0/configs/prompts.py::ADDITIVE_EXTRACTION_PROMPT`
  (orientation only — write our own, do not copy).
- Benchmarks to eventually self-run: LoCoMo, LongMemEval (treat published vendor
  numbers skeptically; our eval harness is the number we trust).
