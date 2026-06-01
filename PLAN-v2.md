# mneme — v2 Build Plan (benchmark-competitive memory)

> Sequel to `PLAN.md`. Same contract: written so an agent with **no prior
> context** can build it end-to-end. Read `PLAN.md` first (it describes the
> shipped v1), then this top to bottom.

---

## 1. The goal of v2

v1 proved the pipeline: additive extraction, dedup, scoped vector search, an
eval harness, dogfooded end-to-end. v2 has exactly one north star:

> **Score competitively with mem0 on the public long-term-memory benchmarks
> (LoCoMo, LongMemEval), and prove it with our own harness — not vendor numbers.**

Everything in this plan is ordered by **score-impact-per-effort**, not by feature
neatness. Two truths drive the ordering:

1. **You cannot compete on a number you do not measure.** v1's eval scores
   *extraction* on 18 hand-authored fixtures. The benchmarks score *end-to-end
   QA* over multi-session dialogues. Until we run the real benchmark, "competitive"
   is unfalsifiable. So the benchmark harness is task #1.
2. **Additive memory caps the ceiling.** v1 keeps every fact forever (`PLAN.md`
   §4, "Approach A"). Benchmark dialogues are full of facts that *change*
   ("moved from Seattle to Austin"). Additive memory recalls both and the QA
   model answers wrong. A consolidation pass that updates/deletes stale facts is
   the single biggest accuracy lever. Task #2.

A non-negotiable discipline carries over from v1: **every change is justified by
a number from the harness.** Build the harness first; measure every step after.

---

## 2. Where v1 stands (the seams already cut for v2)

v2 is mostly *filling in seams v1 deliberately left*. Know these before building:

| v2 feature | Seam already in v1 | File |
|---|---|---|
| Update/Delete pass | The integer-relabel map (`realUUID → "0","1",…`) is already built and returned; v1 just discards it (`existing, _ = relabelExisting(hits)`). The pass references existing facts by that integer id. | `pipeline.go` |
| Prompt versioning | `promptVersions` registry + `systemPrompt(version)` + `WithPromptVersion`. Add a consolidation prompt as a new entry. | `prompt.go` |
| Benchmark harness | `eval.Judge` (LLM semantic match), `eval.Run`, `Config`, `Result`, `VersionReport`, `condMean`. ~60% of a benchmark runner. | `eval/score.go`, `eval/run.go` |
| Pluggable retrieval/store | `Store` interface; `provider.LLM`/`Embedder`; functional `Option`s; `types` leaf package. | `store/store.go`, `memory.go` |
| Defensive parsing | `parseExtraction` tolerates fenced/prose/garbage JSON; reuse for the consolidation response. | `parse.go` |

The public API (`Add`/`Search`/`Get`/`Delete`/`Close`) **stays stable**. Every v2
capability is added behind a functional `Option` or as an additive struct field —
no breaking changes to the four methods.

---

## 3. Decisions for v2 (do not relitigate)

| Decision | Choice | Rationale |
|---|---|---|
| What gates "competitive" | A benchmark harness producing our own LoCoMo/LongMemEval numbers | Vendor numbers are marketing; `PLAN.md` §12 already says trust our harness |
| Memory strategy | Add **Approach B — consolidation (ADD/UPDATE/DELETE/NONE)** behind an option; additive stays the default | The biggest accuracy lever for changing facts; keep the cheap 1-call path for users who want it |
| Vector backend | Still brute-force cosine for accuracy work; sqlite-vec/pgvector are a **scale** play, not a score play | Brute force is *exact* — an index can only lose recall, never gain it |
| Temporal | Reverse the v1 non-goal: store structured event/ingest times | LoCoMo has a dedicated temporal category v1 can't answer |
| Graph memory | Sketch only; defer the full build until the harness shows multi-hop is the weak category | Biggest subsystem, narrowest gain; earn it with data |
| API stability | Four core methods unchanged; everything via `Option`s + additive fields | v1's "small, stable surface" promise holds |
| Cost transparency | Consolidation = 2 LLM calls/`Add`; reranking/multi-query add calls/`Search`. Each is opt-in and logged. | Users must be able to choose accuracy vs. cost |

---

## 4. Tier 1 — the score movers (build these, in order)

### 4.1 Benchmark harness (`bench/`) — **task #1, do first**

A new package, *separate from* `eval/` (eval = our fixtures; bench = public
datasets with a QA protocol).

**Protocol (the standard LoCoMo/LongMemEval loop):**
1. For each sample (a multi-session dialogue + its QA pairs):
   - Ingest every session via `Add` into a per-sample `Scope` (e.g.
     `RunID: sample.ID`) so samples don't cross-contaminate.
2. For each question:
   - `Search(question, scope, k)` → top-k facts.
   - Feed them to an **answer LLM** with a QA prompt ("Answer using only these
     memories; say you don't know if absent").
   - Score the answer against the gold answer: deterministic (F1 / exact-match,
     normalized) **and** `eval.Judge` semantic match. Report both.
3. Aggregate **per LoCoMo category** (single-hop, multi-hop, temporal,
   open-domain, adversarial) — the per-category table is what tells you *what to
   fix next*. A single aggregate hides which lever to pull.

**Package layout:**
```
bench/
  bench.go         dataset types + loaders (LoCoMo, LongMemEval)
  qa.go            retrieve→answer step + answer prompt (versioned, like extraction)
  score.go         F1/EM + reuse eval.Judge; per-category aggregation
  bench_test.go    offline test with fakes (tiny synthetic sample)
cmd/bench/main.go  runner: -dataset locomo|longmemeval -k N -strategy additive|consolidate -out
```

**Tests:** the loop is unit-tested offline with a fake LLM/embedder and a
2-question synthetic sample (no dataset download in `go test`). Real runs need
the dataset files (gitignored, path via flag) + a live model, exactly like
`cmd/eval`.

**Done when:** `go run ./cmd/bench -dataset locomo` prints a per-category +
aggregate table and writes `bench/RESULTS.md`. **This is the v2 baseline; every
later task is measured against it.**

> Get a real LoCoMo number with the *current additive* pipeline before building
> anything else. That number — not intuition — decides whether 4.2/4.3 are
> pulling their weight.

### 4.2 Consolidation pass — ADD/UPDATE/DELETE/NONE (`pipeline.go`, `prompt.go`)

The headline feature. mem0's two-call pipeline, reimplemented with our own prompt.

**New option:**
```go
type Strategy int
const ( Additive Strategy = iota; Consolidate )
func WithStrategy(s Strategy) Option   // default Additive (v1 behavior, unchanged)
```

**Pipeline change (when `Consolidate`):** after extraction + embedding of the
candidate facts (steps 4–6 of `PLAN.md` §4), instead of hash-dedup-then-insert:
1. Retrieve the relevant existing facts in scope (already done for the extractor)
   and keep the **integer-relabel map** v1 currently discards.
2. **Consolidation LLM call** with a new prompt: input = candidate facts +
   existing facts shown under their integer ids; output =
   `{"memory":[{"id":"<int|new>","text":"…","event":"ADD|UPDATE|DELETE|NONE"}]}`.
3. Apply operations, mapping integer ids back to real UUIDs via the map:
   - `ADD` → insert new fact (uuid, embed text).
   - `UPDATE` → replace the referenced fact's text + re-embed (`Store.Update`).
   - `DELETE` → `Store.Delete` the referenced fact.
   - `NONE` → skip.
4. Parse defensively (`parseExtraction`-style) — a malformed consolidation
   response must fall back to additive insert, never corrupt the store.

**Store interface gains one method** (only `store/sqlite` implements it today):
```go
Update(ctx context.Context, rec types.Record) error  // replace text+embedding+event_time by id
```

**Prompt:** add `consolidationPromptV1` to the `promptVersions` registry. Encode:
preserve unrelated facts (NONE), prefer UPDATE over ADD+DELETE when a fact's value
changed, DELETE only on explicit contradiction/obsolescence, never invent ids.

**Tests:** offline with a scripted LLM — assert "moved from X to Y" UPDATEs the X
fact rather than adding a second; an unrelated new fact ADDs; a restated fact is
NONE. Then **measure on the harness**: expect the largest gains in the temporal
and multi-session/multi-hop categories.

### 4.3 Retrieval boosters: rerank + multi-query (`pipeline.go`, new `rerank.go`)

Brute-force cosine over one query phrasing leaves recall on the table.

- **Rerank:** retrieve top-N (~20), rerank to top-k. Define a minimal interface:
  ```go
  type Reranker interface {
      Rerank(ctx, query string, candidates []Fact) ([]Fact, error) // reordered/scored
  }
  ```
  Ship an LLM-based reranker (score each candidate's relevance) in
  `provider/openai`; the interface allows a cross-encoder later.
  `WithReranker(r)` — applied inside `Search`, public signature unchanged.
- **Multi-query:** `WithMultiQuery(n)` — one LLM call expands the question into n
  search queries; union the per-query hits, dedup by id, then rerank. Targets
  multi-hop, where one phrasing won't surface every needed fact.

**Tests:** offline assert union+dedup+reorder logic with fakes; **measure**
recall@k delta on the harness (multi-hop category especially).

---

## 5. Tier 2 — category-specific gains

### 5.1 Structured temporal grounding (`types`, `store/sqlite`, `prompt.go`)
v1 grounds dates *in the fact text*; v2 stores them *structurally*.
- `types.Fact`/`Record` gain `EventTime time.Time` (when the fact became true) vs
  the existing `CreatedAt` (ingestion). Additive field — non-breaking.
- SQLite migration: `ALTER TABLE facts ADD COLUMN event_time TEXT` with a default;
  add a tiny schema-version table so migrations are ordered and idempotent.
- Extraction prompt emits an optional `event_date` per fact (grounded to the
  observation date, which v1 already passes). Store it.
- `Search` gains optional time filters via option/struct (`as-of T`,
  `between [a,b]`) and/or a recency tie-breaker. Targets LoCoMo's temporal QA.

### 5.2 Semantic dedup (`dedup.go`, `pipeline.go`)
`hashText` (md5) only catches verbatim repeats; "works at Shopify" / "is employed
by Shopify" both get stored. Before insert, embed the candidate and search
existing in scope; if max cosine > a threshold (~0.95, tune on the harness),
treat as a duplicate (skip, or hand to the consolidation pass as a NONE/UPDATE
candidate). Augments — does not replace — hash dedup.

### 5.3 Context-aware extraction (`store`, `pipeline.go`)
`PLAN.md` §4 step 1 anticipated this and v1 deferred it. Persist recent raw
messages per scope (a `messages` table), load the last ~k as context for the
extractor so pronoun/reference resolution improves on multi-turn dialogues. Gate
with `WithContextWindow(k)`; default off keeps v1 behavior.

---

## 6. Tier 3 — heavy / narrow (earn these with data)

### 6.1 Graph memory (the mem0g analog) — sketch only
Entity/relationship extraction into a graph (SQLite `nodes`/`edges` tables, no
new dependency), with traversal-augmented retrieval for multi-hop questions.
Real gains, but it is the **largest subsystem in the whole project** and its
benefit is concentrated on one category. **Do not start** until the §4.1 harness
shows multi-hop is the limiting category *after* 4.2/4.3. Full design = its own
plan when that day comes.

### 6.2 Vector backend (`store/vec` sqlite-vec, `store/pg` pgvector) — scale, not score
State this plainly so it is never mistaken for an accuracy task: **brute-force
cosine is exact; an ANN index can only lose recall, never gain it.** This is a
latency/scale play for very large stores, behind the existing `Store` interface.
The pipeline does not change. Build it when a real store gets big enough to hurt,
not to chase a benchmark number.

---

## 7. Free wins to fold in along the way
- **Stronger embedding model** (e.g. `text-embedding-3-large`): one-line config,
  pure recall upgrade. Note: changing the embedder against an existing store
  invalidates its vectors (see README caveats) — benchmark runs use a fresh store.
- **Answer-generation prompt + k tuning:** the benchmarks score QA, so the answer
  prompt ("use only these memories; say you don't know") and the choice of k
  affect the score even with identical retrieval. Version the answer prompt in
  `bench/qa.go` and tune it on the harness like we tune extraction.

---

## 8. Build sequence (each step measured, not vibed)
1. **`bench/` + `cmd/bench`** (4.1). Record the v2 baseline for the *current
   additive* pipeline on LoCoMo. Per-category table.
2. **Consolidation pass** (4.2) behind `WithStrategy(Consolidate)` + `Store.Update`.
   Re-run the harness; expect temporal/multi-session gains.
3. **Rerank + multi-query** (4.3). Re-run; expect multi-hop recall@k gains.
4. **Temporal grounding** (5.1). Re-run; expect temporal-category gains.
5. **Semantic dedup** (5.2) + **context-aware extraction** (5.3). Re-run.
6. Re-read the per-category table. **Only now** decide whether graph (6.1) is
   worth it. Vector backend (6.2) is independent and scheduled by scale, not score.

After each step: `go test ./...` stays green and **offline** (the v1 invariant);
new live work goes behind `cmd/bench`/`cmd/eval` and env checks.

---

## 9. Non-goals for v2 (still deferred)
- Full knowledge-graph product (6.1 is a sketch + a gate, not a commitment).
- Multi-tenant server hardening, auth, rate limiting — the HTTP/MCP server
  (`PLAN.md` §9) is still a thin wrapper to be built when the core is settled.
- Re-ranking model training, custom embedders — we consume providers, not train.
- Anything that breaks the four-method public API.

---

## 10. Definition of done for v2
- `go run ./cmd/bench -dataset locomo` produces a per-category + aggregate table,
  recorded in `bench/RESULTS.md`, for both `-strategy additive` and
  `-strategy consolidate`.
- The consolidate aggregate **beats** the additive baseline (especially temporal +
  multi-session), and our LoCoMo number is in the same ballpark as mem0's
  published figure — measured by *our* harness, with the gap (if any) explained.
- `go test ./...` green and offline; all new accuracy work is reproducible via
  `cmd/bench` against a real OpenAI-compatible endpoint.
- Every shipped lever (consolidation, rerank, multi-query, temporal) has a
  recorded before/after delta justifying its inclusion. No lever ships on vibes.

---

## 11. Orientation references
- mem0 consolidation pipeline (ADD/UPDATE/DELETE/NONE) + mem0g graph variant —
  Apache-2.0, orientation only; reimplement ideas, do not copy prompts (same rule
  as `PLAN.md` §5).
- LoCoMo: multi-session conversational QA with category labels (single-hop,
  multi-hop, temporal, open-domain, adversarial) — the per-category split is the
  point.
- LongMemEval: long-context memory QA. Treat all published vendor numbers
  skeptically; the harness in `bench/` is the number we trust.
