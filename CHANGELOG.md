# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the version is `0.x`, the public API may still change between minor
releases; breaking changes are called out under **Changed**.

## [Unreleased]

Work accumulating toward `v0.2.0`. The headline (a full distribution-weighted
benchmark validating the retrieval and consolidation levers as a set) is the
gate for cutting it — see `bench/RESULTS.md` and `PLAN-v2.md` §10.

### Added
- **Temporal grounding: `Message.Timestamp` and `Fact.ObservedAt`.** A message
  can now carry when it was said, and the fact extracted from it records that
  time — distinct from `CreatedAt`, which stays ingestion time. The extractor's
  date arithmetic is anchored on the conversation's own timestamp instead of on
  today, so "I went yesterday" resolves against when it was said rather than when
  it was ingested. Both fields are optional and default to the previous behavior
  (fall back to the clock; `ObservedAt` stays zero), so no existing caller
  changes. SQLite stores get an `observed_at` column, added by an idempotent
  migration on `Open` — existing databases upgrade in place and their rows read
  back with a zero `ObservedAt`, which is the truth: their source time was never
  recorded.

  `ObservedAt` is **say-time, not event-time**: for "I went yesterday" said on 8
  May it is 8 May, and the event's own date (7 May) is what the extractor resolves
  into the fact text. Do not answer "when did X happen" from the field directly.

  Under `Consolidate`, an `UPDATE` takes the reconciling conversation's
  observation time; if that conversation carried no timestamp, the fact keeps the
  date it already had rather than being blanked — losing a known date is worse
  than not learning a new one. `Store.Update` enforces this, so no caller can
  erase a date by passing a partially-populated record.

  This fixes a silent corruption. Without a grounding date, `gemini-2.5-flash-lite`
  wrote *"Caroline attended an LGBTQ support group on 2026-07-11"* — the ingestion
  date — for a May 2023 event, and nothing downstream could detect it. See
  `bench/RESULTS.md`.
- **Consolidation write strategy** (`WithStrategy(Consolidate)`): a second LLM
  call per `Add` reconciles newly extracted facts against existing ones —
  ADD / UPDATE / DELETE / NONE — so a changed fact replaces the stale one
  instead of piling up beside it. Tunable via `WithConsolidationTopK` (default
  30) and `WithConsolidationVersion`; ships versioned consolidation prompts
  (`v1`, and a conservative `v2`). Reconciliation candidates are retrieved
  per extracted fact, and ADD ops are hash-deduped against what is already
  stored.
- **Retrieval boosters** (`PLAN-v2.md` §4.3), both opt-in and leaving the public
  `Search` signature unchanged:
  - `WithReranker(r)` — over-retrieve a wider pool (`DefaultRerankPoolN`),
    reorder by relevance, keep the top-k. Adds the `Reranker` interface and a
    shipped, parse-defensive `openai.LLMReranker`.
  - `WithMultiQuery(n)` — one LLM call expands the query into up to `n`
    phrasings; their hits are unioned (deduped by id, best score wins) before
    reranking. Targets multi-hop questions.
- **Embedder-identity guard**: a store pins the embedder that wrote it and `New`
  refuses a mismatched one (which would silently degrade search). Override with
  `AllowEmbedderMismatch()`; embedders may expose `provider.Named` for a stable
  identity.
- **Benchmark harness** (`cmd/bench`, `bench/`): runs LoCoMo / LongMemEval with
  EM, token-F1 and LLM-judge metrics, a per-category table, and flags for
  strategy, consolidation prompt, `-rerank` and `-multiquery`.
- **CI and eval-regression-gate** GitHub Actions workflows.
- **Parallel benchmark scoring** (`cmd/bench -concurrency`, default 8): questions
  within a sample are scored through a bounded worker pool (ingestion stays
  sequential), cutting a full LoCoMo run from ~90 min to ~25.
- **Separate answer/judge models** for clean model A/Bs: `cmd/bench`
  `-answer-model` / `-judge-model` and `cmd/eval` `-judge-model` pin those roles
  to a fixed model so only the extraction model under test varies.
- **Transient-failure retries** in the OpenAI-compatible client: network errors,
  429/5xx, and empty/garbled 2xx bodies are retried with exponential backoff (up
  to 5 attempts), so one gateway blip no longer aborts a long ingestion or bench.
- **Benchmark answer prompt v2** (`cmd/bench -answer-version`, now the default).
  The source oracle showed the answer stage lost more score than retrieval did:
  v1 abstained on questions whose answer sat verbatim in the evidence, and echoed
  relative dates ("last week") back instead of resolving them. v2 fixes both,
  worth +0.097 answerable Judge at the oracle. v1 stays registered so past runs
  remain reproducible, and an unknown version is now an error rather than a
  silent fallback to the default.
- **`cmd/replay`**: re-runs the answer step of a finished run against a different
  answer prompt or answer model, reading its prediction dump and writing a new one
  for `cmd/rescore` to compare. The answer prompt cannot affect retrieval, so
  replaying a dump is exact, not approximate — an answer-stage change now costs
  one call per question instead of a full re-ingest.

### Changed
- File-backed SQLite stores open in WAL mode for better concurrent-read behavior.
- **Recommended extraction model is now `google/gemini-2.5-flash`** (was
  `gpt-4o-mini`). The full-LoCoMo extraction A/B showed extraction quality is the
  dominant lever: `flash` lifts end-to-end Judge **0.23 → 0.30** (temporal
  0.22 → 0.45) over `flash-lite`, while a stronger embedder, consolidation, and
  reranking all measured flat. `cmd/bench`/`cmd/eval` `-model` defaults, the
  README config table, and `examples/basic` updated to match. See
  `bench/RESULTS.md`.

  **Superseded.** Those figures predate the adversarial-scoring fix, and the
  recommendation itself no longer holds: flash's edge was almost entirely
  temporal date normalization, which `Message.Timestamp` now supplies to any
  extraction model for free. See `bench/RESULTS.md` for the current numbers.

### Fixed
- Consolidation detects no-op `UPDATE`s against a target that was concurrently
  deleted, instead of reporting a phantom write.
- **Consolidation blast-radius cap**: at most one UPDATE/DELETE of an existing
  memory is applied per extracted candidate, so a hostile or confused
  conversation turn (prompt injection reaching the consolidation LLM) can no
  longer rewrite or wipe the whole reconciliation window in one `Add`. Prompt
  rendering also flattens role/name and indents multi-line message content so
  a message cannot forge speaker turns or section headers.
- **`":memory:"` stores survive connection-pool churn**: the store now uses a
  uniquely named shared-cache in-memory database with a pinned anchor
  connection. Previously, `database/sql` discarding the single pooled
  connection (e.g. after a context cancelled mid-query) silently replaced the
  database with an empty one.
- **Concurrent-`Add` dedup backstop in the store**: `Insert` skips a record
  whose `(scope, hash)` already exists, inside the write transaction, so two
  racing `Add`s extracting the same fact can no longer both insert it.
  Consolidation `UPDATE`s are deliberately not constrained.
- **Silent-corruption guards**: empty embedding vectors are rejected at the
  client, pipeline, and store layers (previously stored and unretrievable
  forever); mixed-dimension batches are rejected; a truncated embedding BLOB
  surfaces a read error instead of decoding short; `Cosine` collapses NaN to 0
  so one corrupt vector cannot scramble search ordering.
- **Reranker no longer corrupts `Fact.Score`**: `openai.LLMReranker` reorders
  candidates without writing model-internal values (including a `-1` sentinel
  for unscored candidates) into the public score field, and degrades to the
  cosine order on a failed LLM call instead of failing `Search`.
- **Client hardening**: an explicit `temperature` rejected by reasoning-family
  models triggers one retry without the parameter; a response body cut off
  mid-read is retried instead of trusted; cancellation during retry backoff
  reports the error that caused the retrying alongside the context error. The
  retry loop is now covered by fast unit tests.
- **Consolidation honors context cancellation** instead of falling back to an
  additive insert with a dead context and misattributing the failure.
- **LoCoMo adversarial scoring inverted** (`bench/`): category-5 questions are
  unanswerable and are now scored as abstention checks; previously the loader
  used the `adversarial_answer` *trap* field as gold, scoring correct
  abstentions 0 across 22% of the question set. Absolute historical numbers in
  `bench/RESULTS.md` are annotated with an erratum; relative deltas stand.
- **Bench provenance**: result files and their `## Reproduce` commands now
  record the answer/judge/embedding models; `cmd/bench -out` no longer
  defaults to overwriting the curated `bench/RESULTS.md`.
- **Workflow injection sinks closed**: release notes are passed to
  `gh release create` via `--notes-file` (changelog text no longer expands in
  a shell holding a write token), and the eval workflow reads its dispatch
  input from `env` (no expansion in a shell holding the API key). The eval
  gate also pins its judge model so the oracle is constant across runs.

## [0.1.0] - 2026-06-01

Baseline release marking the **v1 core**: a standalone Go library giving agents
persistent, searchable long-term memory.

### Added
- Public `Memory` API — `Add`, `Search`, `Get`, `Delete`, `Close` — with
  functional options (`New(...Option)`) and `MNEME_*` environment wiring.
- Additive ingestion pipeline: retrieve existing → integer-relabel → one LLM
  extraction → defensive parse → hash dedup → embed → persist.
- `store/sqlite`: pure-Go (modernc) backend, `float32` BLOB embeddings, cosine
  similarity computed in Go, per-scope isolation.
- Providers: SDK-free OpenAI-compatible HTTP client (`provider/openai`) and a
  deterministic offline `provider/fake` (scripted LLM + feature-hashing
  embedder) for tests.
- Versioned extraction prompt (`extractionPromptV1`), owned in-repo.
- Eval harness (`cmd/eval`, `eval/`): 18 fixtures, LLM-judge and deterministic
  metrics, recording a baseline aggregate of **0.97** on `gpt-4o-mini` with
  `text-embedding-3-small`.

[Unreleased]: https://github.com/AccursedGalaxy/mneme/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/AccursedGalaxy/mneme/releases/tag/v0.1.0
