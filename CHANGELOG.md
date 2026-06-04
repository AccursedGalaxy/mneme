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

### Changed
- File-backed SQLite stores open in WAL mode for better concurrent-read behavior.
- **Recommended extraction model is now `google/gemini-2.5-flash`** (was
  `gpt-4o-mini`). The full-LoCoMo extraction A/B showed extraction quality is the
  dominant lever: `flash` lifts end-to-end Judge **0.23 → 0.30** (temporal
  0.22 → 0.45) over `flash-lite`, while a stronger embedder, consolidation, and
  reranking all measured flat. `cmd/bench`/`cmd/eval` `-model` defaults, the
  README config table, and `examples/basic` updated to match. See
  `bench/RESULTS.md`.

### Fixed
- Consolidation detects no-op `UPDATE`s against a target that was concurrently
  deleted, instead of reporting a phantom write.

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
