# mneme bench results — locomo (full distribution)

> **Erratum (2026-07-12, scoring fix — not yet re-run).** Every run below was
> scored by a harness that graded the adversarial category (446 of 1986
> questions) against LoCoMo's `adversarial_answer` — which is the *trap*
> distractor, not a gold answer. The correct behavior for that category is
> abstention, which is why every run shows adversarial ≈ 0.01–0.02: correctly
> abstaining systems were scored 0. The harness now scores adversarial
> questions as abstention checks. Consequences: (a) the absolute numbers below
> understate performance and are not comparable to published LoCoMo results;
> (b) the *relative* deltas between rows remain valid — every row carried the
> same dead-weight category; (c) the "inferential-fact capture" reading of the
> adversarial column below is wrong — it was a scoring artifact. Re-run the
> matrix under the corrected harness before quoting absolute numbers.

model: `google/gemini-2.5-flash-lite`  ·  dataset: `bench/data/locomo10.json`  ·  k: 5  ·  **full set, 1986 questions, no `-maxq`**

Metrics: **EM** = normalized exact match, **F1** = token-overlap F1, **Judge** = LLM semantic match (the metric we steer on). This supersedes the earlier bounded gpt-4o-mini sample — it is the distribution-weighted v2 DoD baseline (PLAN-v2 §10). Per-run files live in `bench/results/`.

Run on a parallelized harness (`-concurrency 8`, ~35 min/run) and the retry-hardened OpenAI provider. Lever matrix holds the LLM constant so each row is a clean A/B against the additive/3-small baseline.

## Lever matrix — Judge by category

| run | single (841) | multi (282) | temporal (321) | open (96) | adversarial (446) | **overall (1986)** |
|---|---|---|---|---|---|---|
| **additive · 3-small** (baseline) | **0.36** | 0.26 | 0.22 | 0.07 | 0.02 | **0.23** |
| consolidate · 3-small | 0.28 | 0.20 | 0.19 | 0.08 | 0.01 | **0.18** ↓ |
| additive · 3-large | 0.32 | 0.27 | 0.25 | 0.08 | 0.02 | **0.22** ≈ |
| additive · boosters (rerank+mq3) | 0.36 | 0.26 | 0.21 | 0.10 | 0.01 | **0.23** ≈ |
| additive · gemini-embedding-2 | — | — | — | — | — | FAILED (free-tier 1000/day quota; needs BYOK) |

EM/F1 for the baseline: overall 0.11 / 0.22. Full per-run tables in `bench/results/*.md`.

## What this settles (measured, not guessed)

1. **Consolidation v1 is a net loss on the true distribution: 0.23 → 0.18 Judge** (single-hop −0.08, and single-hop is 42% of the set). The old balanced sample hid this as "+0.01 flat". **Decision: `additive` stays the default; do not ship consolidate.** The v1 prompt UPDATE/DELETE/merges facts that were already correct. v2 prompt would need to recover single-hop *without* losing the small multi/temporal gains before this reopens.

2. **A stronger embedder does not help: 3-large 0.23 → 0.22** (temporal +0.03, single-hop −0.04, net flat-negative). 3-small already returns native 1536-dim vectors. **Retrieval *quality* is not the bottleneck** — which also means Voyage/Cohere are not worth wiring up for this workload.

3. **Retrieval boosters (LLM-rerank + multi-query=3) are flat: 0.23 → 0.23** for ~2× the per-question cost. Small open-domain bump (+0.03), nothing on aggregate. The LLM-as-reranker does not earn its cost; don't default it. A real cross-encoder reranker *might*, but #2 makes retrieval-side gains a low bet overall.

## Extraction-model A/B — the lever that works

The matrix above pointed at extraction as the bottleneck. So we varied **only the extraction model**, pinning answer + judge to `gemini-2.5-flash-lite` (so the delta is purely "which model writes facts to the store"); embedder `text-embedding-3-small`, additive, k=5.

**eval** (18 fixtures, oracle pinned to `gemini-2.5-flash`):

| extraction model | recall | precision | specificity | search@k | dedup | aggregate | $/extract |
|---|---|---|---|---|---|---|---|
| gemini-2.5-flash-lite (baseline) | 0.94 | 1.00 | 0.93 | 0.94 | 0.81 | 0.93 | $0.00034 |
| **gemini-2.5-flash** | 1.00 | 0.94 | 1.00 | 1.00 | 1.00 | **0.99** | $0.0017 |
| gpt-5-mini | 0.94 | 0.83 | 1.00 | 1.00 | 1.00 | 0.94 | $0.0029 |
| deepseek-v3.2 | 1.00 | 0.89 | 1.00 | 1.00 | 0.88 | 0.92 | $0.0021 |

**bench** (full LoCoMo Judge, answer+judge = flash-lite):

| extraction model | single | multi | temporal | open | adversarial | **overall** |
|---|---|---|---|---|---|---|
| gemini-2.5-flash-lite (baseline) | 0.36 | 0.26 | 0.22 | 0.07 | 0.02 | **0.23** |
| **gemini-2.5-flash** | 0.42 | 0.26 | **0.45** | 0.10 | 0.02 | **0.30** |
| gpt-5-mini | — | — | — | — | — | FAILED (2h ingest timeout; reasoning extraction too slow) |
| deepseek-v3.2 | — | — | — | — | — | FAILED (provider error mid-run; eval already < baseline) |

**Result: `gemini-2.5-flash` extraction lifts overall Judge 0.23 → 0.30 (+30% relative), all from extraction alone.** Temporal nearly doubles (0.22 → 0.45) — flash-lite was mangling dated facts that flash preserves (consistent with its eval specificity 0.93→1.00, dedup 0.81→1.00). This is the **first lever that moves the number** — bigger than embeddings, consolidation, or reranking combined (all of which were flat).

Losers, usefully: **gpt-5-mini** is slower (reasoning tokens, hit the harness 2h ingest timeout), pricier, *and* worse on eval (precision 0.83 — over-extracts). **deepseek-v3.2** is below baseline on eval and failed mid-bench. Neither is worth pursuing.

**Recommendation: use `gemini-2.5-flash` (or equivalent-tier) for extraction, not flash-lite.** ~5× the per-fact cost ($0.0017 vs $0.00034) for +0.07 Judge — easily worth it. ~~`adversarial` stays at 0.02 even with flash → inferential-fact capture is a *separate, harder* problem (prompt/inference work, tk #11), not a model-tier problem.~~ **Retracted (see erratum):** the flat 0.02 was the harness scoring correct abstentions as wrong; it says nothing about inferential-fact capture.

## Where the signal actually points

Three independent retrieval-side levers — consolidation, a bigger embedder, and reranking — are all flat-or-negative. That triangulates the bottleneck **upstream of retrieval: facts that answer the question are never extracted in the first place**, so no amount of better ranking surfaces them.

- ~~The **adversarial category (446 Q = 22%, stuck at ~0.02 everywhere)** is *not* abstention — its golds are real, short, **inferential** answers ("self-care is important", "researching adoption agencies", "purple"). These are realizations/causes/attributes the additive extractor doesn't capture.~~ **Retracted (see erratum):** those "golds" were the `adversarial_answer` trap field, which the loader wrongly fell back to; per the LoCoMo protocol the category is unanswerable and the correct behavior is abstention. The flat ~0.02 measured the harness bug, not extraction. The single-hop diagnostic (tk task 7/11) still stands on its own evidence.
- **single-hop (0.36)** is the ceiling category and still leaves most questions unanswered — again an extraction/answer-specificity gap, not a ranking gap.

**Next lever: re-run this matrix under the corrected adversarial scoring** to get a trustworthy baseline, then extraction recall on implied/causal facts (tk task 11) and answer-prompt specificity tuning (task 8).

## Reproduce

```sh
# one run
go run ./cmd/bench -dataset locomo -path bench/data/locomo10.json -k 5 -concurrency 8 -strategy additive
# full matrix
bash bench/run_matrix.sh
```
