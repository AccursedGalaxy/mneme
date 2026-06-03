# mneme bench results — locomo (full distribution)

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

## Where the signal actually points

Three independent retrieval-side levers — consolidation, a bigger embedder, and reranking — are all flat-or-negative. That triangulates the bottleneck **upstream of retrieval: facts that answer the question are never extracted in the first place**, so no amount of better ranking surfaces them.

- The **adversarial category (446 Q = 22%, stuck at ~0.02 everywhere)** is *not* abstention — its golds are real, short, **inferential** answers ("self-care is important", "researching adoption agencies", "purple"). These are realizations/causes/attributes the additive extractor doesn't capture. This matches the prior single-hop diagnostic (tk task 7/11): answer facts like "realized self-care matters" were *never written to the store*.
- **single-hop (0.36)** is the ceiling category and still leaves most questions unanswered — again an extraction/answer-specificity gap, not a ranking gap.

**Next lever (highest expected value): extraction recall on implied/causal/inferential facts** (tk task 11) — tighten the extraction prompt to capture realizations, causes, origins, and attributes, then re-run this matrix. Answer-prompt specificity tuning (task 8) is the cheap secondary.

## Reproduce

```sh
# one run
go run ./cmd/bench -dataset locomo -path bench/data/locomo10.json -k 5 -concurrency 8 -strategy additive
# full matrix
bash bench/run_matrix.sh
```
