# mneme bench results — locomo

dataset: `bench/data/locomo10.json`  ·  model: `openai/gpt-4o-mini`  ·  embedder: `openai`  ·  k: 5

Metrics: **EM** = normalized exact match, **F1** = token-overlap F1, **Judge** = LLM semantic match. n = question count.

> ⚠️ **Bounded run — not the full benchmark.** ≤10 category-balanced questions/sample (10 samples, 100 questions). Per-category n is equalized by the balanced cap, so the **overall** row is a flat mean across categories, NOT weighted by LoCoMo's true distribution (single-hop is ~42% of the real 1,986-question set). Use this to track per-category movement; for the distribution-weighted headline, drop `-maxq`.

This file is composed from two runs (additive baseline = PLAN-v2.md §4.1; consolidate = §4.2). Reproduce commands are at the bottom; each regenerates one half.

## Additive vs Consolidate

`additive → consolidate`, with the Judge delta called out (Judge is the semantic metric, the most meaningful of the three).

| category | n | EM | F1 | Judge | Δ Judge |
|---|---|---|---|---|---|
| single_hop | 21 | 0.33 → 0.24 | 0.45 → 0.30 | 0.43 → 0.33 | **−0.10** |
| multi_hop | 20 | 0.10 → 0.15 | 0.26 → 0.35 | 0.20 → 0.30 | **+0.10** |
| temporal | 21 | 0.10 → 0.14 | 0.30 → 0.35 | 0.29 → 0.33 | **+0.04** |
| open_domain | 18 | 0.06 → 0.11 | 0.12 → 0.14 | 0.11 → 0.11 | 0.00 |
| adversarial | 20 | 0.00 → 0.00 | 0.00 → 0.00 | 0.05 → 0.05 | 0.00 |
| **overall** | **100** | **0.12 → 0.13** | **0.23 → 0.23** | **0.22 → 0.23** | **+0.01** |

## Reading

- **Consolidation helps exactly where the plan predicted it would** (§4.2): the
  multi-hop and temporal categories — questions that span multiple sessions and
  involve facts that change over time — improve on every metric (multi-hop Judge
  +0.10, temporal Judge +0.04). Reconciling a changed fact in place instead of
  storing both the stale and the new version is what those categories reward.
- **It regresses single-hop (−0.10 Judge, −0.15 F1).** Single-hop questions are
  answered by one directly-stored fact; the consolidation pass can UPDATE or
  DELETE facts that were already correct, or merge two distinct facts into one
  less-specific statement, losing the precise token the answer needed. This is
  the cost side of the lever and the first thing to tune next.
- **Net is roughly flat on this balanced sample** (+0.01 Judge). Because the
  balanced cap equalizes category counts, the gains and the single-hop loss
  nearly cancel. On LoCoMo's *true* distribution single-hop dominates (~42% of
  questions), so a distribution-weighted run would currently weight that
  regression more heavily — consolidation is **not yet a clear aggregate win**
  and should not ship as the default on this evidence.

## Next levers (measured against this table)

1. Tune `consolidationPromptV1` to be more conservative on single-hop facts
   (prefer NONE/ADD over UPDATE/DELETE unless a value genuinely changed; never
   merge two specific facts into one). Re-run and check single-hop recovers
   without giving back the multi-hop/temporal gains.
2. Run the **full** distribution-weighted benchmark (drop `-maxq`) for both
   strategies to get the headline number the v2 DoD asks for.
3. Retrieval boosters (§4.3: rerank + multi-query) target the same multi-hop
   recall the consolidation gain hints at.

## Reproduce

```sh
# additive baseline (§4.1)
go run ./cmd/bench -dataset locomo -path bench/data/locomo10.json -k 5 -strategy additive   -maxq 10

# consolidation (§4.2)
go run ./cmd/bench -dataset locomo -path bench/data/locomo10.json -k 5 -strategy consolidate -maxq 10
```
