# mneme bench results — locomo

dataset: `bench/data/locomo10.json`  ·  model: `openai/gpt-4o-mini`  ·  embedder: `openai`  ·  k: 5  ·  strategy: `additive`

Metrics: **EM** = normalized exact match, **F1** = token-overlap F1, **Judge** = LLM semantic match. n = question count.

> ⚠️ **Bounded run — not the full benchmark.** ≤10 category-balanced questions/sample (10 samples scored). Per-category n is equalized by the balanced cap, so the **overall** row is a flat mean across categories, NOT weighted by LoCoMo's true distribution (single-hop dominates the real set). Use this to track per-category movement; for the distribution-weighted headline, drop `-maxq`/`-limit`.

| category | n | EM | F1 | judge |
|---|---|---|---|---|
| adversarial | 20 | 0.00 | 0.00 | 0.05 |
| multi_hop | 20 | 0.10 | 0.26 | 0.20 |
| open_domain | 18 | 0.06 | 0.12 | 0.11 |
| single_hop | 21 | 0.33 | 0.45 | 0.43 |
| temporal | 21 | 0.10 | 0.30 | 0.29 |
| **overall** | **100** | **0.12** | **0.23** | **0.22** |

## Reproduce

```sh
go run ./cmd/bench -dataset locomo -path bench/data/locomo10.json -k 5 -strategy additive -maxq 10
```
