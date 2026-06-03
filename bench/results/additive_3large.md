# mneme bench results — locomo

dataset: `bench/data/locomo10.json`  ·  model: `google/gemini-2.5-flash-lite`  ·  embedder: `text-embedding-3-large`  ·  k: 5  ·  strategy: `additive`

Metrics: **EM** = normalized exact match, **F1** = token-overlap F1, **Judge** = LLM semantic match. n = question count.

| category | n | EM | F1 | judge |
|---|---|---|---|---|
| adversarial | 446 | 0.01 | 0.01 | 0.02 |
| multi_hop | 282 | 0.05 | 0.28 | 0.27 |
| open_domain | 96 | 0.04 | 0.07 | 0.08 |
| single_hop | 841 | 0.17 | 0.28 | 0.32 |
| temporal | 321 | 0.06 | 0.20 | 0.25 |
| **overall** | **1986** | **0.09** | **0.20** | **0.22** |

## Reproduce

```sh
go run ./cmd/bench -dataset locomo -path bench/data/locomo10.json -k 5 -strategy additive
```
