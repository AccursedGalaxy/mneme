# mneme bench results — locomo

dataset: `bench/data/locomo10.json`  ·  model: `google/gemini-2.5-flash-lite`  ·  embedder: `text-embedding-3-small`  ·  k: 5  ·  strategy: `additive`  ·  rerank: `on`  ·  multiquery: `3`

Metrics: **EM** = normalized exact match, **F1** = token-overlap F1, **Judge** = LLM semantic match. n = question count.

| category | n | EM | F1 | judge |
|---|---|---|---|---|
| adversarial | 446 | 0.00 | 0.01 | 0.01 |
| multi_hop | 282 | 0.06 | 0.28 | 0.26 |
| open_domain | 96 | 0.06 | 0.09 | 0.10 |
| single_hop | 841 | 0.20 | 0.32 | 0.36 |
| temporal | 321 | 0.06 | 0.19 | 0.21 |
| **overall** | **1986** | **0.10** | **0.21** | **0.23** |

## Reproduce

```sh
go run ./cmd/bench -dataset locomo -path bench/data/locomo10.json -k 5 -strategy additive -rerank -multiquery 3
```
