# mneme bench results — locomo

dataset: `bench/data/locomo10.json`  ·  model: `google/gemini-2.5-flash-lite`  ·  embedder: `text-embedding-3-small`  ·  k: 5  ·  strategy: `additive`

Metrics: **EM** = normalized exact match, **F1** = token-overlap F1, **Judge** = LLM semantic match. n = question count.

| category | n | EM | F1 | judge |
|---|---|---|---|---|
| adversarial | 446 | 0.01 | 0.01 | 0.02 |
| multi_hop | 282 | 0.06 | 0.28 | 0.26 |
| open_domain | 96 | 0.03 | 0.06 | 0.07 |
| single_hop | 841 | 0.21 | 0.32 | 0.36 |
| temporal | 321 | 0.06 | 0.21 | 0.22 |
| **overall** | **1986** | **0.11** | **0.22** | **0.23** |

## Reproduce

```sh
go run ./cmd/bench -dataset locomo -path bench/data/locomo10.json -k 5 -strategy additive
```
