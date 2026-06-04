# mneme bench results — locomo

dataset: `bench/data/locomo10.json`  ·  model: `google/gemini-2.5-flash`  ·  embedder: `text-embedding-3-small`  ·  k: 5  ·  strategy: `additive`

Metrics: **EM** = normalized exact match, **F1** = token-overlap F1, **Judge** = LLM semantic match. n = question count.

| category | n | EM | F1 | judge |
|---|---|---|---|---|
| adversarial | 446 | 0.01 | 0.02 | 0.02 |
| multi_hop | 282 | 0.07 | 0.29 | 0.26 |
| open_domain | 96 | 0.05 | 0.08 | 0.10 |
| single_hop | 841 | 0.24 | 0.39 | 0.42 |
| temporal | 321 | 0.08 | 0.29 | 0.45 |
| **overall** | **1986** | **0.13** | **0.26** | **0.30** |

## Reproduce

```sh
go run ./cmd/bench -dataset locomo -path bench/data/locomo10.json -k 5 -strategy additive
```
