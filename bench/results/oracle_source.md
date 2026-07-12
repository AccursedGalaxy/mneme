# mneme bench results — locomo

dataset: `bench/data/locomo10.json`  ·  model: `google/gemini-2.5-flash-lite`  ·  answer: `google/gemini-2.5-flash-lite`  ·  judge: `google/gemini-2.5-flash-lite`  ·  embedder: `text-embedding-3-small`  ·  k: 5  ·  strategy: `additive`

Metrics: **EM** = normalized exact match, **F1** = token-overlap F1, **Judge** = LLM semantic match. n = question count.

| category | n | EM | F1 | judge |
|---|---|---|---|---|
| adversarial | 446 | 0.93 | 0.93 | 0.93 |
| multi_hop | 282 | 0.18 | 0.50 | 0.56 |
| open_domain | 96 | 0.09 | 0.13 | 0.16 |
| single_hop | 841 | 0.41 | 0.62 | 0.67 |
| temporal | 321 | 0.11 | 0.32 | 0.31 |
| **overall** | **1986** | **0.43** | **0.60** | **0.63** |

## Reproduce

```sh
MNEME_EMBED_MODEL=text-embedding-3-small go run ./cmd/bench -dataset locomo -path bench/data/locomo10.json -k 5 -strategy additive \
  -model google/gemini-2.5-flash-lite -answer-model google/gemini-2.5-flash-lite -judge-model google/gemini-2.5-flash-lite
```
