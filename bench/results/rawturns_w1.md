# mneme bench results — locomo

dataset: `bench/data/locomo10.json`  ·  model: `google/gemini-2.5-flash-lite`  ·  answer: `google/gemini-2.5-flash-lite`  ·  judge: `google/gemini-2.5-flash-lite`  ·  embedder: `text-embedding-3-small`  ·  k: 5  ·  strategy: `rawturns`

Metrics: **EM** = normalized exact match, **F1** = token-overlap F1, **Judge** = LLM semantic match. n = question count.

| category | n | EM | F1 | judge |
|---|---|---|---|---|
| adversarial | 446 | 0.95 | 0.95 | 0.95 |
| multi_hop | 282 | 0.05 | 0.25 | 0.25 |
| open_domain | 96 | 0.04 | 0.06 | 0.07 |
| single_hop | 841 | 0.29 | 0.48 | 0.54 |
| temporal | 321 | 0.04 | 0.27 | 0.21 |
| **overall** | **1986** | **0.35** | **0.50** | **0.51** |

## Reproduce

```sh
MNEME_EMBED_MODEL=text-embedding-3-small go run ./cmd/bench -dataset locomo -path bench/data/locomo10.json -k 5 -strategy rawturns \
  -model google/gemini-2.5-flash-lite -answer-model google/gemini-2.5-flash-lite -judge-model google/gemini-2.5-flash-lite
```
