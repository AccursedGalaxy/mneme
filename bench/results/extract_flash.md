# mneme bench results — locomo

dataset: `bench/data/locomo10.json`  ·  model: `google/gemini-2.5-flash`  ·  answer: `google/gemini-2.5-flash-lite`  ·  judge: `google/gemini-2.5-flash-lite`  ·  embedder: `text-embedding-3-small`  ·  k: 5  ·  strategy: `additive`

Metrics: **EM** = normalized exact match, **F1** = token-overlap F1, **Judge** = LLM semantic match. n = question count.

| category | n | EM | F1 | judge |
|---|---|---|---|---|
| adversarial | 446 | 0.94 | 0.94 | 0.94 |
| multi_hop | 282 | 0.07 | 0.28 | 0.28 |
| open_domain | 96 | 0.07 | 0.10 | 0.14 |
| single_hop | 841 | 0.23 | 0.38 | 0.43 |
| temporal | 321 | 0.09 | 0.31 | 0.49 |
| **overall** | **1986** | **0.34** | **0.47** | **0.52** |

## Reproduce

```sh
MNEME_EMBED_MODEL=text-embedding-3-small go run ./cmd/bench -dataset locomo -path bench/data/locomo10.json -k 5 -strategy additive \
  -model google/gemini-2.5-flash -answer-model google/gemini-2.5-flash-lite -judge-model google/gemini-2.5-flash-lite
```
