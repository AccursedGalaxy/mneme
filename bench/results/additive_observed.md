# mneme bench results — locomo

dataset: `bench/data/locomo10.json`  ·  model: `google/gemini-2.5-flash-lite`  ·  answer: `google/gemini-2.5-flash-lite`  ·  judge: `google/gemini-2.5-flash-lite`  ·  embedder: `text-embedding-3-small`  ·  k: 5  ·  strategy: `additive`

Metrics: **EM** = normalized exact match, **F1** = token-overlap F1, **Judge** = LLM semantic match. n = question count.

| category | n | EM | F1 | judge |
|---|---|---|---|---|
| adversarial | 446 | 0.92 | 0.92 | 0.92 |
| multi_hop | 282 | 0.05 | 0.29 | 0.26 |
| open_domain | 96 | 0.04 | 0.08 | 0.12 |
| single_hop | 841 | 0.24 | 0.42 | 0.48 |
| temporal | 321 | 0.15 | 0.44 | 0.53 |
| **overall** | **1986** | **0.34** | **0.50** | **0.54** |

## Reproduce

```sh
MNEME_EMBED_MODEL=text-embedding-3-small go run ./cmd/bench -dataset locomo -path bench/data/locomo10.json -k 5 -strategy additive \
  -model google/gemini-2.5-flash-lite -answer-model google/gemini-2.5-flash-lite -judge-model google/gemini-2.5-flash-lite
```
