# mneme bench results — locomo

dataset: `bench/data/locomo10.json`  ·  model: `google/gemini-2.5-flash-lite`  ·  answer: `google/gemini-2.5-flash-lite`  ·  judge: `google/gemini-2.5-flash-lite`  ·  embedder: `text-embedding-3-large`  ·  k: 5  ·  strategy: `additive`

Metrics: **EM** = normalized exact match, **F1** = token-overlap F1, **Judge** = LLM semantic match. n = question count.

| category | n | EM | F1 | judge |
|---|---|---|---|---|
| adversarial | 446 | 0.94 | 0.94 | 0.94 |
| multi_hop | 282 | 0.07 | 0.33 | 0.34 |
| open_domain | 96 | 0.05 | 0.08 | 0.09 |
| single_hop | 841 | 0.24 | 0.39 | 0.44 |
| temporal | 321 | 0.06 | 0.21 | 0.27 |
| **overall** | **1986** | **0.34** | **0.46** | **0.49** |

## Reproduce

```sh
MNEME_EMBED_MODEL=text-embedding-3-large go run ./cmd/bench -dataset locomo -path bench/data/locomo10.json -k 5 -strategy additive \
  -model google/gemini-2.5-flash-lite -answer-model google/gemini-2.5-flash-lite -judge-model google/gemini-2.5-flash-lite
```
