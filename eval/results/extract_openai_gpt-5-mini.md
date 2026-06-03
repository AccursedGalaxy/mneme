# mneme eval results

model: `openai/gpt-5-mini`  ·  embedder: `openai`  ·  search k: 5

| version | recall | precision | specificity | search@k | dedup | aggregate |
|---|---|---|---|---|---|---|
| v1 | 0.94 | 0.83 | 1.00 | 1.00 | 1.00 | **0.94** |

## Per-fixture (version v1)

| fixture | recall | precision | spec | search | new-on-redup | aggregate |
|---|---|---|---|---|---|---|
| decision | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| health-fact | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| meaning-preservation-direction | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| meaning-preservation-negation | 1.00 | 0.33 | 1.00 | 1.00 | 0 | 0.87 |
| multi-speaker | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| multi-topic-three | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| no-fabrication | 1.00 | 1.00 | — | 1.00 | 0 | 1.00 |
| nothing-to-extract | 1.00 | 1.00 | — | — | 0 | 1.00 |
| plans-and-dates | 0.00 | 0.00 | 1.00 | 1.00 | 0 | 0.60 |
| preferences | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| promotion-and-pet | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| pronoun-resolution | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| question-not-fact | 1.00 | 0.00 | — | — | 0 | 0.50 |
| relationships | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| relative-date-grounding | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| single-fact | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| specificity-numbers | 1.00 | 0.67 | 1.00 | 1.00 | 0 | 0.93 |
| specificity-product | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
