# mneme eval results

model: `deepseek/deepseek-v3.2`  ·  embedder: `openai`  ·  search k: 5

| version | recall | precision | specificity | search@k | dedup | aggregate |
|---|---|---|---|---|---|---|
| v1 | 1.00 | 0.89 | 1.00 | 1.00 | 0.88 | **0.92** |

## Per-fixture (version v1)

| fixture | recall | precision | spec | search | new-on-redup | aggregate |
|---|---|---|---|---|---|---|
| decision | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| health-fact | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| meaning-preservation-direction | 1.00 | 1.00 | 1.00 | 1.00 | 1 | 0.80 |
| meaning-preservation-negation | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| multi-speaker | 1.00 | 1.00 | 1.00 | 1.00 | 1 | 0.80 |
| multi-topic-three | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| no-fabrication | 1.00 | 1.00 | — | 1.00 | 0 | 1.00 |
| nothing-to-extract | 1.00 | 0.00 | — | — | 0 | 0.50 |
| plans-and-dates | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| preferences | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| promotion-and-pet | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| pronoun-resolution | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| question-not-fact | 1.00 | 0.00 | — | — | 0 | 0.50 |
| relationships | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| relative-date-grounding | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| single-fact | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| specificity-numbers | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| specificity-product | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
