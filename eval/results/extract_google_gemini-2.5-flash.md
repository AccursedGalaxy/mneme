# mneme eval results

model: `google/gemini-2.5-flash`  ·  embedder: `openai`  ·  search k: 5

| version | recall | precision | specificity | search@k | dedup | aggregate |
|---|---|---|---|---|---|---|
| v1 | 1.00 | 0.94 | 1.00 | 1.00 | 1.00 | **0.99** |

## Per-fixture (version v1)

| fixture | recall | precision | spec | search | new-on-redup | aggregate |
|---|---|---|---|---|---|---|
| decision | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| health-fact | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| meaning-preservation-direction | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| meaning-preservation-negation | 1.00 | 0.67 | 1.00 | 1.00 | 0 | 0.93 |
| multi-speaker | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| multi-topic-three | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| no-fabrication | 1.00 | 1.00 | — | 1.00 | 0 | 1.00 |
| nothing-to-extract | 1.00 | 1.00 | — | — | 0 | 1.00 |
| plans-and-dates | 1.00 | 0.67 | 1.00 | 1.00 | 0 | 0.93 |
| preferences | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| promotion-and-pet | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| pronoun-resolution | 1.00 | 0.67 | 1.00 | 1.00 | 0 | 0.93 |
| question-not-fact | 1.00 | 1.00 | — | — | 0 | 1.00 |
| relationships | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| relative-date-grounding | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| single-fact | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| specificity-numbers | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| specificity-product | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
