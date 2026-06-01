# mneme eval results

model: `openai/gpt-4o-mini`  ·  embedder: `text-embedding-3-small` (OpenRouter)  ·  search k: 3

> Baseline for `extractionPromptV1`, recorded 2026-06-01. Regenerate with
> `go run ./cmd/eval -embedder openai -out eval/RESULTS.md`. Every later prompt
> version must not regress the aggregate.

| version | recall | precision | specificity | search@k | dedup | aggregate |
|---|---|---|---|---|---|---|
| v1 | 0.94 | 0.97 | 1.00 | 0.94 | 1.00 | **0.97** |

## Per-fixture (version v1)

| fixture | recall | precision | spec | search | new-on-redup | aggregate |
|---|---|---|---|---|---|---|
| decision | 0.50 | 1.00 | 1.00 | 1.00 | 0 | 0.90 |
| health-fact | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| meaning-preservation-direction | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| meaning-preservation-negation | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| multi-speaker | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| multi-topic-three | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| no-fabrication | 1.00 | 1.00 | — | 1.00 | 0 | 1.00 |
| nothing-to-extract | 1.00 | 1.00 | — | — | 0 | 1.00 |
| plans-and-dates | 1.00 | 1.00 | 1.00 | 0.00 | 0 | 0.80 |
| preferences | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| promotion-and-pet | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| pronoun-resolution | 0.50 | 0.50 | 1.00 | 1.00 | 0 | 0.80 |
| question-not-fact | 1.00 | 1.00 | — | — | 0 | 1.00 |
| relationships | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| relative-date-grounding | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| single-fact | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| specificity-numbers | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
| specificity-product | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 |
