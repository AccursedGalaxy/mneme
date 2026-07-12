# mneme bench results — locomo (full distribution)

model: `google/gemini-2.5-flash-lite` (answer + judge, pinned on every row)  ·  dataset: `bench/data/locomo10.json`  ·  k: 5  ·  **full set, 1986 questions, no `-maxq`**

Metrics: **EM** = normalized exact match, **F1** = token-overlap F1, **Judge** = LLM semantic match (the metric we steer on). n = question count. Per-run files live in `bench/results/`, each with a `.jsonl` beside it holding every question's gold, retrieved facts, and prediction.

Re-run 2026-07-12 under the corrected adversarial scoring. These numbers supersede everything published before that date; see [Scoring history](#scoring-history). Answer and judge are pinned to flash-lite on every row, so a row that changes the extraction model isolates the model that writes facts to the store.

## Lever matrix — Judge by category

| run | single (841) | multi (282) | temporal (321) | open (96) | adversarial (446) | **overall (1986)** |
|---|---|---|---|---|---|---|
| **additive · 3-small** (baseline) | 0.43 | 0.29 | 0.24 | 0.09 | 0.96 | **0.48** |
| consolidate · 3-small | 0.39 | 0.26 | 0.23 | 0.08 | 0.94 | **0.45** ↓ |
| additive · 3-large | 0.44 | 0.34 | 0.27 | 0.09 | 0.94 | **0.49** ≈ |
| **additive · extract=flash** | 0.43 | 0.28 | **0.49** | 0.14 | 0.94 | **0.52** ↑ |
| additive · boosters (rerank+mq3) | 0.44 | **0.36** | 0.27 | 0.11 | 0.94 | **0.50** ↑ |

Baseline EM/F1 overall: 0.33 / 0.46.

The adversarial column is an abstention check: 446 questions whose gold behavior is to decline. It sits at about 0.95 on every row and does not separate them. Since those questions are 22% of the set, they lift every overall number by roughly the same amount, which makes the overall column a poor place to look for levers. Read the answerable-only column instead:

| run | **answerable Judge (1540)** | paired Δ vs baseline [95% CI] | verdict |
|---|---|---|---|
| additive · 3-small (baseline) | **0.34** | — | — |
| consolidate · 3-small | 0.31 | −0.029 [−0.048, −0.010] | real regression |
| additive · 3-large | 0.36 | +0.023 [−0.001, +0.044] | not resolvable |
| additive · boosters (rerank+mq3) | 0.37 | +0.028 [−0.003, +0.056] | not resolvable |
| **additive · extract=flash** | **0.40** | **+0.057 [+0.026, +0.084]** | **real** |

Intervals are 10k bootstrap resamples over conversations (the unit of independence: the ~200 questions about one conversation all share its facts), paired so per-conversation difficulty cancels. Reproduce with `go run ./cmd/rescore -baseline bench/results/additive_3small.jsonl bench/results/*.jsonl`, which reads the dumps and costs nothing.

**Only two effects are resolvable at n=10 conversations.** Flash extraction helps; consolidation hurts. The bigger embedder and the retrieval boosters both trend positive by about the same amount and neither interval excludes zero. Do not read their point estimates as a ranking, and do not ship either on this evidence. Getting these to resolve needs more conversations, not more matrix rows.

## What this settles (measured, not guessed)

1. **The system abstains correctly: adversarial 0.96.** LoCoMo's adversarial questions are unanswerable traps, and mneme declines them almost every time. The answer prompt's explicit "say I don't know" instruction is doing its job. Worth keeping, but it is not a lever, and averaging it into a headline meant to guide work only obscures the rows that differ.

2. **The bottleneck: mneme abstains on 40% of the questions that *do* have answers.** On the 1540 answerable questions, the baseline says "I don't know" to 616 of them. Retrieval is not misranking those facts and the answerer is not fumbling the phrasing. The facts simply are not in the store, so there is nothing to retrieve and declining is the correct move. Mined from `bench/results/additive_3small.jsonl`.

3. **Consolidation v1 is still a net loss: 0.34 → 0.31 answerable.** The dumps say why. It abstains on 47% of answerable questions where the baseline abstains on 40%, which means it is deleting facts the answerer would otherwise have used. Decision: `additive` stays the default. Consolidation remains available as an opt-in for callers who need a current value more than they need recall, and the README says so in those terms. It is not a path to a better score, and the v2 prompt would have to recover that lost recall before it becomes one.

4. **A stronger embedder does not pay: 3-large 0.34 → 0.36 answerable** (+0.02, with multi-hop +0.05). Real but small, against roughly 6.5× the embedding cost, and worth nothing on the 40% of questions where the fact was never stored to begin with. Retrieval quality is not where the number is lost, so Voyage and Cohere are not worth wiring up for this workload either.

5. **Extraction is the biggest single lever: flash 0.34 → 0.40 answerable, 0.48 → 0.52 overall.** Almost all of the gain is temporal, 0.24 → 0.49, where flash preserves dated facts that flash-lite mangles. That matches its eval scores (specificity 0.93 → 1.00, dedup 0.81 → 1.00). It does not fix the abstention rate, which stays at 39.6%: flash makes the facts it extracts more faithful without extracting more of them.

6. **Retrieval boosters were written off too early, and the two levers act on different categories.** The old harness scored rerank + multi-query=3 as flat (0.23 → 0.23) and we published "the LLM-as-reranker does not earn its cost." The per-category picture is more interesting than either verdict. Boosters lift multi-hop by +0.070 [+0.005, +0.135] and do nothing for temporal (+0.033, interval spans zero). Flash extraction is the mirror image: temporal +0.252 [+0.192, +0.306], multi-hop −0.007. They are orthogonal, and **nobody has run them together.**

   Two honest caveats. The boosters' multi-hop interval only barely excludes zero, and it is one of several categories examined, so it deserves the skepticism that any barely-significant subgroup result does. And their aggregate effect stays unresolvable, because a real gain on 282 multi-hop questions dilutes across 1540. Boosters cost roughly 2× per query, so the question is whether that multi-hop gain holds up when it is paid for.

**Recommendation: use `gemini-2.5-flash` for extraction, not flash-lite.** Roughly 5× the per-fact cost ($0.0017 vs $0.00034) for +0.057 [+0.026, +0.084] answerable Judge, most of it temporal. Boosters stay opt-in and undefaulted, pending a combined run and more conversations.

## Extraction-model A/B (eval fixtures)

Extraction models scored against the 18-fixture eval, oracle pinned to `gemini-2.5-flash`. This is what pointed at extraction before the bench confirmed it.

| extraction model | recall | precision | specificity | search@k | dedup | aggregate | $/extract |
|---|---|---|---|---|---|---|---|
| gemini-2.5-flash-lite (baseline) | 0.94 | 1.00 | 0.93 | 0.94 | 0.81 | 0.93 | $0.00034 |
| **gemini-2.5-flash** | 1.00 | 0.94 | 1.00 | 1.00 | 1.00 | **0.99** | $0.0017 |
| gpt-5-mini | 0.94 | 0.83 | 1.00 | 1.00 | 1.00 | 0.94 | $0.0029 |
| deepseek-v3.2 | 1.00 | 0.89 | 1.00 | 1.00 | 0.88 | 0.92 | $0.0021 |

Two models were ruled out on the bench and are not worth revisiting. gpt-5-mini blew the 2h ingest timeout (reasoning tokens make extraction far too slow), costs more, and over-extracts (precision 0.83). deepseek-v3.2 scores below baseline on eval and died mid-bench on a provider error.

## Where the signal points next

The bottleneck sits upstream of retrieval, and finding #2 puts a number on it: 616 answerable questions where mneme has nothing to say. No reranker can surface a fact that was never stored, which is why extraction recall is the headline work. But finding #6 is a warning against over-reading that: reranking does help the questions whose facts *are* in the store, and we nearly discarded it on a scoring artifact.

- **Extraction recall on implied and causal facts (tk #11).** The additive extractor writes down what was stated. Many LoCoMo golds are realizations, causes, and attributes that a speaker implied rather than asserted, and those never enter the store.
- **Temporal** is the second-largest gap now (0.49 with flash, up from 0.24), though no longer the worst.
- **open_domain (0.09 to 0.14) is the weakest category** and nobody has looked at it. Whether that is an extraction gap or an answer-prompt gap is a question the dumps can settle without a re-run.

## Scoring history

**2026-07-12: adversarial scoring fixed, matrix re-run.** Until this date the harness graded the adversarial category (446 of 1986 questions) against LoCoMo's `adversarial_answer` field, which is the trap distractor rather than a gold answer. The correct behavior for that category is abstention, so a system that abstained correctly scored 0. Every number published before this date understated performance. The baseline read 0.23 overall when it was really 0.48, and the adversarial column read 0.02 when it was really 0.96.

Most rankings survived the fix. Consolidation was a loss before and remains one, the bigger embedder was flat and remains flat, flash extraction won and still wins, though its edge narrows from +0.07 to +0.04 overall.

Two published readings did not survive. The claim that the flat adversarial column exposed an "inferential-fact capture" gap was reading a harness bug as a finding. And the retrieval boosters, written off as flat and "not earning their cost," actually lift multi-hop from 0.29 to 0.36 (finding #6). A category worth 22% of the set, pinned near zero for every row, was hiding real differences between rows.

Every run now dumps per-question predictions to `bench/results/<run>.jsonl` alongside its markdown, which is the direct lesson of this erratum. The LLM calls are the expensive part of a run, the aggregate table throws them away, and so a scoring bug cost a full paid re-run of the matrix. The next scoring change will be a local re-score instead. Finding #2 above came out of those files at zero API cost.

## Reproduce

```sh
# one run; predictions are dumped to bench/results/<run>.jsonl automatically
go run ./cmd/bench -dataset locomo -path bench/data/locomo10.json -k 5 -concurrency 8 \
  -strategy additive -model google/gemini-2.5-flash \
  -answer-model google/gemini-2.5-flash-lite -judge-model google/gemini-2.5-flash-lite \
  -out bench/results/extract_flash.md

# full matrix (~2.5h, ~$2)
bash bench/run_matrix.sh
```
