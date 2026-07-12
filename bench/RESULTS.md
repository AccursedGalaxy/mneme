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

## Stage decomposition: where the score is actually lost

Two control runs, neither of which the matrix had ever included. **rawturns** stores the conversation verbatim and never calls an extraction model. **oracle/source** answers from the dataset's own evidence turns, with no store and no retrieval at all, so it bounds what any memory system can score with this answerer and judge.

| run | answerable Judge | paired Δ vs baseline [95% CI] | abstains on answerable |
|---|---|---|---|
| additive · 3-small (baseline) | 0.34 | — | 40.0% |
| **rawturns** (no extraction at all) | **0.39** | **+0.047 [+0.010, +0.092]** | 37.7% |
| additive · extract=flash | 0.40 | +0.057 [+0.024, +0.083] | 39.6% |
| **oracle · source** (gold evidence turns) | **0.54** | **+0.204 [+0.173, +0.234]** | **22.9%** |

Two results here, and both are load-bearing.

**Storing the raw conversation beats the fact-extraction pipeline.** rawturns scores +0.047 over additive with an interval that excludes zero, and it does that with *zero* extraction LLM calls: it is cheaper than the thing it beats. Single-hop, the largest category, goes 0.43 → 0.54. On aggregate it is statistically indistinguishable from the flash-extraction run, which costs 5× per fact. So extraction, as currently implemented, is not earning its cost against simply keeping the text.

**The answer model abstains on 22.9% of answerable questions even when handed the gold evidence turns.** That is the finding that reframes everything below. The 40% abstention rate was read as an extraction-recall failure, and the roadmap was pointed at extraction on that basis. But more than half of it survives perfect retrieval. Of the 40 points, roughly 23 belong to the answerer, its prompt, or the judge, and only ~17 are attributable to memory at all.

The ceiling that follows: **0.54 answerable is the most this answer model and judge can score on LoCoMo even with perfect memory.** Memory work has 0.20 of headroom between the baseline and that ceiling. The remaining 0.46 sits in the answer prompt, the answer model, and the judge, and no amount of extraction or retrieval work will touch it.

One caveat against over-reading the oracle as a hard ceiling: it is *not* an upper bound per category. Flash extraction beats it on temporal (0.49 vs 0.31), because extraction normalizes relative dates ("last Tuesday") into absolute ones, information the raw evidence turns do not carry. Extraction can add derived value, not just lose it. That is the strongest argument left for keeping a fact pipeline at all.

## What this settles (measured, not guessed)

1. **The system abstains correctly: adversarial 0.96.** LoCoMo's adversarial questions are unanswerable traps, and mneme declines them almost every time. The answer prompt's explicit "say I don't know" instruction is doing its job. Worth keeping, but it is not a lever, and averaging it into a headline meant to guide work only obscures the rows that differ.

2. **mneme abstains on 40% of the questions that *do* have answers — but memory owns only about 17 points of that.** On the 1540 answerable questions the baseline says "I don't know" to 616 of them. The obvious reading, which this document asserted before the oracle run existed, was that the answering fact was never extracted. The source oracle refutes it: hand the answer model the exact gold evidence turns and it still abstains on 22.9%. Most of the abstention is downstream of memory, in the answer prompt, the answer model, or the judge. Only the difference (roughly 17 points) is a memory problem. Mined from `bench/results/*.jsonl` at no API cost.

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

The stage decomposition reorders the work, and not in the direction this document previously argued.

1. **The answer stage, not extraction.** 23 of the 40 abstention points survive gold evidence. That is the largest single pool of recoverable score on the board, it is a prompt-and-model problem rather than a memory problem, and nobody has looked at it. Start by reading the oracle's abstentions in `bench/results/oracle_source.jsonl`: they are cases where the model held the answer and declined anyway.
2. **A hybrid store, not a better extractor.** Raw turns beat extracted facts on aggregate, for free, while extraction wins temporal by normalizing dates. Those are complementary, which is the Phase 1 hybrid store: keep the episodes *and* the derived facts, and retrieve over both.
3. **Extraction recall on implied and causal facts (tk #11)** drops down the list. It is real work, but it is chasing a smaller pool than #1, and the raw-turn result suggests the fact representation itself is the weaker part of the design.
4. **open_domain (0.07–0.16) is the weakest category everywhere,** including the oracle, which manages only 0.16. A category the ceiling itself cannot score is a question-format or judge problem, not a memory one. Worth 20 minutes with the dumps before anyone builds anything for it.

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
