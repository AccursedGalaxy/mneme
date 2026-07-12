#!/usr/bin/env bash
# run_matrix.sh — run the v2 lever matrix on the full LoCoMo set, one config per
# run, holding the LLM constant (gemini-2.5-flash-lite) so each row is a clean
# A/B against the additive/3-small baseline. Writes one results file per run
# under bench/results/ plus a combined SUMMARY.md. Sequential on purpose:
# parallel runs would compete for the same OpenRouter rate budget and muddy the
# comparison. Re-run a single row by copying its line out.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a

# ORACLE answers + judges every row (held constant so rows stay comparable, and
# comparable to the pre-fix matrix). MODEL is the baseline extraction model;
# the last row swaps it for flash, the one lever measured to move the number.
ORACLE="google/gemini-2.5-flash-lite"
MODEL="google/gemini-2.5-flash-lite"
FLASH="google/gemini-2.5-flash"
DATA="bench/data/locomo10.json"
OUT="bench/results"
mkdir -p "$OUT"
LOG="$OUT/matrix.log"
: > "$LOG"

# run <name> <embed-model> <extraction-model> <bench-flags...>
# Answer + judge are pinned to $ORACLE on every row, so a row that varies the
# extraction model isolates "which model writes facts to the store" — same
# protocol as the extraction A/B. Each run also drops a .jsonl of every
# prediction beside its .md (cmd/bench derives it from -out): a future scoring
# change re-scores those files for free instead of re-running the matrix.
#
# A single run failing (e.g. an endpoint outage the retries can't ride out) is
# logged but does NOT abort the matrix — the other rows still produce data.
run() {
  local name="$1"; local embed="$2"; local extract="$3"; shift 3
  echo "=== [$(date +%H:%M:%S)] START $name (extract=$extract embed=$embed) ===" | tee -a "$LOG"
  # shellcheck disable=SC2068
  if MNEME_EMBED_MODEL="$embed" go run ./cmd/bench -dataset locomo -path "$DATA" \
     -model "$extract" -answer-model "$ORACLE" -judge-model "$ORACLE" \
     -k 5 -concurrency 8 -out "$OUT/$name.md" $@ >>"$LOG" 2>&1; then
    echo "=== [$(date +%H:%M:%S)] DONE  $name ===" | tee -a "$LOG"
  else
    echo "=== [$(date +%H:%M:%S)] FAILED $name (continuing) ===" | tee -a "$LOG"
  fi
}

# 1) additive baseline (task 7 headline) — text-embedding-3-small
run additive_3small     "text-embedding-3-small"  "$MODEL" -strategy additive

# 2) consolidate (task 7 — consolidation default decision)
run consolidate_3small  "text-embedding-3-small"  "$MODEL" -strategy consolidate

# 3) embedding A/B: text-embedding-3-large (task 9)
run additive_3large     "text-embedding-3-large"  "$MODEL" -strategy additive

# 4) retrieval boosters: rerank + multi-query (task 1)
run additive_boosters   "text-embedding-3-small"  "$MODEL" -strategy additive -rerank -multiquery 3

# 5) extraction lever: gemini-2.5-flash writes the facts (the headline config).
run extract_flash       "text-embedding-3-small"  "$FLASH" -strategy additive

# Dropped: gemini-embedding-2-preview. It failed the last matrix on the free
# tier's 1000 embeddings/day quota and would burn ~11 min failing again; a full
# LoCoMo run needs far more than 1000 embeds. Restore the row behind BYOK.

echo "=== [$(date +%H:%M:%S)] MATRIX COMPLETE ===" | tee -a "$LOG"
