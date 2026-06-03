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

MODEL="google/gemini-2.5-flash-lite"
DATA="bench/data/locomo10.json"
OUT="bench/results"
mkdir -p "$OUT"
LOG="$OUT/matrix.log"
: > "$LOG"

# run <name> <embed-model> <bench-flags...>
# A single run failing (e.g. an endpoint outage the retries can't ride out) is
# logged but does NOT abort the matrix — the other rows still produce data.
run() {
  local name="$1"; local embed="$2"; shift 2
  echo "=== [$(date +%H:%M:%S)] START $name (embed=$embed) ===" | tee -a "$LOG"
  # shellcheck disable=SC2068
  if MNEME_EMBED_MODEL="$embed" go run ./cmd/bench -dataset locomo -path "$DATA" \
     -model "$MODEL" -k 5 -concurrency 8 -out "$OUT/$name.md" $@ >>"$LOG" 2>&1; then
    echo "=== [$(date +%H:%M:%S)] DONE  $name ===" | tee -a "$LOG"
  else
    echo "=== [$(date +%H:%M:%S)] FAILED $name (continuing) ===" | tee -a "$LOG"
  fi
}

# 1) additive baseline (task 7 headline) — text-embedding-3-small
run additive_3small     "text-embedding-3-small"               -strategy additive

# 2) consolidate (task 7 — consolidation default decision)
run consolidate_3small  "text-embedding-3-small"               -strategy consolidate

# 3) embedding A/B: text-embedding-3-large (task 9)
run additive_3large     "text-embedding-3-large"               -strategy additive

# 4) embedding A/B: gemini-embedding-2-preview, current MTEB leader (task 9)
run additive_geminiembed "google/gemini-embedding-2-preview"   -strategy additive

# 5) retrieval boosters: rerank + multi-query (task 1)
run additive_boosters   "text-embedding-3-small"               -strategy additive -rerank -multiquery 3

echo "=== [$(date +%H:%M:%S)] MATRIX COMPLETE ===" | tee -a "$LOG"
