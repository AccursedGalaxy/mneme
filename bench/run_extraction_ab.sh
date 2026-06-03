#!/usr/bin/env bash
# run_extraction_ab.sh — isolate the EXTRACTION model. Everything else is held
# constant: embedder=text-embedding-3-small, additive, k=5, and (for bench) the
# answer + judge models are pinned to gemini-2.5-flash-lite so the only variable
# is which model writes facts to the store. Two views:
#   eval/  — direct extraction recall/precision on 18 fixtures (judge pinned to
#            gemini-2.5-flash, a fixed oracle stronger than the models under test)
#   bench/ — end-to-end LoCoMo Judge (does better extraction => better QA?)
# Sequential, resilient (one failure logs and continues), telegram on completion.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a

DATA="bench/data/locomo10.json"
EMBED="text-embedding-3-small"
ANSWER="google/gemini-2.5-flash-lite"   # bench answer model, pinned
EVAL_JUDGE="google/gemini-2.5-flash"    # eval oracle, pinned (stronger, cheap on 18 fixtures)
EO="eval/results"; BO="bench/results"; mkdir -p "$EO" "$BO"
LOG="$BO/extraction_ab.log"; : > "$LOG"

# Extraction models under test.
MODELS=( "google/gemini-2.5-flash-lite" "google/gemini-2.5-flash" "openai/gpt-5-mini" "deepseek/deepseek-v3.2" )

slug() { echo "$1" | tr '/' '_'; }

echo "=== [$(date +%H:%M:%S)] EVAL phase (extraction recall/precision) ===" | tee -a "$LOG"
for m in "${MODELS[@]}"; do
  s=$(slug "$m")
  echo "--- [$(date +%H:%M:%S)] eval $m ---" | tee -a "$LOG"
  if MNEME_EMBED_MODEL="$EMBED" go run ./cmd/eval -embedder openai -k 5 \
       -model "$m" -judge-model "$EVAL_JUDGE" -out "$EO/extract_$s.md" >>"$LOG" 2>&1; then
    echo "    eval $m OK" | tee -a "$LOG"
  else
    echo "    eval $m FAILED (continuing)" | tee -a "$LOG"
  fi
done

echo "=== [$(date +%H:%M:%S)] BENCH phase (end-to-end LoCoMo) ===" | tee -a "$LOG"
# flash-lite extraction == existing bench/results/additive_3small.md, skip it.
for m in "google/gemini-2.5-flash" "openai/gpt-5-mini" "deepseek/deepseek-v3.2"; do
  s=$(slug "$m")
  echo "--- [$(date +%H:%M:%S)] bench extract=$m (answer+judge=$ANSWER) ---" | tee -a "$LOG"
  if MNEME_EMBED_MODEL="$EMBED" go run ./cmd/bench -dataset locomo -path "$DATA" -k 5 -concurrency 8 \
       -strategy additive -model "$m" -answer-model "$ANSWER" -judge-model "$ANSWER" \
       -out "$BO/extract_$s.md" >>"$LOG" 2>&1; then
    echo "    bench $m OK" | tee -a "$LOG"
  else
    echo "    bench $m FAILED (continuing)" | tee -a "$LOG"
  fi
done

echo "=== [$(date +%H:%M:%S)] EXTRACTION_AB COMPLETE ===" | tee -a "$LOG"
python3 ~/telegram-bridge/tgctl.py notify "✅ mneme extraction-model A/B done! eval + bench results ready." --label mneme-bench 2>/dev/null || true
