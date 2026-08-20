#!/bin/bash
# Supervisor paper tasks: COMET-Kiwi, Arabic cascade, error analysis, bootstrap CI, human-eval sample.
# Run from repo root: bash experiments/stage2_multilingual/scripts/03_paper_evaluation_tasks.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

LOG="experiments/stage2_multilingual/outputs/paper_evaluation_run.log"
mkdir -p experiments/stage2_multilingual/outputs

exec > >(tee -a "$LOG") 2>&1
echo "=== Paper evaluation tasks started: $(date) ==="

if [ -f "venv-stage2/bin/activate" ]; then
  # shellcheck disable=SC1091
  source venv-stage2/bin/activate
elif [ -f "venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

PIVOT="experiments/stage2_multilingual/data/pivot_en_test.txt"
IDS="experiments/stage2_multilingual/data/pivot_en_test_ids.txt"
FR_OUT="experiments/stage2_multilingual/outputs/predictions_fra_test.txt"
AR_OUT="experiments/stage2_multilingual/outputs/predictions_ar_test.txt"
RESULTS="experiments/stage2_multilingual/outputs"

# --- Task 1a: Arabic (pivots already exist for FR) ---
if [ ! -f "$FR_OUT" ]; then
  echo "ERROR: Missing $FR_OUT — run 02_run_cascade_full.sh first"
  exit 1
fi

echo "--- Task 1: Arabic NLLB translation (642 lines) ---"
bash experiments/stage2_multilingual/scripts/01_translate_from_pivot.sh \
  "$PIVOT" "$AR_OUT" arb_Arab eng_Latn

echo "--- Task 1: COMET-Kiwi French ---"
python experiments/stage2_multilingual/scripts/score_comet_kiwi.py \
  --source "$PIVOT" \
  --hypothesis "$FR_OUT" \
  --label "EN_to_FR_test" \
  --output-json "$RESULTS/comet_kiwi_fr_test.json"

echo "--- Task 1: COMET-Kiwi Arabic ---"
python experiments/stage2_multilingual/scripts/score_comet_kiwi.py \
  --source "$PIVOT" \
  --hypothesis "$AR_OUT" \
  --label "EN_to_AR_test" \
  --output-json "$RESULTS/comet_kiwi_ar_test.json"

# --- Task 3: Error analysis ---
echo "--- Task 3: Cascade error analysis ---"
python experiments/stage2_multilingual/scripts/analyze_cascade_errors.py \
  --source "$PIVOT" --hypothesis "$FR_OUT" --target fr \
  --label "FR_cascade_test" \
  --output-json "$RESULTS/cascade_error_analysis_fr.json" \
  --output-csv "$RESULTS/cascade_error_flagged_fr.csv"

python experiments/stage2_multilingual/scripts/analyze_cascade_errors.py \
  --source "$PIVOT" --hypothesis "$AR_OUT" --target ar \
  --label "AR_cascade_test" \
  --output-json "$RESULTS/cascade_error_analysis_ar.json" \
  --output-csv "$RESULTS/cascade_error_flagged_ar.csv"

# --- Task 2: Human eval sample ---
echo "--- Task 2: Sample 20 for human rating ---"
python experiments/stage2_multilingual/scripts/sample_human_eval.py \
  --source "$PIVOT" --hypothesis "$FR_OUT" --ids "$IDS" \
  --n 20 --seed 42 \
  --output-csv "$RESULTS/human_eval_sample_fr_20.csv" \
  --target-lang French

# --- Task 4: Bootstrap CI (main models) ---
echo "--- Task 4: Bootstrap 95% CI ---"
python scripts/bootstrap_metrics_ci.py \
  --n-bootstrap 1000 --seed 42

echo "=== Done: $(date) ==="
echo "Log: $LOG"
