#!/bin/bash
# Export DE+EN-v4 English test pivots and run NLLB translation (Stage 2).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

SPLIT="${1:-test}"
TARGET_LANG="${2:-fra_Latn}"
MODEL_DIR="${3:-sign_sample_model_de_en_v4_langhead}"

DATA_DIR="experiments/stage2_multilingual/data"
OUT_DIR="experiments/stage2_multilingual/outputs"

PIVOT_TXT="${DATA_DIR}/pivot_en_${SPLIT}.txt"
IDS_TXT="${DATA_DIR}/pivot_en_${SPLIT}_ids.txt"
# e.g. fra_Latn -> predictions_fr_test.txt
case "$TARGET_LANG" in
  fra_Latn) LANG_SHORT="fr" ;;
  spa_Latn) LANG_SHORT="es" ;;
  deu_Latn) LANG_SHORT="de" ;;
  arb_Arab) LANG_SHORT="ar" ;;
  *) LANG_SHORT="${TARGET_LANG%%_*}" ;;
esac
PRED_TXT="${OUT_DIR}/predictions_${LANG_SHORT}_${SPLIT}.txt"

echo "=== Stage 2 full cascade ==="
echo "Split:        ${SPLIT}"
echo "Target lang:  ${TARGET_LANG}"
echo "Model dir:    ${MODEL_DIR}"
echo ""

python scripts/export_pivot_for_nllb.py \
  --model-dir "$MODEL_DIR" \
  --split "$SPLIT" \
  --lang en \
  --output "$PIVOT_TXT" \
  --ids-output "$IDS_TXT"

bash experiments/stage2_multilingual/scripts/01_translate_from_pivot.sh \
  "$PIVOT_TXT" \
  "$PRED_TXT" \
  "$TARGET_LANG"

echo ""
echo "Done."
echo "  Pivots:       ${PIVOT_TXT}"
echo "  Sample IDs:   ${IDS_TXT}"
echo "  Translations: ${PRED_TXT}"
