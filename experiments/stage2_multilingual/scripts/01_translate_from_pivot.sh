#!/bin/bash
set -euo pipefail

if [ $# -lt 3 ]; then
  echo "Usage: $0 <input_txt> <output_txt> <target_lang> [source_lang]"
  echo "Example: $0 input.txt output_fr.txt fra_Latn eng_Latn"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

if [ -f "venv-stage2/bin/activate" ]; then
  # shellcheck disable=SC1091
  source venv-stage2/bin/activate
elif [ -f "venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

INPUT_TXT="$1"
OUTPUT_TXT="$2"
TARGET_LANG="$3"
SOURCE_LANG="${4:-eng_Latn}"

python experiments/stage2_multilingual/scripts/translate_pivot_text.py \
  --input "$INPUT_TXT" \
  --output "$OUTPUT_TXT" \
  --target-lang "$TARGET_LANG" \
  --source-lang "$SOURCE_LANG"
