#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

# Optional venv activation (recommended)
if [ -f "venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

# Configure paths (override by exporting env vars before running)
RELEASE_DIR="${RELEASE_DIR:-$ROOT_DIR/../PHOENIX-2014-T-release-v3}"
INPUT_DIR="${INPUT_DIR:-$ROOT_DIR/data/PHOENIX2014T}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/data/PHOENIX2014T_ENGLISH}"

if [ ! -d "$RELEASE_DIR" ]; then
  echo "ERROR: RELEASE_DIR not found: $RELEASE_DIR"
  echo "Set it with: export RELEASE_DIR=/path/to/PHOENIX-2014-T-release-v3"
  exit 1
fi

python -u create_english_dataset.py \
  --from-release-v3 \
  --release-dir "$RELEASE_DIR" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --release-gloss-col orth_english \
  --release-text-col translation_english \
  --splits train dev test

echo "English dataset created under data/PHOENIX2014T_ENGLISH"
