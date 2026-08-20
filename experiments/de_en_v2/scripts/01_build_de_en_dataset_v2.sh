#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

if [ -f "venv/bin/python" ]; then
  VENV_PYTHON="venv/bin/python"
else
  VENV_PYTHON="python"
fi

"$VENV_PYTHON" experiments/de_en_v2/scripts/build_de_en_dataset_v2.py \
  --de-dir data/PHOENIX2014T \
  --en-dir data/PHOENIX2014T_ENGLISH \
  --out-dir data/PHOENIX2014T_DE_EN_V2 \
  --splits train dev test
