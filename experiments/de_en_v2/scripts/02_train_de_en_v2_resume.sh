#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

if [ -f "venv/bin/python" ]; then
  VENV_PYTHON="venv/bin/python"
else
  VENV_PYTHON="python"
fi

"$VENV_PYTHON" -m signjoey train experiments/de_en_v2/configs/sign_de_en_v2_resume.yaml
