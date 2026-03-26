#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

# Optional venv activation (recommended)
if [ -f "venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

python -m signjoey train experiments/english_v4/configs/sign_improved_v4_english.yaml
