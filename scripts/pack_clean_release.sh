#!/usr/bin/env bash
# Pack a clean, shareable source tree (no checkpoints, data, or venvs).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(cd "$ROOT/.." && pwd)"
STAMP="$(date +%Y%m%d)"
ARCHIVE="${OUT_DIR}/SLTR-clean-release-${STAMP}.tar.gz"

cd "$ROOT"

tar -czf "$ARCHIVE" \
  --exclude='.git' \
  --exclude='venv' \
  --exclude='venv-stage2' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='sign_sample_model*' \
  --exclude='data/PHOENIX*' \
  --exclude='docs/paper_draft' \
  --exclude='*.ckpt' \
  --exclude='*.pdf' \
  --exclude='*.docx' \
  --exclude='*.numbers' \
  --exclude='.DS_Store' \
  --exclude='.pptx_deps' \
  --exclude='scripts/build_defense_slides.py' \
  README.md \
  REPRODUCE.md \
  LICENSE \
  requirements.txt \
  requirements-stage2.txt \
  .gitignore \
  configs \
  signjoey \
  scripts \
  data/.gitkeep \
  data/README.md \
  experiments/de_en_v2/scripts \
  experiments/de_en_v2/configs \
  experiments/de_en_v2/README.md \
  experiments/de_en_v3/scripts \
  experiments/de_en_v3/configs \
  experiments/de_en_v3/README.md \
  experiments/de_en_v4_langhead/scripts \
  experiments/de_en_v4_langhead/configs \
  experiments/de_en_v4_langhead/README.md \
  experiments/english_v4/README.md \
  experiments/stage2_multilingual/README.md \
  experiments/stage2_multilingual/scripts \
  experiments/stage2_multilingual/data/.gitkeep \
  experiments/stage2_multilingual/outputs/.gitkeep \
  2>/dev/null || true

# Fallback: pack known paths only if some optional dirs are missing
if [ ! -f "$ARCHIVE" ] || [ ! -s "$ARCHIVE" ]; then
  tar -czf "$ARCHIVE" \
    README.md REPRODUCE.md LICENSE requirements.txt requirements-stage2.txt .gitignore \
    configs signjoey scripts data
fi

echo "Wrote $ARCHIVE"
ls -lh "$ARCHIVE"
