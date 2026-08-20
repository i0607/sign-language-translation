#!/usr/bin/env python3
"""
Export English (or German) pivot sentences from signjoey test/dev hypothesis files
for Stage-2 NLLB translation. One sentence per line, language tags stripped.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
TAG_RE = re.compile(r"^<(de|en)>\s*", re.IGNORECASE)


def strip_tag(text: str) -> str:
    return TAG_RE.sub("", (text or "").strip())


def parse_hyp_file(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        name, hyp = line.split("|", 1)
        out[name.strip()] = hyp.strip()
    return out


def find_best_txt(model_dir: Path, split: str) -> Path:
    candidates = sorted(model_dir.glob(f"best*.BW_*.A_*.{split}.txt"))
    if not candidates:
        candidates = sorted(model_dir.glob(f"*.BW_*.A_*.{split}.txt"))
    if not candidates:
        raise FileNotFoundError(f"No {split} .txt hypotheses in {model_dir}")
    return candidates[-1]


def export_pivots(
    hyp_path: Path,
    lang: str,
    with_ids: bool,
) -> Tuple[List[str], List[str]]:
    hyps = parse_hyp_file(hyp_path)
    suffix = f"__{lang.lower()}"
    names = sorted(n for n in hyps if n.endswith(suffix))
    if not names:
        raise ValueError(f"No samples ending with {suffix!r} in {hyp_path}")

    lines: List[str] = []
    ids: List[str] = []
    for name in names:
        ids.append(name)
        lines.append(strip_tag(hyps[name]))
    return lines, ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Export pivot text for NLLB Stage 2.")
    parser.add_argument(
        "--model-dir",
        default="sign_sample_model_de_en_v4_langhead",
        help="Model directory with best.*.{split}.txt",
    )
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument(
        "--lang",
        choices=("en", "de"),
        default="en",
        help="Pivot language: en (__en samples) or de (__de samples)",
    )
    parser.add_argument(
        "--output",
        default="experiments/stage2_multilingual/data/pivot_en_test.txt",
        help="Output text file (one sentence per line)",
    )
    parser.add_argument(
        "--ids-output",
        default="",
        help="Optional: save sample names (one per line), aligned with output",
    )
    args = parser.parse_args()

    model_dir = ROOT / args.model_dir
    hyp_path = find_best_txt(model_dir, args.split)
    lines, ids = export_pivots(hyp_path, args.lang, bool(args.ids_output))

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    if args.ids_output:
        ids_path = ROOT / args.ids_output
        ids_path.parent.mkdir(parents=True, exist_ok=True)
        ids_path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
        print(f"Wrote {len(ids)} ids to {ids_path}")

    print(f"Source: {hyp_path.relative_to(ROOT)}")
    print(f"Language: {args.lang} ({args.split})")
    print(f"Wrote {len(lines)} pivot lines to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
