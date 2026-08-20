#!/usr/bin/env python3
"""Bootstrap 95% CI for corpus BLEU-4 and WER on PHOENIX test hypotheses."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from signjoey.metrics import bleu, wer_list  # noqa: E402
TAG_RE = re.compile(r"^<(de|en)>\s*", re.IGNORECASE)

MODELS = [
    ("Model-ODE", "sign_sample_model_improved_v4", "data/PHOENIX2014T/phoenix14t.pami0.test"),
    ("Model-OEN", "sign_sample_model_improved_v4_english", "data/PHOENIX2014T_ENGLISH/phoenix14t_english.test"),
    ("Model-A", "sign_sample_model_improved_v5", "data/PHOENIX2014T/phoenix14t.pami0.test"),
    ("DE+EN-v4", "sign_sample_model_de_en_v4_langhead", "data/PHOENIX2014T_DE_EN_V2/phoenix14t_de_en_v2.test"),
]


def load_pickle(path: Path) -> List[dict]:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def strip_tag(s: str) -> str:
    return TAG_RE.sub("", (s or "").strip())


def parse_hyp_file(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        name, hyp = line.split("|", 1)
        out[name.strip()] = hyp.strip()
    return out


def find_best_outputs(model_dir: Path) -> Tuple[Path, Path]:
    txt = sorted(model_dir.glob("best*.BW_*.A_*.test.txt"))[-1]
    gls = sorted(model_dir.glob("best*.BW_*.test.gls"))[-1]
    return txt, gls


def aligned_pairs(model_dir: Path, pickle_path: Path) -> Tuple[List[str], List[str], List[str], List[str]]:
    txt_path, gls_path = find_best_outputs(model_dir)
    text_hyps = parse_hyp_file(txt_path)
    gloss_hyps = parse_hyp_file(gls_path)

    t_ref, t_hyp, g_ref, g_hyp = [], [], [], []
    for item in load_pickle(pickle_path):
        name = str(item["name"])
        if name not in text_hyps:
            continue
        g = " ".join(item["gloss"]) if isinstance(item.get("gloss"), list) else str(item.get("gloss", ""))
        t = str(item.get("text", ""))
        g_ref.append(strip_tag(g))
        g_hyp.append(strip_tag(gloss_hyps.get(name, "")))
        t_ref.append(strip_tag(t))
        t_hyp.append(strip_tag(text_hyps[name]))
    return t_ref, t_hyp, g_ref, g_hyp


def bootstrap_bleu4(refs: List[str], hyps: List[str], n_samples: int, seed: int) -> List[float]:
    rng = random.Random(seed)
    n = len(refs)
    scores = []
    for _ in range(n_samples):
        idx = [rng.randrange(n) for _ in range(n)]
        r = [refs[i] for i in idx]
        h = [hyps[i] for i in idx]
        bleu_scores = bleu(r, h)
        scores.append(float(bleu_scores["bleu4"]))
    return scores


def bootstrap_wer(refs: List[str], hyps: List[str], n_samples: int, seed: int) -> List[float]:
    rng = random.Random(seed)
    n = len(refs)
    scores = []
    for _ in range(n_samples):
        idx = [rng.randrange(n) for _ in range(n)]
        r = [refs[i] for i in idx]
        h = [hyps[i] for i in idx]
        scores.append(float(wer_list(r, h)["wer"]))
    return scores


def ci(values: List[float], alpha: float = 0.05) -> Tuple[float, float]:
    s = sorted(values)
    lo = s[int((alpha / 2) * len(s))]
    hi = s[int((1 - alpha / 2) * len(s)) - 1]
    return lo, hi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-md", default="docs/results/bootstrap_ci.md")
    parser.add_argument("--output-csv", default="docs/results/bootstrap_ci.csv")
    args = parser.parse_args()

    rows = []
    for model_name, ckpt_rel, pickle_rel in MODELS:
        model_dir = ROOT / ckpt_rel
        pickle_path = ROOT / pickle_rel
        t_ref, t_hyp, g_ref, g_hyp = aligned_pairs(model_dir, pickle_path)

        point_bleu = float(bleu(t_ref, t_hyp)["bleu4"])
        point_wer = float(wer_list(g_ref, g_hyp)["wer"])

        bleu_samples = bootstrap_bleu4(t_ref, t_hyp, args.n_bootstrap, args.seed)
        wer_samples = bootstrap_wer(g_ref, g_hyp, args.n_bootstrap, args.seed + 1)

        bleu_lo, bleu_hi = ci(bleu_samples)
        wer_lo, wer_hi = ci(wer_samples)

        rows.append(
            {
                "model": model_name,
                "n": len(t_ref),
                "bleu4": round(point_bleu, 2),
                "bleu4_ci_lo": round(bleu_lo, 2),
                "bleu4_ci_hi": round(bleu_hi, 2),
                "wer": round(point_wer, 2),
                "wer_ci_lo": round(wer_lo, 2),
                "wer_ci_hi": round(wer_hi, 2),
            }
        )
        print(
            f"{model_name}: BLEU {point_bleu:.2f} (95% CI {bleu_lo:.2f}–{bleu_hi:.2f}) | "
            f"WER {point_wer:.2f} (95% CI {wer_lo:.2f}–{wer_hi:.2f})"
        )

    out_csv = ROOT / args.output_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md = [
        "# Bootstrap 95% confidence intervals (PHOENIX-2014-T test)",
        "",
        f"Resamples: **{args.n_bootstrap}** (sentence-level, with replacement). Seed: {args.seed}.",
        "",
        "| Model | n | BLEU-4 | 95% CI | WER ↓ | 95% CI |",
        "|---|---:|---:|---|---:|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['model']} | {r['n']} | {r['bleu4']:.2f} | {r['bleu4_ci_lo']:.2f}–{r['bleu4_ci_hi']:.2f} | "
            f"{r['wer']:.2f} | {r['wer_ci_lo']:.2f}–{r['wer_ci_hi']:.2f} |"
        )
    md.append("")
    md.append("Regenerate: `python scripts/bootstrap_metrics_ci.py`")
    out_md = ROOT / args.output_md
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {out_csv} and {out_md}")


if __name__ == "__main__":
    main()
