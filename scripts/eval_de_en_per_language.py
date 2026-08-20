#!/usr/bin/env python3
"""
Per-language WER / BLEU for DE+EN models from saved hypothesis files + pickle references.

Uses sample names ending in __de / __en (PHOENIX2014T_DE_EN_V2 format).
"""

from __future__ import annotations

import argparse
import gzip
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from signjoey.metrics import bleu, wer_list  # noqa: E402

TAG_RE = re.compile(r"^<(de|en)>\s*", re.IGNORECASE)


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


def refs_from_pickle(split_path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    gloss_refs: Dict[str, str] = {}
    text_refs: Dict[str, str] = {}
    for item in load_pickle(split_path):
        name = str(item["name"])
        gloss_refs[name] = strip_tag(" ".join(item["gloss"]) if isinstance(item.get("gloss"), list) else str(item.get("gloss", "")))
        text_refs[name] = strip_tag(str(item.get("text", "")))
    return gloss_refs, text_refs


def subset_metrics(
    names: List[str],
    gloss_refs: Dict[str, str],
    text_refs: Dict[str, str],
    gloss_hyps: Dict[str, str],
    text_hyps: Dict[str, str],
) -> dict:
    g_ref, g_hyp, t_ref, t_hyp = [], [], [], []
    missing = 0
    for n in names:
        if n not in gloss_refs or n not in gloss_hyps:
            missing += 1
            continue
        g_ref.append(gloss_refs[n])
        g_hyp.append(strip_tag(gloss_hyps[n]))
        t_ref.append(text_refs[n])
        t_hyp.append(strip_tag(text_hyps[n]))
    wer = wer_list(g_ref, g_hyp) if g_ref else {}
    bleu_scores = bleu(t_ref, t_hyp) if t_ref else {}
    return {
        "n": len(g_ref),
        "missing": missing,
        "wer": wer.get("wer"),
        "del_rate": wer.get("del_rate"),
        "ins_rate": wer.get("ins_rate"),
        "sub_rate": wer.get("sub_rate"),
        "bleu4": bleu_scores.get("bleu4") if bleu_scores else None,
    }


def find_best_outputs(model_dir: Path, split: str) -> Tuple[Path, Path]:
    txt_files = sorted(model_dir.glob(f"best*.BW_*.A_*.{split}.txt"))
    gls_files = sorted(model_dir.glob(f"best*.BW_*.{split}.gls"))
    if not txt_files:
        txt_files = sorted(model_dir.glob(f"*.BW_*.A_*.{split}.txt"))
    if not gls_files:
        gls_files = sorted(model_dir.glob(f"*.BW_010.{split}.gls"))
        if not gls_files:
            gls_files = sorted(model_dir.glob(f"best*.BW_*.{split}.gls"))
    if not txt_files or not gls_files:
        raise FileNotFoundError(f"No {split} hyp files in {model_dir}")
    return txt_files[-1], gls_files[-1]


def eval_model(model_dir: Path, data_dir: Path, split: str) -> dict:
    pickle_path = data_dir / f"phoenix14t_de_en_v2.{split}"
    gloss_refs, text_refs = refs_from_pickle(pickle_path)
    txt_path, gls_path = find_best_outputs(model_dir, split)
    text_hyps = parse_hyp_file(txt_path)
    gloss_hyps = parse_hyp_file(gls_path)

    all_names = sorted(set(gloss_refs.keys()) & set(text_hyps.keys()))
    de_names = [n for n in all_names if n.endswith("__de")]
    en_names = [n for n in all_names if n.endswith("__en")]

    return {
        "split": split,
        "hyp_txt": str(txt_path.relative_to(ROOT)),
        "hyp_gls": str(gls_path.relative_to(ROOT)),
        "all": subset_metrics(all_names, gloss_refs, text_refs, gloss_hyps, text_hyps),
        "de": subset_metrics(de_names, gloss_refs, text_refs, gloss_hyps, text_hyps),
        "en": subset_metrics(en_names, gloss_refs, text_refs, gloss_hyps, text_hyps),
    }


def fmt_metrics(m: dict) -> str:
    if not m or m.get("n", 0) == 0:
        return "n/a"
    return (
        f"WER {m['wer']:.2f} (DEL {m['del_rate']:.2f}, INS {m['ins_rate']:.2f}, SUB {m['sub_rate']:.2f}), "
        f"BLEU-4 {m['bleu4']:.2f}, n={m['n']}"
    )


def write_report(model_name: str, model_dir: str, dev: dict, test: dict, path: Path) -> None:
    lines = [
        f"# Per-language metrics: {model_name}",
        "",
        f"Model directory: `{model_dir}`",
        "",
        "References: `data/PHOENIX2014T_DE_EN_V2/phoenix14t_de_en_v2.{split}`",
        "",
        "Hypotheses: dev-selected beams from final `best.*.{split}.txt` / `.gls` in model dir.",
        "",
        "Regenerate: `./venv/bin/python scripts/eval_de_en_per_language.py`",
        "",
        "## Dev",
        "",
        f"- Hypotheses: `{dev['hyp_txt']}`, `{dev['hyp_gls']}`",
        "",
        "| Subset | Metrics |",
        "|---|---|",
        f"| All (bilingual) | {fmt_metrics(dev['all'])} |",
        f"| `<de>` only | {fmt_metrics(dev['de'])} |",
        f"| `<en>` only | {fmt_metrics(dev['en'])} |",
        "",
        "## Test",
        "",
        f"- Hypotheses: `{test['hyp_txt']}`, `{test['hyp_gls']}`",
        "",
        "| Subset | Metrics |",
        "|---|---|",
        f"| All (bilingual) | {fmt_metrics(test['all'])} |",
        f"| `<de>` only | {fmt_metrics(test['de'])} |",
        f"| `<en>` only | {fmt_metrics(test['en'])} |",
        "",
        "## Comparison to monolingual references (test BLEU-4)",
        "",
        "| Setting | Test BLEU-4 (from summary_metrics) |",
        "|---|---:|",
        "| Model-ODE (German only) | 21.23 |",
        "| Model-OEN (English only) | 20.88 |",
        f"| DE+EN-v4 `<de>` subset | {test['de'].get('bleu4', 0):.2f} |",
        f"| DE+EN-v4 `<en>` subset | {test['en'].get('bleu4', 0):.2f} |",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default="sign_sample_model_de_en_v4_langhead",
        help="Directory with best.*.dev/test.txt and .gls",
    )
    parser.add_argument(
        "--data-dir",
        default="data/PHOENIX2014T_DE_EN_V2",
        help="Merged DE+EN pickle directory",
    )
    parser.add_argument(
        "--model-name",
        default="DE+EN-v4 (lang-head)",
    )
    args = parser.parse_args()

    model_dir = ROOT / args.model_dir
    data_dir = ROOT / args.data_dir
    out_dir = ROOT / "docs" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = eval_model(model_dir, data_dir, "dev")
    test = eval_model(model_dir, data_dir, "test")

    md_path = out_dir / "de_en_v4_per_language_metrics.md"
    write_report(args.model_name, args.model_dir, dev, test, md_path)

    import csv

    csv_path = out_dir / "de_en_v4_per_language_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "split", "subset", "n", "wer", "del", "ins", "sub", "bleu4"])
        for block in (dev, test):
            for subset in ("all", "de", "en"):
                m = block[subset]
                w.writerow(
                    [
                        args.model_name,
                        block["split"],
                        subset,
                        m.get("n"),
                        m.get("wer"),
                        m.get("del_rate"),
                        m.get("ins_rate"),
                        m.get("sub_rate"),
                        m.get("bleu4"),
                    ]
                )

    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")
    print("Test DE:", fmt_metrics(test["de"]))
    print("Test EN:", fmt_metrics(test["en"]))


if __name__ == "__main__":
    main()
