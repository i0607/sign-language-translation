#!/usr/bin/env python3
"""Select qualitative test examples and write appendix markdown."""

from __future__ import annotations

import gzip
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from signjoey.metrics import wer_single  # noqa: E402

TAG_RE = re.compile(r"^<(de|en)>\s*", re.IGNORECASE)
OUT_PATH = ROOT / "docs" / "results" / "qualitative_examples.md"


def load_pickle(path: Path) -> List[dict]:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def strip_tag(s: str) -> str:
    return TAG_RE.sub("", (s or "").strip())


def as_gloss(item: dict) -> str:
    g = item.get("gloss", "")
    if isinstance(g, list):
        return strip_tag(" ".join(str(x) for x in g))
    return strip_tag(str(g))


def as_text(item: dict) -> str:
    return strip_tag(str(item.get("text", "")))


def parse_hyp(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "|" not in line:
            continue
        k, v = line.split("|", 1)
        out[k.strip()] = strip_tag(v.strip())
    return out


def find_hyp(model_dir: Path, split: str, kind: str) -> Path:
    if kind == "txt":
        files = sorted(model_dir.glob(f"best*.BW_*.A_*.{split}.txt")) or sorted(
            model_dir.glob(f"*.BW_*.A_*.{split}.txt")
        )
    else:
        files = sorted(model_dir.glob(f"best*.BW_*.{split}.gls")) or sorted(
            model_dir.glob(f"*.BW_010.{split}.gls")
        )
    if not files:
        raise FileNotFoundError(f"No {kind} for {split} in {model_dir}")
    return files[-1]


def text_overlap_score(ref: str, hyp: str) -> float:
    """Rough proxy for picking examples (reference token recall)."""
    ref_t = ref.lower().split()
    hyp_t = hyp.lower().split()
    if not ref_t:
        return 0.0
    ref_set = set(ref_t)
    hit = sum(1 for t in hyp_t if t in ref_set)
    return 100.0 * hit / len(ref_t)


def wer_pct(ref: str, hyp: str) -> float:
    r = wer_single(ref, hyp)
    if r["num_ref"] == 0:
        return 100.0
    return 100.0 * r["num_err"] / r["num_ref"]


def de_rate(ref: str, hyp: str) -> float:
    r = wer_single(ref, hyp)
    if r["num_ref"] == 0:
        return 0.0
    return 100.0 * r["num_del"] / r["num_ref"]


class Example:
    def __init__(
        self,
        category: str,
        name: str,
        ref_gloss: str,
        hyp_gloss: str,
        ref_text: str,
        hyp_text: str,
        wer: float,
        bleu: float,
        del_pct: float,
        note: str = "",
    ):
        self.category = category
        self.name = name
        self.ref_gloss = ref_gloss
        self.hyp_gloss = hyp_gloss
        self.ref_text = ref_text
        self.hyp_text = hyp_text
        self.wer = wer
        self.bleu = bleu
        self.del_pct = del_pct
        self.note = note


def norm_sample_key(name: str) -> str:
    n = str(name).replace("\\", "/")
    if "/" in n:
        n = n.split("/")[-1]
    if "." in n:
        n = n.rsplit(".", 1)[0]
    return n


def score_monolingual_de(
    data_path: Path, model_dir: Path
) -> List[Tuple[str, str, str, str, str, float, float, float]]:
    items = {norm_sample_key(i["name"]): i for i in load_pickle(data_path)}
    txt_hyps = parse_hyp(find_hyp(model_dir, "test", "txt"))
    gls_hyps = parse_hyp(find_hyp(model_dir, "test", "gls"))
    scored = []
    for key, hyp_txt in txt_hyps.items():
        base = norm_sample_key(key)
        item = items.get(base)
        if item is None:
            continue
        ref_g = as_gloss(item)
        ref_t = as_text(item)
        hyp_g = gls_hyps.get(key, gls_hyps.get(base, ""))
        w = wer_pct(ref_g, hyp_g)
        b = text_overlap_score(ref_t, hyp_txt)
        d = de_rate(ref_g, hyp_g)
        scored.append((base, ref_g, hyp_g, ref_t, hyp_txt, w, b, d))
    return scored


def score_bilingual(
    data_path: Path, model_dir: Path, lang: str
) -> List[Tuple[str, str, str, str, str, float, float, float]]:
    refs = {str(i["name"]): i for i in load_pickle(data_path)}
    txt_hyps = parse_hyp(find_hyp(model_dir, "test", "txt"))
    gls_hyps = parse_hyp(find_hyp(model_dir, "test", "gls"))
    scored = []
    for name, item in refs.items():
        if not name.endswith(f"__{lang}"):
            continue
        if name not in txt_hyps:
            continue
        ref_g = as_gloss(item)
        ref_t = as_text(item)
        hyp_t = txt_hyps[name]
        hyp_g = gls_hyps.get(name, "")
        w = wer_pct(ref_g, hyp_g)
        b = text_overlap_score(ref_t, hyp_t)
        d = de_rate(ref_g, hyp_g)
        scored.append((name, ref_g, hyp_g, ref_t, hyp_t, w, b, d))
    return scored


def pick_examples() -> List[Example]:
    ex: List[Example] = []

    ode_dir = ROOT / "sign_sample_model_improved_v4"
    de_test = ROOT / "data/PHOENIX2014T/phoenix14t.pami0.test"
    ode_scores = score_monolingual_de(de_test, ode_dir)
    ode_scores.sort(key=lambda x: (-x[6], x[5]))  # high BLEU, low WER
    for i, (name, rg, hg, rt, ht, w, b, d) in enumerate(ode_scores[:3]):
        ex.append(
            Example(
                "Strong German (Model-ODE)",
                name,
                rg,
                hg,
                rt,
                ht,
                w,
                b,
                d,
                "Joint DE model; dev-selected decode from train.log",
            )
        )

    v4_dir = ROOT / "sign_sample_model_de_en_v4_langhead"
    bi_test = ROOT / "data/PHOENIX2014T_DE_EN_V2/phoenix14t_de_en_v2.test"
    for lang in ("de", "en"):
        scores = score_bilingual(bi_test, v4_dir, lang)
        scores.sort(key=lambda x: (-x[6], x[5]))
        for name, rg, hg, rt, ht, w, b, d in scores[:2 if lang == "de" else 1]:
            ex.append(
                Example(
                    f"Bilingual v4 success (`{lang}`)",
                    name,
                    rg,
                    hg,
                    rt,
                    ht,
                    w,
                    b,
                    d,
                )
            )

    # Recognition failures: high deletion rate on gloss (ODE)
    ode_scores.sort(key=lambda x: (-x[7], x[5]))
    for name, rg, hg, rt, ht, w, b, d in ode_scores[:2]:
        ex.append(
            Example(
                "Recognition failure (high gloss deletions)",
                name,
                rg,
                hg,
                rt,
                ht,
                w,
                b,
                d,
                "Model under-predicts gloss tokens",
            )
        )

    # Translation failures: low sentence BLEU on DE (ODE)
    ode_scores.sort(key=lambda x: (x[6], -x[5]))
    for name, rg, hg, rt, ht, w, b, d in ode_scores[:2]:
        ex.append(
            Example(
                "Translation failure (low text overlap vs reference)",
                name,
                rg,
                hg,
                rt,
                ht,
                w,
                b,
                d,
            )
        )

    return ex


def write_md(examples: List[Example], path: Path) -> None:
    lines = [
        "# Qualitative examples (PHOENIX-2014-T test)",
        "",
        "Selected automatically for thesis appendix / defense slides.",
        "Regenerate: `./venv/bin/python scripts/build_qualitative_examples.py`",
        "",
        "| # | Category | Sample | Sent. WER (gloss) | Text overlap % | DEL% (gloss) |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for i, e in enumerate(examples, 1):
        lines.append(
            f"| {i} | {e.category} | `{e.name}` | {e.wer:.1f} | {e.bleu:.1f} | {e.del_pct:.1f} |"
        )
    lines.append("")
    for i, e in enumerate(examples, 1):
        lines.extend(
            [
                f"## Example {i}: {e.category}",
                "",
                f"**Sample:** `{e.name}`  ",
                f"**Sentence-level gloss WER:** {e.wer:.1f}% · **Text token overlap (proxy):** {e.bleu:.1f}% · **Gloss deletions:** {e.del_pct:.1f}% of reference length  ",
            ]
        )
        if e.note:
            lines.append(f"**Note:** {e.note}  ")
        lines.extend(
            [
                "",
                "| | Gloss | Text |",
                "|---|---|---|",
                f"| **Reference** | {e.ref_gloss} | {e.ref_text} |",
                f"| **Hypothesis** | {e.hyp_gloss} | {e.hyp_text} |",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    examples = pick_examples()
    write_md(examples, OUT_PATH)
    print(f"Wrote {len(examples)} examples to {OUT_PATH}")


if __name__ == "__main__":
    main()
