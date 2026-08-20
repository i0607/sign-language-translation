#!/usr/bin/env python3
"""Extract final DEV/TEST metrics from train.log files into summary CSV and Markdown."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "results"

BLOCK_RE = re.compile(
    r"\[(DEV|TEST)\] partition \[Recognition & Translation\] results:\s*\n"
    r"\s*Best CTC Decode Beam Size: (\d+)\s*\n"
    r"\s*Best Translation Beam Size: (\d+) and Alpha: (-?\d+)\s*\n"
    r"\s*WER ([\d.]+)\s*\(DEL: ([\d.]+),\s*INS: ([\d.]+),\s*SUB: ([\d.]+)\)\s*\n"
    r"\s*BLEU-4 ([\d.]+).*?\n"
    r"\s*CHRF ([\d.]+)\s*ROUGE ([\d.]+)",
    re.MULTILINE,
)

# Camgoz CVPR 2020 PHOENIX-2014-T German test (published joint row)
CAMGOZ = {
    "model": "Camgoz et al. (2020) Sign2(Gloss+Text)",
    "task": "DE joint (published)",
    "checkpoint_dir": "—",
    "config": "Camgoz et al., CVPR 2020",
    "dev_wer": "",
    "test_wer": "26.16",
    "dev_bleu4": "",
    "test_bleu4": "21.32",
    "decode": "published baseline",
    "test_del": "",
    "test_ins": "",
    "test_sub": "",
    "test_chrf": "",
    "test_rouge": "",
    "notes": "German test; not re-run in this repo",
}

MODEL_LOGS: List[Tuple[str, str, str, str]] = [
    ("Model-ODE", "DE optimised (v4)", "sign_sample_model_improved_v4", "configs/sign_improved_v4.yaml"),
    ("Model-OEN", "EN optimised (v4 English)", "sign_sample_model_improved_v4_english", "configs/sign_improved_v4_english.yaml"),
    ("Model-A", "DE ablation (v5, λ_rec=2.0)", "sign_sample_model_improved_v5", "configs/sign_improved_v5.yaml"),
    ("DE+EN-v2", "Bilingual single softmax", "sign_sample_model_de_en_v2", "experiments/de_en_v2/configs/sign_de_en_v2.yaml"),
    ("DE+EN-v3", "Bilingual fine-tune from v2", "sign_sample_model_de_en_v3", "experiments/de_en_v3/configs/sign_de_en_v3_finetune.yaml"),
    ("DE+EN-v4", "Bilingual lang-head", "sign_sample_model_de_en_v4_langhead", "experiments/de_en_v4_langhead/configs/sign_de_en_v4_langhead_finetune.yaml"),
]


def parse_log(path: Path) -> Dict[str, dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    out: Dict[str, dict] = {}
    for m in BLOCK_RE.finditer(text):
        split, ctc, tx, alpha, wer, del_, ins, sub, bleu, chrf, rouge = m.groups()
        out[split] = {
            "ctc_bw": ctc,
            "tx_bw": tx,
            "alpha": alpha,
            "wer": wer,
            "del": del_,
            "ins": ins,
            "sub": sub,
            "bleu4": bleu,
            "chrf": chrf,
            "rouge": rouge,
        }
    return out


def row_from_parsed(model: str, task: str, ckpt: str, config: str, parsed: Dict[str, dict]) -> dict:
    dev = parsed.get("DEV", {})
    test = parsed.get("TEST", {})
    decode = ""
    if test:
        decode = f"CTC={test.get('ctc_bw', dev.get('ctc_bw', ''))}, TX={test.get('tx_bw', dev.get('tx_bw', ''))}, α={test.get('alpha', dev.get('alpha', ''))}"
    return {
        "model": model,
        "task": task,
        "dev_wer": dev.get("wer", ""),
        "test_wer": test.get("wer", ""),
        "dev_bleu4": dev.get("bleu4", ""),
        "test_bleu4": test.get("bleu4", ""),
        "decode": decode,
        "checkpoint_dir": ckpt,
        "config": config,
        "test_del": test.get("del", ""),
        "test_ins": test.get("ins", ""),
        "test_sub": test.get("sub", ""),
        "test_chrf": test.get("chrf", ""),
        "test_rouge": test.get("rouge", ""),
        "notes": "Dev-selected decode; source: train.log final Recognition & Translation block",
    }


def write_csv(rows: List[dict], path: Path) -> None:
    fields = [
        "model",
        "task",
        "dev_wer",
        "test_wer",
        "dev_bleu4",
        "test_bleu4",
        "decode",
        "checkpoint_dir",
        "config",
        "test_del",
        "test_ins",
        "test_sub",
        "test_chrf",
        "test_rouge",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def write_md(rows: List[dict], path: Path) -> None:
    lines = [
        "# Master results summary (PHOENIX-2014-T)",
        "",
        "Auto-generated from final `[DEV]` / `[TEST]` **Recognition & Translation** blocks in each model `train.log`.",
        "Regenerate: `./venv/bin/python scripts/extract_summary_metrics.py`",
        "",
        "**Comparability:** Camgoz and Model-ODE/OEN/A rows are German-only or English-only monolingual evaluation. "
        "DE+EN-v2/v3/v4 rows use the **bilingual manifest** (mixed references); do not compare their WER directly to 26.16 without context.",
        "",
        "| Model | Task | Dev WER ↓ | Test WER ↓ | Dev BLEU-4 ↑ | Test BLEU-4 ↑ | Decode (dev-selected) | Checkpoint dir |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['task']} | {r['dev_wer']} | {r['test_wer']} | "
            f"{r['dev_bleu4']} | {r['test_bleu4']} | {r['decode']} | `{r['checkpoint_dir']}` |"
        )
    lines.extend(
        [
            "",
            "## Test error decomposition (recognition)",
            "",
            "| Model | DEL % | INS % | SUB % | chrF | ROUGE |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for r in rows:
        if r.get("test_wer"):
            lines.append(
                f"| {r['model']} | {r.get('test_del', '')} | {r.get('test_ins', '')} | "
                f"{r.get('test_sub', '')} | {r.get('test_chrf', '')} | {r.get('test_rouge', '')} |"
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = [dict(CAMGOZ)]

    for model, task, ckpt_dir, config in MODEL_LOGS:
        log_path = ROOT / ckpt_dir / "train.log"
        if not log_path.is_file():
            print(f"WARN: missing {log_path}")
            continue
        parsed = parse_log(log_path)
        rows.append(row_from_parsed(model, task, ckpt_dir, config, parsed))
        print(f"OK: {model}")

    write_csv(rows, OUT_DIR / "summary_metrics.csv")
    write_md(rows, OUT_DIR / "summary_metrics.md")
    print(f"Wrote {OUT_DIR / 'summary_metrics.csv'}")
    print(f"Wrote {OUT_DIR / 'summary_metrics.md'}")


if __name__ == "__main__":
    main()
