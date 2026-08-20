#!/usr/bin/env python3
"""Rough error-type counts on cascade pivot + MT outputs (Task 3)."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import List, Tuple


TAG_RE = re.compile(r"^<(de|en)>\s*", re.IGNORECASE)
ENDS_ABRUPT = re.compile(
    r"\b(and|or|but|with|to|of|in|on|at|for|the|a|an|und|oder|aber|mit|tomorrow|today|morgen|heute)\s*$",
    re.IGNORECASE,
)
LATIN_WORD = re.compile(r"\b[a-zA-Z]{3,}\b")
ARABIC_CHAR = re.compile(r"[\u0600-\u06FF]")


def read_lines(path: Path) -> List[str]:
    return [ln.rstrip("\n") for ln in path.read_text(encoding="utf-8").splitlines()]


def strip_tag(s: str) -> str:
    return TAG_RE.sub("", (s or "").strip())


def is_truncated(text: str, target: str) -> bool:
    t = text.strip()
    if not t:
        return True
    if len(t.split()) <= 2:
        return True
    if ENDS_ABRUPT.search(t):
        return True
    # Latin targets: missing terminal punctuation is a weak signal only with short outputs
    if target.lower() != "ar" and t[-1] not in ".!?" and len(t.split()) <= 6:
        return True
    return False


def is_very_short_vs_source(src: str, hyp: str, ratio: float = 0.45) -> bool:
    sw = len(src.split())
    hw = len(hyp.split())
    if sw == 0:
        return hw == 0
    return hw < max(3, int(sw * ratio))


def has_language_tag_leak(text: str) -> bool:
    return bool(TAG_RE.search(text.strip()))


def english_in_non_english_target(hyp: str, target: str) -> bool:
    """Heuristic: many Latin words in Arabic target."""
    if target.lower() != "ar":
        return False
    if not ARABIC_CHAR.search(hyp):
        return len(LATIN_WORD.findall(hyp)) >= 3
    latin = len(LATIN_WORD.findall(hyp))
    arabic = len(ARABIC_CHAR.findall(hyp))
    return latin >= 5 and latin > arabic * 0.3


def analyze(
    sources: List[str],
    hyps: List[str],
    target: str,
) -> Tuple[dict, List[dict]]:
    n = len(sources)
    counts = {
        "truncated_or_incomplete": 0,
        "very_short_vs_source": 0,
        "empty_hypothesis": 0,
        "language_tag_leak": 0,
        "target_language_leak_heuristic": 0,
    }
    rows: List[dict] = []

    for i, (src, hyp) in enumerate(zip(sources, hyps)):
        src_c = strip_tag(src)
        hyp_c = strip_tag(hyp)
        flags = {
            "truncated_or_incomplete": is_truncated(hyp_c, target),
            "very_short_vs_source": is_very_short_vs_source(src_c, hyp_c),
            "empty_hypothesis": len(hyp_c.strip()) == 0,
            "language_tag_leak": has_language_tag_leak(hyp),
            "target_language_leak_heuristic": english_in_non_english_target(hyp_c, target),
        }
        for k, v in flags.items():
            if v:
                counts[k] += 1
        if any(flags.values()):
            rows.append(
                {
                    "line": i + 1,
                    "source": src_c[:200],
                    "hypothesis": hyp_c[:200],
                    **flags,
                }
            )

    pct = {k: round(100.0 * v / n, 2) for k, v in counts.items()}
    summary = {"n": n, "target": target, "counts": counts, "percent": pct}
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--target", default="fr", help="Target lang short code: fr, ar")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default="", help="Optional flagged examples CSV")
    parser.add_argument("--label", default="cascade")
    args = parser.parse_args()

    sources = read_lines(Path(args.source))
    hyps = read_lines(Path(args.hypothesis))
    if len(sources) != len(hyps):
        raise SystemExit(f"Line mismatch: {len(sources)} vs {len(hyps)}")

    summary, flagged = analyze(sources, hyps, args.target)
    summary["label"] = args.label

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))

    if args.output_csv:
        csv_path = Path(args.output_csv)
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            if flagged:
                w = csv.DictWriter(f, fieldnames=list(flagged[0].keys()))
                w.writeheader()
                w.writerows(flagged)
            else:
                f.write("line\n")
        print(f"Wrote flagged examples: {csv_path}")


if __name__ == "__main__":
    main()
