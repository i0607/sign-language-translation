#!/usr/bin/env python3
"""Sample sentence pairs for human adequacy/fluency rating (Task 2)."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import List


def read_lines(path: Path) -> List[str]:
    return [ln.rstrip("\n") for ln in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--ids", default="", help="Optional IDs file (one per line)")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--target-lang", default="French")
    args = parser.parse_args()

    sources = read_lines(Path(args.source))
    hyps = read_lines(Path(args.hypothesis))
    ids = read_lines(Path(args.ids)) if args.ids else [str(i + 1) for i in range(len(sources))]

    if not (len(sources) == len(hyps) == len(ids)):
        raise SystemExit("source, hypothesis, and ids must have same line count")

    rng = random.Random(args.seed)
    indices = list(range(len(sources)))
    rng.shuffle(indices)
    pick = indices[: args.n]

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "line_index",
                "english_pivot",
                "target_output",
                "target_language",
                "adequacy_rater1_1to5",
                "fluency_rater1_1to5",
                "adequacy_rater2_1to5",
                "fluency_rater2_1to5",
                "notes",
            ],
        )
        w.writeheader()
        for j, idx in enumerate(pick, start=1):
            w.writerow(
                {
                    "sample_id": j,
                    "line_index": idx + 1,
                    "english_pivot": sources[idx],
                    "target_output": hyps[idx],
                    "target_language": args.target_lang,
                    "adequacy_rater1_1to5": "",
                    "fluency_rater1_1to5": "",
                    "adequacy_rater2_1to5": "",
                    "fluency_rater2_1to5": "",
                    "notes": "",
                }
            )
    print(f"Wrote {out} ({args.n} samples, seed={args.seed})")


if __name__ == "__main__":
    main()
