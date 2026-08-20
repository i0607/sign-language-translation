#!/usr/bin/env python3
"""Score pivot→target cascade lines with COMET-Kiwi (reference-free)."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import List


def read_lines(path: Path) -> List[str]:
    return [ln.rstrip("\n") for ln in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser(description="COMET-Kiwi scoring for cascade outputs")
    parser.add_argument("--source", required=True, help="Pivot text (one sentence per line)")
    parser.add_argument("--hypothesis", required=True, help="MT output (one sentence per line)")
    parser.add_argument(
        "--model",
        default="Unbabel/wmt22-cometkiwi-da",
        help="COMET-Kiwi checkpoint name",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-json", type=str, default="", help="Optional JSON summary path")
    parser.add_argument("--label", default="cascade", help="Label for printed summary")
    args = parser.parse_args()

    src_path = Path(args.source)
    hyp_path = Path(args.hypothesis)
    sources = read_lines(src_path)
    hyps = read_lines(hyp_path)

    if len(sources) != len(hyps):
        raise SystemExit(f"Line count mismatch: {len(sources)} sources vs {len(hyps)} hyps")

    # filter empty pairs but keep alignment
    data = []
    skipped = 0
    for s, h in zip(sources, hyps):
        if not s.strip() or not h.strip():
            skipped += 1
            continue
        data.append({"src": s.strip(), "mt": h.strip()})

    if not data:
        raise SystemExit("No non-empty sentence pairs to score")

    try:
        from comet import download_model, load_from_checkpoint
    except ImportError as e:
        raise SystemExit("Install: pip install unbabel-comet") from e

    model_candidates = [args.model]
    if args.model == "Unbabel/wmt22-cometkiwi-da":
        # Gated on HF; fallback is reference-free QE (older Kiwi predecessor)
        model_candidates.append("Unbabel/wmt20-comet-qe-da")

    model_path = None
    used_model = None
    last_err = None
    for candidate in model_candidates:
        try:
            print(f"Loading COMET model: {candidate}")
            model_path = download_model(candidate)
            used_model = candidate
            break
        except Exception as e:
            last_err = e
            print(f"  Could not load {candidate}: {e}")

    if model_path is None:
        raise SystemExit(
            "No COMET model could be loaded. For wmt22-cometkiwi-da, accept the HF "
            "license and run: huggingface-cli login\n"
            f"Last error: {last_err}"
        )

    model = load_from_checkpoint(model_path)

    print(f"Scoring {len(data)} pairs (skipped {skipped} empty lines)...")
    result = model.predict(
        data,
        batch_size=args.batch_size,
        gpus=0,
        accelerator="cpu",
        num_workers=1,
        progress_bar=True,
    )

    scores = [float(x) for x in result["scores"]]
    corpus_score = float(result["system_score"])
    mean = statistics.mean(scores)
    stdev = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    sorted_scores = sorted(scores)
    p25 = sorted_scores[len(sorted_scores) // 4]
    p50 = sorted_scores[len(sorted_scores) // 2]
    p75 = sorted_scores[(3 * len(sorted_scores)) // 4]

    summary = {
        "label": args.label,
        "model": used_model,
        "source_file": str(src_path),
        "hypothesis_file": str(hyp_path),
        "n_total_lines": len(sources),
        "n_scored": len(scores),
        "n_skipped_empty": skipped,
        "corpus_comet_kiwi": float(corpus_score),
        "mean_segment": mean,
        "stdev_segment": stdev,
        "min_segment": min(scores),
        "max_segment": max(scores),
        "p25": p25,
        "median": p50,
        "p75": p75,
    }

    print("\n=== COMET-Kiwi summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
