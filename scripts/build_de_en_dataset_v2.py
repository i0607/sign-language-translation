#!/usr/bin/env python3
"""
Build a stricter DE+EN dataset for unified training.

v2 changes vs de_en_v1:
- keep only samples that exist in both DE and EN datasets
- enforce balanced duplication (exactly two outputs per matched sample)
- prepend language tags to BOTH text and gloss targets
"""

import argparse
import gzip
import os
import pickle
from typing import Dict, List


def load_pickle(path: str) -> List[Dict]:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(path: str, data: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(data, f)


def norm_name(name: str) -> str:
    value = (name or "").strip().replace("\\", "/")
    value = value.split("/")[-1]
    if "." in value:
        value = value.rsplit(".", 1)[0]
    return value


def as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value)


def with_tag(tag: str, content: str) -> str:
    content = content.strip()
    return f"<{tag}> {content}" if content else ""


def build_split(german_path: str, english_path: str) -> List[Dict]:
    de_data = load_pickle(german_path)
    en_data = load_pickle(english_path)

    en_by_name = {norm_name(item["name"]): item for item in en_data}
    merged: List[Dict] = []
    skipped_unmatched = 0
    skipped_empty = 0

    for de_item in de_data:
        key = norm_name(de_item["name"])
        en_item = en_by_name.get(key)
        if en_item is None:
            skipped_unmatched += 1
            continue

        de_text = as_text(de_item.get("text"))
        en_text = as_text(en_item.get("text"))
        de_gloss = as_text(de_item.get("gloss"))
        en_gloss = as_text(en_item.get("gloss"))

        # Strict pairing: keep only complete bilingual targets.
        if not (de_text.strip() and en_text.strip() and de_gloss.strip() and en_gloss.strip()):
            skipped_empty += 1
            continue

        merged.append(
            {
                "name": f"{key}__de",
                "signer": de_item.get("signer", ""),
                "sign": de_item["sign"],
                "gloss": with_tag("de", de_gloss),
                "text": with_tag("de", de_text),
            }
        )
        merged.append(
            {
                "name": f"{key}__en",
                "signer": en_item.get("signer", de_item.get("signer", "")),
                "sign": de_item["sign"],
                "gloss": with_tag("en", en_gloss),
                "text": with_tag("en", en_text),
            }
        )

    print(f"  Matched bilingual pairs kept: {len(merged) // 2}")
    print(f"  Skipped unmatched DE samples: {skipped_unmatched}")
    print(f"  Skipped incomplete pairs: {skipped_empty}")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Build strict merged DE+EN dataset (v2)")
    parser.add_argument("--de-dir", default="data/PHOENIX2014T", help="German dataset dir")
    parser.add_argument(
        "--en-dir", default="data/PHOENIX2014T_ENGLISH", help="English dataset dir"
    )
    parser.add_argument(
        "--out-dir", default="data/PHOENIX2014T_DE_EN_V2", help="Output merged dataset dir"
    )
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    args = parser.parse_args()

    for split in args.splits:
        print(f"\n=== Building split: {split} ===")
        de_file = os.path.join(args.de_dir, f"phoenix14t.pami0.{split}")
        en_file = os.path.join(args.en_dir, f"phoenix14t_english.{split}")
        out_file = os.path.join(args.out_dir, f"phoenix14t_de_en_v2.{split}")

        if not os.path.exists(de_file):
            raise FileNotFoundError(f"Missing German file: {de_file}")
        if not os.path.exists(en_file):
            raise FileNotFoundError(f"Missing English file: {en_file}")

        merged = build_split(de_file, en_file)
        save_pickle(out_file, merged)
        print(f"  Saved {len(merged)} samples to {out_file}")


if __name__ == "__main__":
    main()
