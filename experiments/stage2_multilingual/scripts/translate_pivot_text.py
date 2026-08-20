#!/usr/bin/env python3
"""
Translate pivot text lines into a target language using a multilingual MT model.
"""

import argparse
from pathlib import Path
from typing import List

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def write_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate pivot text file.")
    parser.add_argument("--input", required=True, help="Input txt file (one sentence per line)")
    parser.add_argument("--output", required=True, help="Output txt file")
    parser.add_argument("--target-lang", required=True, help="NLLB target language code (e.g., fra_Latn)")
    parser.add_argument("--source-lang", default="eng_Latn", help="NLLB source language code")
    parser.add_argument(
        "--model",
        default="facebook/nllb-200-distilled-600M",
        help="Hugging Face model id",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    lines = read_lines(input_path)
    if not lines:
        write_lines(output_path, [])
        print("Input file is empty. Wrote empty output.")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    tokenizer.src_lang = args.source_lang
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(args.target_lang)
    if forced_bos_token_id is None or forced_bos_token_id < 0:
        raise ValueError(f"Unknown target language token: {args.target_lang}")

    translated: List[str] = []
    for start in range(0, len(lines), args.batch_size):
        batch = lines[start : start + args.batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        out = model.generate(
            **enc,
            forced_bos_token_id=forced_bos_token_id,
            max_length=256,
        )
        translated.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
        print(f"Translated {min(start + len(batch), len(lines))}/{len(lines)}")

    write_lines(output_path, translated)
    print(f"Saved translations to: {output_path}")


if __name__ == "__main__":
    main()
