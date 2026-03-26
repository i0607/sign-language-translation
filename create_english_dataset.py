#!/usr/bin/env python3
"""
Create English version of PHOENIX-2014-T dataset.

This script:
1. Loads existing gzipped pickle files (with German text)
2. Replaces German text with English translations
3. Saves new pickle files with English text

The features (sign) and glosses remain unchanged - only the text field changes.
"""

import pickle
import gzip
import os
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import csv
import re

# Translation options
try:
    from googletrans import Translator
    HAS_GOOGLETRANS = True
except ImportError:
    HAS_GOOGLETRANS = False
    print("Warning: googletrans not installed. Install with: pip install googletrans==4.0.0rc1")


def load_pickle_file(filepath: str) -> List[Dict]:
    """Load gzipped pickle file."""
    print(f"Loading {filepath}...")
    with gzip.open(filepath, "rb") as f:
        data = pickle.load(f)
    print(f"  ✓ Loaded {len(data)} sequences")
    return data


def translate_german_to_english(text: str, translator=None) -> str:
    """
    Translate German text to English.
    
    Args:
        text: German text string
        translator: Translator object (optional)
    
    Returns:
        English translation
    """
    if translator is None:
        if not HAS_GOOGLETRANS:
            raise ImportError("googletrans not available. Install with: pip install googletrans==4.0.0rc1")
        translator = Translator()
    
    try:
        result = translator.translate(text, src='de', dest='en')
        return result.text
    except Exception as e:
        print(f"  ⚠️ Translation error for '{text[:50]}...': {e}")
        return text  # Return original if translation fails


def load_english_from_csv(csv_file: str) -> Dict[str, str]:
    """
    Load English translations from CSV file.
    
    Expected CSV columns:
    - name: sequence ID
    - text_english or translation_english: English translation
    
    Args:
        csv_file: Path to CSV file
    
    Returns:
        Dictionary mapping sequence_id -> english_text
    """
    import pandas as pd
    
    df = pd.read_csv(csv_file)
    
    # Try different possible column names
    english_col = None
    for col in ['text_english', 'translation_english', 'english', 'eng']:
        if col in df.columns:
            english_col = col
            break
    
    if english_col is None:
        raise ValueError(f"No English column found in CSV. Available columns: {df.columns.tolist()}")
    
    name_col = 'name' if 'name' in df.columns else df.columns[0]
    
    translations = {}
    for _, row in df.iterrows():
        seq_id = row[name_col]
        english_text = row[english_col]
        translations[seq_id] = english_text
    
    print(f"  ✓ Loaded {len(translations)} English translations from CSV")
    return translations


def _is_special_gloss_token(token: str) -> bool:
    """Keep technical/special markers unchanged when translating gloss tokens."""
    if not token:
        return True
    if token.startswith("__") and token.endswith("__"):
        return True
    if token.startswith("loc-") or token.startswith("cl-") or token.startswith("poss-"):
        return True
    return False


def _normalize_sequence_id(name: str) -> str:
    """Normalize sample IDs so release annotations and pickle entries can be matched."""
    if not name:
        return ""
    value = name.strip().replace("\\", "/")
    # Keep only the final path fragment (e.g., train/foo -> foo)
    value = value.split("/")[-1]
    # Drop file extension if present.
    if "." in value:
        value = value.rsplit(".", 1)[0]
    return value


def _candidate_sequence_ids(name: str) -> List[str]:
    """Return candidate keys used to match one sample across different naming conventions."""
    raw = (name or "").strip()
    normalized = _normalize_sequence_id(raw)
    candidates = [raw, normalized]
    if "/" in raw:
        # Also try removing the first path segment if there are nested fragments.
        parts = raw.split("/")
        if len(parts) > 1:
            candidates.append("/".join(parts[1:]))
            candidates.append(_normalize_sequence_id("/".join(parts[1:])))
    # Unique while keeping order
    seen = set()
    uniq = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def translate_gloss_to_english(gloss: str, translator) -> str:
    """
    Translate a PHOENIX gloss sequence token-by-token to English.

    We avoid translating special annotation markers and keep them as-is.
    """
    if not gloss:
        return gloss

    translated_tokens = []
    for token in gloss.split():
        if _is_special_gloss_token(token):
            translated_tokens.append(token)
            continue

        token_for_mt = token.replace("-", " ").replace("_", " ")
        try:
            translated = translator.translate(token_for_mt, src="de", dest="en").text
            translated = re.sub(r"\s+", "_", translated.strip().upper())
            translated_tokens.append(translated if translated else token)
        except Exception:
            translated_tokens.append(token)

    return " ".join(translated_tokens)


def load_release_v3_annotations(
    release_dir: str,
    split: str,
    gloss_col: str = None,
    text_col: str = None,
    translator=None,
) -> Dict[str, Tuple[str, str]]:
    """
    Load PHOENIX-2014-T release-v3 annotations and convert to English gloss/text.

    Returns:
        Dict[name] -> (english_gloss, english_text)
    """
    manual_dir = os.path.join(
        release_dir,
        "PHOENIX-2014-T",
        "annotations",
        "manual",
    )
    candidates = [
        f"PHOENIX-2014-T.{split}.corpus_eng.csv",
        f"PHOENIX-2014-T.{split}.corpus-eng.csv",
        f"PHOENIX-2014-T.{split}.corpus_english.csv",
        f"PHOENIX-2014-T.{split}.corpus.csv",
    ]
    ann_file = None
    for filename in candidates:
        path = os.path.join(manual_dir, filename)
        if os.path.exists(path):
            ann_file = path
            break
    if ann_file is None:
        raise FileNotFoundError(
            f"No annotation file found for split={split} in {manual_dir}. "
            f"Tried: {candidates}"
        )

    rows: Dict[str, Tuple[str, str]] = {}
    with open(ann_file, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters="|,;\t")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = "|"

        reader = csv.DictReader(f, delimiter=delimiter)
        normalized_fieldnames = [(name or "").strip().lstrip("\ufeff") for name in (reader.fieldnames or [])]
        if reader.fieldnames:
            reader.fieldnames = normalized_fieldnames
        fieldnames = set(normalized_fieldnames)
        if "name" not in fieldnames:
            raise ValueError(
                f"Missing required column 'name' in {ann_file}. Found: {reader.fieldnames}"
            )

        default_gloss_candidates = [
            "orth_english",
            "gloss_english",
            "orth_en",
            "gloss_en",
            "orth",
            "gloss",
        ]
        default_text_candidates = [
            "translation_english",
            "text_english",
            "translation_en",
            "text_en",
            "translation",
            "text",
        ]

        if gloss_col is None:
            gloss_col = next((c for c in default_gloss_candidates if c in fieldnames), None)
        if text_col is None:
            text_col = next((c for c in default_text_candidates if c in fieldnames), None)

        if gloss_col is None or text_col is None:
            raise ValueError(
                f"Could not resolve gloss/text columns in {ann_file}. "
                f"Found columns: {reader.fieldnames}"
            )
        if gloss_col not in fieldnames or text_col not in fieldnames:
            raise ValueError(
                f"Configured columns not found in {ann_file}: gloss_col={gloss_col}, text_col={text_col}. "
                f"Found columns: {reader.fieldnames}"
            )

        print(
            f"  Using annotation columns for split={split}: "
            f"gloss='{gloss_col}', text='{text_col}', delimiter='{delimiter}'"
        )

        for idx, row in enumerate(reader, start=1):
            name = row["name"].strip()
            source_gloss = row[gloss_col].strip()
            source_text = row[text_col].strip()

            if translator is None:
                # Fallback mode: keep original labels when MT is unavailable.
                english_gloss = source_gloss
                english_text = source_text
            else:
                english_gloss = translate_gloss_to_english(source_gloss, translator)
                english_text = translate_german_to_english(source_text, translator)

            rows[name] = (english_gloss, english_text)
            normalized_name = _normalize_sequence_id(name)
            rows[normalized_name] = (english_gloss, english_text)

            if idx % 500 == 0:
                print(f"  Processed {idx} annotation rows for split={split}...")

    print(f"  ✓ Loaded {len(rows)} release-v3 annotations for split={split}")
    return rows


def create_english_dataset(
    input_file: str,
    output_file: str,
    english_translations: Dict[str, str] = None,
    translate: bool = False
) -> None:
    """
    Create English version of dataset.
    
    Args:
        input_file: Path to input pickle file (German)
        output_file: Path to output pickle file (English)
        english_translations: Dictionary of sequence_id -> english_text
        translate: If True, translate German to English on-the-fly
    """
    # Load original data
    data = load_pickle_file(input_file)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    # Process each sequence
    english_data = []
    translator = None
    
    if translate:
        if not HAS_GOOGLETRANS:
            raise ImportError("googletrans required for translation. Install with: pip install googletrans==4.0.0rc1")
        translator = Translator()
        print("  Translating German to English...")
    
    for i, entry in enumerate(data):
        sequence_id = entry["name"]
        german_text = entry["text"]
        
        # Get English translation
        if english_translations and sequence_id in english_translations:
            english_text = english_translations[sequence_id]
        elif translate:
            english_text = translate_german_to_english(german_text, translator)
            if (i + 1) % 100 == 0:
                print(f"    Translated {i + 1}/{len(data)} sequences...")
        else:
            print(f"  ⚠️ No English translation for {sequence_id}, skipping...")
            continue
        
        # Create new entry with English text
        english_entry = {
            "name": entry["name"],
            "signer": entry["signer"],
            "sign": entry["sign"],  # Features unchanged
            "gloss": entry["gloss"],  # Glosses unchanged
            "text": english_text  # English translation
        }
        english_data.append(english_entry)
    
    # Save new pickle file
    print(f"Saving to {output_file}...")
    with gzip.open(output_file, "wb") as f:
        pickle.dump(english_data, f)
    
    print(f"  ✓ Saved {len(english_data)} sequences with English translations")
    print(f"  ✓ File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")


def create_english_dataset_from_release_v3(
    input_file: str,
    output_file: str,
    release_dir: str,
    split: str,
    gloss_col: str = None,
    text_col: str = None,
    translate: bool = True,
) -> None:
    """
    Build an English pre-extracted dataset by combining:
    - pre-extracted sign features from `input_file`
    - annotations from PHOENIX release-v3 (matched by sample `name`)
    """
    data = load_pickle_file(input_file)
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)

    translator = None
    if translate:
        if not HAS_GOOGLETRANS:
            raise ImportError("googletrans required for translation. Install with: pip install googletrans==4.0.0rc1")
        translator = Translator()
        print("  Translating release-v3 gloss+translation to English...")
    else:
        print("  Translation disabled: output will keep original release-v3 German labels.")

    release_annotations = load_release_v3_annotations(
        release_dir=release_dir,
        split=split,
        gloss_col=gloss_col,
        text_col=text_col,
        translator=translator,
    )

    english_data = []
    missing = 0
    for entry in data:
        sequence_id = entry["name"]
        matched = None
        for candidate in _candidate_sequence_ids(sequence_id):
            if candidate in release_annotations:
                matched = release_annotations[candidate]
                break
        if matched is None:
            missing += 1
            continue

        english_gloss, english_text = matched
        english_data.append(
            {
                "name": entry["name"],
                "signer": entry["signer"],
                "sign": entry["sign"],
                "gloss": english_gloss,
                "text": english_text,
            }
        )

    print(f"  Matched sequences: {len(english_data)} / {len(data)}")
    if missing:
        print(f"  ⚠️ Missing annotations for {missing} sequences")

    print(f"Saving to {output_file}...")
    with gzip.open(output_file, "wb") as f:
        pickle.dump(english_data, f)
    print(f"  ✓ Saved {len(english_data)} sequences")
    print(f"  ✓ File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Create English version of PHOENIX-2014-T dataset"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/PHOENIX2014T",
        help="Directory containing input pickle files (German)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/PHOENIX2014T_ENGLISH",
        help="Directory to save output pickle files (English)"
    )
    parser.add_argument(
        "--english-csv",
        type=str,
        default=None,
        help="CSV file with English translations (optional)"
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Translate German to English using googletrans (if CSV not provided)"
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "dev", "test"],
        help="Dataset splits to process"
    )
    parser.add_argument(
        "--from-release-v3",
        action="store_true",
        help="Use PHOENIX-2014-T release-v3 annotations (orth/translation) matched by sample name"
    )
    parser.add_argument(
        "--release-dir",
        type=str,
        default="PHOENIX-2014-T-release-v3",
        help="Root path of PHOENIX-2014-T-release-v3"
    )
    parser.add_argument(
        "--release-gloss-col",
        type=str,
        default=None,
        help="Annotation column to use as gloss from release-v3 CSV (auto-detected if omitted)"
    )
    parser.add_argument(
        "--release-text-col",
        type=str,
        default=None,
        help="Annotation column to use as text from release-v3 CSV (auto-detected if omitted)"
    )
    
    args = parser.parse_args()
    
    # Legacy mode (CSV/on-the-fly translation from existing pickle text)
    english_translations = None
    if not args.from_release_v3:
        if args.english_csv:
            print(f"Loading English translations from {args.english_csv}...")
            english_translations = load_english_from_csv(args.english_csv)
        elif not args.translate:
            print("⚠️ Warning: No English CSV provided and --translate not set.")
            print("   Either provide --english-csv or use --translate flag")
            return

    # Process each split
    for split in args.splits:
        input_file = os.path.join(args.input_dir, f"phoenix14t.pami0.{split}")
        output_file = os.path.join(args.output_dir, f"phoenix14t_english.{split}")
        
        if not os.path.exists(input_file):
            print(f"⚠️ Input file not found: {input_file}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {split} split...")
        print(f"{'='*60}")
        
        if args.from_release_v3:
            create_english_dataset_from_release_v3(
                input_file=input_file,
                output_file=output_file,
                release_dir=args.release_dir,
                split=split,
                gloss_col=args.release_gloss_col,
                text_col=args.release_text_col,
                translate=args.translate,
            )
        else:
            create_english_dataset(
                input_file=input_file,
                output_file=output_file,
                english_translations=english_translations,
                translate=args.translate
            )
    
    print(f"\n{'='*60}")
    print("✓ English dataset creation complete!")
    print(f"{'='*60}")
    print(f"\nOutput directory: {args.output_dir}")
    print("\nNext steps:")
    print("1. Create config file pointing to English dataset")
    print("2. Update txt_vocab_limit if English vocabulary is larger")
    print("3. Train with: python -m signjoey train configs/sign_improved_v4_english.yaml")


if __name__ == "__main__":
    main()
