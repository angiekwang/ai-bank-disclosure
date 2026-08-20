#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare the Loughran-McDonald financial sentiment dictionary for AI-context scoring.

Default raw dictionary path:
    Loughran-McDonald_MasterDictionary_1993-2025.csv

Fallback raw dictionary paths:
    metadata/Loughran-McDonald_MasterDictionary_1993-2025.csv
    base_data/Loughran-McDonald_MasterDictionary_1993-2025.csv

Default outputs:
    metadata/lm_dictionary_cleaned.csv
    metadata/lm_dictionary_summary.csv

Example usage:
    python src/prepare_lm_dictionary.py
    python src/prepare_lm_dictionary.py --raw-dictionary-file base_data/Loughran-McDonald_MasterDictionary_1993-2025.csv
"""

# ============================================================
# 1. SETTINGS
# ============================================================

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd


script_folder = Path(__file__).resolve().parent
project_root = script_folder.parent

raw_dictionary_file = project_root / "base_data" / "Loughran-McDonald_MasterDictionary_1993-2025.csv"
cleaned_output_file = project_root / "base_data" / "lm_dictionary_cleaned.csv"
summary_file = project_root / "metadata" / "lm_dictionary_summary.csv"

keep_columns = ["word", "negative", "positive", "uncertainty"]

# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

# Convert raw CSV column names into consistent snake_case names
def standardize_column_name(column: str) -> str:
    column = str(column).strip().lower()
    column = re.sub(r"[^a-z0-9]+", "_", column)
    return column.strip("_")

# Clean dictionary words to match format used in preprocessed text
def normalize_word(value: Any) -> str:
    if pd.isna(value):
        return ""
    word = str(value).strip().lower()
    return re.sub(r"[^a-z]", "", word)

# Convert a raw Loughran-McDonald category value into True or False
def category_to_bool(value: Any) -> bool:
    if pd.isna(value):
        return False

    text = str(value).strip()
    if text == "":
        return False

    numeric = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    if not pd.isna(numeric):
        return numeric != 0

    return text.lower() not in {"0", "false", "no", "nan", "none"}

# Find dictionary file
def resolve_raw_dictionary_path(raw_path: Path) -> Path:
    candidates = [raw_path,]
    for candidate in candidates:
        if candidate.exists():
            if candidate != raw_path:
                print(f"WARNING: Default raw dictionary not found. Using fallback: {candidate}")
            return candidate

    raise FileNotFoundError(
        "Could not find the Loughran-McDonald dictionary. Checked:\n"
        + "\n".join(f"  - {candidate}" for candidate in candidates)
    )

# ============================================================
# 3. PREP DICTIONARY
# ============================================================

# Read, clean, and summarize the Loughran-McDonald dictionary
def prepare_dictionary(raw_dictionary_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_path = resolve_raw_dictionary_path(raw_dictionary_file)
    raw = pd.read_csv(source_path, low_memory=False)
    total_raw_rows = len(raw)

    # Standardize all column names
    raw = raw.rename(columns={column: standardize_column_name(column) for column in raw.columns})
    if "word" not in raw.columns:
        raise ValueError(f"Raw dictionary is missing required Word column. Columns found: {list(raw.columns)}")

    # Start a clean working dataframe with normalized dictionary words
    working = pd.DataFrame()
    working["word"] = raw["word"].map(normalize_word)
    blank_words_removed = int(working["word"].eq("").sum())
    working = working.loc[working["word"] != ""].copy()

    # Convert each selected LM sentiment category into a True/False indicator
    for category in ["negative", "positive", "uncertainty"]:
        if category in raw.columns:
            working[category] = raw.loc[working.index, category].map(category_to_bool).astype(bool)
        else:
            print(f"WARNING: Missing LM category column '{category}'. Setting all values to False.")
            working[category] = False

    duplicate_words_removed = int(working.duplicated(subset=["word"]).sum())
    cleaned = (
        working.groupby("word", as_index=False)[["negative", "positive", "uncertainty"]]
        .max()
        .sort_values("word")
    )

    cleaned = cleaned[keep_columns]

    summary = pd.DataFrame(
        [
            {
                "raw_dictionary_file": str(source_path),
                "total_raw_rows": total_raw_rows,
                "total_clean_words": len(cleaned),
                "negative_word_count": int(cleaned["negative"].sum()),
                "positive_word_count": int(cleaned["positive"].sum()),
                "uncertainty_word_count": int(cleaned["uncertainty"].sum()),
                "duplicate_words_removed": duplicate_words_removed,
                "blank_words_removed": blank_words_removed,
            }
        ]
    )

    return cleaned, summary

# ============================================================
# 4. COMMAND-LINE ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the Loughran-McDonald dictionary for sentiment scoring.")
    parser.add_argument("--raw-dictionary-file", type=Path, default=raw_dictionary_file)
    parser.add_argument("--cleaned-output-file", type=Path, default=cleaned_output_file)
    parser.add_argument("--summary-file", type=Path, default=summary_file)
    return parser.parse_args()

# ============================================================
# 5. MAIN
# ============================================================

def main() -> None:
    args = parse_args()
    cleaned, summary = prepare_dictionary(args.raw_dictionary_file)

    args.cleaned_output_file.parent.mkdir(parents=True, exist_ok=True)
    args.summary_file.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.cleaned_output_file, index=False)
    summary.to_csv(args.summary_file, index=False)

    print("\nLoughran-McDonald dictionary preparation complete.")
    print(f"Cleaned dictionary rows: {len(cleaned):,}")
    print(f"Negative words: {int(cleaned['negative'].sum()):,}")
    print(f"Positive words: {int(cleaned['positive'].sum()):,}")
    print(f"Uncertainty words: {int(cleaned['uncertainty'].sum()):,}")
    print(f"Cleaned output: {args.cleaned_output_file}")
    print(f"Summary: {args.summary_file}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
