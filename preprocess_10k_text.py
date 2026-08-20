#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocess combined 10-K item text files.

This script reads raw 10-K item text files, creates cleaned readable text files,
creates tokenized text files, and writes preprocessing diagnostics.

Default input:
    10k_texts_scraped/

Default outputs:
    preprocessed_texts/
    preprocessed_tokens/
    metadata/preprocess_file_level_diagnostics.csv
    metadata/preprocess_summary.csv
    metadata/preprocess_vocabulary_summary.csv

Preprocessing pipeline:
    raw text
        -> decode HTML entities
        -> remove remaining HTML tags
        -> remove numbers
        -> normalize whitespace
        -> cleaned readable text
        -> whitespace tokenization
        -> remove very short tokens except protected terms
        -> tokenized text

Important design choices:
    - Original letter case is preserved.
    - Punctuation is preserved in the cleaned readable text.
    - Stopwords are retained.
    - No stemming or lemmatization is applied.
    - Raw input files are never modified.
"""

from __future__ import annotations

import argparse
import html
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# 1. SETTINGS
# ============================================================

script_folder = Path(__file__).resolve().parent
project_root = script_folder.parent

input_folder = project_root / "10k_texts_scraped"
text_output_folder = project_root / "preprocessed_texts"
token_output_folder = project_root / "preprocessed_tokens"
metadata_output_folder = project_root / "metadata"

diagnostic_file = project_root / "metadata" / "preprocess_file_level_diagnostics.csv"
summary_file = project_root / "metadata" / "preprocess_summary.csv"
vocab_file = project_root / "metadata" / "preprocess_vocabulary_summary.csv"

min_token_length = 3
min_review_word_count = 20
min_review_token_count = 20

protected_short_tokens = {
    "ai",
}

filename_pattern = re.compile(
    r"^(?P<cik>\d{10})_"
    r"(?P<ticker>[^_]+)_"
    r"(?P<filing_date>\d{4}-\d{2}-\d{2})_"
    r"(?P<accession>\d{10}-\d{2}-\d{6})_"
    r"Item_(?P<item>1A|1|7)\.txt$",
    flags=re.IGNORECASE,
)

diagnostic_columns = [
    "file_name",
    "parse_success",
    "cik_10",
    "ticker",
    "filing_date",
    "accession_number",
    "item",
    "raw_word_count",
    "clean_word_count",
    "token_count",
    "unique_token_count",
    "preprocess_status",
    "preprocess_reason",
    "needs_manual_review",
]


# ============================================================
# 2. FILENAME PARSING
# ============================================================

# Normalizes item name to consistent format for diagnostics
def normalize_item(item: str) -> str:
    item = item.upper().strip()

    if item == "1":
        return "Item 1"
    if item == "1A":
        return "Item 1A"
    if item == "7":
        return "Item 7"

    return item

# Parses filename to extract CIK, ticker, filing date, accession number, and item
def parse_filename(file_name: str) -> dict[str, Any]:
    match = filename_pattern.match(file_name)

    if not match:
        return {
            "parse_success": False,
            "cik_10": "",
            "ticker": "",
            "filing_date": "",
            "accession_number": "",
            "item": "",
            "parse_reason": (
                "Filename does not match expected format: "
                "{cik}_{ticker}_{filing_date}_{accession_number}_Item_{item}.txt"
            ),
        }

    return {
        "parse_success": True,
        "cik_10": match.group("cik"),
        "ticker": match.group("ticker").upper(),
        "filing_date": match.group("filing_date"),
        "accession_number": match.group("accession"),
        "item": normalize_item(match.group("item")),
        "parse_reason": "",
    }


# ============================================================
# 3. TEXT PROCESSING
# ============================================================

# Reads text file with multiple encoding attempts, returning the text as a string
def read_text_file(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=encoding, errors="replace")
        except Exception:
            continue

    raise OSError(f"Could not read file with supported encodings: {path}")


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))

def remove_html_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)

def remove_numbers(text: str) -> str:
    return re.sub(r"\d+", " ", text)

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

# Cleans raw text by decoding HTML entities, removing HTML tags, removing numbers, and normalizing whitespace
def clean_text(raw_text: str) -> str:
    text = html.unescape(raw_text)
    text = remove_html_tags(text)
    text = remove_numbers(text)
    text = normalize_whitespace(text)

    return text

# Tokenizes cleaned text by splitting on whitespace and filtering out very short tokens, except for protected terms
def tokenize_text(cleaned_text: str) -> list[str]:
    tokens = cleaned_text.split()
    filtered_tokens: list[str] = []

    for token in tokens:
        is_protected = token.lower() in protected_short_tokens

        if len(token) < min_token_length and not is_protected:
            continue

        filtered_tokens.append(token)

    return filtered_tokens


# ============================================================
# 4. DIAGNOSTICS AND SUMMARY    
# ============================================================

def get_manual_review_reasons(
    parsed: dict[str, Any],
    clean_word_count: int,
    token_count: int,
) -> list[str]:
    reasons: list[str] = []

    if not parsed["parse_success"]:
        reasons.append(parsed["parse_reason"])

    if clean_word_count == 0:
        reasons.append("Cleaned text is empty")

    if token_count == 0:
        reasons.append("Tokenized text is empty")

    if 0 < clean_word_count < min_review_word_count:
        reasons.append(f"Clean word count below {min_review_word_count}")

    if 0 < token_count < min_review_token_count:
        reasons.append(f"Token count below {min_review_token_count}")

    return reasons

# Determines the overall preprocessing status based on manual review reasons
def get_preprocess_status(review_reasons: list[str]) -> tuple[str, str, bool]:
    if not review_reasons:
        return "success", "Preprocessed successfully", False

    return (
        "needs_manual_review",
        "; ".join(review_reasons),
        True,
    )

# Builds a diagnostics row for a successfully processed file
def build_diagnostics_row(
    input_path: Path,
    parsed: dict[str, Any],
    raw_text: str,
    cleaned_text: str,
    tokens: list[str],
) -> dict[str, Any]:
    raw_word_count = count_words(raw_text)
    clean_word_count = count_words(cleaned_text)
    token_count = len(tokens)

    review_reasons = get_manual_review_reasons(
        parsed=parsed,
        clean_word_count=clean_word_count,
        token_count=token_count,
    )

    preprocess_status, preprocess_reason, needs_manual_review = (
        get_preprocess_status(review_reasons)
    )

    return {
        "file_name": input_path.name,
        "parse_success": parsed["parse_success"],
        "cik_10": parsed["cik_10"],
        "ticker": parsed["ticker"],
        "filing_date": parsed["filing_date"],
        "accession_number": parsed["accession_number"],
        "item": parsed["item"],
        "raw_word_count": raw_word_count,
        "clean_word_count": clean_word_count,
        "token_count": token_count,
        "unique_token_count": len(set(tokens)),
        "preprocess_status": preprocess_status,
        "preprocess_reason": preprocess_reason,
        "needs_manual_review": needs_manual_review,
    }

# Builds a diagnostics row for a file that failed preprocessing
def build_failed_diagnostics_row(
    input_path: Path,
    parsed: dict[str, Any],
    error: Exception,
) -> dict[str, Any]:
    return {
        "file_name": input_path.name,
        "parse_success": parsed["parse_success"],
        "cik_10": parsed["cik_10"],
        "ticker": parsed["ticker"],
        "filing_date": parsed["filing_date"],
        "accession_number": parsed["accession_number"],
        "item": parsed["item"],
        "raw_word_count": None,
        "clean_word_count": None,
        "token_count": None,
        "unique_token_count": None,
        "preprocess_status": "failed",
        "preprocess_reason": str(error),
        "needs_manual_review": True,
    }


# ============================================================
# 5. SINGLE-FILE PROCESSING
# ============================================================

# Processes a single input file: reads raw text, cleans it, tokenizes it, writes cleaned and tokenized files, and returns diagnostics and token counts
def process_file(
    input_path: Path,
    clean_output_dir: Path,
    token_output_dir: Path,
) -> tuple[dict[str, Any], Counter]:
    parsed = parse_filename(input_path.name)

    cleaned_text_path = clean_output_dir / input_path.name
    token_text_path = token_output_dir / input_path.name

    raw_text = read_text_file(input_path)
    cleaned_text = clean_text(raw_text)
    tokens = tokenize_text(cleaned_text)

    cleaned_text_path.write_text(cleaned_text, encoding="utf-8")
    token_text_path.write_text(" ".join(tokens), encoding="utf-8")

    diagnostics = build_diagnostics_row(
        input_path=input_path,
        parsed=parsed,
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        tokens=tokens,
    )

    return diagnostics, Counter(tokens)

# ============================================================
# 6. CORPUS SUMMARY AND OUTPUTS
# ============================================================

# Summary dataframe 
def numeric_sum(df: pd.DataFrame, column: str) -> int:
    return int(pd.to_numeric(df[column], errors="coerce").fillna(0).sum())

# Calculates median of numeric column
def numeric_median(df: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(df[column], errors="coerce").dropna()

    if values.empty:
        return 0.0

    return round(float(values.median()), 2)

# Calculates minimum of numeric column
def numeric_min(df: pd.DataFrame, column: str) -> int:
    values = pd.to_numeric(df[column], errors="coerce").dropna()

    if values.empty:
        return 0

    return int(values.min())

# Calculates maximum of numeric column
def numeric_max(df: pd.DataFrame, column: str) -> int:
    values = pd.to_numeric(df[column], errors="coerce").dropna()

    if values.empty:
        return 0

    return int(values.max())

# Builds summary dataframe 
def build_summary(
    diagnostics: pd.DataFrame,
    vocabulary: Counter,
    input_dir: Path,
    clean_output_dir: Path,
    token_output_dir: Path,
) -> pd.DataFrame:
    parse_success = diagnostics["parse_success"].astype(bool)
    non_failed = diagnostics["preprocess_status"].ne("failed")

    empty_after_cleaning_files = int(
        pd.to_numeric(
            diagnostics.loc[non_failed, "clean_word_count"],
            errors="coerce",
        )
        .fillna(0)
        .eq(0)
        .sum()
    )

    empty_after_tokenization_files = int(
        pd.to_numeric(
            diagnostics.loc[non_failed, "token_count"],
            errors="coerce",
        )
        .fillna(0)
        .eq(0)
        .sum()
    )

    distinct_filings = (
        diagnostics.loc[
            parse_success,
            ["cik_10", "filing_date", "accession_number"],
        ]
        .drop_duplicates()
        .shape[0]
    )

    rows = [
        ("input_folder", str(input_dir)),
        ("cleaned_text_output_folder", str(clean_output_dir)),
        ("token_output_folder", str(token_output_dir)),
        ("stopwords_removed", "False"),
        ("stemming_or_lemmatization_used", "False"),
        ("case_preserved", "True"),
        ("punctuation_preserved_in_clean_text", "True"),
        ("total_input_files", len(diagnostics)),
        ("successfully_preprocessed_files",
            int(diagnostics["preprocess_status"].ne("failed").sum()),
        ),
        ("failed_files", int(diagnostics["preprocess_status"].eq("failed").sum())),
        ("parse_failed_files", int((~parse_success).sum())),
        ("empty_after_cleaning_files", empty_after_cleaning_files),
        ("empty_after_tokenization_files", empty_after_tokenization_files),
        ("needs_manual_review_files",
            int(diagnostics["needs_manual_review"].astype(bool).sum()),
        ),
        ("unique_ciks", int(diagnostics.loc[parse_success, "cik_10"].nunique())),
        ("unique_tickers", int(diagnostics.loc[parse_success, "ticker"].nunique())),
        ("distinct_filings", int(distinct_filings)),
        ("item_1_count", int(diagnostics["item"].eq("Item 1").sum())),
        ("item_1a_count", int(diagnostics["item"].eq("Item 1A").sum())),
        ("item_7_count", int(diagnostics["item"].eq("Item 7").sum())),
        ("total_raw_words", numeric_sum(diagnostics, "raw_word_count")),
        ("total_clean_words", numeric_sum(diagnostics, "clean_word_count")),
        ("total_tokens", numeric_sum(diagnostics, "token_count")),
        ("unique_tokens_total", len(vocabulary)),
        ("median_raw_word_count", numeric_median(diagnostics, "raw_word_count")),
        ("median_clean_word_count", numeric_median(diagnostics, "clean_word_count")),
        ("median_token_count", numeric_median(diagnostics, "token_count")),
        ("minimum_clean_word_count", numeric_min(diagnostics, "clean_word_count")),
        ("maximum_clean_word_count", numeric_max(diagnostics, "clean_word_count")),
    ]

    return pd.DataFrame(rows, columns=["metric", "value"])

# Writes outputs to CSV files
def write_outputs(
    diagnostics: pd.DataFrame,
    summary: pd.DataFrame,
    vocabulary: Counter,
    metadata_dir: Path,
) -> tuple[Path, Path, Path]:
    metadata_dir.mkdir(parents=True, exist_ok=True)

    diagnostics_path = metadata_dir / diagnostic_file
    summary_path = metadata_dir / summary_file
    vocab_path = metadata_dir / vocab_file

    diagnostics.reindex(columns=diagnostic_columns).to_csv(
        diagnostics_path,
        index=False,
    )

    summary.to_csv(summary_path, index=False)

    vocabulary_rows = [
        {"token": token, "count": count}
        for token, count in vocabulary.most_common()
    ]

    pd.DataFrame(vocabulary_rows).to_csv(vocab_path, index=False)

    return diagnostics_path, summary_path, vocab_path

# ============================================================
# 7. COMMAND-LINE INTERFACE
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess combined 10-K item text files."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=input_folder,
        help="Folder containing raw combined 10-K item text files.",
    )

    parser.add_argument(
        "--clean-output-dir",
        type=Path,
        default=text_output_folder,
        help="Folder where cleaned readable text files will be written.",
    )

    parser.add_argument(
        "--token-output-dir",
        type=Path,
        default=token_output_folder,
        help="Folder where tokenized text files will be written.",
    )

    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=metadata_output_folder,
        help="Folder where diagnostic and summary CSV files will be written.",
    )

    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Optional limit for testing on the first N files.",
    )

    return parser.parse_args()


# ============================================================
# 8. MAIN WORKFLOW
# ============================================================

def main() -> None:
    args = parse_args()

    # Converts folder paths to absolute paths
    input_dir = args.input_dir.expanduser().resolve()
    clean_output_dir = args.clean_output_dir.expanduser().resolve()
    token_output_dir = args.token_output_dir.expanduser().resolve()
    metadata_dir = args.metadata_dir.expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    clean_output_dir.mkdir(parents=True, exist_ok=True)
    token_output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob("*.txt"))

    if args.limit_files is not None:
        input_files = input_files[: max(args.limit_files, 0)]

    print("Input folder:", input_dir)
    print(f"Number of .txt files found: {len(input_files):,}")
    print("Cleaned text output folder:", clean_output_dir)
    print("Token output folder:", token_output_dir)
    print("Metadata folder:", metadata_dir)
    print("Stopwords removed: False")
    print("Case preserved: True")
    print("Punctuation preserved in cleaned text: True")

    # Creates empty lists
    diagnostics_rows: list[dict[str, Any]] = []
    vocabulary = Counter()

    # Loop through input files and process each one
    for index, input_path in enumerate(input_files, start=1):
        if index == 1 or index % 100 == 0 or index == len(input_files):
            print(f"Processing {index:,}/{len(input_files):,}: {input_path.name}")

        parsed = parse_filename(input_path.name)

        try:
            diagnostics, token_counts = process_file(
                input_path=input_path,
                clean_output_dir=clean_output_dir,
                token_output_dir=token_output_dir,
            )

            diagnostics_rows.append(diagnostics)
            vocabulary.update(token_counts)

        except Exception as error:
            diagnostics_rows.append(
                build_failed_diagnostics_row(
                    input_path=input_path,
                    parsed=parsed,
                    error=error,
                )       
            )

    diagnostics_df = pd.DataFrame(diagnostics_rows).reindex(
        columns=diagnostic_columns
    )

    summary_df = build_summary(
        diagnostics=diagnostics_df,
        vocabulary=vocabulary,
        input_dir=input_dir,
        clean_output_dir=clean_output_dir,
        token_output_dir=token_output_dir,
    )

    diagnostics_path, summary_path, vocab_path = write_outputs(
        diagnostics=diagnostics_df,
        summary=summary_df,
        vocabulary=vocabulary,
        metadata_dir=metadata_dir,
    )

    total_files = len(diagnostics_df)
    successful_files = int(diagnostics_df["preprocess_status"].ne("failed").sum())
    failed_files = int(diagnostics_df["preprocess_status"].eq("failed").sum())
    parse_failed_files = int((~diagnostics_df["parse_success"].astype(bool)).sum())
    manual_review_files = int(
        diagnostics_df["needs_manual_review"].astype(bool).sum()
    )

    print("\n=== PREPROCESSING SUMMARY ===")
    print(f"Total files processed: {total_files:,}")
    print(f"Successfully preprocessed files: {successful_files:,}")
    print(f"Failed files: {failed_files:,}")
    print(f"Parse failed files: {parse_failed_files:,}")
    print(f"Files needing manual review: {manual_review_files:,}")

    print("\nOutputs saved:")
    print(f"  Cleaned text folder: {clean_output_dir}")
    print(f"  Token folder:        {token_output_dir}")
    print(f"  Diagnostics:         {diagnostics_path}")
    print(f"  Summary:             {summary_path}")
    print(f"  Vocabulary:          {vocab_path}")


if __name__ == "__main__":
    main()