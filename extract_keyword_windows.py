#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract sentence windows around AI keyword matches.

Default input:
    preprocessed_texts/

Keywords:
    1. "artificial intelligen" - case insensitive
    2. "AI" - case sensitive, standalone term

The output is one row per keyword occurrence. Each row includes:
    1. the sentence before the keyword sentence,
    2. the sentence containing the keyword,
    3. the sentence after the keyword sentence.

These windows are intended as inputs for later sentiment analysis.
"""

# ============================================================
# 1. SETTINGS
# ============================================================

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any

import pandas as pd

script_folder = Path(__file__).resolve().parent
project_root = script_folder.parent

input_folder = project_root / "preprocessed_texts"
output_file = project_root / "outputs" / "keyword_windows.csv"

# Keyword "artificial intelligen" pattern
artificial_intelligen_re = re.compile(
    re.escape("artificial intelligen"),
    flags=re.IGNORECASE,
)

# Keyword "AI" pattern
ai_re = re.compile(r"(?<![A-Za-z])AI(?![A-Za-z])")

filename_re = re.compile(
    r"^(?P<cik>\d{10})_"
    r"(?P<ticker>[^_]+)_"
    r"(?P<filing_date>\d{4}-\d{2}-\d{2})_"
    r"(?P<accession_number>\d{10}-\d{2}-\d{6})_"
    r"Item_(?P<item>1A|1|7)\.txt$",
    flags=re.IGNORECASE,
)

common_abbreviations = {
    "co",
    "corp",
    "dr",
    "e.g",
    "etc",
    "i.e",
    "inc",
    "jr",
    "mr",
    "mrs",
    "ms",
    "no",
    "sec",
    "sr",
    "u.s",
    "vs",
}

# Normalize whitespace in extracted sentence windows
def clean_context(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

# Build stable IDs for context-window rows
def stable_hash(parts: list[Any]) -> str:
    text = "||".join(clean_value(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]

# Label each keyword match by search rule
def matched_keyword_type(keyword: str) -> str:
    if keyword == "AI":
        return "AI_acronym"
    return "artificial_intelligence_phrase"

# Normalize 10-K item labels to a consistent display format
def normalize_item_label(value: Any) -> str:
    text = clean_value(value).upper().replace("_", " ").strip()
    text = text.replace("ITEM", "").strip()

    if text == "1":
        return "Item 1"
    if text == "1A":
        return "Item 1A"
    if text == "7":
        return "Item 7"

    return clean_value(value)

# Get the word before punctuation to avoid mistaking abbreviations as sentence endings
def ending_word(text: str, punctuation_start: int) -> str:
    prefix = text[max(0, punctuation_start - 40):punctuation_start].rstrip()
    match = re.search(r"([A-Za-z](?:[A-Za-z]|\.)*)$", prefix)
    return match.group(1).lower().rstrip(".") if match else ""

# Decide whether punctuation marks a sentence boundary: Avoid treating periods in common abbreviations like "U.S." or "Inc." as sentence endings
def is_sentence_boundary(text: str, punctuation_match: re.Match) -> bool:
    punctuation = punctuation_match.group(0)
    boundary_end = punctuation_match.end()

    if boundary_end >= len(text):
        return True

    if not text[boundary_end].isspace():
        return False

    if punctuation == ".":
        word = ending_word(text, punctuation_match.start())
        if word in common_abbreviations:
            return False

    return True

# Split text into sentence character spans (character range where keyword is found) in order to identify sentence before & after keyword sentence
def sentence_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    sentence_start = 0

    for match in re.finditer(r"[.!?]+", text):
        if not is_sentence_boundary(text, match):
            continue

        sentence_end = match.end()
        while sentence_end < len(text) and text[sentence_end] in "\"')]}":
            sentence_end += 1

        if text[sentence_start:sentence_end].strip():
            spans.append((sentence_start, sentence_end))

        sentence_start = sentence_end
        while sentence_start < len(text) and text[sentence_start].isspace():
            sentence_start += 1

    if text[sentence_start:].strip():
        spans.append((sentence_start, len(text)))

    return spans

# Find which sentence contains a keyword match
def sentence_index_for_match(
    sentences: list[tuple[int, int]],
    start: int,
    end: int,
) -> int:
    for index, (sentence_start, sentence_end) in enumerate(sentences):
        if sentence_start <= start < sentence_end:
            return index
        if sentence_start < end <= sentence_end:
            return index
    return -1

# Return previous/current/next sentence around a keyword match
def extract_sentence_window(
    text: str,
    start: int,
    end: int,
    sentences: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    if sentences is None:
        sentences = sentence_spans(text)

    sentence_index = sentence_index_for_match(sentences, start, end)

    if sentence_index < 0:
        return {
            "previous_sentence": "",
            "keyword_sentence": clean_context(text[start:end]),
            "next_sentence": "",
            "sentence_window": clean_context(text[start:end]),
            "context_start_char": start,
            "context_end_char": end,
            "keyword_sentence_number": "",
            "sentence_count_in_file": len(sentences),
        }

    previous_sentence = sentences[sentence_index - 1] if sentence_index > 0 else None
    keyword_sentence = sentences[sentence_index]
    next_sentence = sentences[sentence_index + 1] if sentence_index + 1 < len(sentences) else None

    window_start = previous_sentence[0] if previous_sentence else keyword_sentence[0]
    window_end = next_sentence[1] if next_sentence else keyword_sentence[1]

    return {
        "previous_sentence": clean_context(text[previous_sentence[0]:previous_sentence[1]]) if previous_sentence else "",
        "keyword_sentence": clean_context(text[keyword_sentence[0]:keyword_sentence[1]]),
        "next_sentence": clean_context(text[next_sentence[0]:next_sentence[1]]) if next_sentence else "",
        "sentence_window": clean_context(text[window_start:window_end]),
        "context_start_char": window_start,
        "context_end_char": window_end,
        "keyword_sentence_number": sentence_index + 1,
        "sentence_count_in_file": len(sentences),
    }

# Standardize missing values and whitespace before writing output fields
def clean_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()

# Standardize CIKs to 10-digit strings
def clean_cik(value: Any) -> str:
    digits = re.sub(r"\D", "", clean_value(value))
    return digits.zfill(10) if digits else ""

# Read text files using common encodings
def read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=encoding, errors="replace")
        except UnicodeError:
            continue
    return path.read_text(errors="replace")

# Parse expected preprocessed 10-K filenames into filing identifiers
def parse_source_file(file_name: str) -> dict[str, Any]:
    match = filename_re.match(file_name)
    if not match:
        return {
            "cik": "",
            "ticker": "",
            "filing_date": "",
            "filing_year": "",
            "accession_number": "",
            "item": "",
        }

    filing_date = match.group("filing_date")
    return {
        "cik": match.group("cik"),
        "ticker": match.group("ticker").upper(),
        "filing_date": filing_date,
        "filing_year": filing_date[:4],
        "accession_number": match.group("accession_number"),
        "item": normalize_item_label(match.group("item")),
    }

# Build one row per available preprocessed text file, then add bank identifiers when available
def build_document_index(input_dir: Path, bank_id_file: Path) -> pd.DataFrame:
    rows = []

    for path in sorted(input_dir.glob("*.txt")):
        parsed = parse_source_file(path.name)
        rows.append(
            {
                "file_name": path.name,
                "input_path": str(path),
                **parsed,
            }
        )

    documents = pd.DataFrame(rows)

    if bank_id_file.exists() and not documents.empty:
        bank_ids = pd.read_csv(bank_id_file, dtype=str)

        cik_column = "cik" if "cik" in bank_ids.columns else "cik_str"
        if cik_column in bank_ids.columns:
            bank_ids["cik"] = bank_ids[cik_column].map(clean_cik)

            keep_columns = [
                column
                for column in [
                    "cik",
                    "rssd_id",
                    "consolidated_assets_mil",
                    "bank_holding_company_raw",
                    "holding_company_name",
                    "sec_name",
                ]
                if column in bank_ids.columns
            ]

            documents = documents.merge(
                bank_ids[keep_columns].drop_duplicates(subset=["cik"]),
                on="cik",
                how="left",
            )

    return documents

# Find all AI keyword matches and notes character positions
def keyword_matches(text: str) -> list[dict[str, Any]]:
    matches = []

    for match in artificial_intelligen_re.finditer(text):
        matches.append(
            {
                "keyword": "artificial intelligen",
                "matched_text": match.group(0),
                "match_start_char": match.start(),
                "match_end_char": match.end(),
            }
        )

    for match in ai_re.finditer(text):
        matches.append(
            {
                "keyword": "AI",
                "matched_text": match.group(0),
                "match_start_char": match.start(),
                "match_end_char": match.end(),
            }
        )

    return sorted(matches, key=lambda row: row["match_start_char"])

# Build one output row per keyword occurrence
def build_windows(input_dir: Path, bank_id_file: Path) -> pd.DataFrame:
    documents = build_document_index(input_dir, bank_id_file)
    rows = []

    for _, document in documents.iterrows():
        path = Path(clean_value(document.get("input_path", "")))
        if not path.is_file():
            continue

        text = read_text(path)
        sentences = sentence_spans(text)

        for match_number, match in enumerate(keyword_matches(text), start=1):
            window = extract_sentence_window(
                text=text,
                start=match["match_start_char"],
                end=match["match_end_char"],
                sentences=sentences,
            )

            context_window_id = stable_hash(
                [
                    document["file_name"],
                    match["matched_text"],
                    match["match_start_char"],
                    match["match_end_char"],
                    window["context_start_char"],
                    window["context_end_char"],
                ]
            )

            rows.append(
                {
                    "context_window_id": context_window_id,
                    "file_name": document["file_name"],
                    "input_path": document["input_path"],
                    "cik": document.get("cik", ""),
                    "ticker": document.get("ticker", ""),
                    "filing_date": document.get("filing_date", ""),
                    "filing_year": document.get("filing_year", ""),
                    "accession_number": document.get("accession_number", ""),
                    "item": document.get("item", ""),
                    "matched_keyword": match["matched_text"],
                    "matched_keyword_type": matched_keyword_type(match["keyword"]),
                    "context_window_text": window["sentence_window"],
                    "rssd_id": document.get("rssd_id", ""),
                    "consolidated_assets_mil": document.get("consolidated_assets_mil", ""),
                    "match_number_in_file": match_number,
                    "keyword": match["keyword"],
                    "matched_text": match["matched_text"],
                    "match_start_char": match["match_start_char"],
                    "match_end_char": match["match_end_char"],
                    "sentences_before": 1,
                    "sentences_after": 1,
                    **window,
                }
            )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract sentence windows around AI keyword matches."
    )
    parser.add_argument("--input-dir", type=Path, default=input_folder)
    parser.add_argument("--bank-id-file", type=Path, default=project_root / "base_data" / "bank_identifiers_linked.csv")
    parser.add_argument("--output-file", type=Path, default=output_file)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    bank_id_file = args.bank_id_file.expanduser().resolve()
    output_path = args.output_file.expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    windows = build_windows(input_dir=input_dir, bank_id_file=bank_id_file)
    windows.to_csv(output_path, index=False)

    print("AI keyword context-window extraction complete.")
    print(f"Keyword windows written: {len(windows):,}")
    print(f"Output: {output_path}")

if __name__ == "__main__":
    main()