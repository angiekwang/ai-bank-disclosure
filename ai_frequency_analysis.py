#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Purpose:
    Count AI keyword mentions in preprocessed 10-K token files.

Inputs:
    1. preprocessed_tokens folder
    2. 10k_item_extraction_metadata.csv
    3. bank_identifiers_linked.csv

Outputs:
    1. ai_frequency_item_level.csv
       - one row per bank-filing-item

    2. ai_frequency_bank_year.csv
       - one row per bank-year, aggregated across Items 1, 1A, and 7

    3. ai_frequency_diagnostics.csv
       - simple quality-control summary

Keyword rules:
    1. "artificial intelligen"
       - case-insensitive
       - captures artificial intelligence / artificially intelligent stems

    2. "AI"
       - uppercase only
       - standalone acronym only
"""

# ============================================================
# 1. SETTINGS
# ============================================================

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import re

import pandas as pd


script_folder = Path(__file__).resolve().parent
project_root = script_folder.parent

# Inputs
input_folder = project_root / "preprocessed_tokens"
input_metadata_file = project_root / "metadata" / "10k_item_extraction_metadata.csv"
input_bank_id_file = project_root / "base_data" / "bank_identifiers_linked.csv"

# Outputs
output_folder = project_root / "outputs" / "ai_frequency"
item_level_output_file = project_root / "outputs" / "ai_frequency" / "ai_frequency_item_level.csv"
bank_year_output_file = project_root / "outputs" / "ai_frequency" / "ai_frequency_bank_year.csv"
diagnostics_output_file = project_root / "outputs" / "ai_frequency" / "ai_frequency_diagnostics.csv"

# ============================================================
# 2. KEYWORD PATTERNS
# ============================================================

# Search pattern for phrase stem "artificial intelligen"
artificial_intelligen_re = re.compile(
    re.escape("artificial intelligen"),
    flags=re.IGNORECASE,
)

# Search pattern for standalone uppercase "AI" only
ai_re = re.compile(r"(?<![A-Za-z])AI(?![A-Za-z])")


# ============================================================
# 3. CLEANING HELPER FUNCTIONS
# ============================================================

# Cleans and standardizes column values for merging 

def clean_string(value) -> str:
    """Convert missing values to empty strings and strip whitespace."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def clean_cik(value) -> str:
    """Standardize CIK as a 10-digit string."""
    text = clean_string(value)
    digits = re.sub(r"\D", "", text)
    return digits.zfill(10) if digits else ""


def normalize_date(value) -> str:
    """Normalize dates to YYYY-MM-DD strings when possible."""
    if isinstance(value, (datetime, date)):
        return value.strftime("%Y-%m-%d")

    text = clean_string(value)
    if not text:
        return ""

    parsed = pd.to_datetime(text, errors="coerce")
    if not pd.isna(parsed):
        return parsed.strftime("%Y-%m-%d")

    return text


def filing_year(filing_date: str) -> str:
    """Extract filing year from a YYYY-MM-DD filing date."""
    filing_date = normalize_date(filing_date)
    match = re.match(r"^(\d{4})-\d{2}-\d{2}$", filing_date)
    return match.group(1) if match else ""


def normalize_item(value) -> str:
    """Normalize item labels to 1, 1A, or 7."""
    text = clean_string(value).upper()
    text = text.replace("ITEM", "")
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = re.sub(r"\s+", "", text)

    if text in {"1", "1A", "7"}:
        return text

    return text

# ============================================================
# 4. LOAD METADATA
# ============================================================

def load_metadata(metadata_file: Path, input_folder: Path) -> pd.DataFrame:
    """
    Load 10-K item extraction metadata and prepare one row per item file.

    The metadata file should contain:
        cik
        ticker
        bank_name
        filing_date
        accession_number
        form
        item
        output_file
        status
        extraction_method
        char_count
        word_count
        qc_flag
        reason
        needs_manual_review
    """
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    metadata = pd.read_csv(metadata_file, dtype=str)

    required_columns = [
        "cik",
        "ticker",
        "bank_name",
        "filing_date",
        "accession_number",
        "form",
        "item",
        "output_file",
        "status",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in metadata.columns
    ]

    if missing_columns:
        raise ValueError(
            "Metadata file is missing required columns: "
            + ", ".join(missing_columns)
        )

    metadata["cik"] = metadata["cik"].map(clean_cik)
    metadata["ticker"] = metadata["ticker"].map(clean_string).str.upper()
    metadata["bank_name"] = metadata["bank_name"].map(clean_string)
    metadata["filing_date"] = metadata["filing_date"].map(normalize_date)
    metadata["filing_year"] = metadata["filing_date"].map(filing_year)
    metadata["accession_number"] = metadata["accession_number"].map(clean_string)
    metadata["form"] = metadata["form"].map(clean_string).str.upper()
    metadata["item"] = metadata["item"].map(normalize_item)
    metadata["output_file"] = metadata["output_file"].map(clean_string)

    # Build full path to the preprocessed token file.
    metadata["source_path"] = metadata["output_file"].apply(
        lambda file_name: str(input_folder / file_name) if file_name else ""
    )

    metadata["source_file"] = metadata["output_file"]

    # Optional but useful: keep only successfully extracted 10-K item rows.
    metadata = metadata[
        (metadata["status"].map(clean_string).str.lower() == "success")
        & (metadata["form"] == "10-K")
        & (metadata["item"].isin(["1", "1A", "7"]))
    ].copy()

    return metadata

# ============================================================
# 5. LOAD BANK IDENTIFIERS
# ============================================================

def load_bank_identifiers(bank_id_file: Path) -> pd.DataFrame:
    """
    Load bank-level identifiers to merge onto the AI-frequency output.
    """
    if not bank_id_file.exists():
        raise FileNotFoundError(f"Bank identifier file not found: {bank_id_file}")

    bank_ids = pd.read_csv(bank_id_file, dtype=str)

    if "cik" in bank_ids.columns:
        bank_ids["cik"] = bank_ids["cik"].map(clean_cik)
    elif "cik_str" in bank_ids.columns:
        bank_ids["cik"] = bank_ids["cik_str"].map(clean_cik)
    elif "CIK" in bank_ids.columns:
        bank_ids["cik"] = bank_ids["CIK"].map(clean_cik)
    else:
        raise ValueError(
            "Bank identifier file must contain one of: cik, cik_str, or CIK."
        )

    desired_columns = [
        "cik",
        "rssd_id",
        "permco",
        "gvkey",
        "consolidated_assets_mil",
        "bank_holding_company_raw",
        "holding_company_name",
        "sec_name",
    ]

    available_columns = [
        column for column in desired_columns
        if column in bank_ids.columns
    ]

    if "cik" not in available_columns:
        raise ValueError("Bank identifier file could not create standardized cik column.")

    bank_ids = bank_ids[available_columns].drop_duplicates(subset=["cik"])

    return bank_ids

# ============================================================
# 6. READ TEXT AND COUNT AI KEYWORDS
# ============================================================

def read_text(path: Path) -> str:
    """Read a text file using common encodings."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=encoding, errors="replace")
        except UnicodeError:
            continue

    return path.read_text(errors="replace")

def count_ai_keywords(text: str) -> dict:
    """Count AI keyword mentions and document length."""
    artificial_intelligen_count = len(artificial_intelligen_re.findall(text))
    AI_count = len(ai_re.findall(text))
    ai_raw_freq = artificial_intelligen_count + AI_count

    token_count = len(text.split())
    ai_share = ai_raw_freq / token_count if token_count > 0 else 0

    return {
        "AI_count": AI_count,
        "artificial_intelligen_count": artificial_intelligen_count,
        "ai_raw_freq": ai_raw_freq,
        "token_count": token_count,
        "ai_share": ai_share,
        "includes_ai_keyword": int(ai_raw_freq > 0),
    }


# ============================================================
# 7. BUILD AND SAVE OUTPUT DATAFRAMES
# ============================================================

def build_item_level_dataframe(
    metadata: pd.DataFrame,
    bank_ids: pd.DataFrame,
) -> pd.DataFrame:
    """Create one row per bank-filing-item with AI frequency counts."""
    rows = []

    for _, row in metadata.iterrows():
        path = Path(clean_string(row.get("source_path", "")))
        file_found = path.is_file()

        text = read_text(path) if file_found else ""
        counts = count_ai_keywords(text)

        output_row = row.to_dict()
        output_row.update(counts)
        output_row["file_found"] = int(file_found)

        rows.append(output_row)

    item_level = pd.DataFrame(rows)

    item_level = item_level.merge(
        bank_ids,
        on="cik",
        how="left",
    )

    item_level["matched_bank_identifier"] = item_level["rssd_id"].notna().astype(int)


    return item_level

def build_bank_year_dataframe(item_level: pd.DataFrame) -> pd.DataFrame:
    """Aggregate item-level AI counts to the bank-year level."""
    group_columns = [
        "cik",
        "ticker",
        "bank_name",
        "rssd_id",
        "permco",
        "gvkey",
        "consolidated_assets_mil",
        "bank_holding_company_raw",
        "holding_company_name",
        "sec_name",
        "filing_year",
    ]

    group_columns = [
        column for column in group_columns
        if column in item_level.columns
    ]

    bank_year = (
        item_level
        .groupby(group_columns, dropna=False, as_index=False)
        .agg(
            AI_count_total=("AI_count", "sum"),
            artificial_intelligen_count_total=("artificial_intelligen_count", "sum"),
            ai_raw_freq_total=("ai_raw_freq", "sum"),
            token_count_total=("token_count", "sum"),
            includes_ai_keyword=("includes_ai_keyword", "max"),
            num_items_available=("item", "nunique"),
            num_files_found=("file_found", "sum"),
        )
    )

    bank_year["ai_share_total"] = bank_year.apply(
        lambda row: (
            row["ai_raw_freq_total"] / row["token_count_total"]
            if row["token_count_total"] > 0
            else 0
        ),
        axis=1,
    )

    return bank_year


def build_diagnostics(
    metadata: pd.DataFrame,
    item_level: pd.DataFrame,
    bank_year: pd.DataFrame,
) -> pd.DataFrame:
    """Create a simple diagnostics summary for quality control."""

    diagnostics = {
        "metadata_rows_after_filtering": len(metadata),
        "item_level_rows": len(item_level),
        "bank_year_rows": len(bank_year),
        "files_found": int(item_level["file_found"].sum()) if not item_level.empty else 0,
        "files_missing": int((item_level["file_found"] == 0).sum()) if not item_level.empty else 0,
        "rows_with_ai_mentions": int(item_level["includes_ai_keyword"].sum()) if not item_level.empty else 0,
        "rows_without_ai_mentions": int((item_level["includes_ai_keyword"] == 0).sum()) if not item_level.empty else 0,
        "total_AI_count": int(item_level["AI_count"].sum()) if not item_level.empty else 0,
        "total_artificial_intelligen_count": int(item_level["artificial_intelligen_count"].sum()) if not item_level.empty else 0,
        "total_ai_raw_freq": int(item_level["ai_raw_freq"].sum()) if not item_level.empty else 0,
        "total_token_count": int(item_level["token_count"].sum()) if not item_level.empty else 0,
        "unique_ciks": item_level["cik"].nunique() if not item_level.empty else 0,
        "unique_tickers": item_level["ticker"].nunique() if not item_level.empty else 0,
        "unique_filing_years": item_level["filing_year"].nunique() if not item_level.empty else 0,
        "item_1_rows": int((item_level["item"] == "1").sum()) if not item_level.empty else 0,
        "item_1A_rows": int((item_level["item"] == "1A").sum()) if not item_level.empty else 0,
        "item_7_rows": int((item_level["item"] == "7").sum()) if not item_level.empty else 0,
    }

    if not item_level.empty and "rssd_id" in item_level.columns:
        diagnostics["rows_with_bank_identifier_match"] = int(item_level["rssd_id"].notna().sum())
        diagnostics["rows_without_bank_identifier_match"] = int(item_level["rssd_id"].isna().sum())

    return pd.DataFrame(
        [{"metric": key, "value": value} for key, value in diagnostics.items()]
    )


def save_outputs(
    item_level: pd.DataFrame,
    bank_year: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output_folder: Path,
) -> None:
    """Save item-level, bank-year, and diagnostics outputs."""
    output_folder.mkdir(parents=True, exist_ok=True)

    item_level.to_csv(output_folder / "ai_frequency_item_level.csv", index=False)
    bank_year.to_csv(output_folder / "ai_frequency_bank_year.csv", index=False)
    diagnostics.to_csv(output_folder / "ai_frequency_diagnostics.csv", index=False)

# ============================================================
# 8. MAIN
# ============================================================

def main() -> None:
    """Run the AI frequency analysis pipeline."""

    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    output_folder.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(
        metadata_file=input_metadata_file,
        input_folder=input_folder,
    )

    bank_ids = load_bank_identifiers(
        bank_id_file=input_bank_id_file,
    )

    item_level = build_item_level_dataframe(
        metadata=metadata,
        bank_ids=bank_ids,
    )

    bank_year = build_bank_year_dataframe(
        item_level=item_level,
    )

    diagnostics = build_diagnostics(
        metadata=metadata,
        item_level=item_level,
        bank_year=bank_year,
    )

    save_outputs(
        item_level=item_level,
        bank_year=bank_year,
        diagnostics=diagnostics,
        output_folder=output_folder,
    )

    print("AI frequency analysis complete.")
    print(f"Item-level rows: {len(item_level):,}")
    print(f"Bank-year rows: {len(bank_year):,}")
    print(f"Files found: {int(item_level['file_found'].sum()):,}")
    print(f"Files missing: {int((item_level['file_found'] == 0).sum()):,}")
    print(f"Rows with AI mentions: {int(item_level['includes_ai_keyword'].sum()):,}")
    print(f"Total AI mentions: {int(item_level['ai_raw_freq'].sum()):,}")
    print(f"Output folder: {output_folder}")


if __name__ == "__main__":
    main()