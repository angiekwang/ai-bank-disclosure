#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Score Loughran-McDonald sentiment in AI keyword context windows.

Default inputs:
    metadata/ai_keyword_context_windows.csv
    metadata/lm_dictionary_cleaned.csv
    metadata/preprocess_file_level_diagnostics.csv
    base_data/ai_freq.csv

Default outputs:
    metadata/ai_context_sentiment_context_level.csv
    metadata/ai_context_sentiment_document_level.csv
    metadata/bank_year_item_ai_context_sentiment_features.csv
    metadata/bank_year_ai_context_sentiment_features.csv
    metadata/ai_context_sentiment_summary.csv
    metadata/ai_context_sentiment_diagnostics.csv

Example usage:
    python src/score_ai_sentiment.py
    python src/score_ai_sentiment.py --limit-rows 1000
"""

# SETTINGS
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any

import pandas as pd

# Project paths stay relative to the repository root so the script can run
# after the repository is cloned on another computer.
script_folder = Path(__file__).resolve().parent
project_root = script_folder.parent

context_windows_file = project_root / "metadata" / "ai_keyword_context_windows.csv"
lm_dictionary_file = project_root / "metadata" / "lm_dictionary_cleaned.csv"
preprocess_diagnostics_file = project_root / "metadata" / "preprocess_file_level_diagnostics.csv"
bank_identifier_file = project_root / "base_data" / "ai_freq.csv"

# Sentiment score for each AI keyword context window.
context_output_file = project_root / "metadata" / "ai_context_sentiment_context_level.csv"

# Sentiment score aggregated to each filing item document.
document_output_file = project_root / "metadata" / "ai_context_sentiment_document_level.csv"

# Each row = one bank + year + 10-K item
bank_year_item_output_file = (
    project_root / "metadata" / "bank_year_item_ai_context_sentiment_features.csv"
)

# Each row = one bank + year
bank_year_output_file = project_root / "metadata" / "bank_year_ai_context_sentiment_features.csv"

summary_file = project_root / "metadata" / "ai_context_sentiment_summary.csv"
diagnostics_file = project_root / "metadata" / "ai_context_sentiment_diagnostics.csv"

# Columns required to score sentiment (input)
required_context_columns = [
    "context_window_id",
    "file_name",
    "cik_10",
    "ticker",
    "filing_date",
    "accession_number",
    "item",
    "matched_keyword",
    "matched_keyword_type",
    "context_window_text",
]

# Columns written after scoring each context window (output)
context_output_columns = [
    "context_window_id",
    "file_name",
    "cik_10",
    "ticker",
    "consol_assets_millions",
    "filing_date",
    "accession_number",
    "item",
    "matched_keyword",
    "matched_keyword_type",
    "context_window_text",
    "ai_context_token_count",
    "ai_positive_count",
    "ai_negative_count",
    "ai_uncertainty_count",
    "ai_positive_share",
    "ai_negative_share",
    "ai_uncertainty_share",
    "ai_net_sentiment",
    "context_has_positive",
    "context_has_negative",
    "context_has_uncertainty",
    "context_has_any_lm_sentiment",
]

# Reused sentiment variables in aggregate document, bank-year-item, and bank-year output files
aggregate_metric_columns = [
    "number_of_ai_context_windows_scored",
    "ai_context_token_count",
    "ai_positive_count",
    "ai_negative_count",
    "ai_uncertainty_count",
    "ai_positive_share",
    "ai_negative_share",
    "ai_uncertainty_share",
    "ai_net_sentiment",
    "share_ai_contexts_with_positive",
    "share_ai_contexts_with_negative",
    "share_ai_contexts_with_uncertainty",
    "share_ai_contexts_with_any_lm_sentiment",
]

# Divides numbers for net AI sentiment calculation
def safe_divide(numerator: float, denominator: float) -> float:
    if denominator in (0, 0.0) or pd.isna(denominator):
        return 0.0
    return float(numerator) / float(denominator)

# Cleans word token by lowercasing, removing non-letter characters, and returning empty string if missing
def normalize_word(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z]", "", str(value).strip().lower())

# Standardizes columns into booleans (for LM dictionary)
def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip().str.lower()
    return ((numeric.fillna(0) != 0) | text.isin({"true", "yes", "y"})).astype(bool)

# Splits text into lowercase word tokens using letters only
def tokenize_for_lm(text: Any) -> list[str]:
    if pd.isna(text):
        return []
    return [token.lower() for token in re.findall(r"[A-Za-z]+", str(text))]

# Extracts filing year from date
def filing_year_from_date(value: Any) -> str:
    if pd.isna(value):
        return ""
    parsed = pd.to_datetime(str(value), errors="coerce")
    if pd.isna(parsed):
        match = re.match(r"^(\d{4})", str(value))
        return match.group(1) if match else ""
    return str(parsed.year)

# Standardizes SEC CIKs into 10-digit strings
def normalize_cik(value: Any) -> str:
    if pd.isna(value):
        return ""
    digits = re.sub(r"\D", "", str(value))
    return digits.zfill(10) if digits else ""

# Standardizes 10-K item labels
def normalize_item(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).upper().replace("_", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if text in {"1", "ITEM 1"}:
        return "Item 1"
    if text in {"1A", "ITEM 1A"}:
        return "Item 1A"
    if text in {"7", "ITEM 7"}:
        return "Item 7"
    return str(value).strip()

# Cleans an asset value and converts it to a number
def numeric_assets(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    text = str(value).replace(",", "").replace("$", "").strip()
    if text == "":
        return pd.NA
    return pd.to_numeric(text, errors="coerce")

# Looks for the first matching column name in a dataframe (to match columns)
def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        lowered = candidate.lower()
        if lowered in lower_map:
            return lower_map[lowered]
    return None

# Reads CSV input files used by the sentiment pipeline.
def read_table(path: Path) -> pd.DataFrame:
    """
    Purpose:
        Read a CSV input file used by the sentiment pipeline.

    Methodology:
        All replication inputs are plain-text CSV files so the pipeline can run
        without workbook-specific dependencies.

    Parameters:
        path (Path): Input table path ending in .csv.

    Returns:
        pd.DataFrame: Loaded table.
    """
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV input file, got: {path}")
    return pd.read_csv(path)

# Converts value into a clean string
def clean_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()

# Creates a stable unique ID 
def stable_hash(parts: list[Any]) -> str:
    text = "||".join(clean_value(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]

# Figures out what type of AI keyword was matched (either "artificial intelligen" or "AI")
def derive_keyword_type(row: pd.Series) -> str:
    existing = clean_value(row.get("matched_keyword_type", ""))
    if existing:
        return existing

    matched = clean_value(row.get("matched_keyword", "")).strip()
    matched_lower = matched.lower()
    if matched == "AI":
        return "AI_acronym"
    if "artificial" in matched_lower and "intelligen" in matched_lower:
        return "artificial_intelligence_phrase"

    category = clean_value(row.get("category", ""))
    measure_type = clean_value(row.get("measure_type", ""))
    if category and measure_type:
        return f"{measure_type}:{category}"
    if category:
        return category
    if measure_type:
        return measure_type
    return "legacy_context_keyword"


def standardize_context_windows(context_windows: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the context-window input before LM sentiment scoring.

    The preferred GitHub replication schema already contains the required
    columns. A small amount of column-name fallback remains so older local
    pipeline outputs can still be read while keeping the public schema clear.
    """
    standardized = context_windows.copy()

    column_candidates = {
        "file_name": ["file_name", "final_filename", "source_file"],
        "cik_10": ["cik_10", "cik", "CIK"],
        "ticker": ["ticker", "Ticker"],
        "filing_date": ["filing_date", "filing date"],
        "accession_number": ["accession_number", "accession"],
        "item": ["item", "Item/10K section"],
        "matched_keyword": ["matched_keyword", "keyword_text_matched", "term", "keyword"],
        "context_window_text": ["context_window_text", "context_text", "sentence_window", "inner_context_text"],
    }

    used_legacy_columns = False
    for target, candidates in column_candidates.items():
        if target in standardized.columns:
            continue
        source = first_existing_column(standardized, candidates)
        if source is not None:
            standardized[target] = standardized[source]
            used_legacy_columns = True

    if "cik_10" in standardized.columns:
        standardized["cik_10"] = standardized["cik_10"].map(normalize_cik)
    if "ticker" in standardized.columns:
        standardized["ticker"] = standardized["ticker"].astype(str).str.upper().str.strip()
    if "item" in standardized.columns:
        standardized["item"] = standardized["item"].map(normalize_item)

    if "matched_keyword_type" not in standardized.columns:
        standardized["matched_keyword_type"] = standardized.apply(derive_keyword_type, axis=1)
        used_legacy_columns = True
    else:
        missing_type = standardized["matched_keyword_type"].isna() | standardized["matched_keyword_type"].astype(str).str.strip().eq("")
        if missing_type.any():
            standardized.loc[missing_type, "matched_keyword_type"] = standardized.loc[missing_type].apply(
                derive_keyword_type,
                axis=1,
            )

    if "context_window_id" not in standardized.columns:
        standardized["context_window_id"] = standardized.apply(
            lambda row: stable_hash(
                [
                    row.get("file_name", ""),
                    row.get("matched_keyword", ""),
                    row.get("keyword_start_token", row.get("token_start_index", "")),
                    row.get("keyword_end_token", row.get("token_end_index", "")),
                    row.get("context_start_token", row.get("context_start_token_index", "")),
                    row.get("context_end_token", row.get("context_end_token_index", "")),
                    row.get("occurrence_index", ""),
                ]
            ),
            axis=1,
        )
        used_legacy_columns = True

    if used_legacy_columns:
        print("Standardized context-window input columns to the scorer schema.")

    return standardized

# Extracts CIK, ticker, consolidated assets from bank identifier file
def load_bank_assets(path: Path) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        return pd.DataFrame(), f"Bank identifier file missing: {path}"

    bank_ids = read_table(path)

    cik_col = first_existing_column(bank_ids, ["cik_10", "cik_str", "cik", "CIK", "current_cik"])
    ticker_col = first_existing_column(bank_ids, ["ticker", "current_ticker", "Ticker"])
    asset_col = first_existing_column(
        bank_ids,
        ["consol_assets_millions", "consolidated_assets_mil", "consolidated assets"],
    )

    if asset_col is None:
        return pd.DataFrame(), "Bank identifier file missing consolidated assets column"
    if cik_col is None and ticker_col is None:
        return pd.DataFrame(), "Bank identifier file missing CIK and ticker columns"

    assets = pd.DataFrame()
    assets["consol_assets_millions"] = bank_ids[asset_col].map(numeric_assets)
    assets["cik_10"] = bank_ids[cik_col].map(normalize_cik) if cik_col else ""
    assets["ticker"] = (
        bank_ids[ticker_col].astype(str).str.upper().str.strip()
        if ticker_col
        else ""
    )
    assets = assets.loc[assets["consol_assets_millions"].notna()].copy()
    assets = assets.loc[(assets["cik_10"] != "") | (assets["ticker"] != "")].copy()
    cik_assets = assets.loc[assets["cik_10"] != ""].drop_duplicates(subset=["cik_10"], keep="first")
    ticker_assets = assets.loc[assets["cik_10"] == ""].drop_duplicates(subset=["ticker"], keep="first")
    assets = pd.concat([cik_assets, ticker_assets], ignore_index=True)
    return assets[["cik_10", "ticker", "consol_assets_millions"]], (
        f"Using {asset_col} from bank identifier file"
    )

# Adds consol_assets_millions to the scored context-level data
def add_bank_assets(scored: pd.DataFrame, bank_identifier_file: Path) -> tuple[pd.DataFrame, str]:
    scored = scored.copy()
    if "consol_assets_millions" in scored.columns:
        scored = scored.drop(columns=["consol_assets_millions"])

    if scored.empty:
        scored["consol_assets_millions"] = pd.NA
        return scored, "No context rows available for bank asset merge"

    assets, asset_status = load_bank_assets(bank_identifier_file)
    if assets.empty:
        print(f"WARNING: {asset_status}. consol_assets_millions will be blank.")
        scored["consol_assets_millions"] = pd.NA
        return scored, asset_status

    scored = scored.merge(
        assets[["cik_10", "consol_assets_millions"]],
        on="cik_10",
        how="left",
    )

    if scored["consol_assets_millions"].isna().any():
        ticker_assets = (
            assets.loc[assets["ticker"] != "", ["ticker", "consol_assets_millions"]]
            .drop_duplicates(subset=["ticker"], keep="first")
            .rename(columns={"consol_assets_millions": "ticker_consol_assets_millions"})
        )
        scored = scored.merge(ticker_assets, on="ticker", how="left")
        scored["consol_assets_millions"] = scored["consol_assets_millions"].fillna(
            scored["ticker_consol_assets_millions"]
        )
        scored = scored.drop(columns=["ticker_consol_assets_millions"])

    matched = int(scored["consol_assets_millions"].notna().sum())
    return scored, f"{asset_status}; matched assets for {matched:,} context rows"

# Loads the cleaned LM dictionary and turns it into three Python sets: positive_words, negative_words, uncertainty_words
def load_lm_sets(path: Path) -> tuple[set[str], set[str], set[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Cleaned LM dictionary file not found: {path}")

    dictionary = pd.read_csv(path)
    dictionary.columns = [str(col).strip().lower() for col in dictionary.columns]
    missing = [col for col in ["word", "negative", "positive", "uncertainty"] if col not in dictionary.columns]
    if missing:
        raise ValueError(f"Cleaned LM dictionary is missing required columns: {missing}")

    dictionary["word"] = dictionary["word"].map(normalize_word)
    dictionary = dictionary.loc[dictionary["word"] != ""].copy()

    negative = set(dictionary.loc[bool_series(dictionary["negative"]), "word"])
    positive = set(dictionary.loc[bool_series(dictionary["positive"]), "word"])
    uncertainty = set(dictionary.loc[bool_series(dictionary["uncertainty"]), "word"])
    return positive, negative, uncertainty


def score_context_row(
    row: pd.Series,
    positive_words: set[str],
    negative_words: set[str],
    uncertainty_words: set[str],
) -> dict[str, Any]:
    """
    Purpose:
        Score the language surrounding one AI keyword occurrence.

    Methodology:
        tokenizes the context text
        counts positive LM words
        counts negative LM words
        counts uncertainty LM words
        computes shares using token count as denominator
        computes net sentiment
        creates true/false indicators

    Parameters:
        row (pd.Series): One AI keyword context-window record.
        positive_words (set[str]): Cleaned LM positive dictionary words.
        negative_words (set[str]): Cleaned LM negative dictionary words.
        uncertainty_words (set[str]): Cleaned LM uncertainty dictionary words.

    Returns:
        dict[str, Any]: Original context identifiers plus sentiment counts,
        shares, net sentiment, and indicator variables.

    Formulas:
        ai_positive_share = positive_count / token_count
        ai_negative_share = negative_count / token_count
        ai_uncertainty_share = uncertainty_count / token_count
        ai_net_sentiment = (positive_count - negative_count) / token_count
    """
    tokens = tokenize_for_lm(row.get("context_window_text", ""))
    token_count = len(tokens)
    positive_count = sum(token in positive_words for token in tokens)
    negative_count = sum(token in negative_words for token in tokens)
    uncertainty_count = sum(token in uncertainty_words for token in tokens)

    scored = {column: row.get(column, "") for column in required_context_columns}
    scored.update(
        {
            "ai_context_token_count": token_count,
            "ai_positive_count": positive_count,
            "ai_negative_count": negative_count,
            "ai_uncertainty_count": uncertainty_count,
            "ai_positive_share": safe_divide(positive_count, token_count),
            "ai_negative_share": safe_divide(negative_count, token_count),
            "ai_uncertainty_share": safe_divide(uncertainty_count, token_count),
            "ai_net_sentiment": safe_divide(positive_count - negative_count, token_count),
            "context_has_positive": positive_count > 0,
            "context_has_negative": negative_count > 0,
            "context_has_uncertainty": uncertainty_count > 0,
            "context_has_any_lm_sentiment": (positive_count + negative_count + uncertainty_count) > 0,
        }
    )
    return scored

# Checks whether the input context-window file has all required columns
def validate_context_input(context_windows: pd.DataFrame) -> list[str]:
    return [column for column in required_context_columns if column not in context_windows.columns]


def compute_aggregate(scored: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """
    Purpose:
        Convert context-level sentiment into document, bank-year-item, or
        bank-year research variables.

    Methodology:
        Sum LM word counts and AI-context token counts within each group.
        Shares are recomputed after aggregation so denominators match the
        grouped text unit.

    Parameters:
        scored (pd.DataFrame): Context-level sentiment records.
        group_cols (list[str]): Columns defining the aggregation level.

    Returns:
        pd.DataFrame: Aggregated sentiment features for the requested level.
    """
    if scored.empty:
        return pd.DataFrame(columns=group_cols + aggregate_metric_columns)

    grouped = (
        scored.groupby(group_cols, dropna=False)
        .agg(
            number_of_ai_context_windows_scored=("context_window_id", "nunique"),
            ai_context_token_count=("ai_context_token_count", "sum"),
            ai_positive_count=("ai_positive_count", "sum"),
            ai_negative_count=("ai_negative_count", "sum"),
            ai_uncertainty_count=("ai_uncertainty_count", "sum"),
            share_ai_contexts_with_positive=("context_has_positive", "mean"),
            share_ai_contexts_with_negative=("context_has_negative", "mean"),
            share_ai_contexts_with_uncertainty=("context_has_uncertainty", "mean"),
            share_ai_contexts_with_any_lm_sentiment=("context_has_any_lm_sentiment", "mean"),
        )
        .reset_index()
    )

    token_denominator = grouped["ai_context_token_count"].where(
        grouped["ai_context_token_count"] != 0
    )
    grouped["ai_positive_share"] = (
        grouped["ai_positive_count"] / token_denominator
    ).fillna(0)
    grouped["ai_negative_share"] = (
        grouped["ai_negative_count"] / token_denominator
    ).fillna(0)
    grouped["ai_uncertainty_share"] = (
        grouped["ai_uncertainty_count"] / token_denominator
    ).fillna(0)
    grouped["ai_net_sentiment"] = (
        (grouped["ai_positive_count"] - grouped["ai_negative_count"]) / token_denominator
    ).fillna(0)
    return grouped[group_cols + aggregate_metric_columns]


def load_complete_panel_rows(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Build complete bank-year-item and bank-year panels from the base workbook.

    The context-window sentiment data only contain rows for filings with at
    least one AI keyword. The base workbook contains the full filing/item
    universe, so we use it to retain zero-AI rows in aggregate outputs.
    """
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame(), f"Complete panel source missing: {path}"

    panel_source = read_table(path)
    if panel_source.empty:
        return pd.DataFrame(), pd.DataFrame(), f"Complete panel source has no rows: {path}"

    cik_col = first_existing_column(panel_source, ["cik_10", "cik_str", "cik", "CIK", "current_cik"])
    ticker_col = first_existing_column(panel_source, ["ticker", "current_ticker", "Ticker"])
    asset_col = first_existing_column(
        panel_source,
        ["consol_assets_millions", "consolidated_assets_mil", "consolidated assets"],
    )
    year_col = first_existing_column(panel_source, ["filing_year", "filing year", "year"])
    item_col = first_existing_column(panel_source, ["item", "Item/10K section"])

    missing = []
    if cik_col is None and ticker_col is None:
        missing.append("cik/ticker")
    if asset_col is None:
        missing.append("consol_assets_millions")
    if year_col is None:
        missing.append("filing_year")
    if item_col is None:
        missing.append("item")
    if missing:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            f"Complete panel source missing required columns: {', '.join(missing)}",
        )

    base = pd.DataFrame()
    base["cik_10"] = panel_source[cik_col].map(normalize_cik) if cik_col else ""
    base["ticker"] = (
        panel_source[ticker_col].astype(str).str.upper().str.strip()
        if ticker_col
        else ""
    )
    base["consol_assets_millions"] = panel_source[asset_col].map(numeric_assets)
    base["filing_year"] = panel_source[year_col].map(filing_year_from_date)
    base["item"] = panel_source[item_col].map(normalize_item)

    base = base.loc[(base["cik_10"] != "") | (base["ticker"] != "")].copy()
    base = base.loc[base["filing_year"].astype(str).str.strip() != ""].copy()
    base = base.loc[base["item"].isin(["Item 1", "Item 1A", "Item 7"])].copy()

    bank_year_item_base = base.drop_duplicates(
        subset=["cik_10", "ticker", "consol_assets_millions", "filing_year", "item"],
        keep="first",
    ).copy()
    bank_year_base = (
        base.groupby(["cik_10", "ticker", "filing_year"], dropna=False)["consol_assets_millions"]
        .max()
        .reset_index()
    )
    bank_year_base = bank_year_base[["cik_10", "ticker", "consol_assets_millions", "filing_year"]]

    return (
        bank_year_item_base,
        bank_year_base,
        (
            f"Complete panel source added {len(bank_year_item_base):,} bank-year-item rows "
            f"and {len(bank_year_base):,} bank-year rows"
        ),
    )


def complete_aggregate_panel(
    aggregate: pd.DataFrame,
    complete_keys: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    if complete_keys.empty:
        return aggregate[group_cols + aggregate_metric_columns].copy()

    aggregate = aggregate[group_cols + aggregate_metric_columns].copy()
    keys = pd.concat(
        [
            complete_keys[group_cols],
            aggregate[group_cols],
        ],
        ignore_index=True,
    ).drop_duplicates(subset=group_cols, keep="first")

    completed = keys.merge(aggregate, on=group_cols, how="left")
    for column in aggregate_metric_columns:
        completed[column] = pd.to_numeric(completed[column], errors="coerce").fillna(0)

    sort_cols = [column for column in ["ticker", "cik_10", "filing_year", "item"] if column in completed.columns]
    if sort_cols:
        completed = completed.sort_values(sort_cols).reset_index(drop=True)

    return completed[group_cols + aggregate_metric_columns]

# Loads total document token counts from preprocessing diagnostics (used to compute AI disclosure intensity)
def load_token_denominators(path: Path) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        return pd.DataFrame(), f"Preprocessing diagnostics file missing: {path}"

    diagnostics = pd.read_csv(path)
    if "file_name" not in diagnostics.columns:
        return pd.DataFrame(), "Preprocessing diagnostics missing file_name column"

    token_col = ""
    if "token_count" in diagnostics.columns:
        token_col = "token_count"
    elif "clean_word_count" in diagnostics.columns:
        token_col = "clean_word_count"
    else:
        return pd.DataFrame(), "Preprocessing diagnostics missing token_count and clean_word_count columns"

    keep_cols = ["file_name", "cik_10", "ticker", "filing_date", "item", token_col]
    available = [col for col in keep_cols if col in diagnostics.columns]
    denominators = diagnostics[available].copy()
    denominators["file_name"] = denominators["file_name"].astype(str)
    denominators["total_item_token_count"] = pd.to_numeric(denominators[token_col], errors="coerce")

    if "cik_10" in denominators.columns:
        denominators["cik_10"] = denominators["cik_10"].map(normalize_cik)
    if "ticker" in denominators.columns:
        denominators["ticker"] = denominators["ticker"].astype(str).str.upper().str.strip()
    if "filing_date" in denominators.columns:
        denominators["filing_year"] = denominators["filing_date"].map(filing_year_from_date)
    if "item" in denominators.columns:
        denominators["item"] = denominators["item"].map(normalize_item)

    denominators = denominators.drop_duplicates(subset=["file_name"], keep="last")
    return denominators, f"Using {token_col} from preprocessing diagnostics"

# Adds total token denominators to the aggregate outputs
def add_denominators(
    document_level: pd.DataFrame,
    bank_year_item_level: pd.DataFrame,
    bank_year_level: pd.DataFrame,
    denominator_file: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    denominators, denominator_status = load_token_denominators(denominator_file)
    if denominators.empty:
        print(f"WARNING: {denominator_status}. Denominator variables will be blank.")
        document_level["total_item_token_count"] = pd.NA
        document_level["ai_disclosure_share"] = pd.NA
        bank_year_item_level["total_bank_year_item_token_count"] = pd.NA
        bank_year_item_level["bank_year_item_ai_disclosure_share"] = pd.NA
        bank_year_level["total_bank_year_token_count"] = pd.NA
        bank_year_level["bank_year_ai_disclosure_share"] = pd.NA
        return document_level, bank_year_item_level, bank_year_level, denominator_status

    document_level = document_level.merge(
        denominators[["file_name", "total_item_token_count"]],
        on="file_name",
        how="left",
    )
    document_level["ai_disclosure_share"] = document_level.apply(
        lambda row: safe_divide(row["ai_context_token_count"], row["total_item_token_count"]), axis=1
    )

    by_item_denominator = (
        denominators.groupby(["cik_10", "ticker", "filing_year", "item"], dropna=False)["total_item_token_count"]
        .sum(min_count=1)
        .reset_index(name="total_bank_year_item_token_count")
    )
    bank_year_item_level = bank_year_item_level.merge(
        by_item_denominator,
        on=["cik_10", "ticker", "filing_year", "item"],
        how="left",
    )
    bank_year_item_level["bank_year_item_ai_disclosure_share"] = bank_year_item_level.apply(
        lambda row: safe_divide(row["ai_context_token_count"], row["total_bank_year_item_token_count"]), axis=1
    )

    by_year_denominator = (
        denominators.groupby(["cik_10", "ticker", "filing_year"], dropna=False)["total_item_token_count"]
        .sum(min_count=1)
        .reset_index(name="total_bank_year_token_count")
    )
    bank_year_level = bank_year_level.merge(
        by_year_denominator,
        on=["cik_10", "ticker", "filing_year"],
        how="left",
    )
    bank_year_level["bank_year_ai_disclosure_share"] = bank_year_level.apply(
        lambda row: safe_divide(row["ai_context_token_count"], row["total_bank_year_token_count"]), axis=1
    )

    return document_level, bank_year_item_level, bank_year_level, denominator_status

# Creates the output folder if needed and writes a dataframe to CSV
def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

# Writes a diagnostics file when the script cannot run because required input columns are missing
def write_error_diagnostics(path: Path, missing_columns: list[str], reason: str) -> None:
    diagnostics = pd.DataFrame(
        [
            {
                "total_context_windows_input": 0,
                "context_windows_scored_this_run": 0,
                "context_windows_skipped_already_scored": 0,
                "final_context_level_rows": 0,
                "zero_token_contexts": 0,
                "contexts_with_positive": 0,
                "contexts_with_negative": 0,
                "contexts_with_uncertainty": 0,
                "contexts_with_any_lm_sentiment": 0,
                "contexts_with_consol_assets_millions": 0,
                "missing_required_columns": ";".join(missing_columns),
                "status": "error",
                "reason": reason,
            }
        ]
    )
    write_csv(diagnostics, path)

# Defines command-line options
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score LM sentiment in AI context windows.")
    parser.add_argument("--context-windows-file", type=Path, default=context_windows_file)
    parser.add_argument("--lm-dictionary-file", type=Path, default=lm_dictionary_file)
    parser.add_argument("--preprocess-diagnostics-file", type=Path, default=preprocess_diagnostics_file)
    parser.add_argument("--bank-identifier-file", type=Path, default=bank_identifier_file)
    parser.add_argument("--context-output-file", type=Path, default=context_output_file)
    parser.add_argument("--document-output-file", type=Path, default=document_output_file)
    parser.add_argument("--bank-year-item-output-file", type=Path, default=bank_year_item_output_file)
    parser.add_argument("--bank-year-output-file", type=Path, default=bank_year_output_file)
    parser.add_argument("--summary-file", type=Path, default=summary_file)
    parser.add_argument("--diagnostics-file", type=Path, default=diagnostics_file)
    parser.add_argument("--limit-rows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load and validate AI keyword context windows.
    if not args.context_windows_file.exists():
        raise SystemExit(f"ERROR: Context windows file not found: {args.context_windows_file}")

    context_windows = pd.read_csv(args.context_windows_file)
    if args.limit_rows is not None:
        context_windows = context_windows.head(args.limit_rows).copy()
    context_windows = standardize_context_windows(context_windows)

    missing = validate_context_input(context_windows)
    if missing:
        reason = (
            f"Context windows input is missing required columns: {missing}. "
            "Run src/extract_keyword_windows.py first, or pass --context-windows-file to a compatible file."
        )
        write_error_diagnostics(args.diagnostics_file, missing, reason)
        raise SystemExit(f"ERROR: {reason}")

    # 2. Load LM word lists and score every context window from the current input.
    positive_words, negative_words, uncertainty_words = load_lm_sets(args.lm_dictionary_file)
    context_windows["context_window_id"] = context_windows["context_window_id"].astype(str)
    scored_rows = [
        score_context_row(row, positive_words, negative_words, uncertainty_words)
        for _, row in context_windows.iterrows()
    ]
    final_context = pd.DataFrame(scored_rows, columns=context_output_columns)
    if len(final_context):
        final_context = final_context.drop_duplicates(subset=["context_window_id"], keep="last")
        final_context["cik_10"] = final_context["cik_10"].map(normalize_cik)
        final_context["ticker"] = final_context["ticker"].astype(str).str.upper().str.strip()
        final_context["item"] = final_context["item"].map(normalize_item)
        final_context["filing_year"] = final_context["filing_date"].map(filing_year_from_date)

    # 3. Add bank asset data, then aggregate context scores to research panels.
    final_context, asset_status = add_bank_assets(final_context, args.bank_identifier_file)

    document_level = compute_aggregate(
        final_context,
        ["file_name", "cik_10", "ticker", "consol_assets_millions", "filing_date", "accession_number", "item"],
    )
    bank_year_item_level = compute_aggregate(
        final_context,
        ["cik_10", "ticker", "consol_assets_millions", "filing_year", "item"],
    )
    bank_year_level = compute_aggregate(
        final_context,
        ["cik_10", "ticker", "consol_assets_millions", "filing_year"],
    )

    # 4. Keep zero-AI bank-year rows and add total-token denominators.
    complete_bank_year_item_rows, complete_bank_year_rows, complete_panel_status = load_complete_panel_rows(
        args.bank_identifier_file
    )
    bank_year_item_level = complete_aggregate_panel(
        bank_year_item_level,
        complete_bank_year_item_rows,
        ["cik_10", "ticker", "consol_assets_millions", "filing_year", "item"],
    )
    bank_year_level = complete_aggregate_panel(
        bank_year_level,
        complete_bank_year_rows,
        ["cik_10", "ticker", "consol_assets_millions", "filing_year"],
    )

    document_level, bank_year_item_level, bank_year_level, denominator_status = add_denominators(
        document_level,
        bank_year_item_level,
        bank_year_level,
        args.preprocess_diagnostics_file,
    )

    final_context = final_context[context_output_columns]

    # 5. Write output files and run-level quality checks.
    diagnostics = pd.DataFrame(
        [
            {
                "total_context_windows_input": len(context_windows),
                "context_windows_scored_this_run": len(final_context),
                "context_windows_skipped_already_scored": 0,
                "final_context_level_rows": len(final_context),
                "zero_token_contexts": int((final_context["ai_context_token_count"] == 0).sum()) if len(final_context) else 0,
                "contexts_with_positive": int(final_context["context_has_positive"].sum()) if len(final_context) else 0,
                "contexts_with_negative": int(final_context["context_has_negative"].sum()) if len(final_context) else 0,
                "contexts_with_uncertainty": int(final_context["context_has_uncertainty"].sum()) if len(final_context) else 0,
                "contexts_with_any_lm_sentiment": int(final_context["context_has_any_lm_sentiment"].sum())
                if len(final_context)
                else 0,
                "contexts_with_consol_assets_millions": int(final_context["consol_assets_millions"].notna().sum())
                if len(final_context)
                else 0,
                "bank_year_rows_with_zero_ai_contexts": int(
                    (bank_year_level["number_of_ai_context_windows_scored"] == 0).sum()
                )
                if len(bank_year_level)
                else 0,
                "bank_year_item_rows_with_zero_ai_contexts": int(
                    (bank_year_item_level["number_of_ai_context_windows_scored"] == 0).sum()
                )
                if len(bank_year_item_level)
                else 0,
                "missing_required_columns": "",
                "status": "success",
                "reason": f"{complete_panel_status}; {denominator_status}; {asset_status}",
            }
        ]
    )

    total_tokens = final_context["ai_context_token_count"].sum() if len(final_context) else 0
    total_positive = final_context["ai_positive_count"].sum() if len(final_context) else 0
    total_negative = final_context["ai_negative_count"].sum() if len(final_context) else 0
    total_uncertainty = final_context["ai_uncertainty_count"].sum() if len(final_context) else 0
    filing_years = (
        sorted(
            year
            for year in final_context["filing_date"].map(filing_year_from_date).dropna().unique()
            if str(year).strip()
        )
        if len(final_context)
        else []
    )

    summary = pd.DataFrame(
        [
            {
                "total_context_windows": len(final_context),
                "total_AI_context_tokens": int(total_tokens),
                "total_positive_words": int(total_positive),
                "total_negative_words": int(total_negative),
                "total_uncertainty_words": int(total_uncertainty),
                "overall_positive_share": safe_divide(total_positive, total_tokens),
                "overall_negative_share": safe_divide(total_negative, total_tokens),
                "overall_uncertainty_share": safe_divide(total_uncertainty, total_tokens),
                "overall_net_sentiment": safe_divide(total_positive - total_negative, total_tokens),
                "number_of_document_level_rows": len(document_level),
                "number_of_bank_year_item_level_rows": len(bank_year_item_level),
                "number_of_bank_year_level_rows": len(bank_year_level),
                "unique_CIKs": bank_year_level["cik_10"].nunique(dropna=True) if len(bank_year_level) else 0,
                "unique_tickers": bank_year_level["ticker"].nunique(dropna=True) if len(bank_year_level) else 0,
                "unique_CIKs_with_ai_contexts": final_context["cik_10"].nunique(dropna=True) if len(final_context) else 0,
                "unique_tickers_with_ai_contexts": final_context["ticker"].nunique(dropna=True) if len(final_context) else 0,
                "bank_year_rows_with_zero_ai_contexts": int(
                    (bank_year_level["number_of_ai_context_windows_scored"] == 0).sum()
                )
                if len(bank_year_level)
                else 0,
                "bank_year_item_rows_with_zero_ai_contexts": int(
                    (bank_year_item_level["number_of_ai_context_windows_scored"] == 0).sum()
                )
                if len(bank_year_item_level)
                else 0,
                "contexts_with_consol_assets_millions": int(final_context["consol_assets_millions"].notna().sum())
                if len(final_context)
                else 0,
                "filing_years_covered": ";".join(filing_years),
            }
        ]
    )

    write_csv(final_context, args.context_output_file)
    write_csv(document_level, args.document_output_file)
    write_csv(bank_year_item_level, args.bank_year_item_output_file)
    write_csv(bank_year_level, args.bank_year_output_file)
    write_csv(summary, args.summary_file)
    write_csv(diagnostics, args.diagnostics_file)

    print("\nAI context sentiment scoring complete.")
    print(f"Context windows input: {len(context_windows):,}")
    print(f"Context windows scored this run: {len(final_context):,}")
    print(f"Final context-level rows: {len(final_context):,}")
    print(f"Document-level rows: {len(document_level):,}")
    print(f"Bank-year-item-level rows: {len(bank_year_item_level):,}")
    print(f"Bank-year-level rows: {len(bank_year_level):,}")
    print(f"Complete panel status: {complete_panel_status}")
    print(f"Denominator status: {denominator_status}")
    print(f"Asset merge status: {asset_status}")
    print(f"Context-level output: {args.context_output_file}")
    print(f"Bank-year-level output: {args.bank_year_output_file}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
