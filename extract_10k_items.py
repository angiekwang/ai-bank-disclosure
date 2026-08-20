from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
from edgar import Company, set_identity

# ============================================================
# 1. SETTINGS
# ============================================================

script_folder = Path(__file__).resolve().parent
project_root = script_folder.parent

# Input file
input_file = project_root / "base_data" / "bank_identifiers_linked.csv"

# Output folders
text_output_folder = project_root / "10k_texts_scraped"
metadata_output_folder = project_root / "metadata"

# Metadata and report files
metadata_file = metadata_output_folder / "10k_item_extraction_metadata.csv"
manual_review_file = metadata_output_folder / "10k_extraction_manual_review.csv"
no_qualifying_file = metadata_output_folder / "no_qualifying_10k_filings.csv"
excluded_10ka_file = metadata_output_folder / "excluded_10ka_filings.csv"
summary_file = metadata_output_folder / "10k_extraction_summary.csv"

# EdgarTools identity (account required)
sec_user_agent = os.environ.get(
    "SEC_USER_AGENT",
    "Angie Wang aw1407@georgetown.edu",
)

# Date range and items to extract
start_date = "2020-01-01"
end_date = "2026-12-31"
items_to_extract = ["Item 1", "Item 1A", "Item 7"]

# Section length rules
min_section_length = {
    "Item 1": 1_000,
    "Item 1A": 1_000,
    "Item 7": 2_000,
}

max_section_length = {
    "Item 1": 500_000,
    "Item 1A": 800_000,
    "Item 7": 900_000,
}

# Request pacing
edgar_sleep_seconds = 1.0

# Metadata columns
metadata_columns = [
    "cik",
    "ticker",
    "bank_name",
    "filing_date",
    "accession_number",
    "form",
    "item",
    "output_file",
    "status",
    "extraction_method",
    "char_count",
    "word_count",
    "qc_flag",
    "reason",
    "needs_manual_review",
]

manual_review_columns = [
    "cik",
    "ticker",
    "bank_name",
    "filing_date",
    "accession_number",
    "item",
    "output_file",
    "status",
    "qc_flag",
    "reason",
    "suggested_action",
]

no_qualifying_columns = [
    "cik",
    "ticker",
    "bank_name",
    "reason",
]

excluded_10ka_columns = [
    "cik",
    "ticker",
    "filing_date",
    "accession_number",
    "form",
    "exclusion_reason",
]


# ============================================================
# 2. INPUT AND GENERAL HELPER FUNCTIONS
# ============================================================


# Convert raw CIK values into a 10-digit SEC CIK string
def normalize_cik(value: Any) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return ""

    raw = str(value).strip().replace(",", "")

    if raw.endswith(".0"):
        raw = raw[:-2]

    digits = re.sub(r"\D", "", raw)

    if not digits:
        return ""

    if len(digits) > 10:
        raise ValueError(f"CIK has more than 10 digits: {value!r}")

    return digits.zfill(10)


# Clean a value so it can  be used inside a filename
def clean_filename_part(value: Any, fallback: str = "UNKNOWN") -> str:

    if pd.isna(value):
        value = ""

    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")

    return value or fallback


# Count words in extracted text (for quality check)
def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", str(text)))

# Create output folders if they do not already exist
def make_output_folders() -> None:
    text_output_folder.mkdir(parents=True, exist_ok=True)
    metadata_output_folder.mkdir(parents=True, exist_ok=True)

# Read the input file and return one clean row per CIK.
def read_candidate_ciks(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path, dtype=str)

    candidates = pd.DataFrame()
    candidates["cik"] = df["cik"].apply(normalize_cik)
    candidates["ticker"] = df["ticker"].fillna("").str.strip().str.upper()
    candidates["bank_name"] = df["bank_name"].fillna("").str.strip()

    candidates = candidates[candidates["cik"] != ""].copy()
    candidates = candidates.drop_duplicates(subset=["cik"], keep="first")
    candidates = candidates.reset_index(drop=True)

    print(f"Loaded {len(candidates):,} unique CIKs from {path}")

    return candidates

# Creates file path for one extracted filing item, based on CIK, ticker, filing date, accession number, and item (1, 1A, 7)
def build_output_path(
    cik: str,
    ticker: str,
    filing_date: str,
    accession_number: str,
    item: str,
) -> Path:

    item_part = item.replace(" ", "_")
    ticker_part = clean_filename_part(ticker)
    accession_part = clean_filename_part(accession_number)

    filename = (
        f"{cik}_{ticker_part}_{filing_date}_"
        f"{accession_part}_{item_part}.txt"
    )

    return text_output_folder / filename

# ============================================================
# 3. SEC FILING SEARCH HELPER FUNCTIONS
# ============================================================

# Get attribute from EdgarTools filing object, trying several possible attribute names and returning the first one that exists
def get_filing_attr(filing: Any, possible_attrs: list[str], default: Any = "") -> Any:
    for attr in possible_attrs:
        if hasattr(filing, attr):
            value = getattr(filing, attr)
            if value is not None:
                return value

    return default

# Return filing date as string
def get_filing_date(filing: Any) -> str:
    return str(get_filing_attr(filing, ["filing_date", "date"], default=""))

# Return exact filing form, such as 10-K or 10-K/A

def get_filing_form(filing: Any) -> str:
    return str(get_filing_attr(filing, ["form"], default="")).strip().upper()

# Return the unique SEC accession number for a filing
def get_accession_number(filing: Any) -> str:
    return str(
        get_filing_attr(
            filing,
            ["accession_number", "accession_no", "accession"],
            default="unknown_accession",
        )
    ).strip()

# Find original 10-K filings and amended 10-K/A filings for one company and returns lists: original_10ks and amended_10ks

def identify_10k_filings(company: Company) -> tuple[list[Any], list[Any]]:
    original_10ks: list[Any] = []
    amended_10ks_by_accession: dict[str, Any] = {}

    # EdgarTools may return both 10-K and 10-K/A when searching for 10-K, so explicitly check the exact form
    filings = company.get_filings(form="10-K")

    for filing in filings:
        filing_date = get_filing_date(filing)
        form = get_filing_form(filing)

        if not (start_date <= filing_date <= end_date):
            continue

        if form == "10-K":
            original_10ks.append(filing)
        elif form == "10-K/A":
            amended_10ks_by_accession[get_accession_number(filing)] = filing

    # Also search 10-K/A directly, in case amendments are not included above
    try:
        amended_filings = company.get_filings(form="10-K/A")

        for filing in amended_filings:
            filing_date = get_filing_date(filing)
            form = get_filing_form(filing)

            if not (start_date <= filing_date <= end_date):
                continue

            if form == "10-K/A":
                amended_10ks_by_accession[get_accession_number(filing)] = filing

    except Exception as exc:
        print(f"Warning: could not separately search 10-K/A filings: {exc}")

    original_10ks = sorted(
        original_10ks,
        key=lambda filing: (get_filing_date(filing), get_accession_number(filing)),
    )

    amended_10ks = sorted(
        amended_10ks_by_accession.values(),
        key=lambda filing: (get_filing_date(filing), get_accession_number(filing)),
    )

    return original_10ks, amended_10ks


# ============================================================
# 4. EDGARTOOLS ITEM EXTRACTION AND TEXT VALIDATION
# ============================================================

# Extract one item section from one 10-K filing using EdgarTools
def extract_item_with_edgartools(filing: Any, item: str) -> str:
    tenk = filing.obj()

    try:
        text = str(tenk[item]).strip()
    except Exception as exc:
        raise ValueError(f"Could not extract {item} with EdgarTools: {exc}")

    if text.lower() in {"", "none", "nan"}:
        raise ValueError(f"EdgarTools returned empty text for {item}")

    return text

# Detect whether text is probably only a table of contents
def looks_like_table_of_contents_only(text: str) -> bool:
    compact = re.sub(r"\s+", " ", str(text)).strip().lower()

    if not compact:
        return False

    opening = compact[:2_000]

    has_toc_phrase = "table of contents" in opening

    item_heading_count = len(
        re.findall(r"\bitem\s+(?:1a?|1b|1c|2|3|4|5|6|7a?|8|9)\b", opening)
    )

    dotted_page_refs = len(
        re.findall(r"\.{3,}\s*\d{1,4}\b", opening)
    )

    if has_toc_phrase and item_heading_count >= 3 and dotted_page_refs >= 2:
        return True

    return False

# Check whether an extracted item section looks usable, based on length and content
# Returns is_valid: True if the section passed quality check, qc_flag: label describing the result, reason: explanation for failures or warnings
def validate_section(text: str, item: str) -> tuple[bool, str, str]:
    text = str(text).strip()
    char_count = len(text)
    word_count = count_words(text)

    if char_count == 0 or word_count == 0:
        return False, "zero_text", f"{item} extraction had zero words"

    if looks_like_table_of_contents_only(text):
        return (
            True,
            "possible_table_of_contents",
            f"{item} may contain table-of-contents content; manual review recommended",
        )

    if char_count < min_section_length[item]:
        return (
            False,
            "suspiciously_short",
            f"{item} is suspiciously short: {char_count:,} characters",
        )

    if char_count > max_section_length[item]:
        return (
            False,
            "suspiciously_long",
            f"{item} is suspiciously long: {char_count:,} characters",
        )

    return True, "passed", ""

# Write list of dictionaries to a CSV file with a fixed column order
def write_csv(data: list[dict[str, Any]], path: Path, columns: list[str]) -> None:
    df = pd.DataFrame(data)

    if df.empty:
        df = pd.DataFrame(columns=columns)
    else:
        df = df.reindex(columns=columns)

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

# ============================================================
# 5. MAIN EXTRACTION 
# ============================================================

def run_extraction() -> None:
    make_output_folders()
    set_identity(sec_user_agent)

    candidates = read_candidate_ciks(input_file)

    # Create empty lists for diagnostic/quality control outputs
    metadata_rows: list[dict[str, Any]] = []
    manual_review_rows: list[dict[str, Any]] = []
    no_qualifying_rows: list[dict[str, Any]] = []
    excluded_10ka_rows: list[dict[str, Any]] = []

    # Loop through each bank and extract specific items from 10-K filings
    for _, candidate in candidates.iterrows():
        cik = candidate["cik"]
        ticker = candidate["ticker"]
        bank_name = candidate["bank_name"]

        print(f"\nProcessing CIK {cik} | {ticker} | {bank_name}")

        # Create an EdgarTools Company object for the CIK and find 10-K or 10-K/A filings
        try:
            company = Company(cik)
            original_10ks, amended_10ks = identify_10k_filings(company)
        except Exception as exc:
            reason = f"Could not identify filings: {exc}"
            print(f"  FAILED: {reason}")

            no_qualifying_rows.append(
                {
                    "cik": cik,
                    "ticker": ticker,
                    "bank_name": bank_name,
                    "reason": reason,
                }
            )
            continue

        # Record excluded 10-K/A filings 
        for amended in amended_10ks:
            excluded_10ka_rows.append(
                {
                    "cik": cik,
                    "ticker": ticker,
                    "filing_date": get_filing_date(amended),
                    "accession_number": get_accession_number(amended),
                    "form": get_filing_form(amended),
                    "exclusion_reason": "10-K/A excluded; script extracts original 10-K filings only.",
                }
            )

        # If no 10-K filings were found, record the CIK and reason for exclusion
        if not original_10ks:
            reason = f"No original 10-K found from {start_date} to {end_date}"
            print(f"  {reason}")

            no_qualifying_rows.append(
                {
                    "cik": cik,
                    "ticker": ticker,
                    "bank_name": bank_name,
                    "reason": reason,
                }
            )
            continue
        
        print(
            f"  Found {len(original_10ks)} original 10-K filing(s); "
            f"excluded {len(amended_10ks)} 10-K/A filing(s)."
        )

        # Loop through each 10-K filing and extract the specified items
        for filing in original_10ks:
            filing_date = get_filing_date(filing)
            accession_number = get_accession_number(filing)
            form = get_filing_form(filing)

            for item in items_to_extract:
                output_path = build_output_path(
                    cik=cik,
                    ticker=ticker,
                    filing_date=filing_date,
                    accession_number=accession_number,
                    item=item,
                )

                if output_path.exists():
                    existing_text = output_path.read_text(
                        encoding="utf-8",
                        errors="replace",
                    )
                    is_valid, qc_flag, reason = validate_section(existing_text, item)

                    # Append metadata for existing file
                    row = {
                        "cik": cik,
                        "ticker": ticker,
                        "bank_name": bank_name,
                        "filing_date": filing_date,
                        "accession_number": accession_number,
                        "form": form,
                        "item": item,
                        "output_file": str(output_path),
                        "status": "already_exists",
                        "extraction_method": "existing_file",
                        "char_count": len(existing_text),
                        "word_count": count_words(existing_text),
                        "qc_flag": qc_flag,
                        "reason": reason,
                        "needs_manual_review": not is_valid or qc_flag != "passed",
                    }

                    metadata_rows.append(row)

                    # Append to manual review if the existing file fails quality check
                    if row["needs_manual_review"]:
                        manual_review_rows.append(
                            {
                                "cik": cik,
                                "ticker": ticker,
                                "bank_name": bank_name,
                                "filing_date": filing_date,
                                "accession_number": accession_number,
                                "item": item,
                                "output_file": str(output_path),
                                "status": row["status"],
                                "qc_flag": qc_flag,
                                "reason": reason,
                                "suggested_action": f"Manually review {item} extraction.",
                            }
                        )

                    print(f"    {item}: already exists")
                    continue
                
                # Try to extract item using EdgarTools; if it fails, record the failure and reason
                try:
                    text = extract_item_with_edgartools(filing, item)
                    is_valid, qc_flag, reason = validate_section(text, item)

                    status = "extracted"
                    extraction_method = "edgartools"

                    output_path.write_text(text, encoding="utf-8")

                    print(f"    {item}: extracted | qc={qc_flag}")

                except Exception as exc:
                    text = ""
                    is_valid = False
                    qc_flag = "extraction_failed"
                    reason = str(exc)
                    status = "failed"
                    extraction_method = "edgartools"

                    print(f"    {item}: failed | {reason}")

                needs_manual_review = not is_valid or qc_flag != "passed"

                # Append metadata for this extraction attempt, whether it succeeded or failed
                row = {
                    "cik": cik,
                    "ticker": ticker,
                    "bank_name": bank_name,
                    "filing_date": filing_date,
                    "accession_number": accession_number,
                    "form": form,
                    "item": item,
                    "output_file": str(output_path),
                    "status": status,
                    "extraction_method": extraction_method,
                    "char_count": len(text),
                    "word_count": count_words(text),
                    "qc_flag": qc_flag,
                    "reason": reason,
                    "needs_manual_review": needs_manual_review,
                }

                metadata_rows.append(row)

                if needs_manual_review:
                    manual_review_rows.append(
                        {
                            "cik": cik,
                            "ticker": ticker,
                            "bank_name": bank_name,
                            "filing_date": filing_date,
                            "accession_number": accession_number,
                            "item": item,
                            "output_file": str(output_path),
                            "status": status,
                            "qc_flag": qc_flag,
                            "reason": reason,
                            "suggested_action": f"Manually review {item} extraction.",
                        }
                    )

            # Pause between filings
            time.sleep(edgar_sleep_seconds)

    # Write metadata, manual review, and summary CSV files
    write_csv(metadata_rows, metadata_file, metadata_columns)
    write_csv(manual_review_rows, manual_review_file, manual_review_columns)
    write_csv(no_qualifying_rows, no_qualifying_file, no_qualifying_columns)
    write_csv(excluded_10ka_rows, excluded_10ka_file, excluded_10ka_columns)

    # Create summary report
    summary_rows = [
        {"metric": "candidate_ciks", "value": len(candidates)},
        {"metric": "metadata_rows", "value": len(metadata_rows)},
        {"metric": "manual_review_rows", "value": len(manual_review_rows)},
        {"metric": "no_qualifying_10k_ciks", "value": len(no_qualifying_rows)},
        {"metric": "excluded_10ka_filings", "value": len(excluded_10ka_rows)},
        {
            "metric": "item_files_extracted",
            "value": sum(row["status"] == "extracted" for row in metadata_rows),
        },
        {
            "metric": "item_files_already_existing",
            "value": sum(row["status"] == "already_exists" for row in metadata_rows),
        },
        {
            "metric": "item_extraction_failures",
            "value": sum(row["status"] == "failed" for row in metadata_rows),
        },
    ]

    write_csv(summary_rows, summary_file, ["metric", "value"])

    print("\nExtraction complete.")
    print(f"Metadata file: {metadata_file}")
    print(f"Manual review file: {manual_review_file}")
    print(f"Summary file: {summary_file}")



def main() -> None:
    run_extraction()


if __name__ == "__main__":
    main()