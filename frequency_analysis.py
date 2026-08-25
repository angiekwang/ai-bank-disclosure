from pathlib import Path
import re
import pandas as pd

# File paths
DIRECTORY = Path(__file__).resolve().parents[1]
TEXT_TOKENS = DIRECTORY / "10k_item_tokens"
successful_cik_file = DIRECTORY / "metadata" / "successfully_extracted_bank_ciks.txt"
successful_ciks = set(successful_cik_file.read_text(encoding="utf-8").splitlines())
OUTPUT_FILE = DIRECTORY / "metadata" / "ai_keyword_frequency.csv"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# Search pattern for phrase stem "artificial intelligen"
artificial_intelligen_re = re.compile(
    r"\bartificial(?:ly)?\s+intelligen\w*\b"
)

filename_re = re.compile(
    r"^(?P<ticker>.+?)_"
    r"(?P<cik>\d{10})_"
    r"(?P<reporting_date>\d{4}-\d{2}-\d{2})_"
    r"(?P<accession_number>\d{10}-\d{2}-\d{6})_"
    r"item_(?P<item>1A|1|7)\.txt$",
    flags=re.IGNORECASE,
)



def read_filename(file):
    match = filename_re.match(file.name)

    if match is None:
        raise ValueError(
            f"Unexpected filename format: {file.name}"
        )

    identifiers = match.groupdict()

    identifiers["item"] = (
        f"Item {identifiers['item'].upper()}"
    )

    identifiers["reporting_year"] = int(
    identifiers["reporting_date"][:4]
)
    return identifiers

def count_ai_keywords(text):
    tokens = text.split()

    # Count the standalone lowercase token "ai".
    ai_count = tokens.count("ai")

    # Count phrases beginning with artificial/artificially intelligen-.
    artificial_intelligen_count = len(
        artificial_intelligen_re.findall(text)
    )

    ai_raw_freq = (
        ai_count
        + artificial_intelligen_count
    )

    token_count = len(tokens)

    ai_share = (
        ai_raw_freq / token_count
        if token_count > 0
        else 0
    )

    ai_per_1000_tokens = ai_share * 1000

    return {
        "ai_count": ai_count,
        "artificial_intelligen_count": artificial_intelligen_count,
        "ai_raw_freq": ai_raw_freq,
        "token_count": token_count,
        "ai_share": ai_share,
        "ai_per_1000_tokens": ai_per_1000_tokens,
        "includes_ai_keyword": int(ai_raw_freq > 0),
    }

def main():
    input_files = [
    file
    for file in sorted(TEXT_TOKENS.glob("*.txt"))
    if file.name.split("_")[1] in successful_ciks
]

    if not input_files:
        raise FileNotFoundError(
            f"No token files found in {TEXT_TOKENS}"
        )

    rows = []

    for file in input_files:
        text = file.read_text(encoding="utf-8", errors="replace")
        identifiers = read_filename(file)
        keyword_counts = count_ai_keywords(text)

        rows.append(
            {
                **identifiers,
                "filename": file.name,
                **keyword_counts,
            }
        )

    results = pd.DataFrame(rows)
    
    results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(f"Analyzed {len(results):,} filing-Item files.")
    print(
        f"Files containing AI keywords: "
        f"{results['includes_ai_keyword'].sum():,}"
    )
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()