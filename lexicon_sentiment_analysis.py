from pathlib import Path
import pandas as pd
import re
from nltk.tokenize import sent_tokenize

# File paths
DIRECTORY = Path(__file__).resolve().parents[1]
RAW_TEXT_DIR = DIRECTORY / "10k_items"
lm_library_raw = pd.read_csv(DIRECTORY / "base_data" / "Loughran-McDonald_MasterDictionary_1993-2025.csv")
successful_cik_file = DIRECTORY / "metadata" / "successfully_extracted_bank_ciks.txt"
OUTPUT_FILE = DIRECTORY / "metadata" / "ai_context_sentiment.csv"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

successful_ciks = set(successful_cik_file.read_text(encoding="utf-8").splitlines())


# Convert words to lowercase
lm_library_raw["word"] = (
    lm_library_raw["word"]
    .astype(str)
    .str.strip()
    .str.lower()
)

sentiment_columns = [
    "negative",
    "positive",
    "uncertainty",
    "litigious",
    "strong_modal",
    "weak_modal",
    "constraining",
]

# Identify "AI" keyword
ai_re = re.compile(
    r"(?<![A-Za-z])(?:AI|A\.I\.)(?![A-Za-z])"
)

# Identify "artificial intelligen" keyword
artificial_intelligen_re = re.compile(
    r"\bartificial(?:ly)?[\s-]+intelligen\w*\b",
    flags=re.IGNORECASE,
)

# Keep original category year codes and convert active category memberships into binary indicators
for column in sentiment_columns:
   year_column = f"{column}_year_code"

   lm_library_raw[year_column] = pd.to_numeric(
      lm_library_raw[column],
      errors="coerce"
    ).fillna(0)

   lm_library_raw[column] = (
      lm_library_raw[year_column] > 0
    ).astype(int)


# Create set of negative, positive, uncertainty, litigious, and constraining words 
sentiment_word_sets = {
    column: set(
        lm_library_raw.loc[
            lm_library_raw[column] == 1,
            "word"
        ]
    )
    for column in sentiment_columns
}

# Identify AI keyword windows: the sentence preceding, containing, and after the appearance of the AI keyword
def extract_ai_windows(text):
    sentences = [
        sentence.strip()
        for sentence in sent_tokenize(text)
        if sentence.strip()
    ]

    windows = []

    for sentence_index, sentence in enumerate(sentences):
        ai_count = len(ai_re.findall(sentence))

        artificial_intelligen_count = len(
            artificial_intelligen_re.findall(sentence)
        )

        keyword_count = (
            ai_count
            + artificial_intelligen_count
        )

        if keyword_count == 0:
            continue

        previous_sentence = (
            sentences[sentence_index - 1]
            if sentence_index > 0
            else ""
        )

        following_sentence = (
            sentences[sentence_index + 1]
            if sentence_index < len(sentences) - 1
            else ""
        )

        context_text = " ".join(
            part
            for part in [
                previous_sentence,
                sentence,
                following_sentence,
            ]
            if part
        )

        windows.append(
            {
                "sentence_index": sentence_index,
                "previous_sentence": previous_sentence,
                "keyword_sentence": sentence,
                "following_sentence": following_sentence,
                "context_text": context_text,
                "ai_count": ai_count,
                "artificial_intelligen_count": (
                    artificial_intelligen_count
                ),
                "keyword_count": keyword_count,
            }
        )

    return windows

def count_sentiment(text):
   tokens = re.findall(
    r"[A-Za-z]+",
    text.lower()
)
   word_count = len(tokens)

   sentiment_results = {
        "word_count": word_count
    }

   for category in sentiment_columns:
        category_count = sum(
            token in sentiment_word_sets[category]
            for token in tokens
        )

        category_share = (
            category_count / word_count
            if word_count > 0
            else 0
        )

        sentiment_results[f"{category}_count"] = (
            category_count
        )

        sentiment_results[f"{category}_share"] = (
            category_share
        )

   sentiment_results["net_sentiment"] = (
      (
      sentiment_results["positive_count"]
       - sentiment_results["negative_count"]
            )
         / word_count
         if word_count > 0
         else 0
      )
   return sentiment_results


input_files = sorted(RAW_TEXT_DIR.glob("*_item_*.txt"))

# Keep only files belonging to banks in the final sample.
input_files = [
    file
    for file in input_files
    if file.name.split("_")[1] in successful_ciks
]

rows = []
files_without_ai = 0

for file in input_files:
    raw_text = file.read_text(
        encoding="utf-8",
        errors="replace"
    )

    ai_windows = extract_ai_windows(raw_text)

    if not ai_windows:
        files_without_ai += 1
        continue

    for window_number, window in enumerate(
        ai_windows,
        start=1
    ):
        sentiment_results = count_sentiment(
            window["context_text"]
        )

        rows.append(
            {
                "filename": file.name,
                "window_number": window_number,
                **window,
                **sentiment_results,
            }
        )

results = pd.DataFrame(rows)

results.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"Analyzed {len(input_files):,} filing-Item files.")
print(f"Created {len(results):,} AI-context windows.")
print(f"Files without AI keywords: {files_without_ai:,}")
print(f"Results saved to: {OUTPUT_FILE}")