from pathlib import Path
from bs4 import BeautifulSoup
import re

# File paths
DIRECTORY = Path(__file__).resolve().parents[1]
RAW_TENK_ITEMS_DIR = DIRECTORY / "10k_items"
OUTPUT_DIR = DIRECTORY / "10k_item_tokens"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(text):
    text = re.sub(r"(?<!\w)A\.I\.(?!\w)", "AI", text, flags=re.IGNORECASE) # Normalize A.I. into the single token "ai" for keyword analysis
    text = text.lower() # Convert to lowercase
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ") # Remove any remaining HTML
    text = re.sub(r'http\S+|www\S+', " ", text) # Replace URLs with spaces
    text = re.sub(r'\d+', " ", text) # Replace numbers with spaces
    text = text.replace("_", " ") # Replace underscores with spaces
    text = re.sub(r'\W+', ' ', text) # Replace punctuation and other non-word characters with spaces
    text = re.sub(r'\s+', ' ', text).strip() # Normalize repeated whitespace
    return text

def preprocess_files():
   input_files = sorted(RAW_TENK_ITEMS_DIR.glob("*.txt"))

   if not input_files:
    raise FileNotFoundError(f"No .txt files found in {RAW_TENK_ITEMS_DIR}")

   for file in input_files:
      text = file.read_text(encoding="utf-8", errors="replace")

      cleaned_text = clean_text(text)
      tokens = cleaned_text.split()

      output_file = OUTPUT_DIR / file.name
      output_file.write_text(" ".join(tokens), encoding="utf-8")

   print(f"Preprocessed {len(input_files):,} files.")
   print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    preprocess_files()