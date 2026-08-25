from pathlib import Path
import pandas as pd
import time
from edgar import set_identity, Company

# EDGAR API requires an ID for each user (name and email)
SEC_USER_AGENT = "Angie Wang aw1407@georgetown.edu"
set_identity(SEC_USER_AGENT)

# File paths
DIRECTORY = Path(__file__).resolve().parents[1]
OUTPUT_DIR = DIRECTORY / "10k_items"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR = DIRECTORY / "metadata"
METADATA_DIR.mkdir(parents=True, exist_ok=True)
input_file = DIRECTORY / "base_data" / "clean_bank_list.csv"

# Fiscal year 2019 through 2026 --> use fiscal period info in XBRL (later)
DATE_RANGE = "2018-01-01:2026-08-01"
START_DATE = pd.to_datetime("2018-01-01")
END_DATE = pd.to_datetime("2026-08-01")

# Minimum required filings
MIN_REQUIRED_FILINGS = 8

# Load ciks & ticker
clean_bank_list = pd.read_csv(
    input_file, 
    dtype={"cik": str}
) 

banks = clean_bank_list[
   ["cik", "tic", "conm"]
].copy()

# Clean CIK, ticker, and company-name columns
banks["cik"] = (
    banks["cik"]
    .str.strip()
    .str.replace(r"\.0$", "", regex=True)
    .str.zfill(10)
)

banks["tic"] = (
    banks["tic"]
    .fillna("UNKNOWN")
    .astype(str)
    .str.strip()
    .str.upper()
)

banks["conm"] = (
    banks["conm"]
    .fillna("")
    .astype(str)
    .str.strip()
)

def create_filename(tic, cik, report_date, accession):
    return f"{tic}_{cik}_{report_date}_{accession}"

# Extract 10-K filings for each CIK
tenk_files = []
errors = []

for bank in banks.itertuples(index=False): 
      cik = bank.cik
      tic = bank.tic
      company_name = bank.conm

      try:
         tenk_filings = Company(cik).get_filings(
            form = "10-K", 
            amendments = False, 
            filing_date = DATE_RANGE
         )
      except Exception as error:
         errors.append(
        {
            "cik": cik,
            "ticker": tic,
            "company_name": company_name,
            "accession_number": "",
            "error": str(error),
        }
    )
         print(f"error for cik {cik}: {error}")
         continue

      if len(tenk_filings) == 0:
        errors.append(
        {
            "cik": cik,
            "ticker": tic,
            "accession_number": "",
            "company_name": company_name,
      
            "error": "No 10-K filings found in requested date range",
        }
         ) 
        print(f"No 10-K filings found for {company_name} ({tic} - {cik})")
        continue

      print(
        f"{company_name} ({tic} - {cik}): "
        f"found {len(tenk_filings)} filings"
      )

      # Extract items 1, 1A, and 7 from each filing
      for filing in tenk_filings:
         try:
            tenk = filing.obj()

            if tenk is None:
               raise ValueError(f"Filing object is None for CIK {cik} and accession {filing.accession_number}")

            item1 = str(tenk["Item 1"] or "")
            item1a = str(tenk["Item 1A"] or "")
            item7 = str(tenk["Item 7"] or "")

            reporting_date = pd.to_datetime(
                filing.period_of_report,
                errors="coerce",
            )

            # Keep fiscal/report periods beginning in 2019.
            if pd.isna(reporting_date):
                print(
                    "Missing reporting date for "
                    f"{filing.accession_number}"
                )
                continue

            if not (
                START_DATE
                <= reporting_date
                <= END_DATE
            ):
                continue

            accession = filing.accession_number
            report_date_string = reporting_date.strftime("%Y-%m-%d")

            filename = create_filename(
               tic,
               cik,
               report_date_string,
               accession
            )

            items = {
            "item_1": item1,
            "item_1A": item1a,
            "item_7": item7,
         }

            for item_name, item_text in items.items():
               (OUTPUT_DIR / f"{filename}_{item_name}.txt").write_text(
                  item_text,
                  encoding="utf-8"
               )


            tenk_files.append(
               {
                  "cik": cik,
                  "accession_number": accession,
               }
            )

         except Exception as error:
            errors.append(
               {
                  "cik": cik,
                  "ticker": tic,
                  "company_name": company_name,
                  "accession_number": filing.accession_number,
                  "error": str(error)
               }
            )
            print(
                "Error processing "
                f"{filing.accession_number} "
                f"for CIK {cik}: {error}"
            )

      time.sleep(0.5)

         
results = pd.DataFrame(tenk_files)

# Count successfully extracted filings for every input bank.
filing_counts = (
    results
    .groupby("cik")["accession_number"]
    .nunique()
)

bank_filing_counts = (
    banks
    .rename(
        columns={
            "tic": "ticker",
            "conm": "company_name",
        }
    )
    .drop_duplicates(subset="cik")
    .copy()
)

bank_filing_counts["filings_successfully_extracted"] = (
    bank_filing_counts["cik"]
    .map(filing_counts)
    .fillna(0)
    .astype(int)
)

# Banks with at least eight successfully extracted filings.
successful_banks = bank_filing_counts[
    bank_filing_counts["filings_successfully_extracted"]
    >= MIN_REQUIRED_FILINGS
].copy()

# Save successful banks and their identifiers to csv
successful_banks.to_csv(METADATA_DIR / "successfully_extracted_banks.csv", index=False)

# Banks with fewer than eight successfully extracted filings.
incomplete_banks = bank_filing_counts[
    bank_filing_counts["filings_successfully_extracted"]
    < MIN_REQUIRED_FILINGS
].copy()


# Create txt file of CIKs of successfully extracted banks (final sample of banks)
(METADATA_DIR / "successfully_extracted_bank_ciks.txt").write_text(
    "\n".join(successful_banks["cik"]),
    encoding="utf-8"
)

for bank in incomplete_banks.itertuples(index=False):
    errors.append(
        {
            "cik": bank.cik,
            "ticker": bank.ticker,
            "company_name": bank.company_name,
            "accession_number": "",
            "error": (
                "10-K filings did not span the entire sampling period: " f"{bank.filings_successfully_extracted} of " f"{MIN_REQUIRED_FILINGS} filings extracted"
            ),
        }
    )

error_results = pd.DataFrame(
   errors,
   columns = ["cik", "ticker", "company_name", "accession_number", "error"]
)
error_results.to_csv(
   METADATA_DIR / "10k_items_errors.csv", 
   index=False
)


print(f"Extracted {len(results):,} filings.")
print(f"Encountered {len(error_results):,} filing errors.")
print(f"Successfully extracted filings for "f"{len(successful_banks):,} banks.")
