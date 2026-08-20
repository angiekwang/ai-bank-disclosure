import pandas as pd
from pathlib import Path

"""
1. Link RSSD ID to PERMCO via FRBNY CRSP-TIC-RSSD link table base_data/crsp_20240930v2.csv
2. Link PERMCO to GVKEY via WRDS https://wrds-www.wharton.upenn.edu/pages/classroom/using-crspcompustat-merged-database/
3. Link GVKEY to CIK and ticker
4. Create master identifier table

WRDS linking tables: https://wrds-www.wharton.upenn.edu/pages/grid-items/wrds-sec-linking-tables/
"""

# Identify relative paths and input/output files

script_folder = Path(__file__).resolve().parent
project_root = script_folder.parent

clean_bank_list = project_root / "base_data" / "cleaned_bank_list.csv"
frbny_rssd_to_permco_link = project_root / "base_data" / "crsp_20240930v2.csv"
wrds_permco_to_gvkey_link = project_root / "base_data" / # put file name here
wrds_gvkey_to_cik_ticker_link = project_root / "base_data" / # put file name here

output_file = project_root / "base_data" / "bank_identifiers_linked.csv"
removed_banks_file = project_root / "metadata" / "removed_banks.csv"

def rssd_to_permco_link(bank_list: Path, frbny_crsp: Path) -> pd.DataFrame:
   """
   Explanation
   """

   bank_df = pd.read_csv(bank_list)
   frbny_df = pd.read_csv(frbny_crsp, dtype={"entity": str, "permco": str})

   # Standardize column names
   frbny_df = frbny_df.rename(columns={"entity": "rssd_id"})

   # Convert IDs to strings
   bank_df["rssd_id"] = bank_df["rssd_id"].astype(str) 
   frbny_df["rssd_id"] = frbny_df["rssd_id"].astype(str)
   frbny_df["permco"] = frbny_df["permco"].astype(str)

   # Keep only relevant columns and drop duplicates
   frbny_df = frbny_df[["rssd_id", "permco"]].drop_duplicates()

   # Merge the two dataframes on RSSD ID to link to PERMCO
   rssd_to_permco_df = pd.merge(bank_df, frbny_df, on="rssd_id", how="left")
   return rssd_to_permco_df

def permco_to_gvkey_link(rssd_to_permco_df: pd.DataFrame, wrds_permco_to_gvkey_link: Path) -> pd.DataFrame:
   """
   Explanation
   """

   permco_to_gvkey_df = pd.read_csv(wrds_permco_to_gvkey_link, dtype={"permco": str, "gvkey": str})

   # Standardize column names
   # REVISE WITH ACTUAL COLUMN NAMES
   permco_to_gvkey_df = permco_to_gvkey_df.rename(columns={"permco": "permco", "gvkey": "gvkey"}) 

   # Convert IDs to strings
   rssd_to_permco_df["permco"] = rssd_to_permco_df["permco"].astype(str)
   permco_to_gvkey_df["permco"] = permco_to_gvkey_df["permco"].astype(str)
   permco_to_gvkey_df["gvkey"] = permco_to_gvkey_df["gvkey"].astype(str)

   # Keep only relevant columns and drop duplicates
   permco_to_gvkey_df = permco_to_gvkey_df[["permco", "gvkey"]].drop_duplicates()

   # Merge the two dataframes on PERMCO to link to GVKEY
   permco_to_gvkey_linked_df = pd.merge(rssd_to_permco_df, permco_to_gvkey_df, on="permco", how="left")
   return permco_to_gvkey_linked_df

def gvkey_to_cik_ticker_link(permco_to_gvkey_df: pd.DataFrame, wrds_gvkey_to_cik_ticker_link: Path) -> pd.DataFrame:
   """
   Explanation
   """

   gvkey_to_cik_ticker_df = pd.read_csv(wrds_gvkey_to_cik_ticker_link, dtype={"gvkey": str, "cik": str, "ticker": str})

   # Standardize column names
   # REVISE WITH ACTUAL COLUMN NAMES
   gvkey_to_cik_ticker_df = gvkey_to_cik_ticker_df.rename(columns={"gvkey": "gvkey", "cik": "cik", "ticker": "ticker"})

   # Convert IDs to strings
   permco_to_gvkey_df["gvkey"] = permco_to_gvkey_df["gvkey"].astype(str)
   gvkey_to_cik_ticker_df["gvkey"] = gvkey_to_cik_ticker_df["gvkey"].astype(str)
   gvkey_to_cik_ticker_df["cik"] = gvkey_to_cik_ticker_df["cik"].astype(str)
   gvkey_to_cik_ticker_df["ticker"] = gvkey_to_cik_ticker_df["ticker"].astype(str)

   # Keep only relevant columns and drop duplicates
   gvkey_to_cik_ticker_df = gvkey_to_cik_ticker_df[["gvkey", "cik", "ticker"]].drop_duplicates()

   # Merge the two dataframes on GVKEY to link to CIK and ticker
   final_linked_df = pd.merge(permco_to_gvkey_df, gvkey_to_cik_ticker_df, on="gvkey", how="left")
   return final_linked_df


# Record banks without matching CIK to ticker & append to removed_banks.csv
def remove_banks_without_cik_ticker(
    final_df: pd.DataFrame,
    removed_banks_file: Path
   ) -> pd.DataFrame:

    removed_banks_file.parent.mkdir(parents=True, exist_ok=True)

    # Identify banks missing either CIK or ticker
    missing_cik_ticker_df = final_df[
        final_df["cik"].isna()
        | final_df["ticker"].isna()
        | (final_df["cik"] == "nan")
        | (final_df["ticker"] == "nan")
    ].copy()

    # Add reason for removal
    missing_cik_ticker_df["reason_for_removal"] = "no_matching_cik_or_ticker"

    # Keep banks that have both CIK and ticker
    final_kept_df = final_df[
        final_df["cik"].notna()
        & final_df["ticker"].notna()
        & (final_df["cik"] != "nan")
        & (final_df["ticker"] != "nan")
    ].copy()

    # Append removed banks to removed_banks.csv
    if len(missing_cik_ticker_df) > 0:
        if removed_banks_file.exists():
            existing_removed_df = pd.read_csv(removed_banks_file)

            updated_removed_df = pd.concat(
                [existing_removed_df, missing_cik_ticker_df],
                ignore_index=True
            )
        else:
            updated_removed_df = missing_cik_ticker_df

        updated_removed_df = updated_removed_df.drop_duplicates()
        updated_removed_df.to_csv(removed_banks_file, index=False)

        print(f"Removed banks without CIK/ticker: {len(missing_cik_ticker_df)}")

    return final_kept_df

# Main workflow
def main():
    rssd_to_permco_df = rssd_to_permco_link(
        clean_bank_list,
        frbny_rssd_to_permco_link
    )

    permco_to_gvkey_df = permco_to_gvkey_link(
        rssd_to_permco_df,
        wrds_permco_to_gvkey_link
    )

    final_df = gvkey_to_cik_ticker_link(
        permco_to_gvkey_df,
        wrds_gvkey_to_cik_ticker_link
    )

    final_kept_df = remove_banks_without_cik_ticker(
        final_df,
        removed_banks_file
    )
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    final_kept_df.to_csv(output_file, index=False)

    print(f"Saved linked bank identifiers to: {output_file}")
    print(f"Removed-bank log file: {removed_banks_file}")

if __name__ == "__main__":
    main()