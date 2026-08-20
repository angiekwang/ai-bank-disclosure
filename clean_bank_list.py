import os
from pathlib import Path
import pandas as pd

# Relative paths 
script_folder = Path(__file__).resolve().parent
project_root = script_folder.parent

input_file = project_root / "base_data" / "lrg_bnk_lst.csv"
output_dir = project_root / "base_data"

df = pd.read_csv(input_file, skiprows=4) # Skip first 4 rows of data intro

# Identify columns to keep and rename them for clarity
rename_dict = {
        "Nat'l Rank": "national_rank",
        "Bank ID": "rssd_id",
        "Bank Name / Holding Co Name": "bank_name",
        "Bank Headquarters": "bank_headquarters",
        "Consol Assets (Mil $)": "consolidated_assets_mil",
         }
df = df.rename(columns=rename_dict)

columns_to_keep = [
        "national_rank",
        "rssd_id",
        "bank_name",
        "bank_headquarters",
        "consolidated_assets_mil"
    ]

df = df[columns_to_keep]

# Remove banks under $2 billion in consolidated assets
df["consolidated_assets_mil"] = df["consolidated_assets_mil"].astype(str).str.replace(",", "") # remove commas from consolidated assets column
df["consolidated_assets_mil"] = pd.to_numeric(df["consolidated_assets_mil"]) # convert consol assets to numeric

df = df[df["consolidated_assets_mil"] >= 2000] # Keep only banks with consolidated assets of $2 billion or more
print(pd.concat([df.head(10), df.tail(10)])) # Print first and last 10 rows of cleaned dataframe for verification

# Create cleaned output file
output_dir.mkdir(parents=True, exist_ok=True)
cleaned_file_path = output_dir / "cleaned_bank_list.csv"
df.to_csv(cleaned_file_path, index=False)
print(f"Cleaned bank list saved to: {cleaned_file_path}")