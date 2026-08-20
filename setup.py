#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
Create the directory structure required for the AI Risk Text Analysis project.

Run from the repository root:

```
python setup_project.py
```

The script is idempotent: existing folders are preserved, so it can be run
multiple times without deleting or overwriting project data.
"""

from future import annotations

from pathlib import Path

# Repository root is the folder containing this script.

PROJECT_ROOT = Path(file).resolve().parent

# Required project directories.

DIRECTORIES = [
# Source code
"src",

# Original input datasets
"base_data",

# Bank identifiers, filing metadata, dictionaries, and lookup files
"metadata",

# Raw SEC filing sections
"10k_texts_by_cik",

# Cleaned filing text
"preprocessed_texts_by_cik",

# Tokenized filing text
"preprocessed_tokens_by_cik",

# AI keyword frequency outputs
"outputs/ai_frequency",

# AI keyword-level and context-window outputs
"outputs/ai_context_windows",

# AI-specific sentiment outputs
"outputs/ai_sentiment",

# Bank-year disclosure and sentiment datasets
"outputs/bank_year_features",

# Summary tables and descriptive statistics
"outputs/summaries",

# Quality-control and diagnostic files
"outputs/diagnostics",

# Figures and tables for the README or research paper
"outputs/figures",
"outputs/tables",

# Runtime logs
"logs",

]

def create_directories() -> None:
   print(f"Project root: {PROJECT_ROOT}")
   print("\nChecking project directories:")


created_count = 0
existing_count = 0

for relative_path in DIRECTORIES:
    directory = PROJECT_ROOT / relative_path

    if directory.exists():
        print(f"  EXISTS   {relative_path}")
        existing_count += 1
    else:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  CREATED  {relative_path}")
        created_count += 1

    # Keep otherwise-empty folders visible in Git.
    gitkeep_file = directory / ".gitkeep"
    if not any(directory.iterdir()):
        gitkeep_file.touch(exist_ok=True)

print("\nSetup complete.")
print(f"Directories created: {created_count}")
print(f"Directories already present: {existing_count}")


def main() -> None:
   create_directories()

if name == main:
   main()
