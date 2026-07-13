# Artificial Intelligence Disclosures in U.S. Bank 10-K Filings

This repository contains fully replicable code for a project examining how publicly traded U.S. banks discuss artificial intelligence in annual SEC filings. The analysis focuses on Items 1, 1A, and 7 of Form 10-K filings from filing years 2020 through 2026. 

# 1. Overview
Include literature review-esque blurb 

# 2. Methodology & Sample Construction
I obtain a sample of publicly traded U.S. banks with over $2 billion in consolidated assets, with 10-K filings available for all filing years from 2020 through 2026. Banks are removed from the sample for the following reasons: they are owned by foreign parent holding companies (thereby not filing 10-K reports), are a subsidiary of a larger holding company (in which case only the holding company was kept, or do not possess filings spanning the entire seven-year sampling period. 

# 3. Data Sources

This project uses the following sources:
* The initial sample of publicly traded banks comes from the FRB's "U.S. Domestically Chartered Commercial Banks" open-access dataset, available at https://www.federalreserve.gov/releases/lbr/current/
* Bank identifier linking datasets come from WRDS __ and __ (insert names), as well as the SEC's "Company Tickers Exchange," available at https://www.sec.gov/file/company-tickers-exchange
* The Loughran-McDonald dictionary used for sentiment analysis is taken from https://sraf.nd.edu/loughranmcdonald-master-dictionary/ (using file updated in March 2026)
* Include other variables used for regression

# 4. Repo Structure
This repository contains the following scripts (in order):
1. **clean_bank_list.py** --> Cleans FRB's "U.S. Domestically Chartered Commercial Banks" bank list by removing banks under $2 billion in consolidated assets.
2. **link_bank_identifiers.py** --> Uses various bank identifier linking datasets to obtain final sample of banks along with RSSD ID, PERMCO, GVKEY, CIK, and ticker identifiers.
3. **extract_10k_text.py** --> Uses EdgarTools Python library to extract 10-K Item 1, 1A, and 7 sections for all banks in final sample spanning years 2020 through 2026.
4. **preprocess_10k_text.py** --> Cleans raw 10-K text by removing HTML tags, white spaces, and numbers, as well as tokenizing text to be used for frequency and sentiment analysis.
5. **ai_frequency_analysis.py** --> Counts mentions of keywords "AI" and "artificial intelligen" in preprocessed 10-K token files.
6. **extract_keyword_windows.py** --> Extract sentence windows around AI keyword matches to be used for sentiment analysis.
7. **prepare_lm_library.py** --> Cleans Loughran-McDonald library for sentiment analysis.
8. **score_ai_sentiment.py** --> Uses sentence-level context surrounding AI keywords to produce sentiment measures.


# 5. Reproduction Information

Before running the extraction scripts, set a SEC user agent by running:
```bash
export SEC_USER_AGENT="Your Name youremail@example.com"
```

# 6. Main Outputs
