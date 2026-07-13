# Corporate Governance and AI Disclosure in U.S. Bank 10-K Filings

This repository contains fully replicable code for a project examining how publicly traded U.S. banks discuss artificial intelligence in annual SEC filings. The analysis focuses on Items 1, 1A, and 7 of Form 10-K filings from filing years 2020 through 2026. 

# 1. Overview
Emerging technologies create both extraordinary opportunities and significant uncertainty for firms. Artificial intelligence (AI), in particular, has the potential to improve productivity, fraud detection, customer service, and decision-making while simultaneously introducing challenges related to cybersecurity, model risk, operational resilience, regulation, and governance. Corporate governance research argues that boards of directors play a central role in overseeing these emerging strategic risks (Stulz, Tompkins, Williamson, & Ye, 2026).

This project examines how publicly traded U.S. bank holding companies communicate AI in mandatory SEC disclosures and whether corporate governance characteristics help explain variation in those disclosures.

Unlike much of the existing literature, which studies whether AI disclosure predicts firm performance or firm value (e.g., Mishra et al., 2022; Wang & Yen, 2023; Alzeghoul & Alsharari, 2024), this project focuses on the determinants of AI disclosure itself. Rather than treating AI disclosure solely as a proxy for technology adoption, I investigate whether governance characteristics are associated with the prominence of AI within banks' mandatory disclosures.

Although motivated by artificial intelligence, the broader goal of this project is to understand how firms communicate emerging technologies that present both substantial opportunities and strategic risks, and whether corporate governance influences those disclosure decisions.

This repository contains a fully reproducible Python pipeline for constructing a research dataset of AI disclosure measures, AI-specific sentiment, and bank-year variables from SEC Form 10-K filings.

# 2. Research Contributions

This project extends the existing literature in the following ways:

- Examines the **determinants**, rather than the consequences, of AI disclosure.
- Frames AI disclosure within the literature on **corporate governance and board oversight of emerging strategic risks**.
- Constructs a reproducible panel of **1,554 SEC Form 10-K filings** from **222 publicly traded U.S. bank holding companies** spanning filing years **2020–2026**.
- Focuses specifically on **Item 1 (Business)**, **Item 1A (Risk Factors)**, and **Item 7 (Management's Discussion and Analysis)** instead of analyzing complete annual reports.
- Measures AI disclosure using explicit AI terminology ("AI" and "artificial intelligen") rather than broad digital transformation dictionaries.
- Develops **AI-specific sentiment measures** by applying the Loughran–McDonald financial sentiment dictionary only to localized context surrounding AI mentions.
- Provides a fully documented and reproducible Python workflow for SEC data collection, preprocessing, feature engineering, and text analysis.


# 3. Methodology & Sample Construction

The initial sample is drawn from the Federal Reserve Board's "U.S. Domestically Chartered Commercial Banks" dataset. Banks with less than $2 billion in consolidated assets are excluded.

Remaining institutions are linked to publicly traded parent holding companies using SEC and WRDS identifier mapping datasets. Banks are removed from the sample if they:

- are privately held,
- are owned by foreign parent companies,
- are subsidiaries of another publicly traded holding company,
- or do not possess complete Form 10-K coverage for filing years 2020 through 2026.

The final sample consists of __ publicly traded U.S. bank holding companies and __ SEC Form 10-K filings.

For each filing, only the sections most relevant to business strategy and risk disclosure are analyzed: namely, Item 1 (Business), Item 1A (Risk Factors), and Item 7 (Management's Discussion and Analysis)

AI disclosure is measured using two explicit AI indicators:

- `"AI"` (case-sensitive)
- `"artificial intelligen"` (capturing "artificial intelligence" and related variants)

Rather than measuring sentiment across entire filings, this project extracts localized sentence windows surrounding each AI mention and applies the Loughran–McDonald financial sentiment dictionary to construct AI-specific measures of positive, negative, uncertainty, and net sentiment.

I obtain a sample of publicly traded U.S. banks with over $2 billion in consolidated assets, with 10-K filings available for all filing years from 2020 through 2026. Banks are removed from the sample for the following reasons: they are owned by foreign parent holding companies (thereby not filing 10-K reports), are a subsidiary of a larger holding company (in which case only the holding company was kept), or do not possess filings spanning the entire seven-year sampling period. 

# 4. Data Sources

This project combines publicly available regulatory data with financial research databases.

| Source | Purpose |
|---------|---------|
| Federal Reserve Board – *U.S. Domestically Chartered Commercial Banks* | Initial bank dataset |
| SEC Company Tickers Exchange | CIK and ticker identifiers |
| WRDS identifier linking datasets | RSSD ID, PERMCO, and GVKEY identifier mappings |
| SEC EDGAR | Form 10-K filings (retrieved using EdgarTools) |
| Loughran–McDonald Financial Sentiment Dictionary (March 2026 release) | Lexicon-based sentiment analysis |
| WRDS financial databases | Bank financial characteristics and governance variables used in regression analysis |

---


# 4. Repo Structure
This repository contains the following scripts (in order):
| Script | Purpose |
|---------|---------|
| `clean_bank_list.py` | Filters FRB bank list to institutions with consolidated assets exceeding $2 billion. |
| `link_bank_identifiers.py` | Links banks to RSSD ID, PERMCO, GVKEY, CIK, and ticker identifiers while constructing the final bank sample. |
| `extract_10k_text.py` | Downloads Form 10-K Items 1, 1A, and 7 from the SEC EDGAR database using the EdgarTools library. |
| `preprocess_10k_text.py` | Cleans and tokenizes filing text for subsequent frequency and sentiment analyses. |
| `ai_frequency_analysis.py` | Measures AI disclosure frequency using keywords `"AI"` and `"artificial intelligen"`. |
| `extract_keyword_windows.py` | Extracts sentence-level context surrounding each AI keyword occurrence. |
| `prepare_lm_dictionary.py` | Cleans and prepares the Loughran–McDonald financial sentiment dictionary. |
| `score_ai_sentiment.py` | Computes AI-specific positive, negative, uncertainty, and net sentiment measures from keyword context windows. |


# 5. Reproduction Information

Before running the extraction scripts, set a SEC user agent by running:
```bash
export SEC_USER_AGENT="Your Name youremail@example.com"
```

# References

Alzeghoul, A., & Alsharari, N. M. (2024). Impact of AI disclosure on financial reporting and performance: Evidence from U.S. banks. *Journal of Risk and Financial Management, 18*(1), 1–32.

Mishra, S., Ewing, M. T., & Cooper, H. B. (2022). Artificial intelligence focus and firm performance. *Journal of the Academy of Marketing Science, 50*(6), 1176–1197. https://doi.org/10.1007/s11747-022-00876-5

Stulz, R. M., Tompkins, J., Williamson, R., & Ye, Z. (2026). Why do bank boards have risk committees? *Journal of Financial and Quantitative Analysis.*

Wang, T., & Yen, J.-C. (2023). Does AI bring value to firms? Value relevance of AI disclosures. *Die Unternehmung, 77*(2), 134–161.
