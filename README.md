# AI Disclosure in Bank 10-K Filings 

The scripts in this repository (updated as of August 2026) scrape 10-K filing sections from a cleaned list of publicly traded U.S. banks and perform frequency and lexicon-based sentiment analysis. 

# 1. Reproduction Information

Clone the repository and install the required packages by running:

```bash
pip install -r requirements.txt
```

For scraping filings using EDGAR, define an SEC user agent by running:

```bash
export SEC_USER_AGENT="Your Name your.email@example.com"
```

Then, execute each script in the order listed in Section 4.


# 2. Background

Emerging technologies create both extraordinary opportunities and significant uncertainty for firms. As these technologies become increasingly embedded in business operations, boards of directors are expected to oversee not only their strategic potential but also the risks they introduce. Artificial intelligence (AI), in particular, has the potential to improve productivity, fraud detection, customer service, and decision-making while simultaneously introducing challenges related to cybersecurity, model risk, operational resilience, regulation, and governance. Corporate governance research argues that boards of directors play a central role in overseeing these emerging strategic risks (Stulz, Tompkins, Williamson, & Ye, 2026).

This project examines how publicly traded U.S. bank holding companies communicate AI in mandatory SEC disclosures and whether corporate governance characteristics help explain variation in those disclosures.

Unlike much of the existing literature which studies whether AI disclosure predicts firm performance or firm value (e.g., Mishra et al., 2022; Wang & Yen, 2023; Alzeghoul & Alsharari, 2024), this project focuses on the determinants of AI disclosure itself. I investigate whether governance characteristics are associated with the prominence of artificial intelligence-related discussion within banks' mandatory disclosures.

Although motivated by artificial intelligence, the broader goal of this project is to understand how firms communicate emerging technologies that present both opportunities and risks, and whether corporate governance influences those disclosure decisions.

This repository contains a fully reproducible Python pipeline for constructing a research dataset of AI disclosure measures, AI-specific sentiment, and bank-year variables from SEC Form 10-K filings.

# 3. Sample Construction

The initial sample is drawn from the Federal Reserve Board's "U.S. Domestically Chartered Commercial Banks" dataset. Banks with less than $2 billion in consolidated assets are excluded.

Remaining institutions are linked to publicly traded parent holding companies using SEC and WRDS identifier mapping datasets. Banks are removed from the sample if they are privately held, are owned by foreign parent companies, are subsidiaries of another publicly traded holding company, or do not possess complete Form 10-K coverage for filing years 2020 through 2026.

The final sample consists of __ publicly traded U.S. bank holding companies and __ SEC Form 10-K filings.

For each filing, only the sections most relevant to business strategy and risk disclosure are analyzed: namely, Item 1 (Business), Item 1A (Risk Factors), and Item 7 (Management's Discussion and Analysis)

AI disclosure is measured using two keywords: "AI" (case-sensitive), and "artificial intelligen" (capturing "artificial intelligence" and related variants).

---


# 4. Repo Structure
This repository contains the following scripts (in order):
| Script | Purpose |
|---------|---------|
| clean_bank_list.py | Filters FRB bank list to institutions with consolidated assets exceeding $2 billion. |
| link_bank_identifiers.py | Links banks to RSSD ID, PERMCO, GVKEY, CIK, and ticker identifiers while constructing the final bank sample. |
| extract_10k_text.py | Downloads Form 10-K Items 1, 1A, and 7 from the SEC EDGAR database using the EdgarTools library. |
| preprocess_10k_text.py | Cleans and tokenizes filing text for subsequent frequency and sentiment analyses. |
| ai_frequency_analysis.py | Measures AI disclosure frequency using keywords "AI" and "artificial intelligen". |
| extract_keyword_windows.py | Extracts sentence-level context surrounding each AI keyword occurrence. |
| prepare_lm_dictionary.py | Cleans and prepares the Loughran–McDonald financial sentiment dictionary. |
| score_ai_sentiment.py | Computes AI-specific positive, negative, uncertainty, and net sentiment measures from keyword context windows. |


# References

Alzeghoul, A., & Alsharari, N. M. (2024). Impact of AI disclosure on financial reporting and performance: Evidence from U.S. banks. *Journal of Risk and Financial Management, 18*(1), 1–32.

Mishra, S., Ewing, M. T., & Cooper, H. B. (2022). Artificial intelligence focus and firm performance. *Journal of the Academy of Marketing Science, 50*(6), 1176–1197. https://doi.org/10.1007/s11747-022-00876-5

Stulz, R. M., Tompkins, J., Williamson, R., & Ye, Z. (2026). Why do bank boards have risk committees? *Journal of Financial and Quantitative Analysis.*

Wang, T., & Yen, J.-C. (2023). Does AI bring value to firms? Value relevance of AI disclosures. *Die Unternehmung, 77*(2), 134–161.
