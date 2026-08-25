# AI Disclosure in Bank 10-K Filings
The scripts in this repository (updated as of August 2026) scrape 10-K filing sections from a cleaned list of publicly traded U.S. banks and perform frequency and lexicon-based sentiment analysis of artificial intelligence-related keywords. 

## Data and Methodology

I queried WRDS “Bank Fundamentals Annual” database to obtain a sample of publicly traded U.S. and North American financial institutions. 

For each filing, I analyze Item 1 (Business), Item 1A (Risk Factors), and Item 7 (Management's Discussion and Analysis).

AI disclosure is measured using two keywords: "AI" (case-sensitive), and "artificial intelligen" (capturing "artificial intelligence" and related variants). Lexicon-based sentiment analysis is conducted using the Loughran-McDonald Financial Sentiment Library. 


## Background
Emerging technologies create opportunities and uncertainty for firms. Artificial intelligence (AI), in particular, has the potential to improve business operations but also introduces challenges including cybersecurity, regulation, and governance. Corporate governance research argues that boards of directors play a central role in overseeing these emerging strategic risks (Stulz, Tompkins, Williamson, & Ye, 2026).

This project examines how publicly traded U.S. bank holding companies communicate AI in mandatory SEC disclosures and whether corporate governance characteristics help explain variation in those disclosures.

Unlike much of the existing literature which studies whether AI disclosure predicts firm performance or firm value (e.g., Mishra et al., 2022; Wang & Yen, 2023; Alzeghoul & Alsharari, 2024), this project focuses on the determinants of AI disclosure itself. I investigate whether governance characteristics are associated with the prominence of artificial intelligence-related discussion within banks' mandatory disclosures.

The broader goal of this project is to understand how firms communicate their attitude toward emerging technologies that present both opportunities and risks, and whether corporate governance influences those disclosure decisions.

## Results

My initial sample contains 885 financial institutions. After filtering for assets over $2 billion, 225 banks remain. I then exclude 67 banks, either for being foreign (not headquartered in the U.S.), or for not having a sufficient number of 10-K filings for reporting years 2018-2025. The final sample contains 158 publicly-traded, U.S.-headquartered banks and covers 1,264 distinct 10-K filings spanning the entire sampling period from 2018-2025.

The percentage of filings containing at least one AI keyword increased from 12.0% in 2022 to 90.5% in 2025, with 89.2% of all AI mentions appearing in Item 1A. Negative words represent 4.72% of tokens in the context of an AI-related keyword, compared with 1.13% positive words surrounding an AI-related keyword. Uncertainty also accounts for 3.71% of context tokens. Overall, banks are increasingly discussing AI and discussing them in terms of risk: using negative and uncertain language. 
