# AI Disclosure in Bank 10-K Filings
The scripts in this repository (updated as of August 2026) scrape 10-K filing sections from a cleaned list of publicly traded U.S. banks (and related financial institutions) and perform frequency and lexicon-based sentiment analysis of artificial intelligence-related keywords. 

## Sample Construction

I queried WRDS “Bank Fundamentals Annual” database to obtain a sample of publicly traded U.S. and North American depository institutions, commercial banks, savings institutions, and bank holding companies. 

For each filing, I analyze Item 1 (Business), Item 1A (Risk Factors), and Item 7 (Management's Discussion and Analysis).

AI disclosure is measured using two keywords: "AI" (case-sensitive), and "artificial intelligen" (capturing "artificial intelligence" and related variants). Lexicon-based sentiment analysis is conducted using the Loughran-McDonald Financial Sentiment Library. 


## Background
Emerging technologies create both opportunities and uncertainty for firms. As these technologies become increasingly embedded in business operations, boards of directors are expected to oversee not only their strategic potential but also the risks they introduce. Artificial intelligence (AI), in particular, has the potential to improve productivity, fraud detection, customer service, and decision-making while simultaneously introducing challenges related to cybersecurity, model risk, operational resilience, regulation, and governance. Corporate governance research argues that boards of directors play a central role in overseeing these emerging strategic risks (Stulz, Tompkins, Williamson, & Ye, 2026).

This project examines how publicly traded U.S. bank holding companies communicate AI in mandatory SEC disclosures and whether corporate governance characteristics help explain variation in those disclosures.

Unlike much of the existing literature which studies whether AI disclosure predicts firm performance or firm value (e.g., Mishra et al., 2022; Wang & Yen, 2023; Alzeghoul & Alsharari, 2024), this project focuses on the determinants of AI disclosure itself. I investigate whether governance characteristics are associated with the prominence of artificial intelligence-related discussion within banks' mandatory disclosures.

Although motivated by artificial intelligence, the broader goal of this project is to understand how firms communicate emerging technologies that present both opportunities and risks, and whether corporate governance influences those disclosure decisions.
