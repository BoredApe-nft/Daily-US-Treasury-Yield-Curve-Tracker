# 📈 Daily US Treasury Yield Curve Tracker

This repository **automatically** collects and stores the US Treasury yield curve daily.

## 📌 What it does

- Runs a GitHub Actions workflow every day
- Fetches (or simulates) the current yield curve
- Saves it as a CSV file with the date
- Commits and pushes to this repo

## 📂 Data structure

All data is stored in the `data/` folder:

```
data/
2025-07-08.csv
2025-07-09.csv
...
```


Each CSV contains:

| Date       | 1M  | 3M  | 6M  | 1Y  | 2Y  | ... | 30Y |
|------------|-----|-----|-----|-----|-----|-----|-----|
| 2025-07-08 | 5.35| 5.30| ... | ... | ... | ... | ... |

## 🤖 Automation

- GitHub Actions runs daily at 12:00 UTC
- Can also be manually triggered

## 💡 Uses

- Building historical yield curve archives
- Time series analysis
- Macro research
- Training models (e.g. yield curve fitting)

---

## ⚠️ Note

This repo currently includes a **demo** data generator for reliable daily commits.  
For real scraping, replace the Python script to download Treasury's CSV or use FRED API.

