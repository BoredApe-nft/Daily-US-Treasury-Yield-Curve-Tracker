#!/usr/bin/env python3

import requests
import pandas as pd
from datetime import datetime
import os

def fetch_yield_data():
    # US Treasury daily yield CSV - this URL changes sometimes! Check Treasury.gov for the latest direct CSV
    # For demo purposes let's use a FRED CSV link or a static URL.
    # Replace with the correct daily CSV link.
    # Example FRED series (multiple calls would be needed in reality for multiple maturities)
    #
    # For this example, let's show *mock* code that simulates data:
    #
    # --- Replace this with real scrape code ---
    maturities = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
    rates = [5.35, 5.30, 5.25, 5.15, 5.05, 5.00, 4.95, 4.90, 4.85, 4.80, 4.75]

    today = datetime.now().isoformat()

    df = pd.DataFrame([rates], columns=maturities)
    df.insert(0, "Date", today)

    os.makedirs("data", exist_ok=True)
    df.to_csv(f"data/{today}.csv", index=False)
    print(f"Saved yield data for {today}.")

if __name__ == "__main__":
    fetch_yield_data()
