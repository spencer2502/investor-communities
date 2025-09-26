import os
import json
import pandas as pd
from datetime import datetime

# Define the output path for the processed JSON data
OUTFILE = "data/raw_market_data.json"

# Define the input path for your CSV file. 
# ASSUMPTION: The CSV is named 'CompanyValues.csv' and is in the project root.
# !! IMPORTANT: CHANGE THIS FILENAME if yours is different !!
MARKET_FILE = "CompanyValues.csv" 

# Define the column mapping from your CSV names to the desired names
COLUMN_MAPPING = {
    'ticker_symbol': 'Ticker',      # Maps to the 'Stock Name' concept
    'day_date': 'Date',             # The trading date
    'close_value': 'Close',         # Closing price
    'volume': 'Volume',             # Trading volume
    'open_value': 'Open',           # Opening price
    'high_value': 'High',           # Daily high price
    'low_value': 'Low'              # Daily low price
    # NOTE: The CSV does not have 'Adj Close', so it will not be included.
}

def load_market_data():
    """Loads market data, renames columns, and standardizes the Date field."""
    
    # 1. Check for the input file
    if not os.path.exists(MARKET_FILE):
        print(f"❌ Market data file not found at {MARKET_FILE}. Please check the filename and location.")
        return

    # Ensure the 'data' output directory exists
    os.makedirs("data", exist_ok=True)
    print(f"🔄 Loading data from {MARKET_FILE}...")

    try:
        # 2. Load the CSV file
        df = pd.read_csv(MARKET_FILE, low_memory=False)
    except Exception as e:
        print(f"❌ Error reading market data file: {e}")
        return

    # 3. Rename columns using the defined mapping
    # Note: We select only the columns we need to avoid including extras.
    df = df.rename(columns=COLUMN_MAPPING)
    
    # Filter down to the exact columns needed for analysis and ensure they exist
    required_cols = list(COLUMN_MAPPING.values())
    if not all(col in df.columns for col in required_cols):
         print("❌ Error: Not all required columns were found after renaming.")
         print(f"Expected: {required_cols}. Found: {df.columns.tolist()}")
         return
         
    df = df[required_cols]

    # 4. Standardize the Date column
    # Use pandas to_datetime with 'infer_datetime_format=True' for robustness
    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True, errors='coerce')
    
    # Drop any rows where date conversion failed
    df.dropna(subset=['Date', 'Ticker'], inplace=True)
    
    # 5. Check and print unique tickers found
    unique_tickers = df['Ticker'].unique()
    print("-" * 50)
    print(f"✅ Loaded {len(df)} records for {len(unique_tickers)} unique tickers.")
    print("Unique Tickers Found (First 10):", unique_tickers[:10].tolist())
    print("-" * 50)

    # 6. Convert to list of dictionaries for JSON output
    # Convert date back to ISO string format before saving
    records = df.to_dict('records')
    for rec in records:
        if isinstance(rec['Date'], pd.Timestamp):
            rec['Date'] = rec['Date'].isoformat()

    # 7. Save the transformed data
    with open(OUTFILE, "w", encoding="utf8") as f:
        json.dump(records, f, indent=2)
    
    print(f"✅ Saved standardized market data to {OUTFILE}")

if __name__ == "__main__":
    # Ensure pandas is installed: pip install pandas
    load_market_data()