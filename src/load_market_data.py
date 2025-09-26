import os
import json
import pandas as pd
from datetime import datetime

# Define the output path for the processed JSON data
OUTFILE = "data/raw_market_data.json"

# Define the input paths for your two CSV files. 
FILE_PATHS = {
    "CompanyValues": "CompanyValues.csv",
    "YFinanceData": "stock_yfinance_data.csv"
}

# --- Column Mappings for Standardization ---
# This dictionary maps the diverse column names from both files 
# to the desired, uniform output column names.
COLUMN_MAPPER = {
    # CompanyValues.csv columns (Source 1)
    'ticker_symbol': 'Ticker',
    'day_date': 'Date',
    'close_value': 'Close',
    'volume': 'Volume',
    'open_value': 'Open',
    'high_value': 'High',
    'low_value': 'Low',
    
    # stick_yfinance_data.csv columns (Source 2 - already close to desired format)
    'Stock Name': 'Ticker',
    'Date': 'Date',
    'Open': 'Open',
    'High': 'High',
    'Low': 'Low',
    'Close': 'Close',
    'Volume': 'Volume',
    'Adj Close': 'Adj_Close', # Maps to a standardized name
}

# Define the final columns we want in the output
FINAL_COLUMNS = ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj_Close']

def load_market_data():
    """Loads market data from two files, merges, renames columns, and standardizes the Date field."""
    
    os.makedirs("data", exist_ok=True)
    all_dfs = []
    
    # 1. Load and Process Each File Individually
    for file_key, file_path in FILE_PATHS.items():
        if not os.path.exists(file_path):
            print(f"❌ Market data file not found at {file_path}. Skipping this file.")
            continue
            
        print(f"🔄 Loading data from {file_path}...")

        try:
            df = pd.read_csv(file_path, low_memory=False)
            
            # Rename columns using the defined mapping
            # This ensures 'ticker_symbol' becomes 'Ticker', 'day_date' becomes 'Date', etc.
            df.columns = [COLUMN_MAPPER.get(col, col) for col in df.columns]
            
            # --- Specific Date Formatting Handling ---
            if file_key == "CompanyValues":
                # Source 1 Date: 29-05-2020 (DD-MM-YYYY)
                df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
            else:
                # Source 2 Date: Let Pandas infer the format (usually YYYY-MM-DD or MM-DD-YYYY)
                df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True, errors='coerce')

            all_dfs.append(df)
            
        except Exception as e:
            print(f"❌ Error reading data file {file_path}: {e}")
            
    # 2. Consolidate DataFrames
    if not all_dfs:
        print("❌ No data successfully loaded from either file. Aborting.")
        return

    # Merge all DataFrames into one master DataFrame
    df_combined = pd.concat(all_dfs, ignore_index=True)
    print(f"✅ Successfully combined {len(all_dfs)} files into one DataFrame with {len(df_combined)} records.")

    # 3. Final Standardization and Cleanup
    
    # Select only the FINAL_COLUMNS (dropping temporary or unnecessary columns)
    df_final = df_combined[[col for col in FINAL_COLUMNS if col in df_combined.columns]].copy()
    
    # Drop any rows where Ticker or Date is missing (or failed conversion)
    df_final.dropna(subset=['Date', 'Ticker'], inplace=True)
    
    # Remove duplicate rows (e.g., if one stock's data for the same day appears in both files)
    df_final.drop_duplicates(subset=['Date', 'Ticker'], keep='last', inplace=True)
    
    # 4. Final Ticker Check
    unique_tickers = df_final['Ticker'].unique()
    print("-" * 50)
    print(f"✅ Final data: {len(df_final)} records for {len(unique_tickers)} unique tickers.")
    print("Unique Tickers Found (First 10):", unique_tickers[:10].tolist())
    print("-" * 50)

    # 5. Convert to list of dictionaries for JSON output
    records = df_final.to_dict('records')
    for rec in records:
        # Convert date back to ISO string format (YYYY-MM-DDTHH:MM:SS)
        if isinstance(rec['Date'], pd.Timestamp):
            rec['Date'] = rec['Date'].isoformat()

    # 6. Save the transformed data
    with open(OUTFILE, "w", encoding="utf8") as f:
        json.dump(records, f, indent=2)
    
    print(f"✅ Saved standardized market data to {OUTFILE}")

if __name__ == "__main__":
    # Ensure pandas is installed: pip install pandas
    load_market_data()