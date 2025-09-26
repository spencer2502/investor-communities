# src/load_kaggle_twitter.py - FIX APPLIED

import os
import json
import kagglehub
import pandas as pd
from datetime import datetime

OUTFILE = "data/raw_twitter.json"
KAGGLE_DATASET = "zzsleepyzzz/crypto-tweets-data-and-sentiments"

# --- Transformation Function ---
def load_and_transform_tweets():
    os.makedirs("data", exist_ok=True)
    
    # 1. Download the dataset using kagglehub
    try:
        print(f"🔄 Downloading latest version of {KAGGLE_DATASET}...")
        download_path = kagglehub.dataset_download(KAGGLE_DATASET)
        print("✅ Download complete. Dataset stored at:", download_path)
    except Exception as e:
        print(f"❌ KaggleHub Error: Ensure you are authenticated. Details: {e}")
        return

    # 2. Identify and load all relevant CSV files
    all_files = os.listdir(download_path)
    csv_files = [f for f in all_files if f.endswith('.csv') and not f.startswith('__')]
    
    if not csv_files:
        print(f"❌ No suitable CSV files found in the directory: {download_path}")
        return

    all_tweets_df = []
    print(f"🔄 Found {len(csv_files)} files. Consolidating data...")
    
    for filename in csv_files:
        file_path = os.path.join(download_path, filename)
        try:
            # Load each CSV file
            df = pd.read_csv(file_path, low_memory=False)
            all_tweets_df.append(df)
        except Exception as e:
            print(f"⚠️ Could not read {filename}. Skipping. Error: {e}")
            continue

    # Concatenate all DataFrames into one
    if not all_tweets_df:
        print("❌ No data successfully loaded from any file.")
        return
        
    df = pd.concat(all_tweets_df, ignore_index=True)
    
    print(f"✅ Consolidated a total of {len(df)} records.")

    # 3. Transform data to match the desired JSON structure
    tweets_out = []
    
    # Use 'unique_id' for mapping, as the original data may not have a reliable 
    # 'id' column for tweets across different files.
    for index, row in df.iterrows():
        try:
            # Use DataFrame index as a unique ID fallback
            tweet_id = index 
            
            # Date conversion for format '01/15/2024'
            date_str = str(row['date']).strip()
            # Ensure proper handling if the date format is inconsistent across files.
            # We assume the %m/%d/%Y format based on your previous info.
            datetime_obj = datetime.strptime(date_str, '%m/%d/%Y')
            created_at_iso = datetime_obj.isoformat()
            
            tweet_obj = {
                "id": f"t_{tweet_id}",
                "source": "twitter",
                "author": str(row['username']), 
                "text": str(row['tweet_text']), 
                "created_at": created_at_iso
            }

            tweets_out.append(tweet_obj)

        except KeyError as ke:
            # This happens if a required column is missing in one of the files
            # (highly unlikely if all files share the same structure)
            print(f"Skipped row {index}: Missing expected column data ({ke}).")
            continue
        except ValueError as ve:
            # Handles errors if a date string is malformed or in a different format
            print(f"Skipped row {index}: Date format error for value '{row.get('date')}'. Error: {ve}")
            continue
        except Exception as e:
            print(f"Skipped row {index} due to unexpected error: {e}")
            continue

    # 4. Save the transformed data
    with open(OUTFILE, "w", encoding="utf8") as f:
        json.dump(tweets_out, f, indent=2)
    
    print(f"\n✅ Saved {len(tweets_out)} tweets to {OUTFILE}")

if __name__ == "__main__":
    load_and_transform_tweets()