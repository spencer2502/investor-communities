import os
import json
import kagglehub
import pandas as pd
from datetime import datetime, timezone

OUTFILE = "data/raw_twitter.json"
KAGGLE_DATASET = "zzsleepyzzz/crypto-tweets-data-and-sentiments"
LOCAL_CSV_FILE = "stock_tweets.csv" # The user-provided local file name

# --- Transformation Function for Local CSV ---
def load_local_csv_data(df_local, start_id):
    """Transforms a DataFrame loaded from the local CSV into the standardized JSON format."""
    tweets_out = []
    current_id = start_id
    
    print(f"🔄 Processing local data from {LOCAL_CSV_FILE}...")

    for _, row in df_local.iterrows():
        try:
            current_id += 1
            
            # Date Conversion: The format is 'YYYY-MM-DD HH:MM:SS+00:00' (ISO with timezone)
            # Pandas pd.to_datetime handles this complex string directly.
            # Convert to a timezone-aware object (UTC) and then to ISO string format.
            dt_utc = pd.to_datetime(row['Date'], utc=True).replace(tzinfo=timezone.utc)
            created_at_iso = dt_utc.isoformat()
            
            tweet_obj = {
                "id": f"t_{current_id}",
                "source": "twitter",
                # The local CSV lacks a specific author; use a standardized placeholder
                "author": "AnonymousLocalUser", 
                "text": str(row['Tweet']),
                "created_at": created_at_iso
            }
            tweets_out.append(tweet_obj)

        except KeyError as ke:
            # Note: This will catch errors if 'Date' or 'Tweet' column names are wrong.
            print(f"⚠️ Skipped row: Missing column {ke} in local CSV.")
            continue
        except Exception as e:
            print(f"⚠️ Skipped row in local CSV due to unexpected error: {e}")
            continue
            
    print(f"✅ Successfully processed {len(tweets_out)} records from local CSV.")
    return tweets_out, current_id


# --- Main Load and Transform Function (Modified) ---
def load_and_transform_tweets():
    os.makedirs("data", exist_ok=True)
    all_tweets_out = []
    
    # --- PART A: Process Kaggle Data (Runs First) ---
    print("--- Starting Kaggle Data Collection ---")
    
    # 1. Download the dataset using kagglehub
    try:
        download_path = kagglehub.dataset_download(KAGGLE_DATASET)
        print(f"✅ Download complete. Dataset stored at: {download_path}")
    except Exception as e:
        print(f"❌ KaggleHub Error: Ensure you are authenticated. Details: {e}")
        return

    # 2. Identify, load, and consolidate all relevant Kaggle CSV files
    all_files = os.listdir(download_path)
    csv_files = [f for f in all_files if f.endswith('.csv') and not f.startswith('__')]
    
    all_tweets_df = []
    for filename in csv_files:
        file_path = os.path.join(download_path, filename)
        try:
            df = pd.read_csv(file_path, low_memory=False)
            all_tweets_df.append(df)
        except Exception:
            continue
            
    if all_tweets_df:
        df_kaggle = pd.concat(all_tweets_df, ignore_index=True)
        print(f"✅ Consolidated {len(df_kaggle)} records from Kaggle files.")
        
        # 3. Transform Kaggle data
        kaggle_tweets = []
        # Initialize tweet_id counter for Kaggle data
        current_id = 0
        
        for index, row in df_kaggle.iterrows():
            try:
                current_id += 1 # Increment ID first (starts at t_1)
                
                # Date conversion for format '01/15/2024' (MM/DD/YYYY)
                date_str = str(row['date']).strip()
                datetime_obj = datetime.strptime(date_str, '%m/%d/%Y')
                created_at_iso = datetime_obj.isoformat()
                
                kaggle_tweets.append({
                    "id": f"t_{current_id}",
                    "source": "twitter",
                    "author": str(row['username']), 
                    "text": str(row['tweet_text']), 
                    "created_at": created_at_iso
                })
            except Exception:
                continue
        
        all_tweets_out.extend(kaggle_tweets)
        # Store the last assigned ID for the next dataset to pick up
        last_processed_id = current_id 
    else:
        last_processed_id = 0
        
    print(f"--- Kaggle Data Processing Complete. Total: {len(all_tweets_out)} tweets ---")


    # --- PART B: Process Local CSV Data (Runs Second) ---
    print("\n--- Starting Local CSV Processing ---")
    
    if os.path.exists(LOCAL_CSV_FILE):
        try:
            df_local = pd.read_csv(LOCAL_CSV_FILE)
            
            # Pass the next available ID to the local processing function
            local_tweets, final_id = load_local_csv_data(df_local, last_processed_id)
            all_tweets_out.extend(local_tweets)
            
        except Exception as e:
            print(f"❌ Error loading local file {LOCAL_CSV_FILE}: {e}")
    else:
        print(f"Local CSV file '{LOCAL_CSV_FILE}' not found. Skipping.")
        
    # --- PART C: Final Save ---
    
    # 4. Save the transformed data
    with open(OUTFILE, "w", encoding="utf8") as f:
        json.dump(all_tweets_out, f, indent=2)
    
    print(f"\n✅ FINAL SAVE: Total {len(all_tweets_out)} tweets from all sources saved to {OUTFILE}")

if __name__ == "__main__":
    load_and_transform_tweets()