# src/preprocess.py
"""
Preprocess Twitter, Reddit, and Market Price data.
- Cleans text
- Extracts tickers
- Fills missing user_id / ticker if needed
- Saves posts.csv, users.csv
- Normalizes market data into prices/<TICKER>.csv
"""

import os
import re
import json
import argparse
import pandas as pd
import unicodedata
import random
from collections import defaultdict



# ----------------------------
# Utils
# ----------------------------

def load_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def strip_nonprintable(s: str):
    """Remove weird unicode and keep ASCII only."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    return s


TICKER_PATTERN = re.compile(r"\$[A-Z]{1,5}")


def extract_tickers(text: str):
    if not isinstance(text, str):
        return []
    return [t[1:] for t in TICKER_PATTERN.findall(text)]


def random_user_id():
    """Generate a fallback user ID if missing."""
    return f"u_{random.randint(10000, 99999)}"


# ----------------------------
# Process Twitter
# ----------------------------

def process_twitter(raw_file, ticker_pool):
    print(f"[Twitter] loading {raw_file}")
    data = load_json_file(raw_file)
    posts = []

    for post in data:
        tid = post.get("id")
        uid = post.get("author") or random_user_id()
        text = strip_nonprintable(post.get("text") or "")
        tickers = extract_tickers(text)

        # fallback: assign a random ticker if none found
        if not tickers and ticker_pool:
            tickers = [random.choice(ticker_pool)]

        created = post.get("created_at")

        posts.append({
            "platform": "twitter",
            "post_id": tid,
            "user_id": uid,
            "created_at": created,
            "text": text,
            "tickers": json.dumps(tickers)
        })

    print(f"[Twitter] loaded {len(posts)} posts")
    return pd.DataFrame(posts)


# ----------------------------
# Process Reddit
# ----------------------------

def process_reddit(raw_file, ticker_pool):
    print(f"[Reddit] loading {raw_file}")
    data = load_json_file(raw_file)
    posts = []

    for post in data:
        pid = post.get("id")
        uid = post.get("author") or random_user_id()
        text = strip_nonprintable(post.get("text") or "")
        tickers = extract_tickers(text)

        # fallback: assign random ticker if none
        if not tickers and ticker_pool:
            tickers = [random.choice(ticker_pool)]

        created = post.get("created_at")

        posts.append({
            "platform": "reddit",
            "post_id": pid,
            "user_id": uid,
            "created_at": created,
            "text": text,
            "tickers": json.dumps(tickers)
        })

        # also process comments as posts
        for c in post.get("comments", []):
            cid = c.get("id")
            cuid = c.get("author") or random_user_id()
            ctext = strip_nonprintable(c.get("text") or "")
            ctickers = extract_tickers(ctext)
            if not ctickers and ticker_pool:
                ctickers = [random.choice(ticker_pool)]

            posts.append({
                "platform": "reddit_comment",
                "post_id": cid,
                "user_id": cuid,
                "created_at": created,
                "text": ctext,
                "tickers": json.dumps(ctickers)
            })

    print(f"[Reddit] loaded {len(posts)} posts/comments")
    return pd.DataFrame(posts)


# ----------------------------
# Process Prices
# ----------------------------

def process_price_files(price_path, out_dir):
    prices_dir = os.path.join(out_dir, "prices")
    os.makedirs(prices_dir, exist_ok=True)

    if not os.path.isfile(price_path):
        print(f"[PriceLoader] No price file found at {price_path}")
        return {}, []

    try:
        data = load_json_file(price_path)
        df = pd.DataFrame(data)

        df.columns = [c.lower() for c in df.columns]
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        expected = ["ticker", "date", "open", "high", "low", "close", "volume", "adj_close"]
        for col in expected:
            if col not in df.columns:
                df[col] = pd.NA

        price_map = {}
        ticker_pool = []
        for ticker, subdf in df.groupby("ticker"):
            subdf = subdf.sort_values("date")
            outpath = os.path.join(prices_dir, f"{ticker.upper()}.csv")
            subdf.to_csv(outpath, index=False)
            price_map[ticker.upper()] = subdf
            ticker_pool.append(ticker.upper())
            print(f"[PriceLoader] Saved {len(subdf)} rows for {ticker} -> {outpath}")

        return price_map, ticker_pool

    except Exception as e:
        print(f"[PriceLoader] Failed to process {price_path}: {e}")
        return {}, []


# ----------------------------
# Preprocess All
# ----------------------------

def preprocess_all(twitter_file, reddit_file, price_file, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    price_map, ticker_pool = process_price_files(price_file, out_dir)

    df_twitter = process_twitter(twitter_file, ticker_pool)
    df_reddit = process_reddit(reddit_file, ticker_pool)

    df_all = pd.concat([df_twitter, df_reddit], ignore_index=True)
    df_all.to_csv(os.path.join(out_dir, "posts.csv"), index=False)
    print(f"[Output] Saved {len(df_all)} posts -> {os.path.join(out_dir, 'posts.csv')}")

    # build user-level file
    user_groups = defaultdict(list)
    for _, row in df_all.iterrows():
        user_groups[row["user_id"]].append(row.to_dict())

    users = []
    for uid, posts in user_groups.items():
        tickers = set()
        for p in posts:
            tickers.update(p["tickers"].split(",") if p["tickers"] else [])
        users.append({
            "user_id": uid,
            "num_posts": len(posts),
            "tickers": ",".join(sorted(tickers))
        })

    df_users = pd.DataFrame(users)
    df_users.to_csv(os.path.join(out_dir, "users.csv"), index=False)
    print(f"[Output] Saved {len(df_users)} users -> {os.path.join(out_dir, 'users.csv')}")

    return df_all, df_users, price_map


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--twitter", required=True, help="raw_twitter.json")
    parser.add_argument("--reddit", required=True, help="raw_reddit.json")
    parser.add_argument("--prices", required=True, help="raw_market_data.json")
    parser.add_argument("--out_dir", required=True, help="output directory")
    args = parser.parse_args()

    preprocess_all(
        twitter_file=args.twitter,
        reddit_file=args.reddit,
        price_file=args.prices,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
