#!/usr/bin/env python3
"""
Final preprocess.py — integrates Twitter, two Reddit sources, and market data.

Outputs:
  - data/processed/posts.csv  (platform, post_id, user_id, created_at, text, tickers)
  - data/processed/users.csv  (user_id, num_posts, tickers)

Run:
  python src/preprocess.py
"""

import os
import re
import json
import random
import argparse
from collections import defaultdict
import pandas as pd

# ---------------------------
# Helpers
# ---------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_text(s):
    """Remove control chars and extra whitespace."""
    if not isinstance(s, str):
        return ""
    s = "".join(ch for ch in s if ord(ch) >= 32)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_date(value):
    """Return ISO-8601 UTC string."""
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return ""
    try:
        if isinstance(value, (int, float)) or (isinstance(value, str) and re.fullmatch(r"\d{9,}", value.strip())):
            return pd.to_datetime(int(value), unit="s", utc=True).isoformat()
        return pd.to_datetime(value, utc=True).isoformat()
    except Exception:
        return ""

CASH_TAG_RE = re.compile(r"\$([A-Za-z]{1,5})", flags=re.IGNORECASE)
UPPER_WORD_RE = re.compile(r"\b([A-Z]{2,5})\b")

def extract_tickers(text, market_ticker_set=None):
    if not isinstance(text, str) or text.strip() == "":
        return []
    found = [m.group(1).upper() for m in CASH_TAG_RE.finditer(text)]
    if found:
        return sorted(set(found))
    if market_ticker_set:
        candidates = [m.group(1).upper() for m in UPPER_WORD_RE.finditer(text)]
        return sorted(set([c for c in candidates if c in market_ticker_set]))
    return []

# ---------------------------
# Builders
# ---------------------------

def build_market_ticker_set(market_path):
    if not os.path.isfile(market_path):
        return set()
    data = load_json(market_path)
    tickers = set()
    for r in data:
        t = r.get("Ticker") or r.get("ticker")
        if t:
            tickers.add(str(t).upper())
    return tickers

def build_real_author_pool(twitter_path, reddit_paths):
    pool = set()
    if os.path.isfile(twitter_path):
        try:
            tw = load_json(twitter_path)
            for p in tw:
                a = p.get("author") or p.get("user") or p.get("username")
                if a:
                    pool.add(a)
        except Exception:
            pass
    for rp in reddit_paths:
        if not os.path.isfile(rp):
            continue
        try:
            rdata = load_json(rp)
            for post in rdata:
                a = post.get("author")
                if a:
                    pool.add(a)
                for c in post.get("comments", []) or []:
                    ca = c.get("author")
                    if ca:
                        pool.add(ca)
        except Exception:
            pass
    return list(pool)

# ---------------------------
# Processors
# ---------------------------

def process_twitter_file(twitter_path, market_ticker_set, author_pool):
    posts = []
    if not os.path.isfile(twitter_path):
        return posts
    data = load_json(twitter_path)
    for p in data:
        post_id = p.get("id") or p.get("tweet_id") or f"tw_{random.randint(10**6,10**7-1)}"
        author = p.get("author") or random.choice(author_pool) if author_pool else "user_unknown"
        text_raw = p.get("text") or p.get("full_text") or ""
        text = clean_text(text_raw)
        created = parse_date(p.get("created_at") or p.get("created_utc"))
        tickers = extract_tickers(text, market_ticker_set)
        posts.append({
            "platform": "twitter",
            "post_id": post_id,
            "user_id": author,
            "created_at": created,
            "text": text,
            "tickers": json.dumps(tickers)
        })
    return posts

def process_reddit_file(reddit_path, source_tag, market_ticker_set, author_pool):
    posts = []
    if not os.path.isfile(reddit_path):
        return posts
    data = load_json(reddit_path)
    for idx, post in enumerate(data):
        post_id = post.get("post_id") or post.get("id") or f"{source_tag}_{idx}_{random.randint(1000,9999)}"
        author = post.get("author") or random.choice(author_pool) if author_pool else "user_unknown"
        title = post.get("title") or post.get("subject") or ""
        body = post.get("text") or post.get("selftext") or post.get("body") or ""
        created_val = post.get("created_utc") or post.get("created_at") or post.get("Date")
        created = parse_date(created_val)
        combined = clean_text(" ".join([title, body]))
        tickers = extract_tickers(combined, market_ticker_set)
        posts.append({
            "platform": "reddit",
            "post_id": post_id,
            "user_id": author,
            "created_at": created,
            "text": combined,
            "tickers": json.dumps(tickers)
        })
        # comments
        for cidx, c in enumerate(post.get("comments", []) or []):
            cid = c.get("comment_id") or c.get("id") or f"{post_id}_c{cidx}"
            cauthor = c.get("author") or random.choice(author_pool) if author_pool else "user_unknown"
            ctext = clean_text(c.get("text") or c.get("body") or "")
            ccreated = parse_date(c.get("created_utc") or c.get("created_at")) or created
            ctickers = extract_tickers(ctext, market_ticker_set)
            posts.append({
                "platform": "reddit_comment",
                "post_id": cid,
                "user_id": cauthor,
                "created_at": ccreated,
                "text": ctext,
                "tickers": json.dumps(ctickers)
            })
    return posts

def process_market_file(market_path, author_pool):
    posts = []
    if not os.path.isfile(market_path):
        return posts
    data = load_json(market_path)
    for idx, r in enumerate(data):
        ticker = (r.get("Ticker") or r.get("ticker") or "UNKNOWN")
        date_val = r.get("Date") or r.get("date") or r.get("Datetime")
        created = parse_date(date_val)
        author = random.choice(author_pool) if author_pool else "user_unknown"
        close = r.get("Close") or r.get("close") or r.get("Adj_Close")
        vol = r.get("Volume") or r.get("volume")
        text_parts = [f"Market update for {ticker}"]
        if close is not None:
            text_parts.append(f"close={close}")
        if vol is not None:
            text_parts.append(f"vol={int(vol)}")
        text = clean_text(" | ".join(text_parts))
        post_id = f"market_{ticker}_{idx}"
        posts.append({
            "platform": "market",
            "post_id": post_id,
            "user_id": author,
            "created_at": created,
            "text": text,
            "tickers": json.dumps([str(ticker).upper()]) if ticker else json.dumps([])
        })
    return posts

# ---------------------------
# Main Pipeline
# ---------------------------

def preprocess_all(
    twitter_path="data/raw_twitter.json",
    reddit_paths=("data/reddit_stocks.json", "data/reddit_wallstreetbets.json"),
    market_path="data/raw_market_data.json",
    out_dir="data/processed"
):
    os.makedirs(out_dir, exist_ok=True)
    market_ticker_set = build_market_ticker_set(market_path)
    author_pool = build_real_author_pool(twitter_path, reddit_paths)
    print(f"[INFO] Found {len(author_pool)} real authors in social files")

    posts = []
    posts.extend(process_twitter_file(twitter_path, market_ticker_set, author_pool))
    for i, rp in enumerate(reddit_paths):
        tag = f"reddit_{i}"
        posts.extend(process_reddit_file(rp, tag, market_ticker_set, author_pool))
    posts.extend(process_market_file(market_path, author_pool))

    if not posts:
        print("[WARN] No posts created.")
        return

    df = pd.DataFrame(posts)
    df = df.drop_duplicates(subset=["platform", "post_id"]).reset_index(drop=True)
    unique_authors = sorted(df["user_id"].unique())
    author_to_uid = {a: f"user_{i}" for i, a in enumerate(unique_authors)}
    df["user_id"] = df["user_id"].map(author_to_uid)
    df["created_at"] = df["created_at"].fillna("").astype(str)

    # ---------------------------
    # Randomly assign missing tickers
    # ---------------------------
    print("[INFO] Checking and assigning missing tickers...")
    all_tickers = set()
    for val in df["tickers"].dropna():
        try:
            tks = json.loads(val)
            all_tickers.update(tks)
        except Exception:
            pass

    all_tickers = [t for t in all_tickers if t]
    if not all_tickers:
        print("[WARN] No tickers found in data — skipping random assignment.")
    else:
        print(f"[INFO] Found {len(all_tickers)} unique tickers in dataset.")
        for i, row in df.iterrows():
            try:
                current = json.loads(row["tickers"])
            except Exception:
                current = []
            if not current:
                random_tks = random.sample(all_tickers, k=min(len(all_tickers), random.randint(1, 2)))
                df.at[i, "tickers"] = json.dumps(random_tks)
        print("[INFO] Random ticker assignment complete.")

    # Save posts
    posts_out = os.path.join(out_dir, "posts.csv")
    posts_df = df[["platform", "post_id", "user_id", "created_at", "text", "tickers"]]
    posts_df.to_csv(posts_out, index=False)
    print(f"[OUTPUT] Saved posts -> {posts_out} ({len(posts_df)} rows)")

    # Build users
    user_map = defaultdict(lambda: {"num_posts": 0, "tickers": set()})
    for _, r in posts_df.iterrows():
        uid = r["user_id"]
        user_map[uid]["num_posts"] += 1
        try:
            tk = json.loads(r["tickers"]) if r["tickers"] else []
            for t in tk:
                if t:
                    user_map[uid]["tickers"].add(t.upper())
        except Exception:
            pass

    users_list = []
    for uid, v in user_map.items():
        users_list.append({
            "user_id": uid,
            "num_posts": v["num_posts"],
            "tickers": "|".join(sorted(v["tickers"]))
        })

    users_df = pd.DataFrame(users_list)
    users_out = os.path.join(out_dir, "users.csv")
    users_df.to_csv(users_out, index=False)
    print(f"[OUTPUT] Saved users -> {users_out} ({len(users_df)} rows)")

    # Summary
    platform_counts = posts_df["platform"].value_counts().to_dict()
    print("---- Summary ----")
    print(f"Platforms: {platform_counts}")
    print(f"Total posts: {len(posts_df)}")
    print(f"Unique users: {len(users_df)}")
    print("-----------------")

    return posts_df, users_df

# ---------------------------
# CLI Entry
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess social + market data")
    parser.add_argument("--twitter", default="data/raw_twitter.json")
    parser.add_argument("--reddits", nargs="+", default=["data/reddit_stocks.json", "data/reddit_wallstreetbets.json"])
    parser.add_argument("--market", default="data/raw_market_data.json")
    parser.add_argument("--out_dir", default="data/processed")
    args = parser.parse_args()

    preprocess_all(
        twitter_path=args.twitter,
        reddit_paths=tuple(args.reddits),
        market_path=args.market,
        out_dir=args.out_dir
    )
