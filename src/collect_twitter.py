# src/collect_twitter.py
"""
Collect recent tweets with cashtags using Tweepy v2.
Save to data/raw_twitter.json (with checkpointing).
"""
import os, json
from dotenv import load_dotenv
import tweepy
from datetime import datetime

load_dotenv()
BEARER = os.getenv("TWITTER_BEARER_TOKEN")
OUTFILE = "data/raw_twitter.json"
QUERY = "$GME OR $AMC OR $TSLA -is:retweet lang:en"

MAX_RESULTS = 100          # Max allowed per request
TWEET_LIMIT = 300          # Safer cap (adjust <500 requests per month)

if not BEARER:
    raise SystemExit("❌ Twitter BEARER token missing. Set TWITTER_BEARER_TOKEN in .env.")

client = tweepy.Client(bearer_token=BEARER, wait_on_rate_limit=True)

def collect():
    os.makedirs("data", exist_ok=True)
    
    # Load existing tweets (checkpoint)
    if os.path.exists(OUTFILE):
        with open(OUTFILE, "r", encoding="utf8") as f:
            tweets_out = json.load(f)
        print(f"🔄 Resuming... already have {len(tweets_out)} tweets.")
    else:
        tweets_out = []

    try:
        for t in tweepy.Paginator(
            client.search_recent_tweets,
            query=QUERY,
            tweet_fields=["created_at","entities","author_id"],
            max_results=MAX_RESULTS
        ).flatten(limit=TWEET_LIMIT):
            
            tweet_obj = {
                "id": f"t_{t.id}",
                "source": "twitter",
                "author": str(t.author_id),
                "text": t.text,
                "created_at": t.created_at.isoformat() if t.created_at else None
            }

            # Avoid duplicates
            if tweet_obj not in tweets_out:
                tweets_out.append(tweet_obj)

    except Exception as e:
        print("⚠️ Twitter collection error:", e)
    
    with open(OUTFILE, "w", encoding="utf8") as f:
        json.dump(tweets_out, f, indent=2)
    
    print(f"✅ Saved {len(tweets_out)} tweets to {OUTFILE}")

if __name__ == "__main__":
    collect()
