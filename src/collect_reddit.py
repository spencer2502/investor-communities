# src/collect_reddit.py
"""
Collect Reddit posts and comments from r/WallStreetBets
Saves JSON to data/raw_reddit.json
"""
import os, json
from dotenv import load_dotenv
load_dotenv()
import praw
from datetime import datetime, timezone

REDDIT_CLIENT = os.getenv("REDDIT_CLIENT_ID")
REDDIT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
USER_AGENT = os.getenv("REDDIT_USER_AGENT", "investor-communities-app/0.1")

OUTFILE = "data/raw_reddit.json"
SUBREDDIT = "wallstreetbets"
LIMIT_POSTS = 200  # tune

if not (REDDIT_CLIENT and REDDIT_SECRET):
    print("Reddit credentials missing. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env or env vars.")
    raise SystemExit(1)

reddit = praw.Reddit(client_id=REDDIT_CLIENT,
                     client_secret=REDDIT_SECRET,
                     user_agent=USER_AGENT)

def collect():
    os.makedirs("data", exist_ok=True)
    posts = []
    sub = reddit.subreddit(SUBREDDIT)
    for submission in sub.new(limit=LIMIT_POSTS):
        try:
            title = submission.title or ""
            selftext = submission.selftext or ""
            created = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc).isoformat()
            author = str(submission.author) if submission.author else None
            # collect top-level comments (flatten limited)
            submission.comments.replace_more(limit=0)
            comments = []
            for c in submission.comments.list()[:200]:
                comments.append({
                    "id": getattr(c, "id", ""),
                    "author": str(getattr(c, "author", None)),
                    "text": getattr(c, "body", "")
                })
            posts.append({
                "id": f"r_{submission.id}",
                "source": "reddit",
                "author": author,
                "text": title + "\n" + selftext,
                "created_at": created,
                "comments": comments
            })
        except Exception as e:
            print("Skipped submission due to", e)
    with open(OUTFILE, "w", encoding="utf8") as f:
        json.dump(posts, f, indent=2)
    print("Saved reddit posts to", OUTFILE)

if __name__ == "__main__":
    collect()
