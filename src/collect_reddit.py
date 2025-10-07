"""
Collects Reddit posts and comments with rich metadata for community analysis.
"""
import os
import json
import praw
from dotenv import load_dotenv
from datetime import datetime, timezone
import argparse

# Load environment variables from .env file
load_dotenv()

# Define the output directory for raw data
OUT_DIR = "data/raw"

def collect_posts(subreddit_name: str, limit: int, post_category: str):
    """
    Collects posts and top-level comments from a specified subreddit.

    Args:
        subreddit_name (str): The name of the subreddit to scrape.
        limit (int): The maximum number of posts to fetch.
        post_category (str): The category of posts to fetch ('hot', 'new', 'top').
    """
    # --- 1. Authentication (from .env) ---
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT")
    )
    print(f"✅ Authenticated successfully with Reddit API.")

    sub = reddit.subreddit(subreddit_name)
    print(f"Fetching {limit} posts from r/{subreddit_name} using category '{post_category}'...")

    # --- 2. Select Post Category Dynamically ---
    if post_category == 'hot':
        submissions_generator = sub.hot(limit=limit)
    elif post_category == 'new':
        submissions_generator = sub.new(limit=limit)
    elif post_category == 'top':
        submissions_generator = sub.top(limit=limit, time_filter='day')
    else:
        raise ValueError("Invalid post category. Choose from 'hot', 'new', 'top'.")

    collected_data = []
    for i, post in enumerate(submissions_generator):
        # Skip stickied posts as they are not organic community content
        if post.stickied:
            continue

        print(f"--> Processing post {i + 1}/{limit}: {post.id}")
        
        comments_data = []
        post.comments.replace_more(limit=0) # Flatten the comment tree efficiently
        
        for comment in post.comments.list()[:50]: # Limit to 50 top comments
            # Ensure the author exists to build the graph
            if comment.author:
                comments_data.append({
                    "comment_id": comment.id,
                    "author": str(comment.author.name),
                    "text": comment.body,
                    "score": comment.score
                })
        
        # Only include posts with an author
        if post.author:
            # --- 3. Collect Rich Metadata ---
            collected_data.append({
                "post_id": post.id,
                "author": str(post.author.name),
                "title": post.title,
                "text": post.selftext,
                "score": post.score,
                "upvote_ratio": post.upvote_ratio,
                "num_comments": post.num_comments,
                "created_utc": post.created_utc,
                "comments": comments_data
            })

    # --- 4. Save to a Uniquely Named File ---
    os.makedirs(OUT_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(OUT_DIR, f"reddit_{subreddit_name}_{timestamp}.json")
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(collected_data, f, indent=4)
        
    print(f"\n✅ Successfully saved {len(collected_data)} posts to {filename}")


if __name__ == "__main__":
    # --- 5. Add Command-Line Arguments for Flexibility ---
    parser = argparse.ArgumentParser(description="Collect Reddit data for community analysis.")
    parser.add_argument("--subreddit", type=str, default="wallstreetbets", help="Subreddit to scrape.")
    parser.add_argument("--limit", type=int, default=100, help="Number of posts to collect.")
    parser.add_argument("--category", type=str, default="hot", choices=['hot', 'new', 'top'], help="Post category to fetch.")
    
    args = parser.parse_args()
    
    collect_posts(subreddit_name=args.subreddit, limit=args.limit, post_category=args.category)