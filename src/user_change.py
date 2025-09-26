import pandas as pd
import random

# Load data
posts = pd.read_csv("data/processed/posts.csv", low_memory=False)
users = pd.read_csv("data/processed/users.csv")

# Filter valid users (exclude 'None' if desired)
valid_users = users[users['user_id'].notna() & (users['user_id'] != 'None')]['user_id'].tolist()

# If you want to keep the giant 'None' user, include them as well
# valid_users = users['user_id'].tolist()

# Randomly assign user_id to each post
posts['user_id'] = [random.choice(valid_users) for _ in range(len(posts))]

# Also assign tickers from users.csv to posts
user_tickers_map = dict(zip(users['user_id'], users['tickers']))
posts['tickers'] = posts['user_id'].map(user_tickers_map)

# Save new posts CSV
posts.to_csv("data/processed/posts_with_users.csv", index=False)
print(posts.head())
