import pandas as pd
import networkx as nx
from itertools import combinations
from collections import defaultdict, Counter
import ast
import os
import numpy as np
from tqdm import tqdm
import gc

# -----------------------------
# Utility Functions
# -----------------------------

def load_data(posts_file: str, users_file: str):
    """
    Loads posts and users dataframes with memory optimization.
    """
    print("📁 Loading data...")
    posts_cols = ['platform', 'post_id', 'user_id', 'created_at', 'text', 'tickers']
    posts = pd.read_csv(
        posts_file, usecols=posts_cols,
        dtype={"user_id": str, "post_id": str}, low_memory=False
    )
    users = pd.read_csv(users_file, dtype={"user_id": str})

    print(f"📊 Loaded {len(posts):,} posts and {len(users):,} users")
    print(f"💾 Posts memory usage: {posts.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    return posts, users


def safe_parse_tickers(value):
    """
    Safely parse the 'tickers' column.
    """
    if pd.isna(value) or value == "" or value == "[]":
        return []
    try:
        tickers = ast.literal_eval(value)
        if isinstance(tickers, list):
            result = []
            for item in tickers:
                if isinstance(item, list):
                    result.extend([t.strip().upper() for t in item if t])
                elif isinstance(item, str):
                    result.append(item.strip().upper())
            return result
        return []
    except (ValueError, SyntaxError, TypeError):
        # Handle pipe-separated fallback
        if "|" in str(value):
            return [t.strip().upper() for t in str(value).split("|") if t]
        return []


# -----------------------------
# Main Graph Builder
# -----------------------------

def build_smart_ticker_graph(
    posts: pd.DataFrame,
    top_users=3000,
    min_ticker_mentions=5,
    max_users_per_ticker=100,    # ↓ reduced to cap global tickers
    min_common_tickers=2,
    universal_ticker_percentile=0.98,  # remove top 2% global tickers
    sample_large_tickers=True,
):
    """
    Build an optimized user co-mention graph while filtering 'universal' tickers.
    """
    print("\n🚀 BUILDING SMART TICKER GRAPH")
    print("=" * 50)

    # Step 1: Filter most active users
    user_activity = posts.groupby('user_id').size().sort_values(ascending=False)
    top_user_ids = set(user_activity.head(top_users).index)
    filtered_posts = posts[posts['user_id'].isin(top_user_ids)].copy()
    print(f"👥 Kept {len(filtered_posts):,} posts from top {top_users} users")

    # Step 2: Parse tickers
    filtered_posts['author_lc'] = filtered_posts['user_id'].str.lower()
    filtered_posts['parsed_tickers'] = filtered_posts['tickers'].apply(safe_parse_tickers)
    filtered_posts = filtered_posts[filtered_posts['parsed_tickers'].map(len) > 0]

    # Step 3: Count ticker mentions
    ticker_counter = Counter()
    for ticker_list in tqdm(filtered_posts['parsed_tickers'], desc="Counting tickers"):
        ticker_counter.update(ticker_list)

    # Filter frequently mentioned tickers
    popular_tickers = {t for t, c in ticker_counter.items() if c >= min_ticker_mentions}

    # Step 4: Remove universal (too-common) tickers
    counts = np.array(list(ticker_counter.values()))
    cutoff = np.percentile(counts, universal_ticker_percentile * 100)
    universal_tickers = {t for t, c in ticker_counter.items() if c >= cutoff}
    print(f"🚫 Removed {len(universal_tickers)} 'universal' tickers (>{cutoff:.0f} mentions)")

    # Apply ticker filters
    def filter_tickers(tickers):
        return [t for t in tickers if t in popular_tickers and t not in universal_tickers]

    filtered_posts['filtered_tickers'] = filtered_posts['parsed_tickers'].apply(filter_tickers)
    filtered_posts = filtered_posts[filtered_posts['filtered_tickers'].map(len) > 0]
    print(f"✅ Final posts after filtering: {len(filtered_posts):,}")

    # Step 5: Build user-ticker mappings
    ticker_users = defaultdict(set)
    user_tickers = defaultdict(set)
    for _, row in tqdm(filtered_posts.iterrows(), total=len(filtered_posts), desc="Building relationships"):
        user_id = row['author_lc']
        for ticker in row['filtered_tickers']:
            if len(ticker_users[ticker]) < max_users_per_ticker:
                ticker_users[ticker].add(user_id)
                user_tickers[user_id].add(ticker)

    # Step 6: Build edges efficiently
    edge_weights = defaultdict(int)
    for ticker, users in tqdm(ticker_users.items(), desc="Building edges"):
        if len(users) < 2:
            continue
        users_list = list(users)
        if sample_large_tickers and len(users_list) > 200:
            users_list = np.random.choice(users_list, 200, replace=False).tolist()
        for u1, u2 in combinations(sorted(users_list), 2):
            edge_weights[(u1, u2)] += 1

    # Step 7: Create graph
    G = nx.Graph()
    for (u1, u2), weight in edge_weights.items():
        if weight >= min_common_tickers:
            G.add_edge(u1, u2, weight=weight)

    # Step 8: Add node attributes
    print("🧩 Adding node attributes...")
    original_user_activity = posts.groupby('user_id').size().to_dict()
    for user_id in G.nodes():
        G.nodes[user_id]['num_posts'] = original_user_activity.get(user_id, 0)
        G.nodes[user_id]['num_tickers'] = len(user_tickers[user_id])
        G.nodes[user_id]['tickers'] = ','.join(sorted(list(user_tickers[user_id])[:20]))
        user_posts = len(filtered_posts[filtered_posts['author_lc'] == user_id])
        G.nodes[user_id]['ticker_diversity'] = len(user_tickers[user_id]) / max(user_posts, 1)

    # Step 9: Print summary
    print("\n✅ TICKER GRAPH COMPLETED")
    print(f"   Nodes: {G.number_of_nodes():,}")
    print(f"   Edges: {G.number_of_edges():,}")
    print(f"   Density: {nx.density(G):.6f}")
    if G.number_of_nodes() > 0:
        degrees = dict(G.degree())
        print(f"   Avg degree: {sum(degrees.values()) / len(degrees):.1f}")
        print(f"   Max degree: {max(degrees.values())}")

    return G


def save_graph(G, out_dir="data/graphs", filename="ticker_network_optimized.gexf"):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    nx.write_gexf(G, out_path)
    print(f"💾 Graph saved to {out_path} (nodes={G.number_of_nodes():,}, edges={G.number_of_edges():,})")


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    posts_file = "data/processed/posts.csv"
    users_file = "data/processed/users.csv"

    posts, users = load_data(posts_file, users_file)
    if posts is not None:
        G = build_smart_ticker_graph(
            posts,
            top_users=3000,
            min_ticker_mentions=10,
            max_users_per_ticker=100,
            min_common_tickers=2,
            universal_ticker_percentile=0.98,
        )
        if G.number_of_edges() > 0:
            save_graph(G)
        else:
            print("⚠️ No edges found, check ticker thresholds.")
        print("\n🎉 DONE")
