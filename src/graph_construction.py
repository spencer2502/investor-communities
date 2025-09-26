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
    try:
        print("📁 Loading data...")
        # Load posts with specific columns to save memory
        posts_cols = ['platform', 'post_id', 'user_id', 'created_at', 'text', 'tickers']
        posts = pd.read_csv(posts_file, 
                           usecols=posts_cols,
                           dtype={"user_id": str, "post_id": str},
                           low_memory=False)
        
        users = pd.read_csv(users_file, dtype={"user_id": str})
        
        print(f"📊 Loaded {len(posts):,} posts and {len(users):,} users")
        print(f"💾 Posts memory usage: {posts.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        return posts, users
    except FileNotFoundError as e:
        print(f"❌ Error loading data: {e}")
        return None, None


def safe_parse_tickers(value):
    """
    Safely parse the 'tickers' column from your format: ["VZ"] or ["PG"],["VZ"]
    """
    if pd.isna(value) or value == "" or value == "[]":
        return []
    
    try:
        # Handle the format in your data
        tickers = ast.literal_eval(value)
        if isinstance(tickers, list):
            # Flatten nested lists and clean
            result = []
            for item in tickers:
                if isinstance(item, list):
                    result.extend([t.strip().upper() for t in item if t])
                elif isinstance(item, str):
                    result.append(item.strip().upper())
            return result
        return []
    except (ValueError, SyntaxError, TypeError):
        return []


def analyze_dataset(posts: pd.DataFrame):
    """
    Analyze your dataset to understand the scale and suggest optimizations.
    """
    print("\n🔍 DATASET ANALYSIS")
    print("=" * 50)
    
    # Basic stats
    print(f"Total posts: {len(posts):,}")
    print(f"Unique users: {posts['user_id'].nunique():,}")
    print(f"Date range: {posts['created_at'].min()} to {posts['created_at'].max()}")
    
    # Parse tickers for analysis
    posts['parsed_tickers'] = posts['tickers'].apply(safe_parse_tickers)
    posts_with_tickers = posts[posts['parsed_tickers'].map(len) > 0]
    print(f"Posts with tickers: {len(posts_with_tickers):,} ({len(posts_with_tickers)/len(posts)*100:.1f}%)")
    
    # User activity distribution
    user_post_counts = posts.groupby('user_id').size()
    print(f"\nUser Activity:")
    print(f"  Top 10% users post: {user_post_counts.quantile(0.9):.0f}+ posts")
    print(f"  Top 1% users post: {user_post_counts.quantile(0.99):.0f}+ posts")
    print(f"  Most active user: {user_post_counts.max():,} posts")
    
    # Ticker analysis
    all_tickers = []
    for ticker_list in posts_with_tickers['parsed_tickers']:
        all_tickers.extend(ticker_list)
    
    ticker_counts = Counter(all_tickers)
    print(f"\nTicker Analysis:")
    print(f"  Unique tickers: {len(ticker_counts):,}")
    print(f"  Total ticker mentions: {len(all_tickers):,}")
    print(f"  Top 10 tickers: {dict(ticker_counts.most_common(10))}")
    
    # Estimate graph complexity
    potential_edges = 0
    ticker_user_counts = defaultdict(set)
    
    for _, row in posts_with_tickers.iterrows():
        user_id = str(row['user_id']).lower()
        for ticker in row['parsed_tickers']:
            ticker_user_counts[ticker].add(user_id)
    
    for ticker, users in ticker_user_counts.items():
        n_users = len(users)
        if n_users > 1:
            potential_edges += n_users * (n_users - 1) // 2
    
    print(f"\nGraph Complexity Estimate:")
    print(f"  Potential edges (all combinations): {potential_edges:,}")
    
    # Memory cleanup
    del posts['parsed_tickers']
    gc.collect()
    
    return ticker_counts, user_post_counts


def save_graph(G, out_dir="data/graphs", filename="graph.gexf"):
    """
    Saves the NetworkX graph with metadata.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, filename)
    
    # Add graph metadata
    G.graph['created_by'] = 'Social Network Analyzer'
    G.graph['node_count'] = G.number_of_nodes()
    G.graph['edge_count'] = G.number_of_edges()
    
    nx.write_gexf(G, out_file)
    print(f"💾 Graph saved to {out_file}")
    print(f"   Nodes: {G.number_of_nodes():,}, Edges: {G.number_of_edges():,}")


# -----------------------------
# Optimized Graph Builder
# -----------------------------

def build_smart_ticker_graph(posts: pd.DataFrame, 
                           top_users=3000,           # Top N most active users
                           min_ticker_mentions=5,    # Min mentions for ticker to be included
                           max_users_per_ticker=500, # Max users per ticker (prevents explosion)
                           min_common_tickers=2,     # Min shared tickers for edge
                           sample_large_tickers=True): # Sample users for very popular tickers
    """
    Build an optimized ticker co-mention graph for large datasets.
    """
    print("\n🚀 BUILDING SMART TICKER GRAPH")
    print("=" * 50)
    
    # Step 1: Filter to most active users
    print("👥 Filtering to most active users...")
    user_activity = posts.groupby('user_id').size().sort_values(ascending=False)
    top_user_ids = set(user_activity.head(top_users).index)
    
    filtered_posts = posts[posts['user_id'].isin(top_user_ids)].copy()
    print(f"   Kept {len(filtered_posts):,} posts from top {top_users} users")
    
    # Step 2: Parse and filter tickers
    print("📈 Processing tickers...")
    filtered_posts['author_lc'] = filtered_posts['user_id'].str.lower()
    filtered_posts['parsed_tickers'] = filtered_posts['tickers'].apply(safe_parse_tickers)
    
    # Only keep posts with tickers
    filtered_posts = filtered_posts[filtered_posts['parsed_tickers'].map(len) > 0]
    
    # Count ticker mentions
    ticker_counter = Counter()
    for ticker_list in tqdm(filtered_posts['parsed_tickers'], desc="Counting tickers"):
        ticker_counter.update(ticker_list)
    
    # Keep only frequently mentioned tickers
    popular_tickers = {ticker for ticker, count in ticker_counter.items() 
                      if count >= min_ticker_mentions}
    
    print(f"   Popular tickers (≥{min_ticker_mentions} mentions): {len(popular_tickers):,}")
    print(f"   Top 10: {dict(ticker_counter.most_common(10))}")
    
    # Filter tickers in posts
    def filter_popular_tickers(ticker_list):
        return [t for t in ticker_list if t in popular_tickers]
    
    filtered_posts['filtered_tickers'] = filtered_posts['parsed_tickers'].apply(filter_popular_tickers)
    filtered_posts = filtered_posts[filtered_posts['filtered_tickers'].map(len) > 0]
    
    print(f"   Final posts after filtering: {len(filtered_posts):,}")
    
    # Step 3: Build user-ticker mappings with limits
    print("🔗 Building user-ticker relationships...")
    ticker_users = defaultdict(set)
    user_tickers = defaultdict(set)
    
    for _, row in tqdm(filtered_posts.iterrows(), total=len(filtered_posts), desc="Processing posts"):
        user_id = row['author_lc']
        for ticker in row['filtered_tickers']:
            # Limit users per ticker to prevent combinatorial explosion
            if len(ticker_users[ticker]) < max_users_per_ticker:
                ticker_users[ticker].add(user_id)
                user_tickers[user_id].add(ticker)
    
    # Step 4: Build edges efficiently
    print("⚡ Building edges...")
    edge_weights = defaultdict(int)
    
    for ticker, users in tqdm(ticker_users.items(), desc="Processing tickers"):
        if len(users) < 2:
            continue
            
        users_list = list(users)
        
        # Sample users for very popular tickers to manage complexity
        if sample_large_tickers and len(users_list) > 200:
            users_list = np.random.choice(users_list, 200, replace=False).tolist()
            
        # Generate all pairs
        for u1, u2 in combinations(sorted(users_list), 2):
            edge_weights[(u1, u2)] += 1
    
    # Step 5: Create NetworkX graph
    print("🏗️ Creating NetworkX graph...")
    G = nx.Graph()
    
    edges_added = 0
    for (u1, u2), weight in tqdm(edge_weights.items(), desc="Adding edges"):
        if weight >= min_common_tickers:
            G.add_edge(u1, u2, weight=weight)
            edges_added += 1
    
    # Step 6: Add node attributes
    print("📝 Adding node attributes...")
    original_user_activity = posts.groupby('user_id').size().to_dict()
    
    for user_id in tqdm(G.nodes(), desc="Adding node attributes"):
        # Convert back to original case for lookup
        original_user_id = user_id  # May need adjustment based on your data
        
        # Add attributes
        G.nodes[user_id]['num_posts'] = original_user_activity.get(original_user_id, 0)
        G.nodes[user_id]['num_tickers'] = len(user_tickers[user_id])
        G.nodes[user_id]['tickers'] = ','.join(sorted(list(user_tickers[user_id])[:20]))  # Limit length
        
        # Calculate user's ticker diversity (unique tickers / total posts)
        user_posts = len(filtered_posts[filtered_posts['author_lc'] == user_id])
        G.nodes[user_id]['ticker_diversity'] = len(user_tickers[user_id]) / max(user_posts, 1)
    
    print(f"\n✅ TICKER GRAPH COMPLETED")
    print(f"   Nodes: {G.number_of_nodes():,}")
    print(f"   Edges: {G.number_of_edges():,}")
    print(f"   Density: {nx.density(G):.6f}")
    
    # Graph statistics
    if G.number_of_nodes() > 0:
        degrees = dict(G.degree())
        avg_degree = sum(degrees.values()) / len(degrees)
        max_degree = max(degrees.values())
        print(f"   Average degree: {avg_degree:.1f}")
        print(f"   Max degree: {max_degree}")
        
        # Find connected components
        components = list(nx.connected_components(G))
        print(f"   Connected components: {len(components)}")
        if components:
            largest_component_size = len(max(components, key=len))
            print(f"   Largest component: {largest_component_size:,} nodes")
    
    return G


def build_quick_reply_graph(posts: pd.DataFrame, sample_size=50000):
    """
    Quick reply graph builder with sampling.
    """
    if 'in_reply_to_user_id' not in posts.columns:
        print("⚠️ No reply data found in posts")
        return nx.DiGraph()
    
    print(f"\n💬 BUILDING REPLY GRAPH")
    print("=" * 30)
    
    # Sample if too large
    reply_posts = posts.dropna(subset=['user_id', 'in_reply_to_user_id'])
    
    if len(reply_posts) > sample_size:
        reply_posts = reply_posts.sample(n=sample_size, random_state=42)
        print(f"📊 Sampled {sample_size:,} reply posts")
    
    G = nx.DiGraph()
    
    for _, row in tqdm(reply_posts.iterrows(), total=len(reply_posts), desc="Processing replies"):
        u1 = str(row['user_id']).lower()
        u2 = str(row['in_reply_to_user_id']).lower()
        
        if u1 != u2:  # No self-loops
            if G.has_edge(u1, u2):
                G[u1][u2]['weight'] += 1
            else:
                G.add_edge(u1, u2, weight=1)
    
    print(f"✅ Reply Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


# -----------------------------
# Main Execution
# -----------------------------

if __name__ == "__main__":
    posts_file = "data/processed/posts.csv"
    users_file = "data/processed/users.csv"
    
    # Load data
    posts, users = load_data(posts_file, users_file)
    
    if posts is not None:
        # Analyze dataset
        ticker_counts, user_activity = analyze_dataset(posts)
        
        # Build optimized ticker graph
        G_ticker = build_smart_ticker_graph(
            posts,
            top_users=3000,              # Adjust based on your needs
            min_ticker_mentions=10,      # Only tickers mentioned 10+ times
            max_users_per_ticker=300,    # Limit combinations per ticker
            min_common_tickers=3,        # Require 3+ shared tickers for edge
            sample_large_tickers=True    # Sample very popular tickers
        )
        
        if G_ticker.number_of_edges() > 0:
            save_graph(G_ticker, filename="ticker_network.gexf")
            
            # Save additional formats
            if G_ticker.number_of_nodes() < 10000:  # Only for manageable sizes
                try:
                    # Save as GraphML (better for attributes)
                    nx.write_graphml(G_ticker, "data/graphs/ticker_network.graphml")
                    print("💾 Also saved as GraphML format")
                except:
                    pass
        
        # Build reply graph if available
        G_reply = build_quick_reply_graph(posts, sample_size=30000)
        if G_reply.number_of_edges() > 0:
            save_graph(G_reply, filename="reply_network.gexf")
        
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print(f"📁 Check 'data/graphs/' for output files")
        
        # Memory cleanup
        del posts, users
        gc.collect()