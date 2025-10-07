#!/usr/bin/env python3
"""
community_profiling.py

Community detection + profiling module optimized for large social graphs.

Usage:
    python src/community_profiling.py --input data/graphs/ticker_network_optimized.gexf

Key features:
- Prefer Leiden (if igraph+leidenalg installed), fallback to python-louvain.
- Approximate betweenness centrality for large graphs (NetworkX k-sampling).
- Ability to skip betweenness for fast runs.
- Per-community profiling (top tickers, top users, bridgers).
"""

from collections import Counter, defaultdict
import argparse
import os
import time
import math

import networkx as nx
import pandas as pd
import numpy as np

# Try to import Leiden/igraph first (preferred)
HAS_LEIDEN = False
try:
    import igraph as ig  # type: ignore
    import leidenalg  # type: ignore
    HAS_LEIDEN = True
except Exception:
    HAS_LEIDEN = False

# Fallback: python-louvain
HAS_LOUVAIN = False
try:
    import community as community_louvain  # pip install python-louvain
    HAS_LOUVAIN = True
except Exception:
    HAS_LOUVAIN = False


# -------------------------
# Utilities
# -------------------------
def load_graph(path: str) -> nx.Graph:
    print(f"> Loading graph from: {path}")
    G = nx.read_gexf(path)
    if G.is_directed():
        print("> Input graph is directed — converting to undirected (symmetrize).")
        G = G.to_undirected()
    print(f"> Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# -------------------------
# Community Detection
# -------------------------
def detect_communities_leiden(G: nx.Graph, resolution: float = 1.0, random_state: int = 42):
    """Run Leiden via igraph. Returns (node_to_comm_dict, modularity_float)."""
    if not HAS_LEIDEN:
        raise RuntimeError("Leiden/igraph not available in this environment.")
    print("> Converting NetworkX -> igraph (Leiden)...")
    nodes = list(G.nodes())
    mapping = {n: i for i, n in enumerate(nodes)}
    edges = [(mapping[u], mapping[v]) for u, v in G.edges()]
    g_ig = ig.Graph(n=len(nodes), edges=edges, directed=False)

    # transfer weights if present
    if nx.get_edge_attributes(G, 'weight'):
        weights = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
        # igraph expects len == number of edges
        if len(weights) == g_ig.ecount():
            g_ig.es['weight'] = weights

    print("> Running Leiden algorithm...")
    partition = leidenalg.find_partition(
        g_ig,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        weights='weight' if 'weight' in g_ig.edge_attributes() else None,
        seed=random_state
    )

    membership = partition.membership
    node_to_comm = {nodes[i]: int(membership[i]) for i in range(len(nodes))}
    modularity = float(partition.quality()) if hasattr(partition, 'quality') else (partition.modularity if hasattr(partition, 'modularity') else None)
    return node_to_comm, modularity


def detect_communities_louvain(G: nx.Graph, resolution: float = 1.0, random_state: int = 42):
    """Run python-louvain. Returns (node_to_comm_dict, modularity_float)."""
    if not HAS_LOUVAIN:
        raise RuntimeError("python-louvain not available in this environment.")
    # python-louvain best_partition currently doesn't accept resolution consistently across versions;
    # we call best_partition with weight and fallback to default behavior.
    print("> Running Louvain algorithm (python-louvain)...")
    partition = community_louvain.best_partition(G, weight='weight')
    modularity = community_louvain.modularity(partition, G, weight='weight')
    # ensure int communities
    partition = {n: int(c) for n, c in partition.items()}
    return partition, float(modularity)


# -------------------------
# Profiling
# -------------------------
def profile_communities(G: nx.Graph, node_to_comm: dict,
                        top_k_tickers: int = 10, top_k_users: int = 10,
                        skip_betweenness: bool = False, approx_k: int = 300,
                        max_full_betweenness_nodes: int = 3000):
    """
    Given G and node->community mapping, return (node_df, community_df, G_with_attrs).
    - skip_betweenness: skip computing betweenness entirely
    - approx_k: when approximating betweenness, sample k nodes
    - max_full_betweenness_nodes: compute exact betweenness only if graph <= this many nodes
    """
    print("> Profiling communities...")
    nx.set_node_attributes(G, node_to_comm, name='community_id')

    # node-level metrics
    deg = dict(G.degree())
    wdeg = dict(G.degree(weight='weight')) if nx.get_edge_attributes(G, 'weight') else deg

    node_rows = []
    comm_to_tickers = defaultdict(list)
    comm_to_users = defaultdict(list)

    for n in G.nodes():
        comm = node_to_comm.get(n, -1)
        attrs = G.nodes[n]
        num_posts = int(attrs.get('num_posts', 0)) if attrs.get('num_posts') is not None else 0
        num_tickers = int(attrs.get('num_tickers', 0)) if attrs.get('num_tickers') is not None else 0
        tickers_str = attrs.get('tickers', '') or ''
        tickers_list = [t for t in tickers_str.split(',') if t] if tickers_str else []
        node_rows.append({
            'user_id': n,
            'community_id': comm,
            'degree': int(deg.get(n, 0)),
            'weighted_degree': float(wdeg.get(n, 0)),
            'num_posts': num_posts,
            'num_tickers': num_tickers,
            'tickers_preview': ','.join(tickers_list[:20])
        })
        comm_to_tickers[comm].extend(tickers_list)
        comm_to_users[comm].append(n)

    node_df = pd.DataFrame(node_rows)

    # Decide betweenness strategy
    num_nodes = G.number_of_nodes()
    compute_full_bw = (not skip_betweenness) and (num_nodes <= max_full_betweenness_nodes)
    compute_approx_bw = (not skip_betweenness) and (num_nodes > max_full_betweenness_nodes)

    if skip_betweenness:
        print("> Betweenness: SKIPPED by flag.")
    elif compute_full_bw:
        print(f"> Betweenness: Computing FULL betweenness on graph ({num_nodes} nodes).")
    else:
        print(f"> Betweenness: Using APPROXIMATE betweenness with k={approx_k} (graph has {num_nodes} nodes).")

    # Compute global betweenness if needed (approx/full). For large graphs we prefer approximate global bw only if necessary.
    global_bw = None
    if not skip_betweenness:
        try:
            if compute_full_bw:
                global_bw = nx.betweenness_centrality(G, weight='weight', normalized=True)
            else:
                # approximate global betweenness using k samples
                k = min(approx_k, max(1, int(math.sqrt(num_nodes))))  # bound k to something reasonable; default uses approx_k cap
                k = approx_k if approx_k < num_nodes else max(1, num_nodes // 10)
                print(f"> Approximate global betweenness with k={k}")
                global_bw = nx.betweenness_centrality(G, k=k, weight='weight', normalized=True, seed=42)
        except Exception as e:
            print(f"> Warning: global betweenness computation failed or timed out: {e}")
            global_bw = {n: 0.0 for n in G.nodes()}

    # Now compute per-community summaries
    comm_rows = []
    for comm, users in comm_to_users.items():
        size = len(users)
        ticker_counts = Counter([t for t in comm_to_tickers[comm] if t])
        top_tickers = ticker_counts.most_common(top_k_tickers)

        # degree-based top users
        sub_degs = [(u, int(G.degree(u))) for u in users]
        top_users = sorted(sub_degs, key=lambda x: x[1], reverse=True)[:top_k_users]

        # induced subgraph stats
        subG = G.subgraph(users)
        avg_deg = float(np.mean([d for _, d in subG.degree()])) if subG.number_of_nodes() > 0 else 0.0
        density = nx.density(subG) if subG.number_of_nodes() > 1 else 0.0

        # bridgers: prefer community-level betweenness computed on subgraph if small, else use global_bw lookup
        bridgers = {}
        try:
            if not skip_betweenness:
                if subG.number_of_nodes() <= 2000:
                    # compute subgraph betweenness exactly or approx depending on size
                    if subG.number_of_nodes() <= 1000:
                        bw_sub = nx.betweenness_centrality(subG, weight='weight', normalized=True)
                    else:
                        k_sub = min(200, max(10, int(math.sqrt(subG.number_of_nodes()))))
                        bw_sub = nx.betweenness_centrality(subG, k=k_sub, weight='weight', normalized=True, seed=42)
                    for u in users:
                        bridgers[u] = bw_sub.get(u, 0.0)
                else:
                    # fallback: use global betweenness if available
                    if global_bw:
                        for u in users:
                            bridgers[u] = global_bw.get(u, 0.0)
                    else:
                        for u in users:
                            bridgers[u] = 0.0
            else:
                for u in users:
                    bridgers[u] = 0.0
        except Exception as e:
            print(f"> Warning: per-community betweenness error for community {comm}: {e}")
            for u in users:
                bridgers[u] = global_bw.get(u, 0.0) if global_bw else 0.0

        sorted_bridgers = sorted(bridgers.items(), key=lambda x: x[1], reverse=True)[:min(10, len(bridgers))]

        comm_rows.append({
            'community_id': int(comm),
            'size': int(size),
            'avg_degree': float(avg_deg),
            'density': float(density),
            'top_tickers': ';'.join([f"{t}:{c}" for t, c in top_tickers]),
            'top_users': ';'.join([f"{u}:{int(d)}" for u, d in top_users]),
            'top_bridgers': ';'.join([f"{u}:{bw:.6f}" for u, bw in sorted_bridgers]),
        })

    community_df = pd.DataFrame(comm_rows).sort_values('size', ascending=False).reset_index(drop=True)
    return node_df, community_df, G


# -------------------------
# Save Helpers
# -------------------------
def save_outputs(node_df: pd.DataFrame, community_df: pd.DataFrame, G: nx.Graph, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    node_path = os.path.join(out_dir, "node_communities.csv")
    comm_path = os.path.join(out_dir, "community_summary.csv")
    gexf_path = os.path.join(out_dir, "ticker_network_with_communities.gexf")

    node_df.to_csv(node_path, index=False)
    community_df.to_csv(comm_path, index=False)
    nx.write_gexf(G, gexf_path)
    print(f"> Saved node CSV -> {node_path}")
    print(f"> Saved community CSV -> {comm_path}")
    print(f"> Saved GEXF with community attributes -> {gexf_path}")


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Community detection + profiling for investor communities.")
    parser.add_argument("--input", type=str, default="data/graphs/ticker_network_optimized.gexf", help="Input GEXF graph path")
    parser.add_argument("--out_dir", type=str, default="data/communities", help="Output folder for CSVs and GEXF")
    parser.add_argument("--resolution", type=float, default=1.0, help="Resolution parameter for Leiden/Louvain")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--top_k_tickers", type=int, default=10)
    parser.add_argument("--top_k_users", type=int, default=10)
    parser.add_argument("--betweenness_sample_k", type=int, default=300, help="k for approximate betweenness (NetworkX)")
    parser.add_argument("--max_full_betweenness_nodes", type=int, default=3000, help="Compute full betweenness only if nodes <= this")
    parser.add_argument("--skip_betweenness", action="store_true", help="Skip betweenness centrality computations entirely")
    parser.add_argument("--prefer_leiden", action="store_true", help="Prefer Leiden if available (default behavior is Leiden when installed)")
    args = parser.parse_args()

    start = time.time()
    G = load_graph(args.input)

    # Decide algorithm: prefer Leiden if present & requested or available
    use_leiden = HAS_LEIDEN and args.prefer_leiden or (HAS_LEIDEN and not HAS_LOUVAIN)
    if use_leiden:
        print("> Detected leiden/igraph available — using Leiden (recommended).")
        node_to_comm, modularity = detect_communities_leiden(G, resolution=args.resolution, random_state=args.random_state)
    elif HAS_LOUVAIN:
        print("> Leiden not available — falling back to Louvain.")
        node_to_comm, modularity = detect_communities_louvain(G, resolution=args.resolution, random_state=args.random_state)
    else:
        raise RuntimeError("No community detection libraries found. Install 'leidenalg'+ 'python-igraph' or 'python-louvain'.")

    print(f"> Community detection finished. (modularity ≈ {modularity})")

    if node_to_comm is None or not isinstance(node_to_comm, dict):
        raise ValueError("Community detection failed — node_to_comm is None or invalid.")

    node_df, community_df, G = profile_communities(
        G,
        node_to_comm,
        top_k_tickers=args.top_k_tickers,
        top_k_users=args.top_k_users,
        skip_betweenness=args.skip_betweenness,
        approx_k=args.betweenness_sample_k,
        max_full_betweenness_nodes=args.max_full_betweenness_nodes
    )

    community_df['overall_modularity'] = float(modularity) if modularity is not None else None

    save_outputs(node_df, community_df, G, args.out_dir)

    end = time.time()
    print(f"> All done in {end - start:.1f}s")


if __name__ == "__main__":
    main()
