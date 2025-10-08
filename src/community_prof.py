"""
community_prof.py — Detect and Profile Investor Communities
Author: Aman | GPT-5 Optimized Version
"""

import os
import pandas as pd
import networkx as nx
from community import community_louvain
import time

# -------------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------------
GRAPH_PATH = "data/graphs/ticker_network_optimized.gexf"
OUTPUT_DIR = "data/communities"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NODE_CSV = os.path.join(OUTPUT_DIR, "node_communities.csv")
COMMUNITY_CSV = os.path.join(OUTPUT_DIR, "community_summary.csv")
COMMUNITY_GEXF = os.path.join(OUTPUT_DIR, "ticker_network_with_communities.gexf")

# -------------------------------------------------------------------------
# STEP 1: LOAD GRAPH
# -------------------------------------------------------------------------
print(f"> Loading graph from: {GRAPH_PATH}")
G = nx.read_gexf(GRAPH_PATH)
print(f"> Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# -------------------------------------------------------------------------
# STEP 2: COMMUNITY DETECTION
# -------------------------------------------------------------------------
print("> Leiden not available — falling back to Louvain.")
print("> Running Louvain algorithm (python-louvain)...")
start = time.time()
partition = community_louvain.best_partition(G, weight="weight", random_state=42)
end = time.time()

nx.set_node_attributes(G, partition, "community")
num_comms = len(set(partition.values()))
modularity = community_louvain.modularity(partition, G)
print(f"> Community detection finished. (modularity ≈ {modularity:.6f})")
print(f"> Detected {num_comms} communities in {end - start:.2f}s.")

# -------------------------------------------------------------------------
# STEP 3: COMMUNITY PROFILING
# -------------------------------------------------------------------------
print("> Profiling communities...")

# --- Betweenness (auto-switch to fast mode) ---
num_nodes = G.number_of_nodes()
if num_nodes > 1500:
    print(f"> Betweenness: Using approximate betweenness (k=300) on large graph ({num_nodes} nodes)...")
    bw = nx.betweenness_centrality(G, weight='weight', normalized=True, k=300, seed=42)
else:
    print(f"> Betweenness: Computing FULL betweenness on graph ({num_nodes} nodes).")
    bw = nx.betweenness_centrality(G, weight='weight', normalized=True)

# --- Degree features ---
deg = dict(G.degree(weight="weight"))
nx.set_node_attributes(G, deg, "weighted_degree")
nx.set_node_attributes(G, bw, "betweenness")

# -------------------------------------------------------------------------
# STEP 4: SAVE NODE-LEVEL CSV
# -------------------------------------------------------------------------
node_data = []
for n, d in G.nodes(data=True):
    node_data.append({
        "node": n,
        "community": d.get("community"),
        "weighted_degree": d.get("weighted_degree", 0),
        "betweenness": d.get("betweenness", 0)
    })
node_df = pd.DataFrame(node_data)
node_df.to_csv(NODE_CSV, index=False)
print(f"> Saved node CSV → {NODE_CSV}")

# -------------------------------------------------------------------------
# STEP 5: COMMUNITY-LEVEL SUMMARY
# -------------------------------------------------------------------------
comm_summary = []
for comm_id in sorted(set(partition.values())):
    members = [n for n, c in partition.items() if c == comm_id]
    subgraph = G.subgraph(members)
    avg_degree = sum(dict(subgraph.degree(weight="weight")).values()) / len(subgraph)
    avg_bw = sum(bw[n] for n in members) / len(members)
    comm_summary.append({
        "community": comm_id,
        "size": len(members),
        "avg_weighted_degree": round(avg_degree, 3),
        "avg_betweenness": round(avg_bw, 6),
        "density": nx.density(subgraph)
    })

comm_df = pd.DataFrame(comm_summary)
comm_df = comm_df.sort_values("size", ascending=False)
comm_df.to_csv(COMMUNITY_CSV, index=False)
print(f"> Saved community CSV → {COMMUNITY_CSV}")

# -------------------------------------------------------------------------
# STEP 6: SAVE GRAPH WITH COMMUNITY ATTRIBUTES
# -------------------------------------------------------------------------
nx.write_gexf(G, COMMUNITY_GEXF)
print(f"> Saved GEXF with community attributes → {COMMUNITY_GEXF}")

# -------------------------------------------------------------------------
# DONE
# -------------------------------------------------------------------------
print(f"✅ All done in {time.time() - start:.2f}s. ({num_comms} communities, modularity={modularity:.4f})")
