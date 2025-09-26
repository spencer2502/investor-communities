import networkx as nx
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import os
import json
from sklearn.cluster import SpectralClustering

try:
    import community as community_louvain
except ImportError:
    os.system("pip install python-louvain")
    import community as community_louvain


class CommunityDetector:
    def __init__(self, graph_file="data/graphs/ticker_network.gexf"):
        """Initialize with graph file."""
        if not os.path.exists(graph_file):
            raise FileNotFoundError(f"❌ Graph file not found: {graph_file}")
        print(f"📊 Loading network from {graph_file}")
        self.G = nx.read_gexf(graph_file)
        self.ensure_undirected()
        self.results = {}
        print(f"✅ Loaded: {self.G.number_of_nodes():,} nodes, {self.G.number_of_edges():,} edges")

    def ensure_undirected(self):
        if self.G.is_directed():
            self.G = self.G.to_undirected()
            print("Converted to undirected graph")

    def preprocess_graph(self, min_degree=2, largest_component_only=True):
        """Preprocess graph by removing noise and keeping largest component."""
        print("🔧 Preprocessing...")
        original_nodes = self.G.number_of_nodes()

        low_degree_nodes = [n for n, d in self.G.degree() if d < min_degree]
        self.G.remove_nodes_from(low_degree_nodes)
        print(f"   Removed {len(low_degree_nodes)} low-degree nodes")

        if largest_component_only:
            comps = list(nx.connected_components(self.G))
            if len(comps) > 1:
                largest = max(comps, key=len)
                self.G = self.G.subgraph(largest).copy()
                print(f"   Kept largest component: {len(largest)} nodes")

        print(f"✅ Final graph: {self.G.number_of_nodes()} nodes ({original_nodes - self.G.number_of_nodes()} removed)")
        return self.G

    def detect_louvain_communities(self, resolution=1.0, random_state=42):
        """Run Louvain community detection."""
        print("🔍 Running Louvain...")
        partition = community_louvain.best_partition(self.G, resolution=resolution, random_state=random_state)
        nx.set_node_attributes(self.G, partition, 'community')

        modularity = community_louvain.modularity(partition, self.G)
        cluster_sizes = Counter(partition.values())
        self.results['louvain'] = {
            'partition': partition,
            'modularity': modularity,
            'cluster_sizes': cluster_sizes,
            'num_communities': len(cluster_sizes)
        }
        print(f"✅ Louvain: {len(cluster_sizes)} communities, modularity={modularity:.4f}")
        return partition, modularity, cluster_sizes

    def detect_spectral_communities(self, n_clusters=None, random_state=42):
        """Run Spectral Clustering."""
        if self.G.number_of_nodes() > 3000:
            print("⚠️ Skipping spectral (too large)")
            return None, None, None

        if n_clusters is None:
            if 'louvain' not in self.results:
                self.detect_louvain_communities()
            n_clusters = min(self.results['louvain']['num_communities'], 50)

        try:
            adj = nx.adjacency_matrix(self.G)
            spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=random_state)
            labels = spectral.fit_predict(adj)
        except Exception as e:
            print(f"❌ Spectral failed: {e}")
            return None, None, None

        nodes = list(self.G.nodes())
        partition = {nodes[i]: labels[i] for i in range(len(nodes))}
        nx.set_node_attributes(self.G, partition, 'spectral_community')

        cluster_sizes = Counter(partition.values())
        communities = [set([nodes[i] for i, l in enumerate(labels) if l == c]) for c in set(labels)]
        modularity = nx.algorithms.community.modularity(self.G, communities)

        self.results['spectral'] = {
            'partition': partition,
            'modularity': modularity,
            'cluster_sizes': cluster_sizes,
            'num_communities': len(cluster_sizes)
        }
        print(f"✅ Spectral: {len(cluster_sizes)} communities, modularity={modularity:.4f}")
        return partition, modularity, cluster_sizes

    def compare_algorithms(self):
        print("\n📊 Comparison")
        comp = {}
        for algo, r in self.results.items():
            comp[algo] = {
                "num_communities": r["num_communities"],
                "modularity": r["modularity"],
                "largest": max(r["cluster_sizes"].values()),
                "smallest": min(r["cluster_sizes"].values())
            }
        df = pd.DataFrame(comp).T
        print(df)
        return df

    def plot_community_analysis(self, algorithm="louvain", out_dir="outputs"):
        if algorithm not in self.results:
            print(f"❌ No results for {algorithm}")
            return

        os.makedirs(out_dir, exist_ok=True)
        results = self.results[algorithm]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Community Detection - {algorithm.upper()}", fontsize=16, fontweight="bold")

        # 1. Size distribution
        sizes = list(results['cluster_sizes'].values())
        axes[0, 0].hist(sizes, bins=min(30, len(sizes)), color="skyblue", edgecolor="black")
        axes[0, 0].set_title("Community Size Distribution")
        axes[0, 0].set_yscale("log")

        # 2. Top communities
        top = sorted(results['cluster_sizes'].items(), key=lambda x: x[1], reverse=True)[:20]
        ids, vals = zip(*top)
        y = np.arange(len(ids))
        colors = ["red" if i < 5 else "lightcoral" for i in range(len(ids))]
        axes[0, 1].barh(y, vals, color=colors)
        axes[0, 1].set_yticks(y)
        axes[0, 1].set_yticklabels([f"C{i}" for i in ids])
        axes[0, 1].invert_yaxis()
        axes[0, 1].set_title("Top 20 Communities")

        # 3. Avg degree vs size
        degrees = dict(self.G.degree())
        comm_avg = [np.mean([degrees[n] for n, c in results["partition"].items() if c == cid]) for cid in ids]
        axes[1, 0].scatter(vals, comm_avg, s=60, alpha=0.7)
        axes[1, 0].set_xscale("log")
        axes[1, 0].set_title("Size vs Avg Degree")

        # 4. Modularity
        if len(self.results) > 1:
            algos = list(self.results.keys())
            mods = [self.results[a]["modularity"] for a in algos]
            axes[1, 1].bar(algos, mods, color="lightgreen")
            axes[1, 1].set_title("Modularity Comparison")
        else:
            axes[1, 1].text(0.5, 0.5, f"{results['modularity']:.3f}", ha="center", va="center", fontsize=16)
            axes[1, 1].set_title("Modularity")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        file = os.path.join(out_dir, f"community_analysis_{algorithm}.png")
        plt.savefig(file, dpi=300)
        plt.show()
        print(f"📊 Plot saved: {file}")


def run_pipeline(graph_file="data/graphs/ticker_network.gexf", out_dir="outputs"):
    det = CommunityDetector(graph_file)
    det.preprocess_graph()
    det.detect_louvain_communities()
    det.detect_spectral_communities()
    det.compare_algorithms()
    det.plot_community_analysis("louvain", out_dir)
    return det


if __name__ == "__main__":
    run_pipeline()
