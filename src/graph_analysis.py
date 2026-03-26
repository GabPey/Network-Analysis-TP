import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display


def analyze_graph_df(G, name="Graph"):
    """
    Analyze an undirected graph and return results as pandas DataFrames.

    Required:
      1. Minimal, maximal, and average degrees
      2. Degree distribution
      3. Diameter, radius, and average path length
      4. Betweenness centrality
      5. Google PageRank
      6. Visualization handled separately

    Extra:
      7. Clustering coefficient

    Returns
    -------
    dict
        {
            "name": str,
            "degree_stats": DataFrame,
            "degree_distribution": DataFrame,
            "distance_metrics": DataFrame,
            "betweenness_centrality": DataFrame,
            "pagerank": DataFrame,
            "clustering_coefficient": DataFrame,
            "graph_info": DataFrame
        }
    """

    if G.number_of_nodes() == 0:
        raise ValueError("The graph is empty.")

    results = {"name": name}

    # -----------------------------
    # Basic graph info
    # -----------------------------
    is_connected = nx.is_connected(G)
    graph_info_df = pd.DataFrame({
        "metric": ["graph_name", "number_of_nodes", "number_of_edges", "is_connected"],
        "value": [name, G.number_of_nodes(), G.number_of_edges(), is_connected]
    })

    # -----------------------------
    # 1) Minimal, maximal, average degrees
    # -----------------------------
    degrees = dict(G.degree())
    degree_values = list(degrees.values())

    degree_stats_df = pd.DataFrame({
        "metric": ["minimal_degree", "maximal_degree", "average_degree"],
        "value": [
            min(degree_values),
            max(degree_values),
            sum(degree_values) / len(degree_values)
        ]
    })

    # -----------------------------
    # 2) Degree distribution
    # -----------------------------
    degree_distribution_df = (
        pd.Series(degree_values, name="degree")
        .value_counts()
        .sort_index()
        .reset_index()
    )
    degree_distribution_df.columns = ["degree", "count"]

    # -----------------------------
    # 3) Diameter, radius, average path length
    # -----------------------------
    if is_connected:
        H = G
        computed_on = "full graph"
    else:
        largest_cc_nodes = max(nx.connected_components(G), key=len)
        H = G.subgraph(largest_cc_nodes).copy()
        computed_on = "largest connected component"

    distance_metrics_df = pd.DataFrame({
        "metric": ["diameter", "radius", "average_path_length"],
        "value": [
            nx.diameter(H),
            nx.radius(H),
            nx.average_shortest_path_length(H)
        ],
        "computed_on": [computed_on, computed_on, computed_on]
    })

    # -----------------------------
    # 4) Betweenness centrality
    # -----------------------------
    betweenness = nx.betweenness_centrality(G)
    betweenness_df = pd.DataFrame({
        "node": list(betweenness.keys()),
        "betweenness_centrality": list(betweenness.values())
    }).sort_values("betweenness_centrality", ascending=False).reset_index(drop=True)

    # -----------------------------
    # 5) Google PageRank
    # -----------------------------
    pagerank = nx.pagerank(G)
    pagerank_df = pd.DataFrame({
        "node": list(pagerank.keys()),
        "pagerank": list(pagerank.values())
    }).sort_values("pagerank", ascending=False).reset_index(drop=True)

    # -----------------------------
    # 6) Extra: Clustering coefficient
    # -----------------------------
    clustering = nx.clustering(G)
    clustering_df = pd.DataFrame({
        "node": list(clustering.keys()),
        "clustering_coefficient": list(clustering.values())
    }).sort_values("clustering_coefficient", ascending=False).reset_index(drop=True)

    # Store results
    results["graph_info"] = graph_info_df
    results["degree_stats"] = degree_stats_df
    results["degree_distribution"] = degree_distribution_df
    results["distance_metrics"] = distance_metrics_df
    results["betweenness_centrality"] = betweenness_df
    results["pagerank"] = pagerank_df
    results["clustering_coefficient"] = clustering_df

    return results


def visualize_graph_analysis(G, results, name="Graph"):
    """
    Extended compact visualization with:
      1. Graph structure
      2. Degree distribution
      3. Betweenness centrality
      4. PageRank
      5. Clustering coefficient histogram
      6. Graph colored by clustering coefficient
    """

    degree_distribution_df = results["degree_distribution"]
    betweenness_df = results["betweenness_centrality"]
    pagerank_df = results["pagerank"]
    clustering_df = results["clustering_coefficient"]

    fig = plt.figure(figsize=(16, 14))

    # -----------------------------
    # 1) Graph structure
    # -----------------------------
    ax1 = fig.add_subplot(2, 3, 1)
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=600,
        font_size=9,
        ax=ax1
    )
    ax1.set_title(f"{name} - Graph")

    # -----------------------------
    # 2) Degree distribution
    # -----------------------------
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.bar(degree_distribution_df["degree"], degree_distribution_df["count"])
    ax2.set_title("Degree Distribution")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("Count")
    ax2.set_xticks(degree_distribution_df["degree"])
    ax2.grid(True, axis="y", linestyle="--", alpha=0.7)

    # -----------------------------
    # 3) Betweenness centrality
    # -----------------------------
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.bar(
        betweenness_df["node"].astype(str),
        betweenness_df["betweenness_centrality"]
    )
    ax3.set_title("Betweenness Centrality")
    ax3.set_xlabel("Node")
    ax3.set_ylabel("Betweenness")
    ax3.tick_params(axis="x", rotation=45)
    ax3.grid(True, axis="y", linestyle="--", alpha=0.7)

    # -----------------------------
    # 4) Google PageRank
    # -----------------------------
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.bar(
        pagerank_df["node"].astype(str),
        pagerank_df["pagerank"]
    )
    ax4.set_title("Google PageRank")
    ax4.set_xlabel("Node")
    ax4.set_ylabel("PageRank")
    ax4.tick_params(axis="x", rotation=45)
    ax4.grid(True, axis="y", linestyle="--", alpha=0.7)

    # -----------------------------
    # 5) Clustering coefficient histogram
    # -----------------------------
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(clustering_df["clustering_coefficient"], bins=10)
    ax5.set_title("Clustering Coefficient Histogram")
    ax5.set_xlabel("Clustering Coefficient")
    ax5.set_ylabel("Frequency")
    ax5.grid(True, axis="y", linestyle="--", alpha=0.7)

    # -----------------------------
    # 6) Graph colored by clustering coefficient
    # -----------------------------
    ax6 = fig.add_subplot(2, 3, 6)
    clustering_map = dict(
        zip(clustering_df["node"], clustering_df["clustering_coefficient"])
    )
    node_colors = [clustering_map[node] for node in G.nodes()]

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        node_size=600,
        font_size=9,
        cmap=plt.cm.viridis,
        ax=ax6
    )
    ax6.set_title("Graph Colored by Clustering Coefficient")
    ax6.grid(False)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    sm.set_array(node_colors)
    fig.colorbar(sm, ax=ax6, fraction=0.046, pad=0.04, label="Clustering Coefficient")

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt


def visualize_graph_analysis_large(G, results, name="Graph", top_k=15):
    """
    Clean visualization for larger graphs.

    Keeps:
      - distributions (histograms)
      - top-k node rankings

    Removes:
      - full graph drawing
      - dense node-by-node plots

    Uses ONLY existing results (no recomputation)
    """

    degree_distribution_df = results["degree_distribution"]
    betweenness_df = results["betweenness_centrality"]
    pagerank_df = results["pagerank"]
    clustering_df = results["clustering_coefficient"]

    fig = plt.figure(figsize=(14, 10))

    # -----------------------------
    # 1. Degree distribution
    # -----------------------------
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.bar(degree_distribution_df["degree"], degree_distribution_df["count"])
    ax1.set_title("Degree Distribution")
    ax1.set_xlabel("Degree")
    ax1.set_ylabel("Count")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.7)

    # -----------------------------
    # 2. Betweenness distribution
    # -----------------------------
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.hist(betweenness_df["betweenness_centrality"], bins=30)
    ax2.set_title("Betweenness Distribution")
    ax2.set_xlabel("Betweenness")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.7)

    # -----------------------------
    # 3. PageRank distribution
    # -----------------------------
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.hist(pagerank_df["pagerank"], bins=30)
    ax3.set_title("PageRank Distribution")
    ax3.set_xlabel("PageRank")
    ax3.set_ylabel("Frequency")
    ax3.grid(True, axis="y", linestyle="--", alpha=0.7)

    # -----------------------------
    # 4. Clustering distribution
    # -----------------------------
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.hist(clustering_df["clustering_coefficient"], bins=30)
    ax4.set_title("Clustering Coefficient Distribution")
    ax4.set_xlabel("Clustering")
    ax4.set_ylabel("Frequency")
    ax4.grid(True, axis="y", linestyle="--", alpha=0.7)

    # -----------------------------
    # 5. Top-k Betweenness
    # -----------------------------
    top_bet = betweenness_df.head(top_k)
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.bar(top_bet["node"].astype(str), top_bet["betweenness_centrality"])
    ax5.set_title(f"Top {top_k} Betweenness")
    ax5.set_xlabel("Node")
    ax5.set_ylabel("Betweenness")
    ax5.tick_params(axis="x", rotation=45)
    ax5.grid(True, axis="y", linestyle="--", alpha=0.7)

    # -----------------------------
    # 6. Top-k PageRank
    # -----------------------------
    top_pr = pagerank_df.head(top_k)
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.bar(top_pr["node"].astype(str), top_pr["pagerank"])
    ax6.set_title(f"Top {top_k} PageRank")
    ax6.set_xlabel("Node")
    ax6.set_ylabel("PageRank")
    ax6.tick_params(axis="x", rotation=45)
    ax6.grid(True, axis="y", linestyle="--", alpha=0.7)

    plt.suptitle(f"{name} - Large Graph Analysis", fontsize=16)
    plt.tight_layout()
    plt.show()