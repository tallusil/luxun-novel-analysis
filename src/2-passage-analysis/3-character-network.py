import os
import json
import jieba
import re
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from pypinyin import pinyin, Style

script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(script_dir, "..", "..", "data", "combined")
output_dir = os.path.join(script_dir, "..", "..", "output", "passage_analysis")
font_dir = os.path.join(script_dir, "..", "..", "font")
network_dir = os.path.join(output_dir, "character_network")
character_analysis_path = os.path.join(output_dir, "character_analysis.json")

os.makedirs(network_dir, exist_ok=True)

# plt.rcParams["font.sans-serif"] = "SimHei"


def extract_character_cooccurrences(text, characters):
    sentences = re.split("[。！？]", text)
    cooccurrences = defaultdict(int)

    for sentence in sentences:
        sentence_chars = []
        for char in characters:
            if char in sentence:
                sentence_chars.append(char)

        for i in range(len(sentence_chars)):
            for j in range(i + 1, len(sentence_chars)):
                pair = tuple(sorted([sentence_chars[i], sentence_chars[j]]))
                cooccurrences[pair] += 1

    return cooccurrences


def build_network(cooccurrences, min_weight=2):
    G = nx.Graph()

    for (char1, char2), weight in cooccurrences.items():
        if weight >= min_weight:
            G.add_edge(char1, char2, weight=weight)

    return G


def analyze_network(G):
    pagerank = nx.pagerank(G, weight="weight")
    betweenness = nx.betweenness_centrality(G, weight="weight")
    degree_centrality = nx.degree_centrality(G)

    return {
        "pagerank": pagerank,
        "betweenness": betweenness,
        "degree_centrality": degree_centrality,
    }


def visualize_network(G, metrics, work_name):
    plt.figure(figsize=(15, 15))

    pos = nx.spring_layout(G, k=1, iterations=50)

    node_sizes = [metrics["pagerank"][node] * 10000 for node in G.nodes()]
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]

    node_labels = {
        node: " ".join([p[0] for p in pinyin(node, style=Style.NORMAL)])
        # node: node
        for node in G.nodes()
    }

    nx.draw_networkx_nodes(
        G, pos, node_size=node_sizes, node_color="lightblue", alpha=0.7
    )

    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5)

    nx.draw_networkx_labels(
        G,
        pos,
        labels=node_labels,
        font_size=10,
    )

    plt.axis("off")

    output_path = os.path.join(network_dir, f"{work_name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


with open(character_analysis_path, "r", encoding="utf-8") as f:
    character_data = json.load(f)

results = {}

for work_name, data in character_data.items():
    file_path = os.path.join(input_dir, f"{work_name}.txt")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    characters = list(data["characters"].keys())
    cooccurrences = extract_character_cooccurrences(content, characters)

    G = build_network(cooccurrences)
    if len(G.nodes()) < 2:
        continue

    metrics = analyze_network(G)
    visualize_network(G, metrics, work_name)

    results[work_name] = {
        "network_metrics": {
            "pagerank": metrics["pagerank"],
            "betweenness": metrics["betweenness"],
            "degree_centrality": metrics["degree_centrality"],
        },
        "edge_weights": {f"{u}-{v}": G[u][v]["weight"] for u, v in G.edges()},
    }

output_path = os.path.join(network_dir, "data.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
