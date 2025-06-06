"""
鲁迅小说人物关系网络分析器 - 增强版
功能：生成更加美观和清晰的人物关系网络图
"""

import os
import json
import jieba
import re
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import numpy as np
from matplotlib import cm
import seaborn as sns
from pypinyin import pinyin, Style

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 获取脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))

# 设置各种路径
input_dir = os.path.join(script_dir, "..", "..", "data", "combined")
output_dir = os.path.join(script_dir, "..", "..", "output", "passage_analysis")
network_dir = os.path.join(output_dir, "character_network_enhanced")
character_analysis_path = os.path.join(output_dir, "character_analysis.json")

# 确保输出目录存在
os.makedirs(network_dir, exist_ok=True)


def extract_character_cooccurrences(text, characters):
    """提取人物共现关系"""
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
    """构建人物关系网络"""
    G = nx.Graph()
    for (char1, char2), weight in cooccurrences.items():
        if weight >= min_weight:
            G.add_edge(char1, char2, weight=weight)
    return G


def analyze_network(G):
    """分析网络特征"""
    pagerank = nx.pagerank(G, weight="weight")
    betweenness = nx.betweenness_centrality(G, weight="weight")
    degree_centrality = nx.degree_centrality(G)

    return {
        "pagerank": pagerank,
        "betweenness": betweenness,
        "degree_centrality": degree_centrality,
    }


def get_optimal_layout(G, layout_type="auto"):
    """选择最优的布局算法"""
    num_nodes = len(G.nodes())

    if layout_type == "auto":
        if num_nodes <= 10:
            layout_type = "circular"
        elif num_nodes <= 30:
            layout_type = "kamada_kawai"
        else:
            layout_type = "spring"

    if layout_type == "spring":
        return nx.spring_layout(G, k=3 / np.sqrt(num_nodes), iterations=100, seed=42)
    elif layout_type == "kamada_kawai":
        try:
            return nx.kamada_kawai_layout(G)
        except:
            return nx.spring_layout(
                G, k=3 / np.sqrt(num_nodes), iterations=100, seed=42
            )
    elif layout_type == "circular":
        return nx.circular_layout(G)
    elif layout_type == "shell":
        return nx.shell_layout(G)
    else:
        return nx.spring_layout(G, k=3 / np.sqrt(num_nodes), iterations=100, seed=42)


def get_node_colors(G, metrics):
    """根据重要性指标为节点分配颜色"""
    pagerank_values = [metrics["pagerank"][node] for node in G.nodes()]

    # 归一化PageRank值
    min_pr = min(pagerank_values)
    max_pr = max(pagerank_values)

    if max_pr - min_pr == 0:
        normalized_pr = [0.5] * len(pagerank_values)
    else:
        normalized_pr = [(pr - min_pr) / (max_pr - min_pr) for pr in pagerank_values]

    # 使用颜色映射
    colors = cm.viridis(normalized_pr)
    return colors


def get_node_sizes(G, metrics, base_size=300):
    """计算节点大小"""
    pagerank_values = [metrics["pagerank"][node] for node in G.nodes()]

    # 归一化并调整大小范围
    min_pr = min(pagerank_values)
    max_pr = max(pagerank_values)

    if max_pr - min_pr == 0:
        return [base_size] * len(pagerank_values)

    normalized_pr = [(pr - min_pr) / (max_pr - min_pr) for pr in pagerank_values]
    sizes = [base_size + pr * base_size * 2 for pr in normalized_pr]

    return sizes


def get_edge_properties(G):
    """计算边的属性"""
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]

    if not edge_weights:
        return [], []

    # 归一化边权重
    min_weight = min(edge_weights)
    max_weight = max(edge_weights)

    if max_weight - min_weight == 0:
        widths = [1.0] * len(edge_weights)
        alphas = [0.5] * len(edge_weights)
    else:
        normalized_weights = [
            (w - min_weight) / (max_weight - min_weight) for w in edge_weights
        ]
        widths = [0.5 + nw * 3.5 for nw in normalized_weights]  # 0.5-4.0
        alphas = [0.3 + nw * 0.5 for nw in normalized_weights]  # 0.3-0.8

    return widths, alphas


def create_enhanced_network_visualization(G, metrics, work_name, use_chinese=True):
    """创建增强版网络可视化"""

    # 设置图形大小和样式
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    fig.patch.set_facecolor("white")

    # 获取布局
    pos = get_optimal_layout(G, "auto")

    # 获取节点属性
    node_colors = get_node_colors(G, metrics)
    node_sizes = get_node_sizes(G, metrics)

    # 获取边属性
    edge_widths, edge_alphas = get_edge_properties(G)

    # 绘制边
    if len(G.edges()) > 0:
        for i, (u, v) in enumerate(G.edges()):
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u, v)],
                width=edge_widths[i] if edge_widths else 1.0,
                alpha=edge_alphas[i] if edge_alphas else 0.5,
                edge_color="gray",
                ax=ax,
            )

    # 绘制节点
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.8,
        edgecolors="black",
        linewidths=0.5,
        ax=ax,
    )

    # 准备标签
    if use_chinese:
        # 尝试直接使用中文
        labels = {node: node for node in G.nodes()}
    else:
        # 使用拼音
        labels = {
            node: " ".join([p[0] for p in pinyin(node, style=Style.NORMAL)])
            for node in G.nodes()
        }

    # 绘制标签
    nx.draw_networkx_labels(
        G, pos, labels=labels, font_size=8, font_weight="bold", ax=ax
    )

    # 添加图例
    create_legend(fig, ax, metrics, G)

    # 设置标题
    ax.set_title(
        f"{work_name} - 人物关系网络图", fontsize=16, fontweight="bold", pad=20
    )

    # 隐藏坐标轴
    ax.axis("off")

    # 调整布局
    plt.tight_layout()

    # 保存图片
    output_path = os.path.join(network_dir, f"{work_name}_enhanced.png")
    plt.savefig(
        output_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    plt.close()

    print(f"增强版网络图已生成: {work_name}_enhanced.png")


def create_legend(fig, ax, metrics, G):
    """创建图例"""
    # 获取重要性范围
    pagerank_values = list(metrics["pagerank"].values())
    min_pr = min(pagerank_values)
    max_pr = max(pagerank_values)

    # 创建重要性图例
    legend_elements = []

    # 节点大小图例
    sizes = [300, 600, 900]
    size_labels = ["低重要性", "中重要性", "高重要性"]

    for size, label in zip(sizes, size_labels):
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="lightblue",
                markersize=np.sqrt(size / 50),
                label=label,
                markeredgecolor="black",
                markeredgewidth=0.5,
            )
        )

    # 边宽度图例
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()] if G.edges() else [1]
    if edge_weights:
        min_weight = min(edge_weights)
        max_weight = max(edge_weights)

        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                color="gray",
                linewidth=1,
                label=f"关系强度: {min_weight}-{max_weight}",
            )
        )

    # 添加图例
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(1, 1),
        frameon=True,
        fancybox=True,
        shadow=True,
    )


def create_multiple_layout_comparison(G, metrics, work_name):
    """创建多种布局对比图"""
    layouts = {
        "Spring Layout": nx.spring_layout(
            G, k=3 / np.sqrt(len(G.nodes())), iterations=100, seed=42
        ),
        "Kamada-Kawai Layout": (
            nx.kamada_kawai_layout(G)
            if len(G.nodes()) <= 100
            else nx.spring_layout(G, seed=42)
        ),
        "Circular Layout": nx.circular_layout(G),
        "Shell Layout": nx.shell_layout(G),
    }

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(f"{work_name} - 不同布局算法对比", fontsize=16, fontweight="bold")

    for idx, (layout_name, pos) in enumerate(layouts.items()):
        ax = axes[idx // 2, idx % 2]

        # 获取节点属性
        node_colors = get_node_colors(G, metrics)
        node_sizes = get_node_sizes(G, metrics, base_size=200)
        edge_widths, edge_alphas = get_edge_properties(G)

        # 绘制网络
        if len(G.edges()) > 0:
            nx.draw_networkx_edges(
                G, pos, width=edge_widths, alpha=0.5, edge_color="gray", ax=ax
            )

        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.8,
            edgecolors="black",
            linewidths=0.5,
            ax=ax,
        )

        # 标签
        labels = {node: node for node in G.nodes()}
        nx.draw_networkx_labels(
            G, pos, labels=labels, font_size=6, font_weight="bold", ax=ax
        )

        ax.set_title(layout_name, fontsize=12, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()

    # 保存对比图
    output_path = os.path.join(network_dir, f"{work_name}_layouts_comparison.png")
    plt.savefig(
        output_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    plt.close()

    print(f"布局对比图已生成: {work_name}_layouts_comparison.png")


# 主程序
if __name__ == "__main__":
    # 读取人物分析结果
    with open(character_analysis_path, "r", encoding="utf-8") as f:
        character_data = json.load(f)

    # 遍历每部作品
    for work_name, data in character_data.items():
        print(f"正在生成增强版网络图: {work_name}")

        # 读取作品文本
        file_path = os.path.join(input_dir, f"{work_name}.txt")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 获取该作品的所有人物
        characters = list(data["characters"].keys())

        # 提取人物共现关系
        cooccurrences = extract_character_cooccurrences(content, characters)

        # 构建网络图
        G = build_network(cooccurrences)

        # 跳过节点数不足的网络
        if len(G.nodes()) < 2:
            print(f"  作品 {work_name} 的人物网络节点数不足，跳过分析")
            continue

        # 分析网络特征
        metrics = analyze_network(G)

        # 创建增强版可视化
        create_enhanced_network_visualization(G, metrics, work_name, use_chinese=True)

        # 为复杂网络创建布局对比图
        if len(G.nodes()) >= 5:
            create_multiple_layout_comparison(G, metrics, work_name)

    print(f"\n所有增强版网络图已生成完成！")
    print(f"输出目录: {network_dir}/")
