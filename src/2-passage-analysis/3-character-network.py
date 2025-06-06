"""
鲁迅小说人物关系网络分析器
功能：分析人物之间的共现关系，构建社交网络图并计算网络指标
"""

import os
import json
import jieba
import re
import networkx as nx  # 网络分析库
import matplotlib.pyplot as plt
from collections import defaultdict
from pypinyin import pinyin, Style  # 用于中文转拼音，解决matplotlib中文显示问题

# 获取脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))

# 设置各种路径
input_dir = os.path.join(script_dir, "..", "..", "data", "combined")  # 输入文本目录
output_dir = os.path.join(
    script_dir, "..", "..", "output", "passage_analysis"
)  # 输出目录
font_dir = os.path.join(script_dir, "..", "..", "font")  # 字体目录
network_dir = os.path.join(output_dir, "character_network")  # 网络图输出目录
character_analysis_path = os.path.join(
    output_dir, "character_analysis.json"
)  # 人物分析结果路径

# 确保网络图输出目录存在
os.makedirs(network_dir, exist_ok=True)

# 配置matplotlib字体（注释掉的SimHei可能在某些系统上不可用）
# plt.rcParams["font.sans-serif"] = "SimHei"


def extract_character_cooccurrences(text, characters):
    """
    提取人物共现关系
    参数：
        text: 文本内容
        characters: 人物名称列表
    返回：
        cooccurrences: 人物共现次数字典，格式为{(人物1, 人物2): 共现次数}
    """
    # 按标点符号分割文本为句子
    sentences = re.split("[。！？]", text)
    cooccurrences = defaultdict(int)

    # 遍历每个句子
    for sentence in sentences:
        sentence_chars = []

        # 检查句子中出现的人物
        for char in characters:
            if char in sentence:
                sentence_chars.append(char)

        # 计算句子中所有人物的两两共现
        for i in range(len(sentence_chars)):
            for j in range(i + 1, len(sentence_chars)):
                # 按字典序排序，确保(A,B)和(B,A)被视为同一对
                pair = tuple(sorted([sentence_chars[i], sentence_chars[j]]))
                cooccurrences[pair] += 1

    return cooccurrences


def build_network(cooccurrences, min_weight=2):
    """
    构建人物关系网络
    参数：
        cooccurrences: 人物共现关系字典
        min_weight: 最小共现次数阈值，低于此值的关系将被过滤
    返回：
        G: NetworkX图对象
    """
    G = nx.Graph()  # 创建无向图

    # 添加边（人物关系）
    for (char1, char2), weight in cooccurrences.items():
        if weight >= min_weight:  # 只保留共现次数大于等于阈值的关系
            G.add_edge(char1, char2, weight=weight)

    return G


def analyze_network(G):
    """
    分析网络特征
    参数：
        G: NetworkX图对象
    返回：
        metrics: 包含各种网络指标的字典
    """
    # PageRank算法：衡量节点的重要性（考虑边权重）
    pagerank = nx.pagerank(G, weight="weight")

    # 介数中心性：衡量节点在网络中的桥梁作用
    betweenness = nx.betweenness_centrality(G, weight="weight")

    # 度中心性：衡量节点的连接数量
    degree_centrality = nx.degree_centrality(G)

    return {
        "pagerank": pagerank,
        "betweenness": betweenness,
        "degree_centrality": degree_centrality,
    }


def visualize_network(G, metrics, work_name):
    """
    可视化人物关系网络
    参数：
        G: NetworkX图对象
        metrics: 网络指标字典
        work_name: 作品名称（用于保存文件）
    """
    # 创建大尺寸图形
    plt.figure(figsize=(15, 15))

    # 使用spring布局算法排列节点
    pos = nx.spring_layout(G, k=1, iterations=50)

    # 根据PageRank值调整节点大小（重要人物显示更大）
    node_sizes = [metrics["pagerank"][node] * 10000 for node in G.nodes()]

    # 根据共现次数调整边的粗细
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]

    # 将中文人名转换为拼音，避免matplotlib中文显示问题
    node_labels = {
        node: " ".join([p[0] for p in pinyin(node, style=Style.NORMAL)])
        # node: node  # 如果系统支持中文显示，可以使用这一行
        for node in G.nodes()
    }

    # 绘制节点
    nx.draw_networkx_nodes(
        G, pos, node_size=node_sizes, node_color="lightblue", alpha=0.7
    )

    # 绘制边
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5)

    # 绘制标签
    nx.draw_networkx_labels(
        G,
        pos,
        labels=node_labels,
        font_size=10,
    )

    # 隐藏坐标轴
    plt.axis("off")

    # 保存网络图
    output_path = os.path.join(network_dir, f"{work_name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()  # 关闭图形以释放内存
    print(f"网络图已生成: {work_name}.png")


# 读取人物分析结果
with open(character_analysis_path, "r", encoding="utf-8") as f:
    character_data = json.load(f)

# 存储所有作品的网络分析结果
results = {}

# 遍历每部作品
for work_name, data in character_data.items():
    print(f"正在分析作品网络: {work_name}")

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

    # 如果网络中的节点数少于2个，跳过分析
    if len(G.nodes()) < 2:
        print(f"  作品 {work_name} 的人物网络节点数不足，跳过分析")
        continue

    # 分析网络特征
    metrics = analyze_network(G)

    # 可视化网络
    visualize_network(G, metrics, work_name)

    # 存储分析结果
    results[work_name] = {
        "network_metrics": {
            "pagerank": metrics["pagerank"],  # PageRank重要性分数
            "betweenness": metrics["betweenness"],  # 介数中心性
            "degree_centrality": metrics["degree_centrality"],  # 度中心性
        },
        "edge_weights": {
            f"{u}-{v}": G[u][v]["weight"] for u, v in G.edges()
        },  # 边权重（共现次数）
    }

# 保存网络分析数据
output_path = os.path.join(network_dir, "data.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"人物关系网络分析完成，结果已保存到:")
print(f"  数据文件: {output_path}")
print(f"  网络图: {network_dir}/")
