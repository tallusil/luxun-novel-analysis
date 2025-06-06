"""
人物网络图配置工具
功能：提供简单的参数调整和快速生成网络图
"""

import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import re
from collections import defaultdict

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class NetworkConfigTool:
    """网络图配置工具"""

    def __init__(self, project_root):
        self.project_root = project_root
        self.input_dir = os.path.join(project_root, "data", "combined")
        self.output_dir = os.path.join(project_root, "output", "passage_analysis")
        self.network_dir = os.path.join(self.output_dir, "character_network_custom")
        self.character_analysis_path = os.path.join(
            self.output_dir, "character_analysis.json"
        )

        # 确保输出目录存在
        os.makedirs(self.network_dir, exist_ok=True)

        # 加载人物数据
        with open(self.character_analysis_path, "r", encoding="utf-8") as f:
            self.character_data = json.load(f)

    def extract_character_cooccurrences(self, text, characters):
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

    def build_network(self, cooccurrences, min_weight=2):
        """构建人物关系网络"""
        G = nx.Graph()
        for (char1, char2), weight in cooccurrences.items():
            if weight >= min_weight:
                G.add_edge(char1, char2, weight=weight)
        return G

    def analyze_network(self, G):
        """分析网络特征"""
        if len(G.nodes()) == 0:
            return {}

        return {
            "pagerank": nx.pagerank(G, weight="weight"),
            "betweenness": nx.betweenness_centrality(G, weight="weight"),
            "degree_centrality": nx.degree_centrality(G),
            "closeness_centrality": nx.closeness_centrality(G),
        }

    def create_custom_network(self, work_name, config):
        """根据配置创建自定义网络图"""

        # 默认配置
        default_config = {
            "min_weight": 2,  # 最小边权重
            "layout": "auto",  # 布局算法：auto/spring/circular/kamada_kawai
            "node_color": "pagerank",  # 节点颜色依据：pagerank/betweenness/degree/fixed
            "node_size": "pagerank",  # 节点大小依据：pagerank/betweenness/degree/fixed
            "base_size": 300,  # 基础节点大小
            "edge_thickness": True,  # 是否根据权重调整边粗细
            "show_labels": True,  # 是否显示标签
            "use_chinese": True,  # 是否使用中文标签
            "color_scheme": "viridis",  # 颜色方案：viridis/plasma/inferno/cool/warm
            "figure_size": (12, 10),  # 图形大小
            "dpi": 300,  # 分辨率
            "background": "white",  # 背景颜色
            "title_size": 14,  # 标题字体大小
            "label_size": 8,  # 标签字体大小
        }

        # 合并配置
        final_config = {**default_config, **config}

        print(f"正在为 {work_name} 生成自定义网络图...")
        print(f"配置参数: {final_config}")

        # 检查作品数据
        if work_name not in self.character_data:
            print(f"错误：找不到作品 {work_name} 的数据")
            return None

        # 读取作品文本
        file_path = os.path.join(self.input_dir, f"{work_name}.txt")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except FileNotFoundError:
            print(f"错误：找不到文件 {file_path}")
            return None

        # 获取该作品的所有人物
        characters = list(self.character_data[work_name]["characters"].keys())

        # 提取人物共现关系
        cooccurrences = self.extract_character_cooccurrences(content, characters)

        # 构建网络图
        G = self.build_network(cooccurrences, final_config["min_weight"])

        if len(G.nodes()) < 2:
            print(f"节点数不足（{len(G.nodes())}），请降低 min_weight 参数")
            return None

        # 分析网络特征
        metrics = self.analyze_network(G)

        # 创建图形
        fig, ax = plt.subplots(1, 1, figsize=final_config["figure_size"])
        fig.patch.set_facecolor(final_config["background"])

        # 选择布局
        pos = self._get_layout(G, final_config["layout"])

        # 设置节点颜色和大小
        node_colors, node_sizes = self._get_node_properties(G, metrics, final_config)

        # 设置边属性
        edge_widths, edge_alphas = self._get_edge_properties(G, final_config)

        # 绘制边
        if final_config["edge_thickness"]:
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
        else:
            nx.draw_networkx_edges(
                G, pos, width=1.0, alpha=0.5, edge_color="gray", ax=ax
            )

        # 绘制节点
        nodes = nx.draw_networkx_nodes(
            G,
            pos,
            node_color=node_colors,
            node_size=node_sizes,
            cmap=getattr(cm, final_config["color_scheme"]),
            alpha=0.8,
            edgecolors="black",
            linewidths=0.5,
            ax=ax,
        )

        # 绘制标签
        if final_config["show_labels"]:
            if final_config["use_chinese"]:
                labels = {node: node for node in G.nodes()}
            else:
                # 如果不使用中文，可以使用简化的标签
                labels = {
                    node: node[:2] + ".." if len(node) > 3 else node
                    for node in G.nodes()
                }

            nx.draw_networkx_labels(
                G,
                pos,
                labels=labels,
                font_size=final_config["label_size"],
                font_weight="bold",
                ax=ax,
            )

        # 添加颜色条
        if final_config["node_color"] in ["pagerank", "betweenness", "degree"]:
            plt.colorbar(
                nodes, ax=ax, label=f"{final_config['node_color']} 值", shrink=0.8
            )

        # 设置标题
        title = f"{work_name} - 人物关系网络图"
        if final_config["min_weight"] > 2:
            title += f"\n(最小关系强度: {final_config['min_weight']})"

        ax.set_title(
            title, fontsize=final_config["title_size"], fontweight="bold", pad=20
        )

        # 隐藏坐标轴
        ax.axis("off")

        # 调整布局
        plt.tight_layout()

        # 保存图片
        config_str = f"w{final_config['min_weight']}_{final_config['layout']}_{final_config['node_color']}"
        filename = f"{work_name}_custom_{config_str}.png"
        output_path = os.path.join(self.network_dir, filename)

        plt.savefig(
            output_path,
            dpi=final_config["dpi"],
            bbox_inches="tight",
            facecolor=final_config["background"],
            edgecolor="none",
        )
        plt.close()

        print(f"自定义网络图已生成: {filename}")
        print(f"网络信息: {len(G.nodes())} 个节点, {len(G.edges())} 条边")

        return output_path

    def _get_layout(self, G, layout_type):
        """获取布局位置"""
        num_nodes = len(G.nodes())

        if layout_type == "auto":
            if num_nodes <= 10:
                return nx.circular_layout(G)
            elif num_nodes <= 30:
                try:
                    return nx.kamada_kawai_layout(G)
                except:
                    return nx.spring_layout(
                        G, k=3 / np.sqrt(num_nodes), iterations=100, seed=42
                    )
            else:
                return nx.spring_layout(
                    G, k=3 / np.sqrt(num_nodes), iterations=100, seed=42
                )
        elif layout_type == "spring":
            return nx.spring_layout(
                G, k=3 / np.sqrt(num_nodes), iterations=100, seed=42
            )
        elif layout_type == "circular":
            return nx.circular_layout(G)
        elif layout_type == "kamada_kawai":
            try:
                return nx.kamada_kawai_layout(G)
            except:
                return nx.spring_layout(
                    G, k=3 / np.sqrt(num_nodes), iterations=100, seed=42
                )
        else:
            return nx.spring_layout(
                G, k=3 / np.sqrt(num_nodes), iterations=100, seed=42
            )

    def _get_node_properties(self, G, metrics, config):
        """获取节点颜色和大小"""
        # 节点颜色
        if config["node_color"] == "pagerank":
            color_values = [metrics["pagerank"].get(node, 0) for node in G.nodes()]
        elif config["node_color"] == "betweenness":
            color_values = [metrics["betweenness"].get(node, 0) for node in G.nodes()]
        elif config["node_color"] == "degree":
            color_values = [
                metrics["degree_centrality"].get(node, 0) for node in G.nodes()
            ]
        else:
            color_values = [0.5] * len(G.nodes())

        # 节点大小
        if config["node_size"] == "pagerank":
            size_values = [metrics["pagerank"].get(node, 0) for node in G.nodes()]
        elif config["node_size"] == "betweenness":
            size_values = [metrics["betweenness"].get(node, 0) for node in G.nodes()]
        elif config["node_size"] == "degree":
            size_values = [
                metrics["degree_centrality"].get(node, 0) for node in G.nodes()
            ]
        else:
            size_values = [1] * len(G.nodes())

        # 归一化大小
        if max(size_values) > 0:
            size_values = [
                config["base_size"] + (v / max(size_values)) * config["base_size"] * 1.5
                for v in size_values
            ]
        else:
            size_values = [config["base_size"]] * len(size_values)

        return color_values, size_values

    def _get_edge_properties(self, G, config):
        """获取边的属性"""
        if not config["edge_thickness"]:
            return [], []

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
            widths = [0.5 + nw * 2.5 for nw in normalized_weights]  # 0.5-3.0
            alphas = [0.3 + nw * 0.4 for nw in normalized_weights]  # 0.3-0.7

        return widths, alphas

    def list_available_works(self):
        """列出可用的作品"""
        return list(self.character_data.keys())

    def get_network_info(self, work_name, min_weight=2):
        """获取网络基础信息"""
        if work_name not in self.character_data:
            return None

        # 读取作品文本
        file_path = os.path.join(self.input_dir, f"{work_name}.txt")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except FileNotFoundError:
            return None

        # 获取该作品的所有人物
        characters = list(self.character_data[work_name]["characters"].keys())

        # 提取人物共现关系
        cooccurrences = self.extract_character_cooccurrences(content, characters)

        # 构建网络图
        G = self.build_network(cooccurrences, min_weight)

        return {
            "total_characters": len(characters),
            "network_nodes": len(G.nodes()),
            "network_edges": len(G.edges()),
            "min_weight": min_weight,
        }


def main():
    """主函数 - 演示如何使用配置工具"""

    # 获取项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..", "..")

    # 创建配置工具
    tool = NetworkConfigTool(project_root)

    print("人物网络图配置工具")
    print("=" * 40)

    # 列出可用作品
    works = tool.list_available_works()
    print(f"可用作品: {', '.join(works)}")

    # 示例配置
    examples = [
        {
            "work": "呐喊",
            "config": {
                "min_weight": 3,
                "layout": "spring",
                "node_color": "pagerank",
                "node_size": "pagerank",
                "color_scheme": "plasma",
            },
        },
        {
            "work": "彷徨",
            "config": {
                "min_weight": 2,
                "layout": "circular",
                "node_color": "betweenness",
                "node_size": "degree",
                "color_scheme": "viridis",
            },
        },
        {
            "work": "朝花夕拾",
            "config": {
                "min_weight": 1,
                "layout": "kamada_kawai",
                "node_color": "degree",
                "node_size": "betweenness",
                "color_scheme": "cool",
                "base_size": 400,
            },
        },
    ]

    # 生成示例图片
    for example in examples:
        work = example["work"]
        config = example["config"]

        print(f"\n正在生成 {work} 的网络图...")

        # 先查看网络信息
        info = tool.get_network_info(work, config.get("min_weight", 2))
        if info:
            print(f"网络信息: {info}")

        # 生成网络图
        output_path = tool.create_custom_network(work, config)
        if output_path:
            print(f"成功生成: {output_path}")

    print(f"\n所有自定义网络图已保存到: {tool.network_dir}")


if __name__ == "__main__":
    main()
