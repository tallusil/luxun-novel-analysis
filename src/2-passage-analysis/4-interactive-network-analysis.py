"""
鲁迅小说人物关系网络交互式分析工具
功能：提供交互式的网络分析和可视化选项
"""

import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import numpy as np
from matplotlib import cm
import seaborn as sns
from pypinyin import pinyin, Style
import pandas as pd
import re

# 设置中文字体和样式
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")

# 获取脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))

# 设置各种路径
input_dir = os.path.join(script_dir, "..", "..", "data", "combined")
output_dir = os.path.join(script_dir, "..", "..", "output", "passage_analysis")
network_dir = os.path.join(output_dir, "character_network_interactive")
character_analysis_path = os.path.join(output_dir, "character_analysis.json")

# 确保输出目录存在
os.makedirs(network_dir, exist_ok=True)


class CharacterNetworkAnalyzer:
    """人物网络分析器类"""

    def __init__(self, character_data_path):
        """初始化分析器"""
        with open(character_data_path, "r", encoding="utf-8") as f:
            self.character_data = json.load(f)
        self.networks = {}
        self.cooccurrences = {}
        self.metrics = {}

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

        metrics = {}

        # 基础指标
        metrics["pagerank"] = nx.pagerank(G, weight="weight")
        metrics["betweenness"] = nx.betweenness_centrality(G, weight="weight")
        metrics["degree_centrality"] = nx.degree_centrality(G)
        metrics["closeness_centrality"] = nx.closeness_centrality(G)

        # 网络整体指标
        metrics["network_stats"] = {
            "nodes": len(G.nodes()),
            "edges": len(G.edges()),
            "density": nx.density(G),
            "average_clustering": nx.average_clustering(G),
        }

        # 如果网络连通，计算更多指标
        if nx.is_connected(G):
            metrics["network_stats"]["diameter"] = nx.diameter(G)
            metrics["network_stats"]["average_shortest_path"] = (
                nx.average_shortest_path_length(G)
            )
        else:
            # 对于非连通图，计算最大连通分量的指标
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            if len(largest_cc) > 1:
                metrics["network_stats"]["largest_component_size"] = len(largest_cc)
                metrics["network_stats"]["largest_component_diameter"] = nx.diameter(
                    subgraph
                )

        return metrics

    def load_all_networks(self, min_weight=2):
        """加载所有作品的网络"""
        for work_name, data in self.character_data.items():
            print(f"正在处理: {work_name}")

            # 读取作品文本
            file_path = os.path.join(input_dir, f"{work_name}.txt")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except FileNotFoundError:
                print(f"  文件不存在: {file_path}")
                continue

            # 获取该作品的所有人物
            characters = list(data["characters"].keys())

            # 提取人物共现关系
            cooccurrences = self.extract_character_cooccurrences(content, characters)
            self.cooccurrences[work_name] = cooccurrences

            # 构建网络图
            G = self.build_network(cooccurrences, min_weight)

            if len(G.nodes()) >= 2:
                self.networks[work_name] = G
                self.metrics[work_name] = self.analyze_network(G)
                print(f"  网络构建完成: {len(G.nodes())} 个节点, {len(G.edges())} 条边")
            else:
                print(f"  节点数不足，跳过")

    def create_network_summary_report(self):
        """创建网络分析总结报告"""
        if not self.networks:
            print("没有可分析的网络数据")
            return

        # 创建汇总统计
        summary_data = []
        for work_name, G in self.networks.items():
            metrics = self.metrics[work_name]

            # 找出最重要的人物（PageRank最高）
            top_character = max(metrics["pagerank"], key=metrics["pagerank"].get)

            # 找出桥梁人物（介数中心性最高）
            bridge_character = max(
                metrics["betweenness"], key=metrics["betweenness"].get
            )

            summary_data.append(
                {
                    "作品": work_name,
                    "节点数": metrics["network_stats"]["nodes"],
                    "边数": metrics["network_stats"]["edges"],
                    "网络密度": round(metrics["network_stats"]["density"], 3),
                    "平均聚类系数": round(
                        metrics["network_stats"]["average_clustering"], 3
                    ),
                    "最重要人物": top_character,
                    "桥梁人物": bridge_character,
                    "PageRank最高值": round(metrics["pagerank"][top_character], 3),
                    "介数中心性最高值": round(
                        metrics["betweenness"][bridge_character], 3
                    ),
                }
            )

        # 转换为DataFrame并保存
        df = pd.DataFrame(summary_data)

        # 创建可视化报告
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("鲁迅作品人物网络分析总结报告", fontsize=16, fontweight="bold")

        # 1. 网络规模对比
        ax1 = axes[0, 0]
        ax1.bar(df["作品"], df["节点数"], alpha=0.7, color="skyblue")
        ax1.set_title("各作品人物网络规模")
        ax1.set_ylabel("节点数")
        ax1.tick_params(axis="x", rotation=45)

        # 2. 网络密度对比
        ax2 = axes[0, 1]
        ax2.bar(df["作品"], df["网络密度"], alpha=0.7, color="lightcoral")
        ax2.set_title("各作品网络密度")
        ax2.set_ylabel("密度")
        ax2.tick_params(axis="x", rotation=45)

        # 3. 聚类系数对比
        ax3 = axes[1, 0]
        ax3.bar(df["作品"], df["平均聚类系数"], alpha=0.7, color="lightgreen")
        ax3.set_title("各作品平均聚类系数")
        ax3.set_ylabel("聚类系数")
        ax3.tick_params(axis="x", rotation=45)

        # 4. 网络规模散点图
        ax4 = axes[1, 1]
        scatter = ax4.scatter(
            df["节点数"], df["边数"], c=df["网络密度"], s=100, alpha=0.7, cmap="viridis"
        )
        ax4.set_xlabel("节点数")
        ax4.set_ylabel("边数")
        ax4.set_title("网络规模关系")

        # 添加作品名标签
        for i, work in enumerate(df["作品"]):
            ax4.annotate(
                work,
                (df.iloc[i]["节点数"], df.iloc[i]["边数"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        plt.colorbar(scatter, ax=ax4, label="网络密度")

        plt.tight_layout()

        # 保存报告
        report_path = os.path.join(network_dir, "network_summary_report.png")
        plt.savefig(report_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

        # 保存CSV数据
        csv_path = os.path.join(network_dir, "network_summary_data.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        print(f"网络分析总结报告已生成:")
        print(f"  图表: {report_path}")
        print(f"  数据: {csv_path}")

        return df

    def create_character_importance_analysis(self):
        """创建人物重要性分析"""
        # 收集所有人物的重要性数据
        all_characters = []

        for work_name, metrics in self.metrics.items():
            for char in metrics["pagerank"]:
                all_characters.append(
                    {
                        "作品": work_name,
                        "人物": char,
                        "PageRank": metrics["pagerank"][char],
                        "介数中心性": metrics["betweenness"][char],
                        "度中心性": metrics["degree_centrality"][char],
                        "接近中心性": metrics["closeness_centrality"][char],
                    }
                )

        df = pd.DataFrame(all_characters)

        # 创建人物重要性可视化
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("鲁迅作品人物重要性分析", fontsize=16, fontweight="bold")

        # 1. 各作品中最重要的人物（PageRank）
        ax1 = axes[0, 0]
        top_chars_by_work = df.loc[df.groupby("作品")["PageRank"].idxmax()]
        bars = ax1.bar(
            range(len(top_chars_by_work)),
            top_chars_by_work["PageRank"],
            alpha=0.7,
            color="gold",
        )
        ax1.set_title("各作品最重要人物 (PageRank)")
        ax1.set_ylabel("PageRank值")
        ax1.set_xticks(range(len(top_chars_by_work)))
        ax1.set_xticklabels(
            [
                f"{row['作品']}\n{row['人物']}"
                for _, row in top_chars_by_work.iterrows()
            ],
            rotation=45,
            ha="right",
        )

        # 2. 重要性指标散点图
        ax2 = axes[0, 1]
        scatter = ax2.scatter(
            df["PageRank"],
            df["介数中心性"],
            c=df["度中心性"],
            alpha=0.6,
            s=50,
            cmap="plasma",
        )
        ax2.set_xlabel("PageRank")
        ax2.set_ylabel("介数中心性")
        ax2.set_title("人物重要性指标关系")
        plt.colorbar(scatter, ax=ax2, label="度中心性")

        # 3. 各作品人物重要性分布
        ax3 = axes[1, 0]
        works_to_plot = df["作品"].value_counts().head(5).index  # 选择人物最多的5部作品
        df_subset = df[df["作品"].isin(works_to_plot)]

        for work in works_to_plot:
            work_data = df_subset[df_subset["作品"] == work]["PageRank"]
            ax3.hist(work_data, alpha=0.6, label=work, bins=10)

        ax3.set_xlabel("PageRank值")
        ax3.set_ylabel("人物数量")
        ax3.set_title("人物重要性分布")
        ax3.legend()

        # 4. 全局最重要的人物排名
        ax4 = axes[1, 1]
        top_global = df.nlargest(10, "PageRank")
        bars = ax4.barh(
            range(len(top_global)), top_global["PageRank"], alpha=0.7, color="lightblue"
        )
        ax4.set_yticks(range(len(top_global)))
        ax4.set_yticklabels(
            [f"{row['人物']} ({row['作品']})" for _, row in top_global.iterrows()]
        )
        ax4.set_xlabel("PageRank值")
        ax4.set_title("全局最重要人物排名 (Top 10)")

        plt.tight_layout()

        # 保存分析结果
        analysis_path = os.path.join(network_dir, "character_importance_analysis.png")
        plt.savefig(analysis_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

        # 保存详细数据
        importance_csv = os.path.join(network_dir, "character_importance_data.csv")
        df.to_csv(importance_csv, index=False, encoding="utf-8-sig")

        print(f"人物重要性分析已生成:")
        print(f"  图表: {analysis_path}")
        print(f"  数据: {importance_csv}")

        return df

    def create_advanced_network_visualization(
        self,
        work_name,
        node_color_by="pagerank",
        node_size_by="pagerank",
        layout_type="auto",
        show_edge_labels=False,
        min_edge_weight=1,
    ):
        """创建高级网络可视化"""
        if work_name not in self.networks:
            print(f"作品 {work_name} 的网络数据不存在")
            return

        G = self.networks[work_name]
        metrics = self.metrics[work_name]

        # 过滤边
        if min_edge_weight > 1:
            edges_to_remove = [
                (u, v)
                for u, v, d in G.edges(data=True)
                if d["weight"] < min_edge_weight
            ]
            G_filtered = G.copy()
            G_filtered.remove_edges_from(edges_to_remove)
            G = G_filtered

        if len(G.nodes()) == 0:
            print("过滤后没有节点")
            return

        # 创建图形
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        fig.patch.set_facecolor("white")

        # 选择布局
        if layout_type == "auto":
            if len(G.nodes()) <= 10:
                pos = nx.circular_layout(G)
            elif len(G.nodes()) <= 30:
                try:
                    pos = nx.kamada_kawai_layout(G)
                except:
                    pos = nx.spring_layout(
                        G, k=3 / np.sqrt(len(G.nodes())), iterations=100
                    )
            else:
                pos = nx.spring_layout(G, k=3 / np.sqrt(len(G.nodes())), iterations=100)
        elif layout_type == "spring":
            pos = nx.spring_layout(G, k=3 / np.sqrt(len(G.nodes())), iterations=100)
        elif layout_type == "circular":
            pos = nx.circular_layout(G)
        elif layout_type == "random":
            pos = nx.random_layout(G)

        # 设置节点颜色
        if node_color_by == "pagerank":
            color_values = [metrics["pagerank"].get(node, 0) for node in G.nodes()]
            cmap = cm.Reds
        elif node_color_by == "betweenness":
            color_values = [metrics["betweenness"].get(node, 0) for node in G.nodes()]
            cmap = cm.Blues
        elif node_color_by == "degree":
            color_values = [
                metrics["degree_centrality"].get(node, 0) for node in G.nodes()
            ]
            cmap = cm.Greens
        else:
            color_values = [0.5] * len(G.nodes())
            cmap = cm.viridis

        # 设置节点大小
        if node_size_by == "pagerank":
            size_values = [metrics["pagerank"].get(node, 0) for node in G.nodes()]
        elif node_size_by == "betweenness":
            size_values = [metrics["betweenness"].get(node, 0) for node in G.nodes()]
        elif node_size_by == "degree":
            size_values = [
                metrics["degree_centrality"].get(node, 0) for node in G.nodes()
            ]
        else:
            size_values = [1] * len(G.nodes())

        # 归一化大小
        if max(size_values) > 0:
            size_values = [300 + (v / max(size_values)) * 500 for v in size_values]
        else:
            size_values = [300] * len(size_values)

        # 绘制边
        edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
        if edge_weights:
            max_weight = max(edge_weights)
            edge_widths = [w / max_weight * 3 + 0.5 for w in edge_weights]
            edge_alphas = [w / max_weight * 0.7 + 0.3 for w in edge_weights]
        else:
            edge_widths = [1]
            edge_alphas = [0.5]

        nx.draw_networkx_edges(
            G, pos, width=edge_widths, alpha=0.6, edge_color="gray", ax=ax
        )

        # 绘制节点
        nodes = nx.draw_networkx_nodes(
            G,
            pos,
            node_color=color_values,
            node_size=size_values,
            cmap=cmap,
            alpha=0.8,
            edgecolors="black",
            linewidths=0.5,
            ax=ax,
        )

        # 绘制标签
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax)

        # 绘制边标签（如果需要）
        if show_edge_labels:
            edge_labels = {(u, v): str(G[u][v]["weight"]) for u, v in G.edges()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6, ax=ax)

        # 添加颜色条
        if max(color_values) > min(color_values):
            plt.colorbar(nodes, ax=ax, label=f"{node_color_by} 值")

        # 设置标题
        ax.set_title(
            f"{work_name} - 高级人物关系网络图\n"
            f"节点颜色/大小: {node_color_by}, 布局: {layout_type}",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        ax.axis("off")
        plt.tight_layout()

        # 保存图片
        filename = f"{work_name}_advanced_{node_color_by}_{layout_type}.png"
        output_path = os.path.join(network_dir, filename)
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close()

        print(f"高级网络图已生成: {filename}")
        return output_path

    def run_complete_analysis(self):
        """运行完整分析"""
        print("开始加载所有网络数据...")
        self.load_all_networks()

        print("\n生成网络分析总结报告...")
        self.create_network_summary_report()

        print("\n生成人物重要性分析...")
        self.create_character_importance_analysis()

        print("\n为主要作品生成高级可视化...")
        # 选择几个重要作品进行详细可视化
        important_works = ["呐喊", "彷徨", "朝花夕拾", "故事新编"]

        for work in important_works:
            if work in self.networks:
                print(f"\n正在处理: {work}")
                # 不同指标的可视化
                for metric in ["pagerank", "betweenness", "degree"]:
                    self.create_advanced_network_visualization(
                        work,
                        node_color_by=metric,
                        node_size_by=metric,
                        layout_type="auto",
                    )


# 主程序
if __name__ == "__main__":
    print("鲁迅小说人物关系网络交互式分析工具")
    print("=" * 50)

    # 创建分析器
    analyzer = CharacterNetworkAnalyzer(character_analysis_path)

    # 运行完整分析
    analyzer.run_complete_analysis()

    print(f"\n所有分析结果已保存到: {network_dir}/")
    print("\n分析完成！")
