"""
鲁迅小说情感变化趋势可视化生成器 (Lu Xun Novel Sentiment Trend Visualizer)
功能：基于情感趋势分析数据生成折线图，展示情感随文本进展的变化
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
import seaborn as sns
import numpy as np
from scipy.interpolate import make_interp_spline
from pypinyin import lazy_pinyin, Style

# 获取脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))

# 设置路径
sentiment_data_path = os.path.join(
    script_dir, "..", "..", "output", "passage_analysis", "sentiment_trend_analysis.csv"
)
output_dir = os.path.join(script_dir, "..", "..", "output", "visualization")
font_dir = os.path.join(script_dir, "..", "..", "font")

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)
sentiment_chart_dir = os.path.join(output_dir, "sentiment_trend_charts")
os.makedirs(sentiment_chart_dir, exist_ok=True)


# 设置中文字体
def setup_chinese_font():
    """
    设置中文字体显示
    """
    font_candidates = [
        "SimHei",
        "Arial Unicode MS",
        "PingFang SC",
        "Hiragino Sans GB",
        "WenQuanYi Micro Hei",
        "DejaVu Sans",
    ]

    # 检查项目字体目录
    if os.path.exists(font_dir):
        for font_file in os.listdir(font_dir):
            if font_file.endswith((".ttf", ".ttc", ".otf")):
                font_path = os.path.join(font_dir, font_file)
                try:
                    font_prop = fm.FontProperties(fname=font_path)
                    rcParams["font.family"] = font_prop.get_name()
                    print(f"使用项目字体: {font_file}")
                    return
                except:
                    continue

    # 尝试系统字体
    for font_name in font_candidates:
        try:
            rcParams["font.family"] = font_name
            plt.text(0, 0, "测试", fontfamily=font_name)
            plt.close()
            print(f"使用系统字体: {font_name}")
            return
        except:
            continue

    print("警告: 未找到合适的中文字体，可能无法正确显示中文")


def setup_plot_style():
    """
    设置matplotlib绘图样式
    """
    setup_chinese_font()
    rcParams["font.size"] = 12
    rcParams["axes.unicode_minus"] = False
    rcParams["figure.figsize"] = (15, 8)
    rcParams["figure.dpi"] = 300
    rcParams["savefig.dpi"] = 300
    rcParams["savefig.bbox"] = "tight"
    sns.set_style("whitegrid")


def convert_to_pinyin(text):
    """
    将中文文本转换为拼音
    参数：
        text: 中文文本
    返回：
        拼音字符串
    """
    if not isinstance(text, str):
        return str(text)
    pinyin_list = lazy_pinyin(text, style=Style.NORMAL)
    return ' '.join(pinyin_list).title()


def create_overall_sentiment_trend(work_data, work_name):
    """
    创建单部作品的整体情感变化趋势图
    参数：
        work_data: 单部作品的情感数据
        work_name: 作品名称
    返回：
        figure: matplotlib图形对象
    """
    # 筛选整体数据
    overall_data = work_data[work_data["character"] == "OVERALL"].copy()
    overall_data = overall_data.sort_values("relative_position")

    if len(overall_data) == 0:
        return None

    fig, ax = plt.subplots(figsize=(15, 8))

    # 绘制原始数据点
    ax.scatter(
        overall_data["relative_position"],
        overall_data["sentiment_score"],
        alpha=0.6,
        s=30,
        color="lightblue",
        label="Original Sentiment Score",
    )

    # 绘制平滑曲线
    ax.plot(
        overall_data["relative_position"],
        overall_data["smoothed_sentiment_score"],
        color="darkblue",
        linewidth=3,
        label="Smoothed Sentiment Trend",
    )

    # 添加情感区域标记
    ax.axhline(y=5, color="gray", linestyle="--", alpha=0.7, label="Neutral Line (5)")
    ax.fill_between(
        overall_data["relative_position"],
        1,
        4,
        alpha=0.1,
        color="red",
        label="Negative Area",
    )
    ax.fill_between(
        overall_data["relative_position"],
        6,
        9,
        alpha=0.1,
        color="green",
        label="Positive Area",
    )

    # 设置标题和标签
    work_name_pinyin = convert_to_pinyin(work_name)
    ax.set_title(
        f"{work_name_pinyin} Overall Sentiment Trend",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Text Progress (%)", fontsize=14)
    ax.set_ylabel("Sentiment Score (1-9)", fontsize=14)

    # 设置坐标轴范围
    ax.set_xlim(0, 100)
    ax.set_ylim(1, 9)

    # 添加网格和图例
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # 添加统计信息
    avg_sentiment = overall_data["sentiment_score"].mean()
    volatility = overall_data["sentiment_score"].std()
    textstr = f"Average Sentiment: {avg_sentiment:.2f}\nSentiment Volatility: {volatility:.2f}"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=props,
    )

    plt.tight_layout()
    return fig


def create_character_sentiment_comparison(work_data, work_name, max_characters=5):
    """
    创建主要人物情感变化对比图
    参数：
        work_data: 单部作品的情感数据
        work_name: 作品名称
        max_characters: 最多显示的人物数量
    返回：
        figure: matplotlib图形对象
    """
    # 获取主要人物（除了OVERALL）
    characters = work_data[work_data["character"] != "OVERALL"]["character"].unique()

    if len(characters) == 0:
        return None

    # 限制人物数量
    characters = characters[:max_characters]

    fig, ax = plt.subplots(figsize=(15, 10))

    # 为每个人物绘制趋势线
    colors = plt.cm.tab10(np.linspace(0, 1, len(characters)))

    for i, character in enumerate(characters):
        char_data = work_data[work_data["character"] == character].copy()
        char_data = char_data.sort_values("relative_position")

        if len(char_data) < 2:
            continue

        char_pinyin = convert_to_pinyin(character)
        # 绘制散点和趋势线
        ax.scatter(
            char_data["relative_position"],
            char_data["sentiment_score"],
            alpha=0.6,
            s=25,
            color=colors[i],
        )
        ax.plot(
            char_data["relative_position"],
            char_data["smoothed_sentiment_score"],
            color=colors[i],
            linewidth=2.5,
            label=f"{char_pinyin}",
            marker="o",
            markersize=4,
        )

    # 添加中性线
    ax.axhline(y=5, color="gray", linestyle="--", alpha=0.7, label="Neutral Line")

    # 设置标题和标签
    work_name_pinyin = convert_to_pinyin(work_name)
    ax.set_title(
        f"{work_name_pinyin} Character Sentiment Comparison",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Text Progress (%)", fontsize=14)
    ax.set_ylabel("Sentiment Score (1-9)", fontsize=14)

    # 设置坐标轴范围
    ax.set_xlim(0, 100)
    ax.set_ylim(1, 9)

    # 添加网格和图例
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    return fig


def create_multi_work_comparison(all_data, selected_works=None):
    """
    创建多部作品的整体情感对比图
    参数：
        all_data: 所有作品的情感数据
        selected_works: 选择对比的作品列表，None表示全部
    返回：
        figure: matplotlib图形对象
    """
    works = all_data["work"].unique()
    if selected_works:
        works = [w for w in works if w in selected_works]

    # 限制作品数量以保证可读性
    works = works[:8]

    fig, ax = plt.subplots(figsize=(18, 10))

    colors = plt.cm.tab20(np.linspace(0, 1, len(works)))

    for i, work in enumerate(works):
        work_data = all_data[
            (all_data["work"] == work) & (all_data["character"] == "OVERALL")
        ].copy()
        work_data = work_data.sort_values("relative_position")

        if len(work_data) < 2:
            continue

        work_pinyin = convert_to_pinyin(work)
        # 绘制趋势线
        ax.plot(
            work_data["relative_position"],
            work_data["smoothed_sentiment_score"],
            color=colors[i],
            linewidth=2.5,
            label=f"{work_pinyin}",
            alpha=0.8,
        )

    # 添加中性线和区域
    ax.axhline(y=5, color="gray", linestyle="--", alpha=0.7, label="Neutral Line")
    ax.fill_between([0, 100], 1, 4, alpha=0.05, color="red", label="Negative Area")
    ax.fill_between([0, 100], 6, 9, alpha=0.05, color="green", label="Positive Area")

    # 设置标题和标签
    ax.set_title("Lu Xun Works Sentiment Trend Comparison", fontsize=18, fontweight="bold", pad=20)
    ax.set_xlabel("Text Progress (%)", fontsize=14)
    ax.set_ylabel("Sentiment Score (1-9)", fontsize=14)

    # 设置坐标轴范围
    ax.set_xlim(0, 100)
    ax.set_ylim(1, 9)

    # 添加网格和图例
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=1)

    plt.tight_layout()
    return fig


def create_sentiment_statistics_chart(all_data):
    """
    创建情感统计分析图表
    参数：
        all_data: 所有作品的情感数据
    返回：
        figure: matplotlib图形对象
    """
    # 计算每部作品的统计信息
    works = all_data["work"].unique()
    stats_data = []

    for work in works:
        overall_data = all_data[
            (all_data["work"] == work) & (all_data["character"] == "OVERALL")
        ]

        if len(overall_data) > 0:
            avg_sentiment = overall_data["sentiment_score"].mean()
            sentiment_std = overall_data["sentiment_score"].std()
            max_sentiment = overall_data["sentiment_score"].max()
            min_sentiment = overall_data["sentiment_score"].min()

            stats_data.append(
                {
                    "work": work,
                    "work_pinyin": convert_to_pinyin(work),
                    "avg_sentiment": avg_sentiment,
                    "sentiment_volatility": sentiment_std,
                    "max_sentiment": max_sentiment,
                    "min_sentiment": min_sentiment,
                    "sentiment_range": max_sentiment - min_sentiment,
                }
            )

    stats_df = pd.DataFrame(stats_data)

    # 创建组合图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

    # 平均情感分数
    bars1 = ax1.bar(
        stats_df["work_pinyin"],
        stats_df["avg_sentiment"],
        color=plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(stats_df))),
    )
    ax1.axhline(y=5, color="gray", linestyle="--", alpha=0.7)
    ax1.set_title("Average Sentiment Score by Work", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Average Sentiment Score")
    ax1.tick_params(axis="x", rotation=45)
    ax1.set_ylim(1, 9)

    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.05,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 情感波动度
    bars2 = ax2.bar(
        stats_df["work_pinyin"],
        stats_df["sentiment_volatility"],
        color=plt.cm.Reds(np.linspace(0.3, 0.9, len(stats_df))),
    )
    ax2.set_title("Sentiment Volatility by Work", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Sentiment Volatility (Std)")
    ax2.tick_params(axis="x", rotation=45)

    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 情感范围
    bars3 = ax3.bar(
        stats_df["work_pinyin"],
        stats_df["sentiment_range"],
        color=plt.cm.Greens(np.linspace(0.3, 0.9, len(stats_df))),
    )
    ax3.set_title("Sentiment Range by Work", fontsize=14, fontweight="bold")
    ax3.set_ylabel("Sentiment Range")
    ax3.tick_params(axis="x", rotation=45)

    for bar in bars3:
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.05,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 箱线图显示情感分布
    sentiment_by_work = []
    work_labels = []
    for work in works:
        overall_data = all_data[
            (all_data["work"] == work) & (all_data["character"] == "OVERALL")
        ]
        if len(overall_data) > 0:
            sentiment_by_work.append(overall_data["sentiment_score"].values)
            work_labels.append(convert_to_pinyin(work))

    bp = ax4.boxplot(sentiment_by_work, labels=work_labels, patch_artist=True)
    ax4.axhline(y=5, color="gray", linestyle="--", alpha=0.7)
    ax4.set_title("Sentiment Score Distribution by Work", fontsize=14, fontweight="bold")
    ax4.set_ylabel("Sentiment Score")
    ax4.tick_params(axis="x", rotation=45)
    ax4.set_ylim(1, 9)

    # 为箱线图添加颜色
    colors = plt.cm.tab20(np.linspace(0, 1, len(bp["boxes"])))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.suptitle("Lu Xun Works Sentiment Statistics Overview", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


# 设置绘图样式
setup_plot_style()

# 读取情感趋势数据
print("正在加载情感趋势数据...")
try:
    sentiment_data = pd.read_csv(sentiment_data_path, encoding="utf-8")
    print(f"成功加载数据，共{len(sentiment_data)}条记录")
except Exception as e:
    print(f"读取数据失败: {e}")
    exit(1)

# 获取所有作品名称
works = sentiment_data["work"].unique()
print(f"发现{len(works)}部作品: {', '.join(works)}")

# 为每部作品生成情感趋势图
print("\nGenerating sentiment trend charts...")

for work_name in works:
    print(f"\nGenerating charts for {convert_to_pinyin(work_name)}...")
    work_data = sentiment_data[sentiment_data["work"] == work_name]

    # 生成整体情感趋势图
    try:
        overall_fig = create_overall_sentiment_trend(work_data, work_name)
        if overall_fig:
            overall_path = os.path.join(
                sentiment_chart_dir, f"{work_name}_整体情感趋势.png"
            )
            overall_fig.savefig(
                overall_path, dpi=300, bbox_inches="tight", facecolor="white"
            )
            plt.close(overall_fig)
            print(f"  ✓ Overall trend chart saved: {overall_path}")
    except Exception as e:
        print(f"  ✗ Failed to generate overall trend chart: {e}")

    # 生成人物对比图
    try:
        char_fig = create_character_sentiment_comparison(work_data, work_name)
        if char_fig:
            char_path = os.path.join(
                sentiment_chart_dir, f"{work_name}_人物情感对比.png"
            )
            char_fig.savefig(char_path, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(char_fig)
            print(f"  ✓ Character comparison chart saved: {char_path}")
    except Exception as e:
        print(f"  ✗ Failed to generate character comparison chart: {e}")

# 生成跨作品对比图表
print("\nGenerating cross-work comparison charts...")

# 作品对比图
try:
    comparison_fig = create_multi_work_comparison(sentiment_data)
    comparison_path = os.path.join(sentiment_chart_dir, "鲁迅作品情感变化对比.png")
    comparison_fig.savefig(
        comparison_path, dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.close(comparison_fig)
    print(f"✓ Works comparison chart saved: {comparison_path}")
except Exception as e:
    print(f"✗ Failed to generate works comparison chart: {e}")

# 生成统计分析图
try:
    stats_fig = create_sentiment_statistics_chart(sentiment_data)
    stats_path = os.path.join(sentiment_chart_dir, "鲁迅作品情感统计分析.png")
    stats_fig.savefig(stats_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(stats_fig)
    print(f"✓ Statistics analysis chart saved: {stats_path}")
except Exception as e:
    print(f"✗ Failed to generate statistics analysis chart: {e}")

print(f"\nSentiment trend visualization completed!")
print(f"All charts have been saved to: {sentiment_chart_dir}")
total_charts = len(works) * 2 + 2  # 每部作品2个图 + 2个对比图
print(f"Total charts generated: {total_charts}")
