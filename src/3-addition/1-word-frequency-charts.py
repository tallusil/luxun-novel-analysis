"""
鲁迅小说词频可视化生成器 (Lu Xun Novel Word Frequency Visualizer)
功能：基于词频统计数据生成柱状图和饼图，展示高频词分布
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
import seaborn as sns
import numpy as np
from pypinyin import lazy_pinyin, Style

# 获取脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))

# 设置路径
word_freq_dir = os.path.join(
    script_dir, "..", "..", "output", "word_frequency"
)  # 词频数据目录
output_dir = os.path.join(
    script_dir, "..", "..", "output", "visualization"
)  # 可视化输出目录
font_dir = os.path.join(script_dir, "..", "..", "font")  # 字体目录

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)
word_freq_chart_dir = os.path.join(output_dir, "word_frequency_charts")
os.makedirs(word_freq_chart_dir, exist_ok=True)


# 设置中文字体
def setup_chinese_font():
    """
    设置中文字体显示
    """
    # 尝试使用系统自带的中文字体
    font_candidates = [
        "SimHei",  # Windows
        "Arial Unicode MS",  # macOS
        "PingFang SC",  # macOS
        "Hiragino Sans GB",  # macOS
        "WenQuanYi Micro Hei",  # Linux
        "DejaVu Sans",  # 备用
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
            # 测试字体是否可用
            plt.text(0, 0, "测试", fontfamily=font_name)
            plt.close()
            print(f"使用系统字体: {font_name}")
            return
        except:
            continue

    print("警告: 未找到合适的中文字体，可能无法正确显示中文")


# 设置绘图样式
def setup_plot_style():
    """
    设置matplotlib绘图样式
    """
    # 设置字体
    setup_chinese_font()

    # 设置其他样式
    rcParams["font.size"] = 12
    rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
    rcParams["figure.figsize"] = (12, 8)
    rcParams["figure.dpi"] = 300
    rcParams["savefig.dpi"] = 300
    rcParams["savefig.bbox"] = "tight"

    # 设置seaborn样式
    sns.set_style("whitegrid")
    sns.set_palette("husl")


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


def create_bar_chart(word_freq_data, work_name, top_n=20):
    """
    创建词频柱状图
    参数：
        word_freq_data: 词频数据DataFrame
        work_name: 作品名称
        top_n: 显示前N个高频词
    返回：
        figure: matplotlib图形对象
    """
    # 取前N个高频词
    top_words = word_freq_data.head(top_n)

    # 创建图形
    fig, ax = plt.subplots(figsize=(15, 8))

    # 创建柱状图
    bars = ax.bar(
        range(len(top_words)),
        top_words["count"],
        color=plt.cm.viridis(np.linspace(0, 1, len(top_words))),
    )

    # 设置x轴标签
    ax.set_xticks(range(len(top_words)))
    pinyin_labels = [convert_to_pinyin(word) for word in top_words["word"]]
    ax.set_xticklabels(pinyin_labels, rotation=45, ha="right")

    # 设置标题和标签
    work_name_pinyin = convert_to_pinyin(work_name)
    ax.set_title(
        f"{work_name_pinyin} Word Frequency (Top {top_n})",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Words (Pinyin)", fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)

    # 在柱子上添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # 添加网格
    ax.grid(True, alpha=0.3)

    # 设置y轴从0开始
    ax.set_ylim(0, max(top_words["count"]) * 1.1)

    plt.tight_layout()
    return fig


def create_pie_chart(word_freq_data, work_name, top_n=10):
    """
    创建词频饼图
    参数：
        word_freq_data: 词频数据DataFrame
        work_name: 作品名称
        top_n: 显示前N个高频词
    返回：
        figure: matplotlib图形对象
    """
    # 取前N个高频词
    top_words = word_freq_data.head(top_n)
    other_count = (
        word_freq_data.iloc[top_n:]["count"].sum() if len(word_freq_data) > top_n else 0
    )

    # 准备数据
    labels = [convert_to_pinyin(word) for word in top_words["word"]]
    sizes = list(top_words["count"])

    if other_count > 0:
        labels.append("Others")
        sizes.append(other_count)

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10))

    # 设置颜色
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

    # 创建饼图
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        textprops={"fontsize": 10},
    )

    # 设置标题
    work_name_pinyin = convert_to_pinyin(work_name)
    ax.set_title(
        f"{work_name_pinyin} Word Distribution (Top {top_n})",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # 确保饼图是圆形
    ax.axis("equal")

    # 调整文本样式
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")

    plt.tight_layout()
    return fig


def create_combined_chart(word_freq_data, work_name, top_n=15):
    """
    创建组合图表（柱状图+词云样式布局）
    参数：
        word_freq_data: 词频数据DataFrame
        work_name: 作品名称
        top_n: 显示前N个高频词
    返回：
        figure: matplotlib图形对象
    """
    top_words = word_freq_data.head(top_n)
    work_name_pinyin = convert_to_pinyin(work_name)

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    # 上半部分：水平柱状图
    bars = ax1.barh(
        range(len(top_words)),
        top_words["count"],
        color=plt.cm.plasma(np.linspace(0.2, 0.8, len(top_words))),
    )

    ax1.set_yticks(range(len(top_words)))
    pinyin_labels = [convert_to_pinyin(word) for word in top_words["word"]]
    ax1.set_yticklabels(pinyin_labels)
    ax1.set_xlabel("Frequency", fontsize=12)
    ax1.set_title(f"{work_name_pinyin} Word Frequency Ranking", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # 在柱子上添加数值
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(
            width + max(top_words["count"]) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width)}",
            ha="left",
            va="center",
            fontsize=10,
        )

    # 下半部分：词频分布面积图
    ax2.fill_between(
        range(len(top_words)),
        top_words["count"],
        alpha=0.7,
        color="skyblue",
        label="Frequency",
    )
    ax2.plot(
        range(len(top_words)),
        top_words["count"],
        "o-",
        color="darkblue",
        linewidth=2,
        markersize=6,
    )

    ax2.set_xticks(range(len(top_words)))
    ax2.set_xticklabels(pinyin_labels, rotation=45, ha="right")
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.set_title(f"{work_name_pinyin} Word Frequency Trend", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    return fig


def generate_summary_chart(all_works_data):
    """
    生成所有作品的词频对比图
    参数：
        all_works_data: 所有作品的词频数据字典
    返回：
        figure: matplotlib图形对象
    """
    # 统计每部作品的总词数和独特词数
    summary_data = []
    for work_name, data in all_works_data.items():
        total_words = data["count"].sum()
        unique_words = len(data)
        avg_freq = data["count"].mean()
        max_freq = data["count"].max()
        top_word = data.iloc[0]["word"]
        top_word_pinyin = convert_to_pinyin(top_word)

        summary_data.append(
            {
                "work": work_name,
                "work_pinyin": convert_to_pinyin(work_name),
                "total_words": total_words,
                "unique_words": unique_words,
                "avg_freq": avg_freq,
                "max_freq": max_freq,
                "top_word": top_word_pinyin,
            }
        )

    summary_df = pd.DataFrame(summary_data)

    # 创建组合图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))

    # 总词数对比
    bars1 = ax1.bar(
        summary_df["work_pinyin"],
        summary_df["total_words"],
        color=plt.cm.Blues(np.linspace(0.4, 1, len(summary_df))),
    )
    ax1.set_title("Total Words Comparison", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Total Words")
    ax1.tick_params(axis="x", rotation=45)

    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 独特词数对比
    bars2 = ax2.bar(
        summary_df["work_pinyin"],
        summary_df["unique_words"],
        color=plt.cm.Greens(np.linspace(0.4, 1, len(summary_df))),
    )
    ax2.set_title("Unique Words Comparison", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Unique Words")
    ax2.tick_params(axis="x", rotation=45)

    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 平均词频对比
    bars3 = ax3.bar(
        summary_df["work_pinyin"],
        summary_df["avg_freq"],
        color=plt.cm.Oranges(np.linspace(0.4, 1, len(summary_df))),
    )
    ax3.set_title("Average Word Frequency Comparison", fontsize=14, fontweight="bold")
    ax3.set_ylabel("Average Frequency")
    ax3.tick_params(axis="x", rotation=45)

    for bar in bars3:
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 最高频词对比
    bars4 = ax4.bar(
        summary_df["work_pinyin"],
        summary_df["max_freq"],
        color=plt.cm.Reds(np.linspace(0.4, 1, len(summary_df))),
    )
    ax4.set_title("Highest Word Frequency Comparison", fontsize=14, fontweight="bold")
    ax4.set_ylabel("Highest Frequency")
    ax4.tick_params(axis="x", rotation=45)

    for i, bar in enumerate(bars4):
        height = bar.get_height()
        top_word = summary_df.iloc[i]["top_word"]
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}\n({top_word})",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.suptitle("Lu Xun Works Word Frequency Analysis", fontsize=18, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


# 设置绘图样式
setup_plot_style()

# 读取所有词频数据
print("正在加载词频数据...")
all_works_data = {}

for filename in os.listdir(word_freq_dir):
    if filename.endswith(".csv"):
        work_name = filename[:-4]  # 去掉.csv扩展名
        file_path = os.path.join(word_freq_dir, filename)

        try:
            df = pd.read_csv(file_path, encoding="utf-8")
            all_works_data[work_name] = df
            print(f"已加载: {work_name} ({len(df)}个词)")
        except Exception as e:
            print(f"读取{filename}时出错: {e}")

print(f"共加载了{len(all_works_data)}部作品的词频数据")

# 为每部作品生成可视化图表
print("\nGenerating word frequency charts...")

for work_name, word_freq_data in all_works_data.items():
    print(f"\nGenerating charts for {convert_to_pinyin(work_name)}...")

    # 生成柱状图
    try:
        bar_fig = create_bar_chart(word_freq_data, work_name, top_n=20)
        bar_path = os.path.join(word_freq_chart_dir, f"{work_name}_词频柱状图.png")
        bar_fig.savefig(bar_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(bar_fig)
        print(f"  ✓ Bar chart saved: {bar_path}")
    except Exception as e:
        print(f"  ✗ Failed to generate bar chart: {e}")

    # 生成饼图
    try:
        pie_fig = create_pie_chart(word_freq_data, work_name, top_n=10)
        pie_path = os.path.join(word_freq_chart_dir, f"{work_name}_词频饼图.png")
        pie_fig.savefig(pie_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(pie_fig)
        print(f"  ✓ Pie chart saved: {pie_path}")
    except Exception as e:
        print(f"  ✗ Failed to generate pie chart: {e}")

    # 生成组合图表
    try:
        combined_fig = create_combined_chart(word_freq_data, work_name, top_n=15)
        combined_path = os.path.join(word_freq_chart_dir, f"{work_name}_词频组合图.png")
        combined_fig.savefig(
            combined_path, dpi=300, bbox_inches="tight", facecolor="white"
        )
        plt.close(combined_fig)
        print(f"  ✓ Combined chart saved: {combined_path}")
    except Exception as e:
        print(f"  ✗ Failed to generate combined chart: {e}")

# 生成总体对比图表
print("\nGenerating overall comparison chart...")
try:
    summary_fig = generate_summary_chart(all_works_data)
    summary_path = os.path.join(word_freq_chart_dir, "鲁迅作品词频对比分析.png")
    summary_fig.savefig(summary_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(summary_fig)
    print(f"✓ Comparison chart saved: {summary_path}")
except Exception as e:
    print(f"✗ Failed to generate comparison chart: {e}")

print(f"\nWord frequency visualization completed!")
print(f"All charts have been saved to: {word_freq_chart_dir}")
print(f"Total charts generated: {len(all_works_data) * 3 + 1}")
