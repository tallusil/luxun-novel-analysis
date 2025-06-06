"""
鲁迅头像词云生成器
功能：合并所有作品的词频数据，生成以鲁迅头像为模板的词云图
"""

import os
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image

# 获取脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))

# 设置各种路径
output_dir = os.path.join(script_dir, "..", "..", "output")  # 输出根目录
font_dir = os.path.join(script_dir, "..", "..", "font")  # 字体文件目录
word_frequency_dir = os.path.join(output_dir, "word_frequency")  # 词频数据目录
word_cloud_dir = os.path.join(output_dir, "word_cloud")  # 词云输出目录

# 确保词云输出目录存在
os.makedirs(word_cloud_dir, exist_ok=True)


def combine_word_frequencies():
    """
    合并所有作品的词频数据
    返回：
        combined_freq: 合并后的词频字典
    """
    combined_freq = {}

    # 遍历词频目录中的所有CSV文件
    for filename in os.listdir(word_frequency_dir):
        if filename.endswith(".csv"):
            # 读取单个作品的词频数据
            df = pd.read_csv(
                os.path.join(word_frequency_dir, filename), encoding="utf-8"
            )

            # 累加每个词的频次
            for _, row in df.iterrows():
                word = row["word"]
                count = row["count"]
                # 如果词已存在，累加频次；否则新增
                combined_freq[word] = combined_freq.get(word, 0) + count

    return combined_freq


def generate_luxun_wordcloud():
    """
    生成鲁迅头像词云图
    """
    # 获取合并后的词频数据
    word_freq = combine_word_frequencies()

    # 加载鲁迅头像遮罩图片
    mask_path = os.path.join(script_dir, "..", "..", "data", "assets", "luxun_mask.png")
    mask = np.array(Image.open(mask_path))

    # 创建词云对象，使用鲁迅头像作为遮罩
    wordcloud = WordCloud(
        font_path=os.path.join(
            font_dir, "SourceHanSerifCN-Regular-Min.otf"
        ),  # 中文字体路径
        width=1200,  # 图片宽度
        height=1600,  # 图片高度
        mask=mask,  # 使用鲁迅头像作为遮罩
        background_color="white",  # 背景颜色
        contour_width=1,  # 轮廓线宽度
        contour_color="black",  # 轮廓线颜色
        min_font_size=10,  # 最小字体大小
        max_font_size=200,  # 最大字体大小
        prefer_horizontal=0.9,  # 水平文字比例（0.9表示90%的文字是水平的）
        relative_scaling=0.5,  # 词频相对缩放因子
    ).generate_from_frequencies(
        word_freq
    )  # 基于合并的词频数据生成词云

    # 创建高分辨率图形
    plt.figure(figsize=(12, 16), dpi=300)  # 设置图形大小和分辨率
    plt.imshow(wordcloud, interpolation="bilinear")  # 显示词云图像
    plt.axis("off")  # 隐藏坐标轴

    # 保存词云图片
    output_path = os.path.join(word_cloud_dir, "鲁迅全集.png")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)  # 紧凑保存，去除边距
    plt.close()  # 关闭图形以释放内存
    print(f"鲁迅人像词云图已生成: {output_path}")


if __name__ == "__main__":
    generate_luxun_wordcloud()
