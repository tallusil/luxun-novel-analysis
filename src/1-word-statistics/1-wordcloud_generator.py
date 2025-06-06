"""
鲁迅小说词云生成器
功能：基于词频统计结果生成词云图
"""

import os
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 获取脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))

# 设置各种路径
output_dir = os.path.join(script_dir, "..", "..", "output")  # 输出根目录
font_dir = os.path.join(script_dir, "..", "..", "font")  # 字体文件目录
word_frequency_dir = os.path.join(output_dir, "word_frequency")  # 词频数据目录
word_cloud_dir = os.path.join(output_dir, "word_cloud")  # 词云输出目录

# 确保词云输出目录存在
os.makedirs(word_cloud_dir, exist_ok=True)


def generate_wordcloud_from_csv(csv_path, filename):
    """
    根据CSV词频文件生成词云图
    参数：
        csv_path: 词频CSV文件路径
        filename: 输出文件名（不含扩展名）
    """
    # 读取词频数据
    df = pd.read_csv(csv_path, encoding="utf-8")

    # 将词频数据转换为字典格式（WordCloud需要的格式）
    word_freq = dict(zip(df["word"], df["count"]))

    # 创建WordCloud对象并配置参数
    wordcloud = WordCloud(
        font_path=os.path.join(
            font_dir, "SourceHanSerifCN-Regular-Min.otf"
        ),  # 中文字体路径
        width=800,  # 图片宽度
        height=400,  # 图片高度
        background_color="white",  # 背景颜色
    ).generate_from_frequencies(
        word_freq
    )  # 基于词频字典生成词云

    # 创建图形并显示词云
    plt.figure(figsize=(10, 5), dpi=300)  # 设置图形大小和分辨率
    plt.imshow(wordcloud, interpolation="bilinear")  # 显示词云图像
    plt.axis("off")  # 隐藏坐标轴

    # 保存词云图片
    plt.savefig(os.path.join(word_cloud_dir, f"{filename}.png"))
    plt.close()  # 关闭图形以释放内存
    print(f"词云图已生成: {filename}.png")


def main():
    """
    主函数：为所有词频文件生成对应的词云图
    """
    # 遍历词频目录中的所有CSV文件
    for filename in os.listdir(word_frequency_dir):
        if filename.endswith(".csv"):
            print(f"正在生成词云: {filename}")

            # 构建CSV文件的完整路径
            csv_path = os.path.join(word_frequency_dir, filename)

            # 获取基础文件名（去掉.csv扩展名）
            base_filename = filename.replace(".csv", "")

            # 生成词云图
            generate_wordcloud_from_csv(csv_path, base_filename)


if __name__ == "__main__":
    main()
