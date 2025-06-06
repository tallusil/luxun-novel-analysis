"""
鲁迅小说词频统计分析器
功能：对鲁迅小说进行中文分词并统计词频，输出词频表
"""

import os
import jieba
import pandas as pd
from collections import Counter

# 获取脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))

# 设置各种路径
file_dir = os.path.join(script_dir, "..", "..", "data", "combined")  # 输入文件目录
output_dir = os.path.join(script_dir, "..", "..", "output")  # 输出根目录
stopwords_path = os.path.join(
    script_dir, "..", "..", "data", "stopwords", "stopwords.txt"
)  # 停用词文件路径

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)
word_frequency_dir = os.path.join(output_dir, "word_frequency")  # 词频结果目录
os.makedirs(word_frequency_dir, exist_ok=True)


def load_stopwords():
    """
    加载停用词表
    返回：停用词集合
    """
    with open(stopwords_path, "r", encoding="utf-8") as f:
        stopwords = set([line.strip() for line in f])
    return stopwords


# 加载停用词
stopwords = load_stopwords()


def process_text(file_path):
    """
    处理文本文件，进行分词和过滤
    参数：
        file_path: 文本文件路径
    返回：
        words: 过滤后的词语列表
    """
    # 读取文本文件
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 使用jieba进行中文分词
    words = jieba.cut(text)

    # 词语过滤条件：
    # 1. 长度大于1个字符
    # 2. 不在停用词表中
    # 3. 不是纯标点符号
    # 4. 不是纯数字（包括中文数字）
    # 5. 不包含特殊符号
    words = [
        word
        for word in words
        if len(word) > 1  # 长度大于1
        and word not in stopwords  # 不在停用词中
        and not all(not c.isalnum() for c in word)  # 不是纯标点
        and not word.isdigit()  # 不是阿拉伯数字
        and not word.isnumeric()  # 不是数字
        and not word.isdecimal()  # 不是十进制数字
        and not all(
            c.isdigit() or c in "一二三四五六七八九十百千万亿零" for c in word
        )  # 不是中文数字
        and not any(
            c in '（）()【】[]{}《》<>「」『』""' "、，。！？；：…—" for c in word
        )  # 不包含标点符号
    ]
    return words


def count_frequent_words(words, top_n=None):
    """
    统计词频
    参数：
        words: 词语列表
        top_n: 返回前n个高频词，None表示返回全部
    返回：
        word_counts: 词频统计结果，格式为[(词语, 频次), ...]
    """
    word_counts = Counter(words)
    return word_counts.most_common(top_n)


def save_to_csv(word_counts, filename):
    """
    将词频结果保存为CSV文件
    参数：
        word_counts: 词频统计结果
        filename: 输出文件名（不含扩展名）
    """
    # 创建DataFrame
    df = pd.DataFrame(word_counts, columns=["word", "count"])

    # 保存到CSV文件
    output_path = os.path.join(word_frequency_dir, f"{filename}.csv")
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"词频统计已保存到: {output_path}")


def main():
    """
    主函数：处理所有文本文件并生成词频统计
    """
    # 遍历输入目录中的所有文本文件
    for filename in os.listdir(file_dir):
        if filename.endswith(".txt"):
            print(f"正在处理文件: {filename}")

            # 构建完整文件路径
            file_path = os.path.join(file_dir, filename)

            # 处理文本并分词
            words = process_text(file_path)

            # 统计词频
            frequent_words = count_frequent_words(words)

            # 保存结果（去掉.txt扩展名作为输出文件名）
            save_to_csv(frequent_words, filename[:-4])


if __name__ == "__main__":
    main()
