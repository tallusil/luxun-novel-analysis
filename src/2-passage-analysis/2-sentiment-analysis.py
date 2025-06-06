"""
鲁迅小说人物情感分析器
功能：基于情感词典分析每个人物在文本中的情感倾向
"""

import os
import json
import jieba
import re
import pandas as pd

# 获取脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))

# 设置路径
input_dir = os.path.join(script_dir, "..", "..", "data", "combined")  # 输入文本目录
output_dir = os.path.join(
    script_dir, "..", "..", "output", "passage_analysis"
)  # 输出目录
character_analysis_path = os.path.join(
    output_dir, "character_analysis.json"
)  # 人物分析结果路径


def load_sentiment_dict():
    """
    从Excel文件加载情感词典
    返回：
        sentiment_dict: 情感词典，格式为{词语: {type, intensity, polarity, ...}}
    """
    # 读取情感词典Excel文件
    df = pd.read_excel(
        os.path.join(script_dir, "..", "..", "data", "sentiment", "dict.xlsx")
    )

    sentiment_dict = {}

    # 遍历每一行，构建情感词典
    for _, row in df.iterrows():
        word = row["词语"]

        # 跳过空值
        if pd.isna(word):
            continue

        # 提取主要情感信息
        sentiment_type = row["情感分类"]  # 情感类型（如：快乐、愤怒等）
        intensity = row["强度"]  # 情感强度（1-9级）
        polarity = row["极性"]  # 情感极性（正负性）

        # 如果主要情感信息完整，则添加到词典中
        if pd.notna(sentiment_type) and pd.notna(intensity) and pd.notna(polarity):
            sentiment_dict[word] = {
                "type": sentiment_type,
                "intensity": intensity,
                "polarity": polarity,
            }

        # 提取辅助情感信息
        aux_sentiment_type = row["辅助情感分类"]
        aux_intensity = row["强度.1"]
        aux_polarity = row["极性.1"]

        # 如果辅助情感信息完整
        if (
            pd.notna(aux_sentiment_type)
            and pd.notna(aux_intensity)
            and pd.notna(aux_polarity)
        ):
            # 如果该词已有主要情感信息，则添加辅助情感
            if word in sentiment_dict:
                sentiment_dict[word]["aux_type"] = aux_sentiment_type
                sentiment_dict[word]["aux_intensity"] = aux_intensity
                sentiment_dict[word]["aux_polarity"] = aux_polarity
            else:
                # 如果没有主要情感信息，则将辅助情感作为主要情感
                sentiment_dict[word] = {
                    "type": aux_sentiment_type,
                    "intensity": aux_intensity,
                    "polarity": aux_polarity,
                }

    return sentiment_dict


def calculate_sentiment_score(sentiment_info):
    """
    计算情感分数
    参数：
        sentiment_info: 词语的情感信息字典
    返回：
        score: 加权平均情感分数
    """
    if not sentiment_info:
        return 0

    score = 0
    weight = 0

    # 计算主要情感分数
    if "type" in sentiment_info:
        intensity = sentiment_info["intensity"]  # 强度
        polarity = sentiment_info["polarity"]  # 极性
        score += intensity * polarity  # 情感分数 = 强度 × 极性
        weight += 1

    # 计算辅助情感分数（权重降低）
    if "aux_type" in sentiment_info:
        aux_intensity = sentiment_info["aux_intensity"]
        aux_polarity = sentiment_info["aux_polarity"]
        score += aux_intensity * aux_polarity * 0.5  # 辅助情感权重为0.5
        weight += 0.5

    # 返回加权平均分数
    return score / weight if weight > 0 else 0


def analyze_sentiment(text, character, sentiment_dict):
    """
    分析特定人物在文本中的情感倾向
    参数：
        text: 文本内容
        character: 目标人物名称
        sentiment_dict: 情感词典
    返回：
        average_score: 该人物的平均情感分数
    """
    # 按标点符号分割文本为句子
    sentences = re.split("[。！？]", text)
    sentiment_scores = []

    # 遍历每个句子
    for sentence in sentences:
        # 只分析包含目标人物的句子
        if character in sentence:
            # 对句子进行分词
            words = jieba.lcut(sentence)
            if not words:  # 跳过空句子
                continue

            score = 0
            sentiment_word_count = 0

            # 计算句子中所有情感词的分数
            for word in words:
                if word in sentiment_dict:
                    score += calculate_sentiment_score(sentiment_dict[word])
                    sentiment_word_count += 1

            # 只有当句子中包含情感词时才计算分数
            if sentiment_word_count > 0:
                # 归一化处理：考虑句子中情感词的数量
                normalized_score = score / sentiment_word_count
                sentiment_scores.append(normalized_score)

    if not sentiment_scores:
        return 0

    # 返回所有有效句子的平均情感分数
    return sum(sentiment_scores) / len(sentiment_scores)


# 加载情感词典
print("正在加载情感词典...")
sentiment_dict = load_sentiment_dict()
print(f"情感词典加载完成，共{len(sentiment_dict)}个词语")

# 读取人物分析结果
with open(character_analysis_path, "r", encoding="utf-8") as f:
    character_data = json.load(f)

# 存储情感分析结果
results = {}
csv_data = []

# 遍历每部作品
for work_name, data in character_data.items():
    print(f"正在分析作品: {work_name}")

    # 读取作品文本
    file_path = os.path.join(input_dir, f"{work_name}.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 获取该作品的主要人物（这里使用所有识别出的人物）
    top_characters = list(data["characters"].items())  # [:10] 可以限制为前10个人物
    character_sentiments = {}

    # 分析每个人物的情感
    for character, _ in top_characters:
        sentiment_score = analyze_sentiment(content, character, sentiment_dict)

        # 根据分数判断情感类别
        # 分数 > 6: 正面情感
        # 分数 < 4: 负面情感
        # 4 <= 分数 <= 6: 中性情感
        sentiment = (
            "positive"
            if sentiment_score > 6
            else "negative" if sentiment_score < 4 else "neutral"
        )

        # 存储人物情感分析结果
        character_sentiments[character] = {
            "sentiment_score": sentiment_score,
            "sentiment": sentiment,
        }

        # 为CSV输出准备数据
        csv_data.append(
            {"work": work_name, "character": character, "sentiment": sentiment}
        )

    # 存储当前作品的所有人物情感分析结果
    results[work_name] = character_sentiments

# 保存JSON格式的结果
output_path = os.path.join(output_dir, "sentiment_analysis.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# 保存CSV格式的结果
csv_df = pd.DataFrame(csv_data)
csv_output_path = os.path.join(output_dir, "sentiment_analysis.csv")
csv_df.to_csv(csv_output_path, index=False, encoding="utf-8")

print(f"情感分析完成，结果已保存到:")
print(f"  JSON格式: {output_path}")
print(f"  CSV格式: {csv_output_path}")
