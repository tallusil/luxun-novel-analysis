"""
鲁迅小说情感变化趋势分析器
功能：分析文本中情感随段落位置的变化趋势，支持整体情感趋势和人物情感趋势分析
"""

import os
import json
import jieba
import re
import pandas as pd
import numpy as np
from collections import defaultdict

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

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)


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


def split_text_into_segments(text, segment_size=500):
    """
    将文本按段落或固定长度分割成片段
    参数：
        text: 原文本
        segment_size: 每个片段的大约字符数
    返回：
        segments: 文本片段列表
    """
    # 首先按双换行符分割段落
    paragraphs = re.split(r"\n\s*\n", text.strip())

    segments = []
    current_segment = ""

    for paragraph in paragraphs:
        # 清理段落，去除多余空白
        paragraph = re.sub(r"\s+", " ", paragraph.strip())
        if not paragraph:
            continue

        # 如果当前片段加上新段落不超过长度限制，则合并
        if len(current_segment) + len(paragraph) <= segment_size:
            if current_segment:
                current_segment += "\n" + paragraph
            else:
                current_segment = paragraph
        else:
            # 如果当前片段不为空，先保存
            if current_segment:
                segments.append(current_segment)

            # 如果单个段落就超过长度限制，需要进一步分割
            if len(paragraph) > segment_size:
                # 按句号分割长段落
                sentences = re.split("[。！？]", paragraph)
                temp_segment = ""
                for sentence in sentences:
                    if sentence.strip():
                        sentence = sentence.strip()
                        if len(temp_segment) + len(sentence) <= segment_size:
                            if temp_segment:
                                temp_segment += "。" + sentence
                            else:
                                temp_segment = sentence
                        else:
                            if temp_segment:
                                segments.append(temp_segment + "。")
                            temp_segment = sentence
                if temp_segment:
                    current_segment = temp_segment + "。"
                else:
                    current_segment = ""
            else:
                current_segment = paragraph

    # 保存最后一个片段
    if current_segment:
        segments.append(current_segment)

    return segments


def analyze_segment_sentiment(segment, sentiment_dict):
    """
    分析单个文本片段的情感
    参数：
        segment: 文本片段
        sentiment_dict: 情感词典
    返回：
        sentiment_score: 情感分数
        sentiment_words: 检测到的情感词列表
    """
    # 对文本片段进行分词
    words = jieba.lcut(segment)

    sentiment_scores = []
    sentiment_words = []

    # 分析每个词的情感
    for word in words:
        if word in sentiment_dict:
            score = calculate_sentiment_score(sentiment_dict[word])
            sentiment_scores.append(score)
            sentiment_words.append(
                {
                    "word": word,
                    "score": score,
                    "type": sentiment_dict[word].get("type", "unknown"),
                }
            )

    # 计算平均情感分数
    if sentiment_scores:
        avg_score = sum(sentiment_scores) / len(sentiment_scores)
    else:
        avg_score = 5.0  # 中性分数

    return avg_score, sentiment_words


def analyze_character_sentiment_trend(
    text, character, sentiment_dict, segment_size=500
):
    """
    分析特定人物在文本中的情感变化趋势
    参数：
        text: 文本内容
        character: 目标人物名称
        sentiment_dict: 情感词典
        segment_size: 片段大小
    返回：
        trend_data: 情感变化趋势数据
    """
    segments = split_text_into_segments(text, segment_size)

    trend_data = []
    character_segments = []

    # 找出包含目标人物的片段
    for i, segment in enumerate(segments):
        if character in segment:
            character_segments.append((i, segment))

    # 分析包含该人物的片段的情感
    for position, segment in character_segments:
        sentiment_score, sentiment_words = analyze_segment_sentiment(
            segment, sentiment_dict
        )

        # 计算相对位置（百分比）
        relative_position = (position + 1) / len(segments) * 100

        trend_data.append(
            {
                "character": character,
                "position": position + 1,
                "relative_position": relative_position,
                "sentiment_score": sentiment_score,
                "sentiment_words_count": len(sentiment_words),
                "segment_preview": (
                    segment[:100] + "..." if len(segment) > 100 else segment
                ),
            }
        )

    return trend_data


def analyze_overall_sentiment_trend(text, sentiment_dict, segment_size=500):
    """
    分析文本的整体情感变化趋势
    参数：
        text: 文本内容
        sentiment_dict: 情感词典
        segment_size: 片段大小
    返回：
        trend_data: 整体情感变化趋势数据
    """
    segments = split_text_into_segments(text, segment_size)

    trend_data = []

    for i, segment in enumerate(segments):
        sentiment_score, sentiment_words = analyze_segment_sentiment(
            segment, sentiment_dict
        )

        # 计算相对位置（百分比）
        relative_position = (i + 1) / len(segments) * 100

        trend_data.append(
            {
                "position": i + 1,
                "relative_position": relative_position,
                "sentiment_score": sentiment_score,
                "sentiment_words_count": len(sentiment_words),
                "segment_length": len(segment),
                "segment_preview": (
                    segment[:100] + "..." if len(segment) > 100 else segment
                ),
            }
        )

    return trend_data


def smooth_trend_data(trend_data, window_size=3):
    """
    对趋势数据进行平滑处理
    参数：
        trend_data: 原始趋势数据
        window_size: 滑动窗口大小
    返回：
        smoothed_data: 平滑后的数据
    """
    if len(trend_data) < window_size:
        return trend_data

    smoothed_data = []
    scores = [item["sentiment_score"] for item in trend_data]

    # 使用简单的移动平均进行平滑
    for i in range(len(trend_data)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(scores), i + window_size // 2 + 1)

        smoothed_score = sum(scores[start_idx:end_idx]) / (end_idx - start_idx)

        smoothed_item = trend_data[i].copy()
        smoothed_item["smoothed_sentiment_score"] = smoothed_score
        smoothed_data.append(smoothed_item)

    return smoothed_data


# 加载情感词典和人物数据
print("正在加载情感词典...")
sentiment_dict = load_sentiment_dict()
print(f"情感词典加载完成，共{len(sentiment_dict)}个词语")

print("正在加载人物分析结果...")
with open(character_analysis_path, "r", encoding="utf-8") as f:
    character_data = json.load(f)

# 存储所有趋势分析结果
all_results = {}
csv_data = []

# 遍历每部作品
for work_name, data in character_data.items():
    print(f"正在分析作品: {work_name}")

    # 读取作品文本
    file_path = os.path.join(input_dir, f"{work_name}.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 分析整体情感趋势
    print(f"  - 分析整体情感趋势...")
    overall_trend = analyze_overall_sentiment_trend(content, sentiment_dict)
    overall_trend_smoothed = smooth_trend_data(overall_trend)

    # 获取主要人物（前5个）
    top_characters = list(data["characters"].items())[:5]
    character_trends = {}

    # 分析主要人物的情感趋势
    for character, count in top_characters:
        print(f"  - 分析人物 '{character}' 的情感趋势...")
        character_trend = analyze_character_sentiment_trend(
            content, character, sentiment_dict
        )
        if character_trend:  # 只保存有数据的人物
            character_trend_smoothed = smooth_trend_data(character_trend)
            character_trends[character] = {
                "raw_trend": character_trend,
                "smoothed_trend": character_trend_smoothed,
                "character_count": count,
            }

            # 为CSV输出准备数据
            for item in character_trend_smoothed:
                csv_data.append(
                    {
                        "work": work_name,
                        "character": character,
                        "position": item["position"],
                        "relative_position": item["relative_position"],
                        "sentiment_score": item["sentiment_score"],
                        "smoothed_sentiment_score": item.get(
                            "smoothed_sentiment_score", item["sentiment_score"]
                        ),
                        "sentiment_words_count": item["sentiment_words_count"],
                    }
                )

    # 为整体趋势准备CSV数据
    for item in overall_trend_smoothed:
        csv_data.append(
            {
                "work": work_name,
                "character": "OVERALL",  # 标记为整体分析
                "position": item["position"],
                "relative_position": item["relative_position"],
                "sentiment_score": item["sentiment_score"],
                "smoothed_sentiment_score": item.get(
                    "smoothed_sentiment_score", item["sentiment_score"]
                ),
                "sentiment_words_count": item["sentiment_words_count"],
            }
        )

    # 存储当前作品的分析结果
    all_results[work_name] = {
        "overall_trend": {
            "raw_trend": overall_trend,
            "smoothed_trend": overall_trend_smoothed,
        },
        "character_trends": character_trends,
        "statistics": {
            "total_segments": len(overall_trend),
            "analyzed_characters": len(character_trends),
            "avg_sentiment_score": sum(
                item["sentiment_score"] for item in overall_trend
            )
            / len(overall_trend),
            "sentiment_volatility": np.std(
                [item["sentiment_score"] for item in overall_trend]
            ),
        },
    }

# 保存JSON格式的详细结果
output_path = os.path.join(output_dir, "sentiment_trend_analysis.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

# 保存CSV格式的结果（用于可视化）
csv_df = pd.DataFrame(csv_data)
csv_output_path = os.path.join(output_dir, "sentiment_trend_analysis.csv")
csv_df.to_csv(csv_output_path, index=False, encoding="utf-8")

print(f"\n情感变化趋势分析完成！")
print(f"详细结果已保存到: {output_path}")
print(f"CSV数据已保存到: {csv_output_path}")

# 输出简要统计信息
print(f"\n分析统计:")
print(f"- 分析作品数量: {len(all_results)}")
print(f"- 生成数据点: {len(csv_data)}")

for work_name, result in all_results.items():
    stats = result["statistics"]
    print(
        f"- {work_name}: 片段{stats['total_segments']}个, 人物{stats['analyzed_characters']}个, "
        f"平均情感{stats['avg_sentiment_score']:.2f}, 波动度{stats['sentiment_volatility']:.2f}"
    )
