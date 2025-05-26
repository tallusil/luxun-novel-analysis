import os
import json
import jieba
import re
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(script_dir, "..", "..", "data", "combined")
output_dir = os.path.join(script_dir, "..", "..", "output", "passage_analysis")
character_analysis_path = os.path.join(output_dir, "character_analysis.json")


def load_sentiment_dict():
    df = pd.read_excel(
        os.path.join(script_dir, "..", "..", "data", "sentiment", "dict.xlsx")
    )

    sentiment_dict = {}
    for _, row in df.iterrows():
        word = row["词语"]
        if pd.isna(word):
            continue

        # 主要情感
        sentiment_type = row["情感分类"]
        intensity = row["强度"]
        polarity = row["极性"]

        if pd.notna(sentiment_type) and pd.notna(intensity) and pd.notna(polarity):
            sentiment_dict[word] = {
                "type": sentiment_type,
                "intensity": intensity,
                "polarity": polarity,
            }

        # 辅助情感
        aux_sentiment_type = row["辅助情感分类"]
        aux_intensity = row["强度.1"]
        aux_polarity = row["极性.1"]

        if (
            pd.notna(aux_sentiment_type)
            and pd.notna(aux_intensity)
            and pd.notna(aux_polarity)
        ):
            if word in sentiment_dict:
                sentiment_dict[word]["aux_type"] = aux_sentiment_type
                sentiment_dict[word]["aux_intensity"] = aux_intensity
                sentiment_dict[word]["aux_polarity"] = aux_polarity
            else:
                sentiment_dict[word] = {
                    "type": aux_sentiment_type,
                    "intensity": aux_intensity,
                    "polarity": aux_polarity,
                }

    return sentiment_dict


def calculate_sentiment_score(sentiment_info):
    if not sentiment_info:
        return 0

    score = 0
    weight = 0

    # 主要情感
    if "type" in sentiment_info:
        intensity = sentiment_info["intensity"]
        polarity = sentiment_info["polarity"]
        score += intensity * polarity
        weight += 1

    # 辅助情感
    if "aux_type" in sentiment_info:
        aux_intensity = sentiment_info["aux_intensity"]
        aux_polarity = sentiment_info["aux_polarity"]
        score += aux_intensity * aux_polarity * 0.5  # 辅助情感权重降低
        weight += 0.5

    # 如果有情感信息，返回加权平均分数
    return score / weight if weight > 0 else 0


def analyze_sentiment(text, character, sentiment_dict):
    sentences = re.split("[。！？]", text)
    sentiment_scores = []

    for sentence in sentences:
        if character in sentence:
            words = jieba.lcut(sentence)
            if not words:  # 跳过空句子
                continue

            score = 0
            sentiment_word_count = 0

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


sentiment_dict = load_sentiment_dict()

with open(character_analysis_path, "r", encoding="utf-8") as f:
    character_data = json.load(f)

results = {}
csv_data = []

for work_name, data in character_data.items():
    file_path = os.path.join(input_dir, f"{work_name}.txt")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    top_characters = list(data["characters"].items())  # [:10]
    character_sentiments = {}

    for character, _ in top_characters:
        sentiment_score = analyze_sentiment(content, character, sentiment_dict)
        sentiment = (
            "positive"
            if sentiment_score > 6
            else "negative" if sentiment_score < 4 else "neutral"
        )
        character_sentiments[character] = {
            "sentiment_score": sentiment_score,
            "sentiment": sentiment,
        }
        csv_data.append(
            {"work": work_name, "character": character, "sentiment": sentiment}
        )

    results[work_name] = character_sentiments

output_path = os.path.join(output_dir, "sentiment_analysis.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

csv_df = pd.DataFrame(csv_data)
csv_output_path = os.path.join(output_dir, "sentiment_analysis.csv")
csv_df.to_csv(csv_output_path, index=False, encoding="utf-8")
