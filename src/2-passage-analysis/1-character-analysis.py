"""
鲁迅小说人物分析器
功能：使用jieba词性标注提取文本中的人物名称，并统计出现频次
"""

import os
import json
import jieba
import jieba.posseg as pseg  # jieba词性标注模块
import csv

# 获取脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))

# 设置输入和输出路径
input_dir = os.path.join(script_dir, "..", "..", "data", "combined")  # 输入文本目录
output_dir = os.path.join(
    script_dir, "..", "..", "output", "passage_analysis"
)  # 输出目录

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)


def extract_characters(text):
    """
    从文本中提取人物名称并统计频次
    参数：
        text: 输入文本
    返回：
        characters: 人物名称及其出现频次的字典
    """
    # 使用jieba进行词性标注
    words = pseg.cut(text)
    characters = {}

    # 遍历所有词语和词性
    for word, flag in words:
        # 筛选人名（词性为'nr'且长度>=2的词）
        # 'nr'表示人名，长度>=2可以过滤掉单字人名，提高准确性
        if flag == "nr" and len(word) >= 2:
            # 统计人物出现频次
            if word not in characters:
                characters[word] = 0
            characters[word] += 1

    return characters


# 存储所有作品的分析结果
results = {}

# 遍历输入目录中的所有文本文件
for file in os.listdir(input_dir):
    # 只处理.txt文件
    if not file.endswith(".txt"):
        continue

    # 获取作品名称（去掉.txt扩展名）
    work_name = file[:-4]
    file_path = os.path.join(input_dir, file)

    print(f"正在分析作品: {work_name}")

    # 读取文件内容
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 提取人物名称
    characters = extract_characters(content)

    # 存储当前作品的人物分析结果
    results[work_name] = {
        "character_count": len(characters),  # 人物总数
        "characters": {
            name: count
            for name, count in sorted(
                characters.items(),
                key=lambda x: x[1],
                reverse=True,  # 按出现频次降序排列
            )
        },
    }

# 将结果保存为JSON格式
output_path = os.path.join(output_dir, "character_analysis.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# 将结果保存为CSV格式，便于进一步分析
csv_path = os.path.join(output_dir, "character_analysis.csv")
with open(csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)

    # 写入表头
    writer.writerow(["work", "character", "count"])

    # 写入每部作品的人物数据
    for work_name, data in results.items():
        for character, count in data["characters"].items():
            writer.writerow([work_name, character, count])

print(f"人物分析完成，结果已保存到:")
print(f"  JSON格式: {output_path}")
print(f"  CSV格式: {csv_path}")
