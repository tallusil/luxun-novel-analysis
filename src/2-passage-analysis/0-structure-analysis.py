"""
鲁迅小说结构分析器
功能：分析每部作品的篇章结构，统计段落数量、字符数量和平均段落长度
"""

import os
import json
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

    # 读取文件内容
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 按行分割并过滤空行，得到段落列表
    paragraphs = [p for p in content.split("\n") if p.strip()]

    # 计算总字符数（所有段落的字符数之和）
    char_count = sum(len(p) for p in paragraphs)

    # 存储当前作品的统计数据
    results[work_name] = {
        "paragraph_count": len(paragraphs),  # 段落数量
        "character_count": char_count,  # 字符总数
        "average_paragraph_length": (
            char_count / len(paragraphs) if paragraphs else 0
        ),  # 平均段落长度
    }

# 将结果保存为JSON格式
output_path = os.path.join(output_dir, "structure_analysis.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# 将结果保存为CSV格式，便于进一步分析
csv_path = os.path.join(output_dir, "structure_analysis.csv")
with open(csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)

    # 写入表头
    writer.writerow(
        ["work", "paragraph_count", "character_count", "average_paragraph_length"]
    )

    # 写入每部作品的数据
    for work_name, data in results.items():
        writer.writerow(
            [
                work_name,  # 作品名称
                data["paragraph_count"],  # 段落数量
                data["character_count"],  # 字符总数
                data["average_paragraph_length"],  # 平均段落长度
            ]
        )

print(f"结构分析完成，结果已保存到:")
print(f"  JSON格式: {output_path}")
print(f"  CSV格式: {csv_path}")
