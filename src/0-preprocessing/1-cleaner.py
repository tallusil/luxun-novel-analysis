"""
鲁迅小说文本清理器
功能：清理合并后的文本文件，去除不必要的元数据信息
"""

import os
import re

# 获取脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))

# 设置输入和输出目录路径（这里输入和输出是同一个目录）
combined_dir = os.path.join(script_dir, "..", "..", "data", "combined")
cleaned_dir = os.path.join(script_dir, "..", "..", "data", "combined")

# 确保输出目录存在
os.makedirs(cleaned_dir, exist_ok=True)

# 遍历合并目录中的所有文本文件
for file in os.listdir(combined_dir):
    # 只处理.txt文件
    if not file.endswith(".txt"):
        continue

    # 设置输入和输出文件路径
    input_path = os.path.join(combined_dir, file)
    output_path = os.path.join(cleaned_dir, file)

    # 读取文件内容
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 文本清理操作
    # 去除"鲁迅《作品名》"这样的标记
    content = re.sub(r"鲁迅《[^》]*》", "", content)
    # 去除"·鲁迅·"标记
    content = content.replace("·鲁迅·", "")
    # 去除"著者：鲁迅"标记
    content = content.replace("著者：鲁迅", "")

    # 将清理后的内容写入文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
