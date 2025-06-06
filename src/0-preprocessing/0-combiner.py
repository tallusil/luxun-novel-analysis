"""
鲁迅小说文本合并器
功能：将散布在各个文件夹中的markdown文件合并成统一的文本文件
"""

import os

# 获取脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))

# 设置原始数据目录和输出目录的路径
raw_dir = os.path.join(script_dir, "..", "..", "data", "raw")
output_dir = os.path.join(script_dir, "..", "..", "data", "combined")

# 确保输出目录存在，如果不存在则创建
os.makedirs(output_dir, exist_ok=True)

# 遍历原始数据目录中的每个文件夹
for folder in os.listdir(raw_dir):
    folder_path = os.path.join(raw_dir, folder)

    # 跳过非目录项
    if not os.path.isdir(folder_path):
        continue

    # 存储当前文件夹中所有文本内容
    contents = []

    # 遍历文件夹中的所有markdown文件
    for file in os.listdir(folder_path):
        if file.endswith(".md"):
            file_path = os.path.join(folder_path, file)

            # 读取markdown文件内容
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # 去除markdown代码块标记（```）
            # 如果第一行是```，则删除
            if lines and lines[0].strip() == "```":
                lines = lines[1:]
            # 如果最后一行是```，则删除
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            # 将处理后的内容添加到列表中
            contents.append("".join(lines))

    # 如果有内容，则写入到合并文件中
    if contents:
        out_path = os.path.join(output_dir, f"{folder}.txt")
        with open(out_path, "w", encoding="utf-8") as out_f:
            # 用换行符连接所有内容
            out_f.write("\n".join(contents))
