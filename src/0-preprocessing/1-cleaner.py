import os
import re

script_dir = os.path.dirname(os.path.abspath(__file__))
combined_dir = os.path.join(script_dir, "..", "..", "data", "combined")
cleaned_dir = os.path.join(script_dir, "..", "..", "data", "combined")

os.makedirs(cleaned_dir, exist_ok=True)

for file in os.listdir(combined_dir):
    if not file.endswith(".txt"):
        continue

    input_path = os.path.join(combined_dir, file)
    output_path = os.path.join(cleaned_dir, file)

    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    content = re.sub(r"鲁迅《[^》]*》", "", content)
    content = content.replace("·鲁迅·", "")
    content = content.replace("著者：鲁迅", "")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
