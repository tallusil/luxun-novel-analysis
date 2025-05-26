import os
import json
import jieba
import jieba.posseg as pseg
import csv

script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(script_dir, "..", "..", "data", "combined")
output_dir = os.path.join(script_dir, "..", "..", "output", "passage_analysis")

os.makedirs(output_dir, exist_ok=True)


def extract_characters(text):
    words = pseg.cut(text)
    characters = {}

    for word, flag in words:
        if flag == "nr" and len(word) >= 2:
            if word not in characters:
                characters[word] = 0
            characters[word] += 1

    return characters


results = {}

for file in os.listdir(input_dir):
    if not file.endswith(".txt"):
        continue

    work_name = file[:-4]
    file_path = os.path.join(input_dir, file)

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    characters = extract_characters(content)

    results[work_name] = {
        "character_count": len(characters),
        "characters": {
            name: count
            for name, count in sorted(
                characters.items(), key=lambda x: x[1], reverse=True
            )
        },
    }

output_path = os.path.join(output_dir, "character_analysis.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

csv_path = os.path.join(output_dir, "character_analysis.csv")
with open(csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["work", "character", "count"])
    for work_name, data in results.items():
        for character, count in data["characters"].items():
            writer.writerow([work_name, character, count])
