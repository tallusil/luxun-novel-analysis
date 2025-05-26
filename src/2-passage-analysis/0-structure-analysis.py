import os
import json
import csv

script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(script_dir, "..", "..", "data", "combined")
output_dir = os.path.join(script_dir, "..", "..", "output", "passage_analysis")

os.makedirs(output_dir, exist_ok=True)

results = {}

for file in os.listdir(input_dir):
    if not file.endswith(".txt"):
        continue

    work_name = file[:-4]
    file_path = os.path.join(input_dir, file)

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    paragraphs = [p for p in content.split("\n") if p.strip()]
    char_count = sum(len(p) for p in paragraphs)

    results[work_name] = {
        "paragraph_count": len(paragraphs),
        "character_count": char_count,
        "average_paragraph_length": char_count / len(paragraphs) if paragraphs else 0,
    }

output_path = os.path.join(output_dir, "structure_analysis.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

csv_path = os.path.join(output_dir, "structure_analysis.csv")
with open(csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        ["work", "paragraph_count", "character_count", "average_paragraph_length"]
    )
    for work_name, data in results.items():
        writer.writerow(
            [
                work_name,
                data["paragraph_count"],
                data["character_count"],
                data["average_paragraph_length"],
            ]
        )
