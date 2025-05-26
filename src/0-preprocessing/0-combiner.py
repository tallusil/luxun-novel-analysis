import os

script_dir = os.path.dirname(os.path.abspath(__file__))

raw_dir = os.path.join(script_dir, "..", "..", "data", "raw")
output_dir = os.path.join(script_dir, "..", "..", "data", "combined")

os.makedirs(output_dir, exist_ok=True)

for folder in os.listdir(raw_dir):
    folder_path = os.path.join(raw_dir, folder)
    if not os.path.isdir(folder_path):
        continue
    contents = []
    for file in os.listdir(folder_path):
        if file.endswith(".md"):
            file_path = os.path.join(folder_path, file)
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if lines and lines[0].strip() == "```":
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            contents.append("".join(lines))
    if contents:
        out_path = os.path.join(output_dir, f"{folder}.txt")
        with open(out_path, "w", encoding="utf-8") as out_f:
            out_f.write("\n".join(contents))
