import os
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "..", "..", "output")
font_dir = os.path.join(script_dir, "..", "..", "font")
word_frequency_dir = os.path.join(output_dir, "word_frequency")
word_cloud_dir = os.path.join(output_dir, "word_cloud")
os.makedirs(word_cloud_dir, exist_ok=True)


def generate_wordcloud_from_csv(csv_path, filename):
    df = pd.read_csv(csv_path, encoding="utf-8")

    word_freq = dict(zip(df["word"], df["count"]))

    wordcloud = WordCloud(
        font_path=os.path.join(font_dir, "SourceHanSerifCN-Regular-Min.otf"),
        width=800,
        height=400,
        background_color="white",
    ).generate_from_frequencies(word_freq)

    plt.figure(figsize=(10, 5), dpi=300)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(os.path.join(word_cloud_dir, f"{filename}.png"))
    plt.close()
    print(f"词云图已生成: {filename}.png")


def main():
    for filename in os.listdir(word_frequency_dir):
        if filename.endswith(".csv"):
            csv_path = os.path.join(word_frequency_dir, filename)
            base_filename = filename.replace(".csv", "")
            generate_wordcloud_from_csv(csv_path, base_filename)


if __name__ == "__main__":
    main()
