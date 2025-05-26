import os
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "..", "..", "output")
font_dir = os.path.join(script_dir, "..", "..", "font")
word_frequency_dir = os.path.join(output_dir, "word_frequency")
word_cloud_dir = os.path.join(output_dir, "word_cloud")
os.makedirs(word_cloud_dir, exist_ok=True)


def combine_word_frequencies():
    combined_freq = {}
    for filename in os.listdir(word_frequency_dir):
        if filename.endswith(".csv"):
            df = pd.read_csv(
                os.path.join(word_frequency_dir, filename), encoding="utf-8"
            )
            for _, row in df.iterrows():
                word = row["word"]
                count = row["count"]
                combined_freq[word] = combined_freq.get(word, 0) + count
    return combined_freq


def generate_luxun_wordcloud():
    word_freq = combine_word_frequencies()

    mask_path = os.path.join(script_dir, "..", "..", "data", "assets", "luxun_mask.png")
    mask = np.array(Image.open(mask_path))

    wordcloud = WordCloud(
        font_path=os.path.join(font_dir, "SourceHanSerifCN-Regular-Min.otf"),
        width=1200,
        height=1600,
        mask=mask,
        background_color="white",
        contour_width=1,
        contour_color="black",
        min_font_size=10,
        max_font_size=200,
        prefer_horizontal=0.9,
        relative_scaling=0.5,
    ).generate_from_frequencies(word_freq)

    plt.figure(figsize=(12, 16), dpi=300)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    output_path = os.path.join(word_cloud_dir, "鲁迅全集.png")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"鲁迅人像词云图已生成: {output_path}")


if __name__ == "__main__":
    generate_luxun_wordcloud()
