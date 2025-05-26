import os
import jieba
import pandas as pd
from collections import Counter

script_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(script_dir, "..", "..", "data", "combined")
output_dir = os.path.join(script_dir, "..", "..", "output")
stopwords_path = os.path.join(
    script_dir, "..", "..", "data", "stopwords", "stopwords.txt"
)
os.makedirs(output_dir, exist_ok=True)
word_frequency_dir = os.path.join(output_dir, "word_frequency")
os.makedirs(word_frequency_dir, exist_ok=True)


def load_stopwords():
    with open(stopwords_path, "r", encoding="utf-8") as f:
        stopwords = set([line.strip() for line in f])
    return stopwords


stopwords = load_stopwords()


def process_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    words = jieba.cut(text)
    words = [
        word
        for word in words
        if len(word) > 1
        and word not in stopwords
        and not all(not c.isalnum() for c in word)
        and not word.isdigit()
        and not word.isnumeric()
        and not word.isdecimal()
        and not all(c.isdigit() or c in "一二三四五六七八九十百千万亿零" for c in word)
        and not any(
            c in '（）()【】[]{}《》<>「」『』""' "、，。！？；：…—" for c in word
        )
    ]
    return words


def count_frequent_words(words, top_n=None):
    word_counts = Counter(words)
    return word_counts.most_common(top_n)


def save_to_csv(word_counts, filename):
    df = pd.DataFrame(word_counts, columns=["word", "count"])
    output_path = os.path.join(word_frequency_dir, f"{filename}.csv")
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"词频统计已保存到: {output_path}")


def main():
    for filename in os.listdir(file_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(file_dir, filename)
            words = process_text(file_path)

            frequent_words = count_frequent_words(words)

            save_to_csv(frequent_words, filename[:-4])


if __name__ == "__main__":
    main()
