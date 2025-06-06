"""
鲁迅小说主题与风格分析器 (Lu Xun Novel Theme and Style Analyzer)
功能：
1. 使用LDA模型分析小说的潜在主题
2. 分析作者的写作风格特点
3. 比较不同章节的风格差异
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import jieba
import jieba.posseg as pseg
from typing import List, Dict, Tuple
import json
from scipy import stats
import warnings
from pypinyin import lazy_pinyin, Style

warnings.filterwarnings("ignore")

# 获取脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))

# 设置路径
data_dir = os.path.join(script_dir, "..", "..", "data", "combined")
output_dir = os.path.join(script_dir, "..", "..", "output", "theme_style_analysis")
os.makedirs(output_dir, exist_ok=True)


class ThemeStyleAnalyzer:
    def __init__(self, segmented_texts: Dict[str, List[str]], n_topics=5):
        """
        初始化分析器
        Args:
            segmented_texts: 分词后的文本数据
            n_topics: 主题数量
        """
        self.segmented_texts = segmented_texts
        self.n_topics = n_topics
        self.vectorizer = CountVectorizer(max_features=2000, max_df=0.95, min_df=2)
        self.lda_model = None
        self.document_topics = {}

    def preprocess_for_lda(self) -> Tuple[np.ndarray, List[str]]:
        """
        为LDA主题建模预处理文本数据
        Returns:
            处理后的文档-词项矩阵和词表
        """
        # 将分词后的文本转换为字符串列表
        documents = []
        for work, segments in self.segmented_texts.items():
            doc = " ".join(segments)
            documents.append(doc)

        # 构建文档-词项矩阵
        dtm = self.vectorizer.fit_transform(documents)
        return dtm, self.vectorizer.get_feature_names_out()

    def train_lda_model(self):
        """
        训练LDA主题模型
        """
        print("训练LDA主题模型...")
        dtm, vocab = self.preprocess_for_lda()

        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            max_iter=50,
            learning_method="online",
            random_state=42,
        )

        # 训练模型并获取文档-主题分布
        doc_topics = self.lda_model.fit_transform(dtm)

        # 保存每个文档的主题分布
        for idx, (work_name, _) in enumerate(self.segmented_texts.items()):
            self.document_topics[work_name] = doc_topics[idx]

    def get_top_words_per_topic(self, n_words=10) -> List[List[str]]:
        """
        获取每个主题的前N个关键词
        Args:
            n_words: 每个主题返回的关键词数量
        Returns:
            每个主题的关键词列表
        """
        vocab = self.vectorizer.get_feature_names_out()
        top_words = []

        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_word_indices = topic.argsort()[: -n_words - 1 : -1]
            top_words.append([vocab[i] for i in top_word_indices])

        return top_words

    def analyze_writing_style(self) -> Dict[str, Dict]:
        """
        分析写作风格特征
        Returns:
            各作品的风格特征统计
        """
        print("分析写作风格特征...")
        style_features = {}

        for work_name, segments in self.segmented_texts.items():
            # 初始化特征字典
            style_features[work_name] = {
                "avg_sentence_length": 0,
                "word_frequency": Counter(),
                "pos_distribution": Counter(),
                "punctuation_frequency": Counter(),
                "vocabulary_richness": 0,
            }

            # 统计词频和词性分布
            all_words = []
            for segment in segments:
                words = pseg.cut(segment)
                for word, flag in words:
                    all_words.append(word)
                    style_features[work_name]["word_frequency"][word] += 1
                    style_features[work_name]["pos_distribution"][flag] += 1

            # 计算平均句长
            style_features[work_name]["avg_sentence_length"] = len(all_words) / len(
                segments
            )

            # 计算词汇丰富度（Type-Token Ratio）
            unique_words = len(set(all_words))
            total_words = len(all_words)
            style_features[work_name]["vocabulary_richness"] = (
                unique_words / total_words
            )

        return style_features

    def compare_chapter_styles(self) -> Dict[str, List[float]]:
        """
        比较不同章节的风格差异
        Returns:
            各章节的风格特征指标
        """
        print("比较章节风格差异...")
        chapter_features = {}

        for work_name, segments in self.segmented_texts.items():
            # 将文本分成若干个章节（这里简单地按固定长度划分）
            chunk_size = len(segments) // 5  # 将每部作品划分为5个部分
            if chunk_size == 0:
                continue

            chapter_features[work_name] = {
                "sentence_lengths": [],
                "vocabulary_richness": [],
                "punctuation_density": [],
            }

            for i in range(0, len(segments), chunk_size):
                chunk = segments[i : i + chunk_size]

                # 计算该章节的特征
                all_words = []
                punctuation_count = 0

                for segment in chunk:
                    words = list(jieba.cut(segment))
                    all_words.extend(words)
                    punctuation_count += sum(
                        1 for char in segment if char in "，。！？；："
                    )

                # 计算特征值
                avg_sent_len = len(all_words) / len(chunk)
                vocab_richness = len(set(all_words)) / len(all_words)
                punct_density = punctuation_count / len(chunk)

                # 保存特征值
                chapter_features[work_name]["sentence_lengths"].append(avg_sent_len)
                chapter_features[work_name]["vocabulary_richness"].append(
                    vocab_richness
                )
                chapter_features[work_name]["punctuation_density"].append(punct_density)

        return chapter_features

    def visualize_topics(self):
        """
        可视化主题分布
        """
        print("生成主题分布可视化...")
        top_words = self.get_top_words_per_topic()

        # 创建主题-词云图
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.ravel()

        for idx, topic_words in enumerate(top_words):
            if idx >= len(axes):
                break

            # 创建词频分布
            word_freq = {
                convert_to_pinyin(word): (len(top_words) - i)
                for i, word in enumerate(topic_words)
            }

            # 绘制条形图
            ax = axes[idx]
            words = list(word_freq.keys())
            freqs = list(word_freq.values())

            ax.barh(range(len(words)), freqs, align="center")
            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words)
            ax.invert_yaxis()
            ax.set_title(f"Topic {idx+1}")

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "topic_distribution.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def visualize_style_features(self, style_features: Dict[str, Dict]):
        """
        可视化风格特征
        Args:
            style_features: 风格特征数据
        """
        print("生成风格特征可视化...")

        # 创建多子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))

        # 1. 平均句长对比
        avg_lengths = [
            features["avg_sentence_length"] for features in style_features.values()
        ]
        ax1.bar(style_features.keys(), avg_lengths)
        ax1.set_title("Average Sentence Length")
        ax1.set_xticklabels(
            [convert_to_pinyin(work) for work in style_features.keys()], rotation=45
        )

        # 2. 词汇丰富度对比
        richness = [
            features["vocabulary_richness"] for features in style_features.values()
        ]
        ax2.bar(style_features.keys(), richness)
        ax2.set_title("Vocabulary Richness")
        ax2.set_xticklabels(
            [convert_to_pinyin(work) for work in style_features.keys()], rotation=45
        )

        # 3. 词性分布热图
        pos_data = {}
        for work, features in style_features.items():
            pos_data[work] = features["pos_distribution"]

        # 获取所有词性标签
        all_pos = set()
        for pos_dist in pos_data.values():
            all_pos.update(pos_dist.keys())

        # 创建热图数据
        pos_matrix = []
        for work in style_features.keys():
            row = [pos_data[work].get(pos, 0) for pos in all_pos]
            # 归一化
            total = sum(row)
            row = [count / total for count in row] if total > 0 else row
            pos_matrix.append(row)

        sns.heatmap(
            pos_matrix,
            ax=ax3,
            xticklabels=list(all_pos),
            yticklabels=[convert_to_pinyin(work) for work in style_features.keys()],
            cmap="YlOrRd",
        )
        ax3.set_title("POS Distribution")

        # 4. 主题分布
        topic_matrix = np.array(
            [self.document_topics[work] for work in style_features.keys()]
        )
        sns.heatmap(
            topic_matrix,
            ax=ax4,
            xticklabels=[
                f"Topic {i+1} ({convert_to_pinyin(', '.join(self.get_top_words_per_topic()[i][:3]))})"
                for i in range(self.n_topics)
            ],
            yticklabels=[convert_to_pinyin(work) for work in style_features.keys()],
            cmap="YlOrRd",
        )
        ax4.set_title("Topic Distribution")

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "style_features.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()

    def visualize_chapter_comparison(self, chapter_features: Dict[str, List[float]]):
        """
        可视化章节对比
        Args:
            chapter_features: 章节特征数据
        """
        print("生成章节对比可视化...")

        for work_name, features in chapter_features.items():
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

            # 绘制句长变化
            ax1.plot(features["sentence_lengths"], marker="o")
            ax1.set_title("Sentence Length Variation")
            ax1.set_xlabel("Chapter")
            ax1.set_ylabel("Average Length")

            # 绘制词汇丰富度变化
            ax2.plot(features["vocabulary_richness"], marker="o")
            ax2.set_title("Vocabulary Richness Variation")
            ax2.set_xlabel("Chapter")
            ax2.set_ylabel("TTR")

            # 绘制标点密度变化
            ax3.plot(features["punctuation_density"], marker="o")
            ax3.set_title("Punctuation Density Variation")
            ax3.set_xlabel("Chapter")
            ax3.set_ylabel("Density")

            plt.suptitle(f"Style Evolution: {work_name}")
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"{work_name}_chapter_comparison.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()


def load_text_data() -> Dict[str, str]:
    """
    加载文本数据
    Returns:
        作品名到文本内容的映射
    """
    print("加载文本数据...")
    texts = {}

    for file in os.listdir(data_dir):
        if not file.endswith(".txt"):
            continue

        work_name = file[:-4]  # 移除.txt后缀
        file_path = os.path.join(data_dir, file)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            texts[work_name] = content
            print(f"  ✓ 已加载：{work_name}")
        except Exception as e:
            print(f"  ✗ 加载失败 {work_name}: {e}")

    return texts


def segment_text(text: str) -> List[str]:
    """
    对文本进行分词
    Args:
        text: 原始文本
    Returns:
        分词后的文本列表
    """
    # 使用结巴分词
    words = jieba.cut(text)
    # 过滤停用词和标点符号
    words = [
        w
        for w in words
        if len(w.strip()) > 0 and not w.strip() in '，。！？；：（）【】《》""'
    ]
    return words


def convert_to_pinyin(text: str) -> str:
    """
    将中文文本转换为拼音
    Args:
        text: 中文文本
    Returns:
        拼音字符串
    """
    if not isinstance(text, str):
        return str(text)
    pinyin_list = lazy_pinyin(text, style=Style.NORMAL)
    return " ".join(pinyin_list).title()


def main():
    # 加载原始文本数据
    texts = load_text_data()
    if not texts:
        print("没有找到文本数据！")
        return

    # 对文本进行分词
    print("\n对文本进行分词...")
    segmented_texts = {}
    for work_name, content in texts.items():
        print(f"正在处理：{work_name}")
        segments = segment_text(content)
        segmented_texts[work_name] = segments

    # 创建分析器实例
    analyzer = ThemeStyleAnalyzer(segmented_texts)

    # 执行分析
    analyzer.train_lda_model()
    style_features = analyzer.analyze_writing_style()
    chapter_features = analyzer.compare_chapter_styles()

    # 生成可视化
    analyzer.visualize_topics()
    analyzer.visualize_style_features(style_features)
    analyzer.visualize_chapter_comparison(chapter_features)

    # 保存分析结果
    print("\n保存分析结果...")
    analysis_results = {
        "topics": {
            f"topic_{i+1}": words
            for i, words in enumerate(analyzer.get_top_words_per_topic())
        },
        "style_features": style_features,
        "chapter_features": chapter_features,
    }

    with open(
        os.path.join(output_dir, "analysis_results.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2)

    print(f"\n分析完成！所有结果已保存至: {output_dir}")
    print("生成的文件包括：")
    print("1. topic_distribution.png - 主题分布可视化")
    print("2. style_features.png - 写作风格特征可视化")
    print("3. [作品名]_chapter_comparison.png - 各作品的章节风格对比")
    print("4. analysis_results.json - 详细分析结果数据")


if __name__ == "__main__":
    main()
