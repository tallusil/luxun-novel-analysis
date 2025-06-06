# 鲁迅全集文本处理与分析项目

本项目对鲁迅全集进行了系统的数据清洗、文本分析和可视化展示。

## 任务完成情况

### 1. 基础任务

#### 1) 小说选择与数据准备 ✓
- 选取鲁迅小说作为分析对象 ✓
- 数据存放：`data/raw/` 目录
- 文本清洗与预处理 ✓
  - 文本合并：`src/0-preprocessing/0-combiner.py`
  - 数据清洗：`src/0-preprocessing/1-cleaner.py`

#### 2) 基础文本分析 ✓
- 词频统计与分析 ✓
  - 高频词统计：`src/1-word-statistics/0-word_frequency.py`
- 情感分析 ✓
  - 基础情感分析：`src/2-passage-analysis/2-sentiment-analysis.py`
  - 情感趋势分析：`src/2-passage-analysis/4-sentiment-trend-analysis.py`
- 人物识别 ✓
  - 人物分析：`src/2-passage-analysis/1-character-analysis.py`
- 章节结构分析 ✓
  - 结构分析：`src/2-passage-analysis/0-structure-analysis.py`

#### 3) 可视化展示 ✓
- 词云图 ✓
  - 基础词云：`src/1-word-statistics/1-wordcloud_generator.py`
  - 人物肖像词云：`src/1-word-statistics/2-luxun_portrait_wordcloud.py`
- 词频统计图表 ✓
  - 词频可视化：`src/3-addition/1-word-frequency-charts.py`
- 情感变化趋势图 ✓
  - 情感趋势图：`src/3-addition/2-sentiment-trend-charts.py`
- 人物关系网络图 ✓
  - 人物网络：`src/2-passage-analysis/3-character-network.py`

### 2. 进阶任务

#### 1) 主题与风格分析 ✓
- 主题模型分析 ✓
- 写作风格分析 ✓
- 章节风格对比 ✓
  - 实现文件：`src/3-addition/3-theme-style-analysis.py`

分析结果输出目录：`output/`

可视化所需字体文件：`font/`

## 技术栈

- Python 3.8+
- 核心依赖：
  - jieba
  - wordcloud
  - networkx
  - pandas
  - matplotlib
  - sklearn

## 快速开始

1. 克隆项目
```bash
git clone [repository-url]
cd luxun-novel-analysis
```

2. 创建并激活虚拟环境
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac

# 或

.venv\Scripts\activate  # Windows
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 运行
```bash
# 按顺序执行各个模块

python src/0-preprocessing/0-combiner.py
python src/0-preprocessing/1-cleaner.py
python src/1-word-statistics/0-word_frequency.py

# ...
```

## 输出示例

分析结果将保存在 `output` 目录下，包括：
- 词云图
- 词频统计表
- 文本特征评价表
- 篇章结构统计表
- 人物关系网络图

## 许可证

本项目采用 MIT 许可证。