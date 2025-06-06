# 鲁迅全集文本处理与分析项目

本项目对鲁迅全集进行了系统的数据清洗、文本分析和可视化展示。

## 项目结构

```bash
project-root/
│
├── data/        # 存放原始与清洗后的鲁迅全集文本
│   ├── raw/        # 原始文本
│   ├─── combined/        # 清洗、合并后的文本
│   └── ...
│
├── src/        # 源代码目录
│   ├── 0-preprocessing/        # 文本预处理模块
│   ├── 1-word-statistics/        # 词频统计模块
│   └── 2-passage-analysis/        # 段落分析模块
│
├── output/        # 分析结果输出目录
│
├── font/        # 字体文件目录
│
├── requirements.txt        # 项目依赖
└── README.md        # 项目说明文件
```

## 主要功能

### 1. 文本预处理
- 文本合并
- 数据清洗与规范化
- 中文分词处理
- 停用词过滤

### 2. 词频统计与分析
- 高频词统计
- 词云生成
- 词性分析

### 3. 段落分析
- 段落结构分析
- 文本特征提取
- 人物关系网络可视化

## 技术栈

- Python 3.8+
- 核心依赖：
  - jieba
  - wordcloud
  - networkx
  - pandas
  - matplotlib

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