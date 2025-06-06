# 鲁迅小说人物网络图可视化指南

## 概述

本项目提供了多种人物网络图可视化工具，帮助你分析鲁迅小说中的人物关系。根据不同的需求，你可以选择不同的工具：

## 可用工具

### 1. 基础网络图 (`3-character-network.py`)
- **用途**: 生成基本的人物关系网络图
- **特点**: 简单、快速，适合初步分析
- **输出**: PNG格式的网络图

### 2. 增强版网络图 (`3-character-network-enhanced.py`)
- **用途**: 生成美观的增强版网络图
- **特点**: 
  - 自动选择最佳布局算法
  - 根据重要性调整节点颜色和大小
  - 提供多种布局对比
  - 包含图例和统计信息
- **输出**: 增强版PNG图片 + 布局对比图

### 3. 交互式分析工具 (`4-interactive-network-analysis.py`)
- **用途**: 生成全面的网络分析报告
- **特点**:
  - 网络统计分析
  - 人物重要性排名
  - 多种可视化方式
  - CSV数据导出
- **输出**: 分析报告图表 + 详细数据

### 4. 自定义配置工具 (`5-network-config-tool.py`)
- **用途**: 根据个人喜好自定义网络图
- **特点**: 可调整所有参数
- **输出**: 完全自定义的网络图

## 使用方法

### 快速开始

1. **生成基础网络图**：
```bash
python src/2-passage-analysis/3-character-network.py
```

2. **生成增强版网络图**：
```bash
python src/2-passage-analysis/3-character-network-enhanced.py
```

3. **生成完整分析报告**：
```bash
python src/2-passage-analysis/4-interactive-network-analysis.py
```

4. **生成自定义网络图**：
```bash
python src/2-passage-analysis/5-network-config-tool.py
```

### 自定义网络图参数

使用配置工具时，可以调整以下参数：

#### 基础参数
- `min_weight`: 最小边权重（1-5），值越大显示的关系越强
- `layout`: 布局算法
  - `auto`: 自动选择（推荐）
  - `spring`: 弹簧布局（适合复杂网络）
  - `circular`: 圆形布局（适合小网络）
  - `kamada_kawai`: KK布局（适合中等网络）

#### 视觉参数
- `node_color`: 节点颜色依据
  - `pagerank`: PageRank重要性（推荐）
  - `betweenness`: 介数中心性（桥梁作用）
  - `degree`: 度中心性（连接数）
- `node_size`: 节点大小依据（同上）
- `color_scheme`: 颜色方案
  - `viridis`: 蓝绿色（推荐）
  - `plasma`: 紫红色
  - `cool`: 冷色调
  - `warm`: 暖色调

#### 高级参数
- `base_size`: 基础节点大小（200-500）
- `figure_size`: 图形大小，如 `(12, 10)`
- `dpi`: 分辨率（300推荐）
- `show_labels`: 是否显示标签
- `use_chinese`: 是否使用中文标签

### 示例配置

#### 1. 突出重要人物的配置
```python
config = {
    'min_weight': 3,
    'layout': 'spring',
    'node_color': 'pagerank',
    'node_size': 'pagerank',
    'color_scheme': 'plasma',
    'base_size': 400
}
```

#### 2. 展示桥梁人物的配置
```python
config = {
    'min_weight': 2,
    'layout': 'circular',
    'node_color': 'betweenness',
    'node_size': 'degree',
    'color_scheme': 'viridis'
}
```

#### 3. 详细分析配置
```python
config = {
    'min_weight': 1,
    'layout': 'kamada_kawai',
    'node_color': 'degree',
    'node_size': 'betweenness',
    'color_scheme': 'cool',
    'base_size': 300,
    'figure_size': (16, 12)
}
```

## 输出文件位置

各个工具的输出文件保存在以下目录：

- **基础网络图**: `output/passage_analysis/character_network/`
- **增强版网络图**: `output/passage_analysis/character_network_enhanced/`
- **交互式分析**: `output/passage_analysis/character_network_interactive/`
- **自定义网络图**: `output/passage_analysis/character_network_custom/`

## 网络图解读

### 节点含义
- **节点大小**: 表示人物的重要性
- **节点颜色**: 根据选择的指标显示人物特征
- **节点标签**: 人物名称

### 边的含义
- **边的存在**: 两个人物在文本中有共现关系
- **边的粗细**: 表示共现次数，越粗关系越密切
- **边的透明度**: 同边的粗细，越不透明关系越强

### 重要性指标

1. **PageRank**: 
   - 衡量人物在网络中的整体重要性
   - 考虑直接和间接连接
   - 适合找出核心人物

2. **介数中心性**:
   - 衡量人物的桥梁作用
   - 高值表示连接不同群体的关键人物
   - 适合找出情节转折的关键人物

3. **度中心性**:
   - 衡量人物的直接连接数
   - 高值表示与很多人物都有关系
   - 适合找出社交活跃的人物

## 优化建议

### 提高网络图清晰度

1. **调整最小权重**:
   - 对于复杂网络，提高 `min_weight` 到 3-4
   - 对于简单网络，降低到 1-2

2. **选择合适布局**:
   - 节点少于10个：使用 `circular`
   - 节点10-30个：使用 `kamada_kawai`
   - 节点多于30个：使用 `spring`

3. **调整图形大小**:
   - 复杂网络使用更大的 `figure_size`
   - 建议 `(14, 12)` 或 `(16, 14)`

4. **优化标签显示**:
   - 节点过多时可以关闭标签：`show_labels=False`
   - 或者使用简化标签：`use_chinese=False`

### 分析不同角度

1. **重要性分析**: 使用 `pagerank` 颜色和大小
2. **桥梁分析**: 使用 `betweenness` 颜色
3. **社交分析**: 使用 `degree` 颜色和大小

## 常见问题

### Q: 网络图节点太多，看不清怎么办？
A: 
1. 提高 `min_weight` 参数
2. 增大 `figure_size`
3. 关闭标签显示
4. 使用分层分析

### Q: 如何找出最重要的人物？
A: 使用 `pagerank` 作为节点颜色和大小依据，深色大节点就是最重要的人物

### Q: 如何分析人物的桥梁作用？
A: 使用 `betweenness` 作为节点颜色，高值节点连接不同的人物群体

### Q: 生成的图片中文显示有问题？
A: 设置 `use_chinese=False` 使用简化标签，或者安装中文字体

## 进阶使用

### 批量生成多个作品的网络图

```python
from src.2-passage-analysis.5-network-config-tool import NetworkConfigTool

tool = NetworkConfigTool(".")
works = ["呐喊", "彷徨", "朝花夕拾"]

for work in works:
    config = {
        'min_weight': 2,
        'layout': 'auto',
        'node_color': 'pagerank',
        'color_scheme': 'viridis'
    }
    tool.create_custom_network(work, config)
```

### 对比分析

生成同一作品的不同视角分析图：

```python
configs = [
    {'node_color': 'pagerank', 'color_scheme': 'plasma'},
    {'node_color': 'betweenness', 'color_scheme': 'viridis'},
    {'node_color': 'degree', 'color_scheme': 'cool'}
]

for i, config in enumerate(configs):
    tool.create_custom_network("呐喊", {**config, 'min_weight': 3})
```

通过这些工具和技巧，你应该能够生成清晰、美观且信息丰富的人物网络图了！ 