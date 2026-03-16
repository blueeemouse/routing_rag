# 数据格式规范

本文档说明路由训练系统支持的数据文件格式。

---

## 硬标签格式 (Hard Label)

用于明确标注最优策略的数据（配合 `RouterLabelDataset` 使用）。

### 示例

```json
{
  "samples": [{
    "question": "Which magazine was started first...",
    "optimal_strategy": "naive_rag",
    "index": 0,
    "no_rag_em": 0,
    "no_rag_f1": 0.0,
    "naive_rag_em": 1,
    "naive_rag_f1": 1.0
  }]
}
```

### 字段说明

| 字段 | 必需 | 说明 |
|------|------|------|
| `question` | ✅ | 问题文本，数据集核心输入 |
| `optimal_strategy` | ✅ | 最优策略名称，必须与 `config.model.strategy_names` 中的名称匹配 |
| `index` | ❌ | 样本索引，用于 `cluster_id`，缺失时使用列表序号。主要用于聚类相关的训练策略（如对比学习、课程学习）|
| `{strategy}_em` | ❌ | 各策略的 Exact Match 分数，用于软标签计算或分析 |
| `{strategy}_f1` | ❌ | 各策略的 F1 分数，用于软标签计算或分析 |
| `{strategy}_score` | ❌ | 兼容性字段，功能类似 em/f1 |

---

## 二分类软标签格式 (Binary Soft Label)

用于 `SoftLabelRouterDataset`（纯统计特征模型）。

### 示例

```json
{
  "samples": [{
    "question": "Which magazine was started first...",
    "soft_label": 0.85,
    "label": "naive_rag",
    "index": 0
  }]
}
```

### 字段说明

| 字段 | 必需 | 说明 |
|------|------|------|
| `question` | ✅ | 问题文本 |
| `soft_label` | ✅ | 二分类软标签，表示 `naive_rag` 的概率（0~1之间），`no_rag` 概率 = 1 - soft_label |
| `label` / `optimal_strategy` | ❌ | 硬标签字符串，用于评估阶段计算准确率 |
| `index` | ❌ | 样本索引，用于 `cluster_id`。主要用于聚类相关的训练策略（如对比学习、课程学习），普通训练可省略 |

---

## 多分类软标签格式 (Multi-class Soft Label)

用于 `FusionSoftLabelDataset`（融合模型，语义+统计特征）。

### 示例

**向量格式（推荐）：**
```json
{
  "samples": [{
    "question": "Which magazine was started first...",
    "soft_label_vector": [0.1, 0.7, 0.2],
    "label": "naive_rag",
    "index": 0
  }]
}
```

**二分类兼容格式（自动转换）：**
```json
{
  "samples": [{
    "question": "Which magazine was started first...",
    "soft_label": 0.85,
    "optimal_strategy": "naive_rag",
    "index": 0,
    "no_rag_em": 0,
    "no_rag_f1": 0.1,
    "naive_rag_em": 0.8,
    "naive_rag_f1": 0.9
  }]
}
```

### 字段说明

| 字段 | 必需 | 说明 |
|------|------|------|
| `question` | ✅ | 问题文本 |
| `soft_label_vector` | ✅* | 多分类软标签概率分布，长度与策略数相同 |
| `soft_label` | ✅* | 二分类单值（二分类时自动转为 `[1-p, p]` 向量）|
| `label` / `optimal_strategy` | ❌ | 硬标签字符串，用于评估，缺失时从 `soft_label` argmax 推断 |
| `index` | ❌ | 样本索引，用于 `cluster_id`。主要用于聚类相关的训练策略（如对比学习、课程学习），普通训练可省略 |
| `{strategy}_em` / `{strategy}_f1` | ❌ | 策略指标，用于无软标签时计算 fallback |

> **注意：** `soft_label_vector` 和 `soft_label` 至少提供一个。

---

## 格式兼容性总结

| 数据集类 | 支持格式 | 必需字段 | 自动推断逻辑 |
|----------|----------|----------|--------------|
| `RouterLabelDataset` | 硬标签 | `question`, `optimal_strategy` | 从 em/f1 计算 scores |
| `SoftLabelRouterDataset` | 二分类软标签 | `question`, `soft_label` | 无硬标签时无法评估准确率 |
| `FusionSoftLabelDataset` | 多分类软标签 | `question` + `soft_label*` | 无软标签时从 em/f1 计算；无硬标签时从 soft_label argmax 推断 |

---

## 配置与数据格式对应

在训练配置 YAML 中，通过 `data.source` 字段指定数据类型：

```yaml
data:
  source: "hard_label"          # 使用 RouterLabelDataset
  # source: "soft_label"        # 使用 SoftLabelRouterDataset
  # source: "fusion_soft_label" # 使用 FusionSoftLabelDataset
  train_path: "path/to/data.json"
```

---

## 常见问题

### Q1: 硬标签数据可以用软标签训练器吗？
可以。`FusionSoftLabelTrainer` 会自动将硬标签（one-hot）转为软标签格式。

### Q2: 软标签数据可以用硬标签训练器吗？
不建议。硬标签训练器（如 `DCTrainer`）期望离散标签，软标签会被强制转为 argmax 硬标签，丢失概率信息。

### Q3: 如何验证数据格式是否正确？
使用数据集类直接加载测试：
```python
from trainable_router.datasets.fusion_soft_label_dataset import FusionSoftLabelDataset
from trainable_router.config import TrainableRouterConfig

config = TrainableRouterConfig.from_yaml("config.yaml")
dataset = FusionSoftLabelDataset(config)
dataset.load_data("path/to/data.json")
print(f"加载成功: {len(dataset)} 条样本")
sample = dataset[0]
print(f"样本字段: {sample.keys()}")
```
