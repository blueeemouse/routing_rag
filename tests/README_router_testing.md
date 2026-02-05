# Router 路由决策测试指南

## 功能说明

`test_router_routing.py` 用于测试训练好的Router模型在HotpotQA测试集上的路由决策能力。

### 支持的功能

1. **路由决策测试**
   - 对每个问题预测应该使用哪个RAG策略（no_rag 或 naive_rag）
   - 统计策略分布
   - 记录模型置信度

2. **准确率评估**（可选）
   - 如果提供NoRAG和NaiveRAG的结果文件
   - 计算真实最优策略（基于 `0.5 * EM + 0.5 * F1` 公式）
   - 对比模型预测与真实标签，计算准确率
   - 提供召回率、精确率和F1分数

## 使用方法

### 基础用法：只测试路由决策

```bash
# 测试路由决策（不计算准确率）
python tests/test_router_routing.py \
  --model router_models/no_rag_vs_naive/final/model.pt \
  --test HotpotQA/hotpot_dev_distractor_1000_samples.jsonl
```

### 完整用法：测试路由决策并计算准确率

```bash
# 使用改进的模型
python tests/test_router_routing.py \
  --model router_models/no_rag_vs_naive_improved/final/model.pt \
  --test HotpotQA/hotpot_dev_distractor_1000_samples.jsonl \
  --no-rag-results HotpotQA/NoRag_results.json \
  --naive-rag-results HotpotQA/Naiverag_results.json \
  --output tests/router_test_improved_results.json
```

### 限制测试样本数

```bash
# 只测试前100个样本
python tests/test_router_routing.py \
  --model router_models/no_rag_vs_naive/final/model.pt \
  --test HotpotQA/hotpot_dev_distractor_1000_samples.jsonl \
  --num-samples 100
```

## 命令行参数说明

| 参数 | 是否必填 | 说明 | 示例 |
|------|----------|------|--------|
| `--model` | **是** | 模型权重文件路径 | `router_models/no_rag_vs_naive/final/model.pt` |
| `--config` | 否 | 配置文件路径（可选，默认从checkpoint读取） | `router_models/no_rag_vs_naive/final/config.json` |
| `--test` | **是** | 测试数据文件路径（JSONL格式） | `HotpotQA/hotpot_dev_distractor_1000_samples.jsonl` |
| `--no-rag-results` | 否 | NoRAG结果文件路径（JSON格式） | `HotpotQA/NoRag_results.json` |
| `--naive-rag-results` | 否 | NaiveRAG结果文件路径（JSON格式） | `HotpotQA/Naiverag_results.json` |
| `--output` | 否 | 输出文件路径 | `tests/router_test_results.json` |
| `--num-samples` | 否 | 测试样本数（None表示全部） | `100` |

## 输出文件结构

### 结果文件（JSON格式）

```json
{
  "questions": [
    {
      "_id": "问题ID",
      "question": "问题文本",
      "predicted_strategy": "no_rag 或 naive_rag",
      "strategy_index": 0 或 1,
      "confidence": 0.85,
      // 如果提供了策略结果文件，会包含以下字段
      "true_strategy": "no_rag 或 naive_rag",
      "no_rag_score": 0.75,
      "naive_rag_score": 0.60,
      "correct": true 或 false
    },
    ...
  ],
  "statistics": {
    "total_questions": 1000,
    "no_rag_count": 200,
    "naive_rag_count": 800,
    "no_rag_ratio": 0.20,
    "naive_rag_ratio": 0.80,
    // 如果提供了策略结果文件，会包含以下字段
    "eval_count": 1000,
    "correct_count": 750,
    "accuracy": 0.75,
    "no_rag_accuracy": 0.70,
    "naive_rag_accuracy": 0.76,
    "no_rag_recall": 0.70,
    "no_rag_precision": 0.60,
    "naive_rag_recall": 0.76,
    "naive_rag_precision": 0.84,
    "no_rag_f1": 0.65,
    "naive_rag_f1": 0.80
  }
}
```

### 统计指标说明

#### 基础统计
- `total_questions`: 测试问题总数
- `no_rag_count`: 路由到no_rag的问题数
- `naive_rag_count`: 路由到naive_rag的问题数
- `no_rag_ratio`: no_rag占比
- `naive_rag_ratio`: naive_rag占比

#### 准确率评估（仅当提供策略结果时）
- `eval_count`: 可评估的样本数
- `correct_count`: 预测正确的样本数
- `accuracy`: 总准确率 = correct_count / eval_count

#### 策略级评估
- `no_rag_accuracy`: no_rag的准确率
- `naive_rag_accuracy`: naive_rag的准确率
- `no_rag_recall`: no_rag的召回率
- `no_rag_precision`: no_rag的精确率
- `no_rag_f1`: no_rag的F1分数
- `naive_rag_recall`: naive_rag的召回率
- `naive_rag_precision`: naive_rag的精确率
- `naive_rag_f1`: naive_rag的F1分数

## 准备策略结果文件

要计算准确率，需要先准备NoRAG和NaiveRAG的结果文件。

### 结果文件格式要求

策略结果文件应该是JSON数组格式，每个元素包含：

```json
[
  {
    "_id": "问题ID（与测试数据一致）",
    "answer": "模型预测的答案",
    "em": 0.0 或 1.0,
    "f1": 0.85,
    // ... 其他字段可选
  },
  ...
]
```

### 如果没有策略结果文件？

如果不想计算准确率，可以只测试路由决策：
- 模型会预测每个问题应该使用哪个策略
- 统计策略分布
- 不会验证决策的正确性

## 评估标准

真实最优策略通过以下公式确定：
```
score = 0.5 * EM + 0.5 * F1

if no_rag_score > naive_rag_score:
    true_strategy = 'no_rag'
elif naive_rag_score > no_rag_score:
    true_strategy = 'naive_rag'
else:
    # 分数相等时，按照argmax规则返回第一个（no_rag）
    true_strategy = 'no_rag'
```

这与训练时使用的标签生成逻辑一致。

## 示例测试流程

### 1. 准备数据

```bash
# 假设已有以下文件
ls HotpotQA/
  - hotpot_dev_distractor_1000_samples.jsonl  # 测试问题
  - NoRag_results.json                      # NoRAG结果（可选）
  - Naiverag_results.json                    # NaiveRAG结果（可选）
```

### 2. 运行测试

```bash
cd d:\Develop\all_RAG\routing_rag

# 测试原始模型（不计算准确率）
python tests/test_router_routing.py \
  --model router_models/no_rag_vs_naive/final/model.pt \
  --test HotpotQA/hotpot_dev_distractor_1000_samples.jsonl \
  --output tests/router_test_original.json

# 测试改进模型（计算准确率）
python tests/test_router_routing.py \
  --model router_models/no_rag_vs_naive_improved/final/model.pt \
  --test HotpotQA/hotpot_dev_distractor_1000_samples.jsonl \
  --no-rag-results HotpotQA/NoRag_results.json \
  --naive-rag-results HotpotQA/Naiverag_results.json \
  --output tests/router_test_improved.json
```

### 3. 查看结果

```bash
# 查看统计
cat tests/router_test_improved.json | grep -A 20 "statistics"
```

或直接用文本编辑器打开结果文件查看。

## 分析建议

### 查看策略分布

- 如果模型对所有问题都路由到同一策略 → 模型训练失败
- 如果策略分布与训练数据接近（58.2% no_rag, 41.8% naive_rag）→ 模型泛化良好
- 如果策略分布差异很大 → 可能需要检查训练数据或模型配置

### 查看准确率

- **总准确率 > 60%**: 模型有基本路由能力
- **总准确率 > 70%**: 模型路由能力较强
- **总准确率 > 80%**: 模型路由能力优秀

### 查看置信度

- 高置信度（>0.9）→ 模型决策明确
- 中等置信度（0.7-0.9）→ 模型决策较为确定
- 低置信度（<0.7）→ 模型对某些问题不确定

对于低置信度的问题，可以：
1. 分析问题特征
2. 查看模型是否需要更多类似样本
3. 考虑调整温度参数

## 常见问题

### Q1: 提示 "找不到模型文件"

检查：
- 路径是否正确
- 文件是否存在于指定位置
- 是否使用了相对路径（建议使用绝对路径）

### Q2: 提示 "找不到测试数据"

检查：
- 测试数据文件路径是否正确
- 文件格式是否为JSONL（每行一个JSON对象）

### Q3: 准确率为0或异常低

可能原因：
- 策略结果文件中的 `_id` 与测试数据不匹配
- 策略结果文件格式不正确
- 模型确实没有训练好

调试方法：
```bash
# 检查ID匹配情况
python -c "import json; d1=json.load(open('HotpotQA/NoRag_results.json')); d2=json.load(open('HotpotQA/Naiverag_results.json')); print(len(d1), len(d2))"

# 只测试5个样本，快速验证
python tests/test_router_routing.py --model ... --test ... --no-rag-results ... --naive-rag-results ... --num-samples 5
```

### Q4: 如何在训练集上测试？

使用相同的命令，但将测试数据替换为训练数据：

```bash
python tests/test_router_routing.py \
  --model router_models/no_rag_vs_naive/final/model.pt \
  --test HotpotQA_train_data.jsonl
```

## 与其他测试工具的关系

| 工具 | 用途 | 特点 |
|------|------|------|
| `test_router_routing.py` | 测试路由决策 | 专注于路由准确率 |
| `test_retrieval_strategies.py` | 测试检索质量 | 专注于检索和答案生成 |
| `diagnose_router.py` | 诊断模型问题 | 分析embedding分布和预测 |

建议同时使用这些工具全面评估Router性能。
