# NoRAG vs NaiveRAG 训练数据标签分析报告

## 分析配置
- **评分公式**: `score = 0.5 * EM + 0.5 * F1`
- **总数据量**: 5000条
- **数据源**: `HotpotQA_train_data`

## 整体标签分布

| 标签 | 数量 | 百分比 | 说明 |
|------|------|--------|------|
| **Routing到NoRAG** | 361 | 7.22% | NoRAG分数高于NaiveRAG |
| **Routing到NaiveRAG** | 2090 | 41.80% | NaiveRAG分数高于NoRAG |
| **分数相等** | 2549 | 50.98% | 两者分数相同 |

## 分数相等的详细分析

在2549条分数相等的查询中：

| 情况 | 数量 | 占tie比例 | 占总数比例 |
|------|------|-----------|-----------|
| **两者都答对（EM=1.0）** | 574 | 22.52% | 11.48% |
| **两者都答错（EM=0.0）** | 1975 | 77.48% | 39.50% |

## 完整分布总结

| 情况 | 数量 | 百分比 |
|------|------|--------|
| 两者都答对 | 574 | 11.48% |
| 两者都答错 | 1975 | 39.50% |
| 仅NoRAG答对 | 361 | 7.22% |
| 仅NaiveRAG答对 | 2090 | 41.80% |
| **总计** | **5000** | **100.00%** |

## 关键发现

1. **NaiveRAG明显占优**：在41.80%的查询中，NaiveRAG的表现优于NoRAG
2. **NoRAG优势有限**：仅在7.22%的查询中，NoRAG表现更好
3. **大部分查询困难**：39.50%的查询两者都答错
4. **简单查询能答对**：11.48%的查询两者都能正确回答

## 数据文件说明

所有分析结果保存在 `label_analysis/` 目录：

- `label_distribution_summary.json` - 统计摘要
- `no_rag_queries.json` - 361条应该routing到NoRAG的查询
- `naive_rag_queries.json` - 2090条应该routing到NaiveRAG的查询
- `tie_queries.json` - 2549条分数相等的查询（进一步分为都答对和都答错）
- `all_labels.json` - 完整的5000条标签分配

## 示例分析

### 两者都答对的例子（11.48%）
```
Which tennis player won more Grand Slam titles, Henri Leconte or Jonathan Stark?
NoRAG: EM=1.0, F1=1.0
NaiveRAG: EM=1.0, F1=1.0
```
这类问题相对简单，两种方法都能正确回答。

### 两者都答错的例子（39.50%）
```
What nationality was James Henry Miller's wife?
NoRAG: EM=0.0, F1=0.0
NaiveRAG: EM=0.0, F1=0.0
```
这类问题难度较高，或者需要更深入的知识，两种方法都无法正确回答。

### 仅NoRAG答对的例子（7.22%）
```
Which magazine was started first Arthur's Magazine or First for Women?
NoRAG: EM=1.0, F1=1.0
NaiveRAG: EM=0.0, F1=0.0
```
这类问题可能在检索时引入了噪声，NoRAG直接回答反而更好。

### 仅NaiveRAG答对的例子（41.80%）
```
The Oberoi family is part of a hotel company that has a head office in what city?
NoRAG: EM=0.0, F1=0.0
NaiveRAG: EM=1.0, F1=1.0
```
这类问题需要外部知识支持，检索后的回答更准确。

## 训练建议

1. **数据不平衡问题**：NaiveRAG标签占41.80%，NoRAG仅占7.22%，存在明显的数据不平衡
2. **Tie数据处理**：50.98%的数据两者表现相同，可能需要特殊处理（如随机分配、使用其他指标细化等）
3. **困难样本**：39.50%的查询两者都答错，可能需要引入更强大的模型或改进prompt
