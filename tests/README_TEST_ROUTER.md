# Router 模型测试说明

## 概述

使用训练好的 Router 模型在 HotpotQA 测试集上进行路由决策和端到端评估。

## 测试脚本

### 阶段 1: Router 路由测试

**脚本**: `tests/test_router_routing.py`

**功能**: 测试 Router 模型的路由决策能力

**输入**:
- HotpotQA 测试集: `HotpotQA/hotpot_dev_distractor_1000_samples.jsonl`
- 训练好的 Router 模型: `router_models/no_rag_vs_naive/final/model.pt`

**输出**: `tests/router_test_results.json`

**运行方式**:
```powershell
# 测试所有样本
python tests/test_router_routing.py

# 测试前 100 个样本
python tests/test_router_routing.py --num-samples 100
```

**输出格式**:
```json
{
  "questions": [
    {
      "_id": "xxx",
      "question": "...",
      "predicted_strategy": "no_rag",
      "strategy_index": 0,
      "confidence": 0.85
    }
  ],
  "statistics": {
    "total_questions": 1000,
    "no_rag_count": 650,
    "naive_rag_count": 350,
    "no_rag_ratio": 0.65,
    "naive_rag_ratio": 0.35
  }
}
```

### 阶段 2: 端到端测试

**脚本**: `tests/test_router_end_to_end.py`（待创建）

**功能**: 测试 Router + RAG 的完整端到端性能

**流程**:
1. 读取阶段 1 的路由决策 (`tests/router_test_results.json`)
2. 根据路由决策调用对应的 RAG 方法：
   - `no_rag`: 调用 NoRAG.execute()
   - `naive_rag`: 调用 NaiveRAG.execute()
3. 计算 EM/F1 分数和执行时间
4. 统计平均检索时间、生成时间、总时间

**输出**: `tests/router_end_to_end_results.json`

## RAG 实现说明

### NoRAG
- **类**: `rag_implementations/no_rag/no_rag_impl.py`
- **方法**: `execute(question, context=None)`
- **功能**: 直接调用 LLM 回答，不进行检索

### NaiveRAG
- **类**: `rag_implementations/naive_rag/naive_rag_impl.py`
- **方法**: `execute(question, context=None)`
- **功能**: 使用 LlamaIndex 检索 + LLM 生成
- **时间统计**: `last_retrieval_time`（检索时间）

## 测试流程

### 第一步：运行路由测试

```powershell
python tests/test_router_routing.py
```

这会：
- 加载训练好的 Router 模型
- 对 1000 个测试问题进行路由决策
- 输出 `router_test_results.json`

### 第二步：创建端到端测试脚本

根据路由决策调用对应的 RAG 方法，评估端到端性能（EM/F1、时间）。

### 第三步：运行端到端测试

```powershell
python tests/test_router_end_to_end.py
```

## 预期输出

### 路由决策统计
- 路由分布：no_rag vs naive_rag 的比例
- 置信度：模型对路由选择的自信度
- 路由准确性：如果后续有 ground truth，可以计算准确率

### 端到端指标
- **EM (Exact Match)**: 精确匹配分数
- **F1 Score**: F1 分数
- **平均检索时间**: NaiveRAG 的平均检索时间
- **平均生成时间**: 生成答案的时间
- **平均总时间**: 检索 + 生成的总时间

## 关于训练时间问题

训练日志显示 19:42:58 开始，21:39:32 结束，只用了 33 秒就完成了 20 个 epoch。可能原因：

1. **数据未正确加载**：虽然日志显示 5000 样本，但实际可能有问题
2. **提前退出**：可能有错误导致提前退出
3. **max_steps 限制**：之前测试时使用了 `--max_steps 5`

建议检查 `router_models/no_rag_vs_naive/logs/router_training_20260131_194258.log` 详细日志。

## 目录结构

```
tests/
├── test_router_routing.py      # 阶段 1: 路由测试
├── router_test_results.json     # 阶段 1 输出
├── test_router_end_to_end.py  # 阶段 2: 端到端测试
└── router_end_to_end_results.json  # 阶段 2 输出
```
