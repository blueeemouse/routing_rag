# HotpotQA评测说明

## 📋 评测概述

参考RouteRAG的评测方式，测试NoRAG（仅用LLM回答）和NaiveRAG（检索增强生成）在HotpotQA数据集上的性能对比。

## 🎯 评测目标

1. **对比NoRAG vs NaiveRAG**：评估检索增强对回答质量的提升
2. **验证系统性能**：在真实数据集上测试现有实现
3. **指导优化方向**：通过评测结果发现改进空间

## 📊 评测指标

- **EM (Exact Match)**：精确匹配率，预测答案与标准答案完全一致的比例
- **F1 Score**：F1分数，衡量预测答案与标准答案的重叠度
- **Accuracy**：准确率（等同于EM）

## 📁 文件说明

### 1. `evaluate_hotpotqa.py` - 完整评测脚本
- **用途**：完整的HotpotQA评测流程
- **功能**：
  - 从HotpotQA数据集提取文档和查询
  - 构建NaiveRAG索引
  - 批量评测NoRAG和NaiveRAG
  - 计算EM和F1分数
  - 保存详细结果和对比报告
- **适用场景**：正式评测、性能分析

### 2. `quick_evaluate_hotpotqa.py` - 快速评测脚本
- **用途**：快速测试少量样本
- **功能**：
  - 加载少量样本（默认5个）
  - 实时显示每个查询的结果
  - 快速对比NoRAG和NaiveRAG
- **适用场景**：快速验证、调试、演示

## 🚀 使用方法

### 前置准备

1. **确保环境配置正确**：
   ```bash
   cd D:\Develop\all_RAG\routing_rag
   ```

2. **加载环境变量**（Windows）：
   ```powershell
   Get-Content .env | ForEach-Object {
       if ($_ -match '^([^=]+)=(.*)$') {
           [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2])
       }
   }
   ```

3. **确保HotpotQA数据集存在**：
   - 文件路径：`HotpotQA/hotpot_1000_samples.jsonl`
   - 如果不存在，需要先准备数据集

### 快速评测（推荐先运行）

```bash
cd tests
python quick_evaluate_hotpotqa.py
```

**预期输出**：
```
================================================================================
HotpotQA快速评测 - NoRAG vs NaiveRAG
================================================================================

正在从 D:\Develop\all_RAG\routing_rag\HotpotQA\hotpot_1000_samples.jsonl 读取前 5 个样本...
已加载 5 个样本，获得 20 个文档块。

初始化NoRAG...

初始化NaiveRAG...
构建NaiveRAG索引...
索引构建成功，包含 20 个文档。

开始评测 NoRAG...

[1/5] 问题: Donnie Smith who plays as a left back for New England Revolution...
预测答案: ...
标准答案: ['Major League Soccer']
EM: 1.00, F1: 1.00
...

================================================================================
评测结果: NoRAG
================================================================================
查询数量: 5
平均EM: 0.6000
平均F1: 0.7500
================================================================================

开始评测 NaiveRAG...

[1/5] 问题: Donnie Smith who plays as a left back for New England Revolution...
预测答案: ...
标准答案: ['Major League Soccer']
EM: 1.00, F1: 1.00
...

================================================================================
评测结果: NaiveRAG
================================================================================
查询数量: 5
平均F1: 0.8500
================================================================================

================================================================================
对比结果
================================================================================
模型            EM         F1
--------------------------------------------------------------------------------
NoRAG           0.6000     0.7500
NaiveRAG        0.8000     0.8500
--------------------------------------------------------------------------------
提升            0.2000     0.1000
================================================================================
```

### 完整评测

```bash
cd tests
python evaluate_hotpotqa.py
```

**配置参数**：
- 修改脚本中的`NUM_SAMPLES`来控制评测样本数
- 修改`OUTPUT_DIR`来改变输出目录

**输出文件**：
- `evaluation_results/no_rag_results.json`：NoRAG详细结果
- `evaluation_results/naive_rag_results.json`：NaiveRAG详细结果
- `evaluation_results/comparison_results.json`：对比结果

## 📈 结果分析

### 评测结果文件格式

**单个模型结果** (`no_rag_results.json`)：
```json
{
  "model_name": "NoRAG",
  "predictions": [
    {
      "question": "问题文本",
      "gold_answer": ["标准答案1", "标准答案2"],
      "prediction": "预测答案",
      "em": 1.0,
      "f1": 1.0
    }
  ],
  "exact_matches": [1.0, 0.0, 1.0, ...],
  "f1_scores": [1.0, 0.5, 1.0, ...],
  "avg_em": 0.75,
  "avg_f1": 0.85,
  "accuracy": 0.75,
  "num_errors": 0
}
```

**对比结果** (`comparison_results.json`)：
```json
{
  "models": [...],
  "improvement": {
    "em": 0.15,
    "f1": 0.10
  }
}
```

### 解读要点

1. **EM vs F1**：
   - EM严格，要求完全匹配
   - F1宽松，允许部分匹配
   - NaiveRAG通常F1提升更明显

2. **性能提升**：
   - 正向提升：NaiveRAG > NoRAG，说明检索有帮助
   - 负向提升：NaiveRAG < NoRAG，可能原因：
     - 检索质量差
     - 索引构建问题
     - 查询复杂度高

3. **错误分析**：
   - 查看`predictions`中的具体案例
   - 分析错误原因（检索错误、生成错误等）

## 🔧 常见问题

### 1. API调用失败
**问题**：`NoRAG调用LLM时出错` 或 `NaiveRAG索引构建失败`

**解决方案**：
- 检查`.env`文件中的API密钥是否正确
- 确认API endpoint是否可访问
- 检查网络连接

### 2. 索引构建慢
**问题**：`构建NaiveRAG索引`时间过长

**解决方案**：
- 减少评测样本数（`NUM_SAMPLES`）
- 使用更快的嵌入模型
- 考虑使用预构建索引

### 3. 内存不足
**问题**：构建索引时内存溢出

**解决方案**：
- 减少文档数量
- 分批构建索引
- 使用更小的chunk_size

### 4. 评测结果不理想
**问题**：NaiveRAG性能不如NoRAG

**可能原因**：
- 索引构建不完整
- 检索参数需要调优（top_k、chunk_size等）
- 数据集样本量太小

**解决方案**：
- 增加评测样本数
- 调整检索参数
- 检查索引构建质量

## 📝 扩展功能

### 添加更多模型

在评测脚本中添加新的模型评测：

```python
# 初始化其他模型
other_rag = OtherRAG()
other_rag.build_index_from_data(documents)

# 评测
other_rag_results = evaluate_model(other_rag, queries, "OtherRAG")
```

### 自定义评测指标

在评测脚本中添加新的指标计算函数：

```python
def compute_custom_metric(gold_answers, prediction):
    # 自定义指标计算逻辑
    return score

# 在evaluate_model函数中调用
custom_score = compute_custom_metric(gold_answers, prediction)
```

### 保存详细日志

修改评测脚本，添加日志记录：

```python
import logging

logging.basicConfig(filename='evaluation.log', level=logging.INFO)
logging.info(f"Evaluating query: {question}")
```

## 🎓 参考资源

- **RouteRAG评测方式**：参考`other-RouteRAG/RouteRAG/test.py`
- **HotpotQA数据集**：https://hotpotqa.github.io/
- **SQuAD评测指标**：https://rajpurkar.github.io/SQuAD-explorer/

## 📞 联系与反馈

如有问题或建议，请通过以下方式联系：
- 创建Issue
- 提交Pull Request
- 联系项目维护者

---

**最后更新**：2026-01-09