# Router训练问题诊断和改进建议

## 当前问题

### 1. TensorBoard没有保存
- **症状**: 日志显示"TensorBoard 日志将保存到: router_models/no_rag_vs_naive/tensorboard"
- **实际**: tensorboard目录不存在
- **原因**: 可能是TensorBoard库未安装，或者初始化失败但没有报错
- **影响**: 无法监控训练过程，loss曲线、准确率变化等都看不到

### 2. 模型全部路由到naive_rAG
- **症状**: 训练后的router对所有的query都返回naive_rAG
- **期望**: 应该部分路由到no_rag（约58%），部分路由到naive_rAG（约42%）
- **原因**: 模型没有学到正确的映射关系

## 数据分布分析

基于5000条训练数据的标签分布：
```
两者都答对（EM=1.0, F1=1.0）:     574条 (11.48%) → 标签: no_rag
两者都答错（EM=0.0, F1=0.0）:    1975条 (39.50%) → 标签: no_rag
仅NoRAG答对:                            361条 (7.22%)  → 标签: no_rag
仅NaiveRAG答对:                          2090条 (41.80%) → 标签: naive_rag

实际训练标签分布:
no_rag:     2910条 (58.20%)
naive_rAG:   2090条 (41.80%)
```

**结论**: 数据分布还算相对均衡，不应该导致模型全部偏向naive_rAG

## 可能的根本原因

### 1. 温度参数问题
- **当前**: temperature = 1.0
- **问题**: 温度太高，softmax输出过于平滑，模型难以收敛
- **建议**: 降低温度到 0.1-0.5

### 2. 归一化分数后的范围问题
- **当前**: normalize_scores = true
- **问题**: 归一化后，分数范围[0,1]，但分布可能很集中
- **示例**:
  - 原始分数：no_rag=[0, 1.0], naive_rAG=[0, 1.0]
  - 归一化后：no_rag=[0, 1.0], naive_rAG=[0, 1.0]
  - 很多样本的分数差距很小（如0.8 vs 0.9），模型难以区分

### 3. 学习率问题
- **当前**: learning_rate = 5e-5
- **问题**: 可能太小，特别是在温度参数不合适的情况下
- **建议**: 尝试更大的学习率 1e-4 或 5e-4

### 4. 策略embedding初始化问题
- **当前**: xavier_uniform_初始化
- **问题**: 如果初始化值都接近0，且backbone embedding归一化后，相似度矩阵可能接近均匀分布

### 5. 损失函数设计问题
- **当前**: sample_llm_loss_weight = 1.0, top_k=1, last_k=1
- **问题**: 
  - top_k=1只取1个正样本，last_k=1只取1个负样本
  - 对于2策略，这个设置可能导致模型难以学到明显偏好
  - log-sigmoid损失可能太温和

### 6. 没有验证集监控
- **当前**: val_path = ""
- **问题**: 无法监控训练是否过拟合，或验证模型泛化能力
- **建议**: 划分训练/验证集，如 train: 90%, val: 10%

## 改进建议

### 方案1: 调整超参数（推荐先尝试）

```yaml
# model配置
model:
  similarity_function: "cos"
  temperature: 0.2  # 从1.0降到0.2，让预测更明确

# training配置
training:
  learning_rate: 1e-4  # 从5e-5提升到1e-4
  batch_size: 32
  epochs: 20
  eval_steps: 50  # 每50步评估一次
  save_steps: 80  # 每80步保存一次
  max_grad_norm: 1.0
  
  # 温和学习率调整
  top_k: 1
  last_k: 1
```

### 方案2: 添加验证集

```yaml
# data配置
data:
  train_path: "HotpotQA_train_data"
  val_path: "HotpotQA_train_data"  # 使用相同数据，但会在代码中划分
  val_split_ratio: 0.1  # 划分10%作为验证集
```

### 方案3: 改进损失函数

```python
# 当前: log-sigmoid loss
loss += torch.nn.functional.logsigmoid(pos_score - neg_score)

# 改进: BCE loss (更强的梯度)
loss += -torch.log(torch.sigmoid(pos_score - neg_score))

# 或者: Triplet loss
loss += torch.relu(margin + neg_score - pos_score)
```

### 方案4: 调整数据标签（处理tie）

对于tie的情况（都答对或都答错），可以：
1. **随机分配**: 随机标记为no_rag或naive_rAG，增加数据多样性
2. **排除tie数据**: 只使用有明确偏好的361+2090=2451条数据
3. **使用额外指标**: 对tie数据，使用其他指标（如推理时间）来区分

```python
# 示例：随机分配tie数据
if no_rag_score == naive_rag_score:
    label = 'no_rag' if random.random() < 0.5 else 'naive_rag'
```

### 方案5: 强制平衡数据集

在训练时对no_rag和naive_rAG标签进行重采样，使其完全均衡：

```python
# 在DataLoader中添加重采样
# 确保每个batch中no_rag和naive_rAG样本数量相等
```

### 方案6: 检查TensorBoard

1. 确认tensorboard是否安装：
```bash
pip install tensorboard
```

2. 检查训练日志，看是否有"TensorBoard 未安装"的消息

3. 如果可用，启动tensorboard：
```bash
cd router_models/no_rag_vs_naive
tensorboard --logdir tensorboard
```

## 推荐的实验顺序

### 第1步: 安装TensorBoard并重新训练（监控训练过程）
```bash
pip install tensorboard
python router/train_router.py --config config/train_no_rag_vs_naive.yaml
# 然后启动 tensorboard --logdir router_models/no_rag_vs_naive/tensorboard
```

### 第2步: 降低temperature + 提高学习率
修改配置文件：
```yaml
model:
  temperature: 0.2  # 从1.0降到0.2
  
training:
  learning_rate: 1e-4  # 从5e-5提升到1e-4
```

### 第3步: 添加验证集监控
修改代码或配置，确保每个epoch都有验证

### 第4步: 调整数据标签（如果前3步还不够）
修改数据加载逻辑，随机分配tie数据

## 诊断脚本

建议创建一个诊断脚本来检查训练后的模型：

```python
# diagnose_model.py
# 功能：
# 1. 检查策略embedding的分布
# 2. 在验证集上评估
# 3. 分析不同分数范围的预测分布
# 4. 可视化loss曲线（如果tensorboard不可用）
```

## 关键指标

训练时应该关注：
1. **Training Loss**: 应该稳定下降
2. **Validation Accuracy**: 应该逐渐提升，至少>50%
3. **每个策略的准确率**: 
   - 应该都能学习到一些代表性样本
   - no_rag准确率 ≈ 58%
   - naive_rAG准确率 ≈ 42%
4. **Loss的组成部分**: sample_llm_loss应该下降

## 预期结果

如果训练成功，模型应该：
1. 对简单的query（两者都能答对的）能正确路由
2. 对困难的query（两者都答错的）可能偏向某一方
3. 对有明显偏好的query能正确路由
4. 整体准确率在60-70%之间（比随机50%要好）
