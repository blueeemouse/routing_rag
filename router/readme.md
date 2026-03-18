# 查询路由器模块 (Query Router Module)

此模块负责查询路由策略的决策，包括基于规则的 LLM 路由和基于学习的可训练路由器。

## 功能

### 1. 基于规则的 LLM 路由（Router）
- 根据子查询类型决定处理策略
- 支持多种 RAG 处理方式：无 RAG、朴素 RAG、图 RAG
- 提供统一的接口供其他模块调用
- 支持 API 和本地 Ollama 模型

### 2. 可训练路由器（Trainable Router）
- 基于对比学习训练路由决策模型
- 支持自定义评分公式（EM, F1, BERT Score, LLM Judge 等）
- 支持 DC Router（Double Contrastive）模型
- 通过配置文件灵活配置训练参数

## 文件结构

### 基础路由器
- `router.py` - 查询路由器的实现（基于 LLM 的规则路由）
- `llm_router.py` - LLM Router 实现（支持 zero-shot 和 few-shot 模式）
- `__init__.py` - 模块初始化文件

### 可训练路由器
- `trainable_router/` - 可训练路由器模块
  - `config.py` - 配置管理（支持 score_formula 自定义评分公式）
  - `factory.py` - 工厂模式（创建模型/训练器/数据集）
  - `data_utils.py` - 数据预处理工具（ScoreComputer 支持任意指标组合）
  - `datasets/hotpotqa_dataset.py` - HotpotQA 数据集实现
  - `models/dc_model.py` - DC Router 模型
  - `trainers/dc_trainer.py` - DC Router 训练器
  - `evaluate_router.py` - 评估脚本
  - `config_example.yaml` - 配置示例
  - `demo_usage.py` - 使用示例
  - `temp_dev_note.md` - 开发笔记（记录训练相关改动和决策）

## 配置

### 基础路由器配置

在 `config/settings.yaml` 中配置：

#### 使用 API 服务

```yaml
router:
  api_url: "${API_BASE_URL}/chat/completions"
  api_key: "${ROUTER_API_KEY}"
  model: "deepseek-v3"
  prompt: "确定查询处理策略：no_rag, naive_rag, 或 graph_rag。查询：{sub_query}\n策略："
```

#### 使用本地 Ollama

```yaml
router:
  api_url: "${API_BASE_URL}/chat/completions"
  api_key: "${ROUTER_API_KEY}"
  model: "qwen2.5:3b"
  temperature: 0.1
  prompt: |
    任务：确定查询处理策略

    可选策略：
    - no_rag: 直接回答，无需检索
    - naive_rag: 使用向量检索
    - graph_rag: 使用图检索

    规则：
    1. 简单事实性问题 → no_rag
    2. 需要信息检索的问题 → naive_rag
    3. 复杂关系查询 → graph_rag
    4. 只输出策略名称，不要任何解释

    查询：{sub_query}
    策略：
```

### LLM Router 配置

LLM Router 支持两种模式：**zero-shot**（仅使用prompt）和 **few-shot**（使用ICL样例）：

```yaml
llm_router:
  api_url: "${API_BASE_URL}"
  api_key: "${ROUTER_API_KEY}"
  model: "qwen2.5:3b"
  temperature: 0.0
  max_tokens: 20
  
  # 模式: "zero_shot" 或 "few_shot"
  mode: "zero_shot"
  
  # few-shot 配置
  few_shot_k: 5
  examples_file: "evaluation_results/router_test_labels.json"
  
  # 策略名称列表
  strategy_names: ["no_rag", "naive_rag"]
```

### 可训练路由器配置

在训练配置文件中指定（例如 `trainable_router/config/train_dc_em_f1.yaml`）：

```yaml
model_type: "dc"
model:
  backbone_name: "sentence-transformers/all-MiniLM-L6-v2"
  strategy_names: ["no_rag", "naive_rag", "graph_rag"]
  similarity_function: "cos"
  temperature: 1.0
  device: "cuda"
training:
  batch_size: 32
  learning_rate: 5.0e-5
  epochs: 10
  top_k: 3
  last_k: 3
data:
  source: "hotpotqa"
  train_path: "evaluation_results/train"
  score_formula: "em * 0.3 + f1 * 0.7"  # 自定义评分公式
```

#### 评分公式示例

- 单指标: `"em"`, `"f1"`, `"llm_judge"`, `"bert_score"`
- 组合指标: `"em * 0.3 + f1 * 0.7"`
- 成本权衡: `"em - 0.01 * total_time"`
- 任意组合: `"llm_judge * 0.6 + bert_score * 0.4"`

## 路由策略

- **no_rag**: 简单事实性问题，无需检索
- **naive_rag**: 需要信息检索的问题
- **graph_rag**: 复杂关系查询

## 模型支持

### 基础路由器
- **API 模型**: gpt-4o, gpt-3.5-turbo, deepseek-v3 等
- **Ollama 模型**: qwen2.5:3b, llama3.1:8b 等

### 可训练路由器
- **DC Router (Double Contrastive)**: 基于对比学习的路由器
  - 学习查询编码器
  - 学习策略嵌入
  - 支持自定义评分公式
- **Backbone 模型**: sentence-transformers/all-MiniLM-L6-v2 等

## 使用示例

### 基础路由器

```python
from router.router import Router

router = Router()
strategy = router.route("什么是人工智能？")
print(strategy)  # 输出: no_rag, naive_rag, 或 graph_rag
```

### LLM Router

```python
from router.llm_router import LLMRouter
from config.config import settings

# 方式1: 从配置创建
config = settings.llm_router_config
router = LLMRouter.from_config(config)

# 方式2: 直接创建
router = LLMRouter(
    api_url="http://localhost:11434/v1",
    api_key="ollama",
    model="qwen2.5:3b",
    mode="zero_shot"  # 或 "few_shot"
)

# 路由决策
strategy = router.route("谁是2020年美国总统？")
print(strategy)  # 输出: no_rag 或 naive_rag

# 批量路由
queries = ["1+1等于几？", "Python创始人是谁？"]
strategies = router.route_batch(queries)
print(strategies)

# 评估性能
test_data = [{"question": "...", "optimal_strategy": "no_rag"}, ...]
results = router.evaluate(test_data)
print(f"准确率: {results['accuracy']:.2%}")
```

### 可训练路由器

```python
from trainable_router.config import TrainableRouterConfig
from trainable_router.factory import TrainableRouterFactory

# 加载配置
config = TrainableRouterConfig.from_yaml("config/train_dc_em_f1.yaml")

# 创建数据集
dataset = TrainableRouterFactory.create_dataset(config, "generic", tokenizer)

# 创建模型和训练器
model = TrainableRouterFactory.create_model(config)
trainer = TrainableRouterFactory.create_trainer(model, config)

# 训练
trainer.train()
```

## 测试

### 测试基础路由器

#### 测试 API 模型
```bash
python tests/test_router.py
```

#### 测试 Ollama 模型
```bash
python tests/test_ollama_router.py
```

### 测试可训练路由器

```bash
# 运行训练
python router/train_router.py --config router/trainable_router/config/train_dc_em_f1.yaml

# 或使用 PowerShell 脚本
powershell -ExecutionPolicy Bypass -File router/train_router.ps1
```

## 开发文档

详细的开发笔记和改动记录请参考 `temp_dev_note.md`，包括：
- ScoreComputer 优化记录（删除 PRESETS，动态提取公式指标）
- 配置系统改进（添加 score_formula 支持）
- 设计决策（为什么不需要为不同指标创建单独的类）
- 训练流程和超参数配置