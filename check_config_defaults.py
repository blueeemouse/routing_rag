# -*- coding: utf-8 -*-
"""
对比用户配置与GraphRAG默认值
"""

print("="*80)
print("GraphRAG配置对比分析")
print("="*80)

print("""
根据GraphRAG源码分析（graphrag/config/defaults.py），以下是你配置中与默认值不同的参数：

【1. chunks配置】
你配置：
  size: 300
  overlap: 50

GraphRAG默认值（ChunksDefaults）：
  size: 1200
  overlap: 100

X 不同！你的chunk size和overlap都比默认值小4倍

---

【2. community_reports配置】
你配置：
  max_input_length: 2000
  max_length: 500
  concurrent_requests: 8  (新增配置)
  async_mode: asyncio  (新增配置)

GraphRAG默认值（CommunityReportDefaults）：
  max_input_length: 8000
  max_length: 2000

X 不同！你的max_input_length和max_length都比默认值小4倍
[注意] concurrent_requests和async_mode不是community_reports的默认配置项
     这些参数可能来自models部分的LanguageModelConfig

---

【3. embed_graph配置】
你配置：
  dimensions: 768
  enabled: false

GraphRAG默认值（EmbedGraphDefaults）：
  dimensions: 1536
  enabled: false

OK enabled与默认值相同
X dimensions不同（你用的是768，默认1536）
   这是因为你使用的nomic-embed-text模型输出768维embeddings

---

【4. embeddings配置】
你配置：
  batch_max_tokens: 8191
  batch_size: 16

GraphRAG默认值（EmbedTextDefaults）：
  batch_max_tokens: 8191
  batch_size: 16

OK 与默认值完全相同

---

【5. extract_graph配置】
你配置：
  entity_types: [person, organization, event, location, gpe]
  max_gleanings: 1
  concurrent_requests: 8  (新增配置)
  batch_size: 10  (新增配置)

GraphRAG默认值（ExtractGraphDefaults）：
  entity_types: [organization, person, geo, event]
  max_gleanings: 1

X entity_types不同！
   你的: [person, organization, event, location, gpe]
   默认: [organization, person, geo, event]
   你把"geo"改成了"location"和"gpe"
[注意] concurrent_requests和batch_size不是extract_graph的默认配置项
     这些参数可能来自extract_graph_nlp配置

---

【6. extract_graph_nlp配置】
你配置的concurrent_requests: 8实际上对应的是：

GraphRAG默认值（ExtractGraphNLPDefaults）：
  concurrent_requests: 25
  async_mode: AsyncType.Threaded  (即threaded)

X 不同！
   你：concurrent_requests: 8, async_mode: asyncio
   默认：concurrent_requests: 25, async_mode: threaded

   这解释了为什么你在extract_graph下配置的concurrent_requests=8
   实际上这个配置应该放在extract_graph_nlp下

---

【7. models - default_chat_model】
你配置：
  api_base: http://127.0.0.1:11434/v1
  api_key: ollama
  concurrent_requests: 16
  max_tokens: 4096
  model: qwen2.5:3b
  model_provider: openai
  request_timeout: 1200.0
  temperature: 0
  type: chat

GraphRAG默认值（LanguageModelDefaults）：
  model: gpt-4-turbo-preview
  model_provider: (未指定，会根据type推断)
  max_tokens: (未指定)
  temperature: 0
  request_timeout: 180.0
  concurrent_requests: 25
  max_retries: 10
  max_retry_wait: 10.0
  retry_strategy: exponential_backoff
  async_mode: threaded

X 不同！
   - model: 你用qwen2.5:3b，默认gpt-4-turbo-preview
   - model_provider: 你用openai（适配Ollama），默认未指定
   - concurrent_requests: 你用16，默认25
   - request_timeout: 你用1200.0，默认180.0
   - max_tokens: 你用4096，默认未指定

---

【8. models - default_embedding_model】
你配置：
  api_base: http://127.0.0.1:11434/v1
  api_key: ollama
  concurrent_requests: 16
  model: nomic-embed-text
  model_provider: openai
  request_timeout: 1200.0
  type: embedding

GraphRAG默认值（LanguageModelDefaults）：
  model: text-embedding-3-small
  request_timeout: 180.0
  concurrent_requests: 25
  max_retries: 10
  max_retry_wait: 10.0
  retry_strategy: exponential_backoff
  async_mode: threaded

X 不同！
   - model: 你用nomic-embed-text，默认text-embedding-3-small
   - concurrent_requests: 你用16，默认25
   - request_timeout: 你用1200.0，默认180.0

---

【9. cluster_graph配置】
你配置：
  max_cluster_size: 10
  use_lcc: true

GraphRAG默认值（ClusterGraphDefaults）：
  max_cluster_size: 10
  use_lcc: true

OK 与默认值完全相同

---

【10. prune_graph配置】
你配置：
  min_edge_weight_pct: 40.0
  min_node_degree: 1

GraphRAG默认值（PruneGraphDefaults）：
  min_edge_weight_pct: 40.0
  min_node_degree: 1

OK 与默认值完全相同

---

【11. summarize_descriptions配置】
你配置：
  max_input_tokens: 1000
  max_length: 200

GraphRAG默认值（SummarizeDescriptionsDefaults）：
  max_input_tokens: 4000
  max_length: 500

X 不同！你的max_input_tokens和max_length都比默认值小

---

【12. snapshots配置】
你配置：
  embeddings: false
  graphml: false
  raw_graph: false

GraphRAG默认值（SnapshotsDefaults）：
  embeddings: false
  graphml: false
  raw_graph: false

OK 与默认值完全相同
""")

print("="*80)
print("总结：与GraphRAG默认值不同的配置项")
print("="*80)
print("""
[X] 不同的配置项：
1. chunks.size: 300 (默认1200)
2. chunks.overlap: 50 (默认100)
3. community_reports.max_input_length: 2000 (默认8000)
4. community_reports.max_length: 500 (默认2000)
5. embed_graph.dimensions: 768 (默认1536) - 合理，因为你用了nomic-embed-text
6. extract_graph.entity_types: [person, organization, event, location, gpe]
   (默认[organization, person, geo, event])
7. extract_graph_nlp.concurrent_requests: 8 (默认25)
8. extract_graph_nlp.async_mode: asyncio (默认threaded)
9. models.concurrent_requests: 16 (默认25)
10. models.request_timeout: 1200.0 (默认180.0)
11. models.model: qwen2.5:3b / nomic-embed-text (默认gpt-4-turbo-preview / text-embedding-3-small)
12. summarize_descriptions.max_input_tokens: 1000 (默认4000)
13. summarize_descriptions.max_length: 200 (默认500)

[注意] 关于你注释中提到的"新增配置"：
   - community_reports.concurrent_requests: 这个参数在community_reports配置中不存在
     实际应该放在models.concurrent_requests
   - community_reports.async_mode: 同上
   - extract_graph.concurrent_requests: 这个参数在extract_graph配置中不存在
     实际应该放在extract_graph_nlp.concurrent_requests
   - extract_graph.batch_size: 这个参数在extract_graph配置中不存在
     可能来自embeddings.batch_size（但那个用于embedding，不是graph extraction）
""")

print("\n建议：")
print("""
1. 如果你想要更小的chunks和更短的报告长度，当前配置是合理的
   但这会影响entity extraction和community reports的详细程度

2. concurrent_requests从25降到16/8，可以减少并发压力
   对于本地Ollama是合理的调整

3. request_timeout从180s增加到1200s，给qwen2.5:3b更多时间
   这对小型模型是必要的

4. entity_types中的"location"和"gpe"可能重复
   建议查看是否会导致重复提取

5. embed_graph.enabled=false是正确的，避免finalize_graph阶段的ZeroDivisionError
""")
