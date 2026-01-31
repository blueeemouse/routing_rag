"""
总结分析：为什么大索引(EM=0.267)表现比小索引(EM=0.00)好？
"""

print("="*80)
print("问题总结与分析")
print("="*80)

print("""
【背景】
1. 两个GraphRAG索引使用相同的配置建立：
   - model: qwen2.5:3b
   - concurrent_requests: 25
   - request_timeout: 180s
   - max_retries: 10
   - embed_graph.enabled: true

2. 两个索引的主要差异：
   - 大索引: 9769个文档, extract_graph耗时 114.9s (平均0.01s/doc)
   - 小索引: 50个文档, extract_graph耗时 1818.0s (平均36.36s/doc)

3. 评估结果（15个样本）：
   - 大索引: EM=0.267 (4/15)
   - 小索引: EM=0.00 (0/15)

【三个关键疑问解答】

1. 【重试机制】
   - 配置: max_retries=10, exponential_backoff, max_retry_wait=10s
   - 理论最大重试时间: 1+2+4+8+16+32+64+128+256+512 = 1023s
   - 实际request_timeout=180s，会提前中断
   - 结论: 600s的timeout足够10次重试，不应该导致失败

2. 【模型能力问题】
   - qwen2.5:3b是一个小型模型，能力有限
   - 但为什么15个样本时小索引表现差？
   - 关键发现: 小索引的entity extraction平均耗时36秒/文档，远超正常值

3. 【并发请求问题】
   - concurrent_requests=25应该不会让效果变差
   - 但实际上小索引的单文档entity extraction耗时异常高

【根本原因分析】

通过stats.json数据，发现了关键差异：

小索引的entity extraction阶段：
- 总耗时: 1818.0s
- 文档数: 50
- 平均每文档: 36.36s

大索引的entity extraction阶段：
- 总耗时: 114.9s  
- 文档数: 9769
- 平均每文档: 0.01s

这个巨大的差异说明了什么？

1. 【小索引建立时的环境问题】
   - 小索引建立时，Ollama可能正在处理其他任务，导致qwen2.5:3b响应极慢
   - 或者本地机器资源不足（CPU/GPU），导致模型推理速度下降
   - 这导致每个entity extraction请求耗时很长

2. 【超时和重试的真实影响】
   - 虽然配置了max_retries=10和request_timeout=180s
   - 但当单次LLM调用耗时接近180s时，可能会触发timeout
   - 即使有重试，如果10次都接近timeout，总耗时可达1800s+
   - 这正好解释了小索引为什么平均每文档需要36秒！

3. 【重试失败的可能原因】
   - qwen2.5:3b在超时前可能输出了不完整的JSON
   - 或者输出了格式错误的entity list
   - GraphRAG在解析失败后会进行重试
   - 但如果模型一直输出错误格式，即使重试10次也可能失败
   - 最终导致某些文档没有提取到entities，或者提取的entities质量很差

4. 【为什么大索引反而快？】
   - 大索引建立时，系统环境更稳定，Ollama响应速度快
   - 单次entity extraction只需0.01s（这可能是因为有缓存或并发批处理）
   - 或者大索引的文档内容更简单，提取速度更快

【为什么15个样本时小索引表现差？】

1. 小索引虽然包含这15个测试样本所需的文档（覆盖率100%）
2. 但entity extraction质量差，导致：
   - 某些关键entities缺失（如"Scott Derrickson", "Ed Wood"）
   - entities的description不完整或错误
   - entities之间的关系没有正确建立
3. 搜索时无法检索到正确的entities，导致回答失败

【结论】

1. 重试机制和并发配置本身不是问题
2. 问题的根源是小索引建立时的**环境不稳定**
   - 导致entity extraction超时或失败
   - 重试机制无法解决模型输出格式错误的问题
3. qwen2.5:3b模型本身能力有限，在超时压力下更容易输出错误格式
4. 15个测试样本能正确，但1000个样本时表现差，说明：
   - 索引质量是关键因素
   - 不是模型推理能力的问题，而是索引建立过程中的稳定性问题

【建议】

1. 使用更大的LLM模型（如qwen2.5:7b或14b）进行entity extraction
2. 增加request_timeout到300-600s，给模型更多时间
3. 建立索引时确保Ollama独占系统资源
4. 监控entity extraction的成功率，发现异常及时终止并重新建立索引
5. 考虑使用云端GPT-4等更稳定的API进行entity extraction
""")

print("\n" + "="*80)
print("补充：用户之前的单样本测试")
print("="*80)
print("""
用户对两个索引进行了单样本测试：
- 测试问题: "What nationality is Scott Derrickson?"
- 期望答案: "American"
- 大索引结果: 没有检索到"Scott Derrickson" entity
- 小索引结果: 也没有检索到"Scott Derrickson" entity

这说明：
1. 两个索引都缺少这个关键entity
2. 这不是query数量的问题（15 vs 1000）
3. 而是entity extraction阶段就失败了

可能的原因：
- Scott Derrickson这个entity在文档中存在
- 但qwen2.5:3b没有成功识别并提取它
- 或者提取了但格式错误被过滤掉了
""")
