import json
import jsonlines

# 把json文件转换成jsonl格式（仿照HopRAG做法）
# 输入输出文件路径
input_file = "D:\\Develop\\all_RAG\\routing_rag\\HotpotQA\\hotpot_train_v1.1_5000_samples.json"
output_file = "D:\\Develop\\all_RAG\\routing_rag\\HotpotQA\\hotpot_train_v1.1_5000_samples.jsonl"

# 读取 JSON 数据
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 写入 JSON Lines 格式
with jsonlines.open(output_file, mode='w') as writer:
    for item in data:
        writer.write(item)

print(f"转换完成: {input_file} -> {output_file}")