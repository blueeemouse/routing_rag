import json
import jsonlines

# 输入输出文件路径
input_file = "D:\\Develop\\all_RAG\\routing_rag\\HotpotQA\\hotpot_dev_distractor_1000_samples.json"
output_file = "D:\\Develop\\all_RAG\\routing_rag\\HotpotQA\\hotpot_dev_distractor_1000_samples.jsonl"

# 读取 JSON 数据
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 写入 JSON Lines 格式
with jsonlines.open(output_file, mode='w') as writer:
    for item in data:
        writer.write(item)

print(f"转换完成: {input_file} -> {output_file}")