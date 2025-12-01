import json
from pathlib import Path

# 你的原始HotpotQA文件路径
input_json = Path("D:/Develop/all_RAG/routing_rag/HotpotQA/raw/qa_test.json")
# 转换后的JSONL文件路径
output_jsonl = Path("D:/Develop/all_RAG/routing_rag/HotpotQA/raw/qa_test.jsonl")

# 读取原始JSON数组
with open(input_json, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# 转换逻辑：
# 1. 外层JSON数组 → 每行一个JSON对象（JSONL）
# 2. 内层context列表 → 拼接为单个字符串（用换行/空格分隔）
with open(output_jsonl, "w", encoding="utf-8") as f:
    for idx, doc in enumerate(raw_data):
        # 确保每个文档有唯一id（没有的话手动生成）
        doc["id"] = doc.get("id", f"hotpotqa_{idx}")
        # 扁平化context列表为纯文本（用两个换行分隔，方便后续分块）
        if isinstance(doc.get("context"), list):
            doc["context"] = "\n\n".join(doc["context"])
        # 写入JSONL（每行一个对象）
        json.dump(doc, f, ensure_ascii=False)
        f.write("\n")

print(f"转换完成！生成文件：{output_jsonl}")
print(f"转换后文档数量：{len(raw_data)}")