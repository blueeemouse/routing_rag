"""
HotpotQA快速评测脚本
快速测试NoRAG和NaiveRAG在少量样本上的性能
"""

import json
import sys
import os
from typing import List, Dict, Any
from collections import Counter
import re
import string
import time

# Add routing_rag path
ROUTING_RAG_ROOT = r"D:\Develop\all_RAG\routing_rag"
sys.path.insert(0, ROUTING_RAG_ROOT)

# 加载环境变量（从.env文件）
ENV_FILE = os.path.join(ROUTING_RAG_ROOT, '.env')
if os.path.exists(ENV_FILE):
    print(f"正在加载环境变量: {ENV_FILE}")
    with open(ENV_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()
    print("环境变量加载完成")
else:
    print(f"警告: 未找到.env文件: {ENV_FILE}")

# 打印关键环境变量（用于调试）
print(f"API_BASE_URL: {os.getenv('API_BASE_URL', 'NOT SET')}")
print(f"NAIVE_RAG_API_KEY: {os.getenv('NAIVE_RAG_API_KEY', 'NOT SET')[:10]}..." if os.getenv('NAIVE_RAG_API_KEY') else "NAIVE_RAG_API_KEY: NOT SET")

from rag_implementations.naive_rag.naive_rag_impl import NaiveRAG
from rag_implementations.no_rag.no_rag_impl import NoRAG


def normalize_answer(s: str) -> str:
    """标准化答案文本"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(gold_answers: List[str], prediction: str) -> float:
    """计算精确匹配分数"""
    normalized_prediction = normalize_answer(prediction)
    for gold_answer in gold_answers:
        if normalize_answer(gold_answer) == normalized_prediction:
            return 1.0
    return 0.0


def compute_f1(gold_answers: List[str], prediction: str) -> float:
    """计算F1分数"""
    normalized_prediction = normalize_answer(prediction)
    prediction_tokens = normalized_prediction.split()

    max_f1 = 0.0
    for gold_answer in gold_answers:
        normalized_gold = normalize_answer(gold_answer)
        gold_tokens = normalized_gold.split()

        common = Counter(prediction_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            continue

        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(gold_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        max_f1 = max(max_f1, f1)

    return max_f1


def extract_gold_answers(sample: Dict[str, Any]) -> List[str]:
    """从HotpotQA样本中提取标准答案"""
    gold_answers = []

    if 'answer' in sample:
        answer = sample['answer']
        if isinstance(answer, str):
            gold_answers.append(answer)
        elif isinstance(answer, list):
            gold_answers.extend(answer)

    if 'answer_aliases' in sample:
        gold_answers.extend(sample['answer_aliases'])

    gold_answers = list(set(gold_answers))
    return gold_answers if gold_answers else [""]


def load_hotpotqa_samples(jsonl_file_path: str, num_samples: int = 10) -> tuple[List[str], List[Dict[str, Any]]]:
    """加载HotpotQA样本"""
    documents = []
    queries = []

    print(f"正在从 {jsonl_file_path} 读取前 {num_samples} 个样本...")
    count = 0
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if count >= num_samples:
                break

            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                # 提取文档
                context = data.get('context', [])
                for title, sentence_list in context:
                    doc_text = "\n\n".join(sentence_list)
                    documents.append(doc_text)

                # 提取查询
                queries.append(data)
                count += 1

            except json.JSONDecodeError as e:
                print(f"警告: 解析第 {count+1} 行 JSON 时出错: {e}")
                continue

    print(f"已加载 {count} 个样本，获得 {len(documents)} 个文档块。")
    return documents, queries


def quick_evaluate(
    model,
    queries: List[Dict[str, Any]],
    model_name: str,
    delay_seconds: float = 2.5,
    record_retrieval_time: bool = False,
) -> Dict[str, Any]:
    """快速评测"""
    results = {
        'model_name': model_name,
        'predictions': [],
        'exact_matches': [],
        'f1_scores': [],
        'total_time': 0.0,
        'avg_time': 0.0,
        'total_retrieval_time': 0.0,
        'avg_retrieval_time': 0.0,
        'total_generation_time': 0.0,
        'avg_generation_time': 0.0,
        'total_generation_tokens': 0,
        'avg_generation_tokens': 0
    }

    print(f"\n开始评测 {model_name}...")

    for i, query_data in enumerate(queries):
        question = query_data['question']
        gold_answers = extract_gold_answers(query_data)

        try:
            print(f"\n[{i+1}/{len(queries)}] 问题: {question[:50]}...")

            # 记录开始时间
            start_time = time.time()

            # 执行查询
            prediction = model.execute(question)

            # 记录结束时间
            end_time = time.time()

            # 计算总时间
            total_time = end_time - start_time
            results['total_time'] += total_time

            # 获取生成时间
            if hasattr(model, 'last_generation_time'):
                generation_time = model.last_generation_time
                generation_tokens = getattr(model, 'last_generation_tokens', 0)
            else:
                generation_time = total_time
                generation_tokens = 0

            # 获取检索时间
            if record_retrieval_time and hasattr(model, 'last_retrieval_time'):
                retrieval_time = model.last_retrieval_time
                results['total_retrieval_time'] += retrieval_time
            else:
                retrieval_time = 0.0

            # 累计生成时间和tokens
            results['total_generation_time'] += generation_time
            results['total_generation_tokens'] += generation_tokens

            # 计算指标
            em = compute_exact_match(gold_answers, prediction)
            f1 = compute_f1(gold_answers, prediction)

            print(f"EM: {em:.2f}, F1: {f1:.2f}")
            print(f"总时间: {total_time:.2f}s", end='')
            print(f", 生成时间: {generation_time:.2f}s", end='')
            if retrieval_time > 0:
                print(f", 检索时间: {retrieval_time:.2f}s", end='')
            if generation_tokens > 0:
                print(f", 生成tokens: {generation_tokens}", end='')
            print()
            print(f"预测答案: {prediction[:100]}...")
            print(f"标准答案: {gold_answers}")

            results['predictions'].append({
                'question': question,
                'gold_answer': gold_answers,
                'prediction': prediction,
                'em': em,
                'f1': f1,
                'total_time': total_time,
                'retrieval_time': retrieval_time,
                'generation_time': generation_time,
                'generation_tokens': generation_tokens
            })

            results['exact_matches'].append(em)
            results['f1_scores'].append(f1)

            # 延迟
            if delay_seconds > 0 and i < len(queries) - 1:
                time.sleep(delay_seconds)

        except Exception as e:
            print(f"错误: {e}")
            results['exact_matches'].append(0.0)
            results['f1_scores'].append(0.0)

    # 计算平均值
    num_queries = len(results['exact_matches'])
    results['avg_em'] = sum(results['exact_matches']) / num_queries
    results['avg_f1'] = sum(results['f1_scores']) / num_queries
    results['avg_time'] = results['total_time'] / num_queries
    results['avg_generation_time'] = results['total_generation_time'] / num_queries
    if results['total_retrieval_time'] > 0:
        results['avg_retrieval_time'] = results['total_retrieval_time'] / num_queries
    if results['total_generation_tokens'] > 0:
        results['avg_generation_tokens'] = results['total_generation_tokens'] / num_queries

    return results


def load_and_filter_settings(settings_path: str) -> Dict[str, Any]:
    """
    加载settings.yaml并过滤敏感信息（保留prompt）

    Args:
        settings_path: settings.yaml文件路径

    Returns:
        过滤后的配置字典
    """
    import yaml
    from copy import deepcopy

    # 读取配置文件
    with open(settings_path, 'r', encoding='utf-8') as f:
        settings = yaml.safe_load(f)

    # 深拷贝一份，避免修改原始配置
    filtered_settings = deepcopy(settings)

    # 敏感字段列表（但保留prompt）
    sensitive_patterns = ['api_key', 'secret', 'token', 'password']

    def filter_sensitive(obj):
        """递归过滤敏感信息"""
        if isinstance(obj, dict):
            filtered = {}
            for key, value in obj.items():
                # 检查是否是敏感字段（但保留prompt）
                if key.lower() in sensitive_patterns:
                    filtered[key] = "***FILTERED***"
                else:
                    filtered[key] = filter_sensitive(value)
            return filtered
        elif isinstance(obj, list):
            return [filter_sensitive(item) for item in obj]
        else:
            return obj

    return filter_sensitive(filtered_settings)


def save_results(results: Dict[str, Any], output_dir: str, timestamp: str, settings_config: Dict[str, Any] = None):
    """
    保存评测结果到JSON文件（包含配置信息）

    Args:
        results: 评测结果字典
        output_dir: 输出目录
        timestamp: 时间戳（用于文件名）
        settings_config: 配置信息（可选）
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{results['model_name']}_results_{timestamp}.json")

    # 准备保存的数据
    save_data = {
        'model_name': results['model_name'],
        'settings': settings_config,  # 添加配置信息
        'results': results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)

    print(f"结果已保存到: {output_file}")


def print_results(results: Dict[str, Any]):
    """打印结果"""
    print(f"\n{'='*80}")
    print(f"评测结果: {results['model_name']}")
    print(f"{'='*80}")
    print(f"查询数量: {len(results['exact_matches'])}")
    print(f"平均EM: {results['avg_em']:.4f}")
    print(f"平均F1: {results['avg_f1']:.4f}")

    # 打印时间统计
    if 'total_time' in results:
        print(f"总时间: {results['total_time']:.2f}s")
        print(f"平均时间: {results['avg_time']:.2f}s")

    if 'total_generation_time' in results:
        print(f"总生成时间: {results['total_generation_time']:.2f}s")
        print(f"平均生成时间: {results['avg_generation_time']:.2f}s")

    if 'total_retrieval_time' in results and results['total_retrieval_time'] > 0:
        print(f"总检索时间: {results['total_retrieval_time']:.2f}s")
        print(f"平均检索时间: {results['avg_retrieval_time']:.2f}s")

    if 'total_generation_tokens' in results and results['total_generation_tokens'] > 0:
        print(f"总生成tokens: {results['total_generation_tokens']}")
        print(f"平均生成tokens: {results['avg_generation_tokens']:.1f}")

    print(f"{'='*80}\n")


def main():
    """主流程"""
    # 配置
    HOTPOTQA_FILE = r"D:\Develop\all_RAG\routing_rag\HotpotQA\hotpot_1000_samples.jsonl"
    NUM_SAMPLES = 1000  # 快速测试只用5个样本
    OUTPUT_DIR = r"D:\Develop\all_RAG\routing_rag\evaluation_results"  # 输出目录
    SETTINGS_FILE = r"D:\Develop\all_RAG\routing_rag\config\settings.yaml"  # 配置文件路径

    print("="*80)
    print("HotpotQA快速评测 - NoRAG vs NaiveRAG")
    print("="*80)

    # 加载并过滤配置
    print("\n加载配置文件...")
    filtered_settings = load_and_filter_settings(SETTINGS_FILE)
    print(f"配置已加载（敏感信息已过滤，prompt已保留）")

    # 1. 加载数据
    documents, queries = load_hotpotqa_samples(HOTPOTQA_FILE, NUM_SAMPLES)

    if not documents or not queries:
        print("未能加载数据，退出。")
        return

    # 2. 初始化NoRAG
    print("\n初始化NoRAG...")
    no_rag = NoRAG()

    # 3. 初始化NaiveRAG并构建索引
    print("\n初始化NaiveRAG...")
    naive_rag = NaiveRAG()

    print("构建NaiveRAG索引...")
    success = naive_rag.build_index_from_data(documents)

    if not success:
        print("NaiveRAG索引构建失败，退出。")
        return

    print(f"索引构建成功，包含 {len(naive_rag.documents)} 个文档。")

    # 4. 评测NoRAG
    no_rag_results = quick_evaluate(no_rag, queries, "NoRAG")
    print_results(no_rag_results)

    # 5. 评测NaiveRAG（记录检索时间）
    naive_rag_results = quick_evaluate(naive_rag, queries, "NaiveRAG", record_retrieval_time=True)
    print_results(naive_rag_results)

    # 6. 对比
    print(f"\n{'='*80}")
    print("对比结果")
    print(f"{'='*80}")
    print(f"{'模型':<15} {'EM':<10} {'F1':<10}")
    print(f"{'-'*80}")
    print(f"{'NoRAG':<15} {no_rag_results['avg_em']:<10.4f} {no_rag_results['avg_f1']:<10.4f}")
    print(f"{'NaiveRAG':<15} {naive_rag_results['avg_em']:<10.4f} {naive_rag_results['avg_f1']:<10.4f}")
    print(f"{'-'*80}")
    print(f"{'提升':<15} {naive_rag_results['avg_em'] - no_rag_results['avg_em']:<10.4f} {naive_rag_results['avg_f1'] - no_rag_results['avg_f1']:<10.4f}")
    print(f"{'='*80}\n")

    # 7. 保存结果（传入配置信息）
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_results(no_rag_results, OUTPUT_DIR, timestamp, filtered_settings)
    save_results(naive_rag_results, OUTPUT_DIR, timestamp, filtered_settings)

    print("快速评测完成！")


if __name__ == "__main__":
    main()
