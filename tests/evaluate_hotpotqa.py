"""
HotpotQA评测脚本
参考RouteRAG的评测方式，测试NoRAG和NaiveRAG在HotpotQA数据集上的性能

评测指标：
- EM (Exact Match): 精确匹配
- F1 Score: F1分数
- Accuracy: 准确率
"""

import json
import sys
import os
from typing import List, Dict, Any, Tuple
from collections import Counter
import re
from tqdm import tqdm

# Add routing_rag path
ROUTING_RAG_ROOT = r"D:\Develop\all_RAG\routing_rag"
sys.path.insert(0, ROUTING_RAG_ROOT)

from rag_implementations.naive_rag.naive_rag_impl import NaiveRAG
from rag_implementations.no_rag.no_rag_impl import NoRAG
from config.config import settings


def normalize_answer(s: str) -> str:
    """
    标准化答案文本
    来自SQuAD评测脚本
    """
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
    """
    计算精确匹配分数
    """
    normalized_prediction = normalize_answer(prediction)
    for gold_answer in gold_answers:
        if normalize_answer(gold_answer) == normalized_prediction:
            return 1.0
    return 0.0


def compute_f1(gold_answers: List[str], prediction: str) -> float:
    """
    计算F1分数
    来自SQuAD评测脚本
    """
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
    """
    从HotpotQA样本中提取标准答案
    """
    gold_answers = []

    if 'answer' in sample:
        answer = sample['answer']
        if isinstance(answer, str):
            gold_answers.append(answer)
        elif isinstance(answer, list):
            gold_answers.extend(answer)

    # 如果有answer_aliases，也加入
    if 'answer_aliases' in sample:
        gold_answers.extend(sample['answer_aliases'])

    # 去重
    gold_answers = list(set(gold_answers))

    return gold_answers if gold_answers else [""]


def extract_documents_from_hotpotqa(jsonl_file_path: str, num_samples: int = None) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    从HotpotQA数据集中提取文档和查询数据

    Returns:
        documents: 文档文本列表
        queries: 查询数据列表
    """
    documents = []
    queries = []

    print(f"正在从 {jsonl_file_path} 读取数据...")
    count = 0
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if num_samples and count >= num_samples:
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

                if count % 10 == 0:
                    print(f"  已处理 {count} 个样本...")

            except json.JSONDecodeError as e:
                print(f"警告: 解析第 {count+1} 行 JSON 时出错: {e}")
                continue
            except Exception as e:
                print(f"警告: 处理第 {count+1} 行时出错: {e}")
                continue

    print(f"数据提取完成，共处理了 {count} 个样本，获得 {len(documents)} 个文档块。")
    return documents, queries


def evaluate_model(model, queries: List[Dict[str, Any]], model_name: str, max_samples: int = None) -> Dict[str, Any]:
    """
    评测单个模型

    Args:
        model: RAG模型实例
        queries: 查询数据列表
        model_name: 模型名称
        max_samples: 最大评测样本数

    Returns:
        评测结果字典
    """
    results = {
        'model_name': model_name,
        'predictions': [],
        'exact_matches': [],
        'f1_scores': [],
        'errors': []
    }

    num_queries = len(queries)
    if max_samples:
        num_queries = min(num_queries, max_samples)

    print(f"\n开始评测 {model_name}，共 {num_queries} 个查询...")

    for i in tqdm(range(num_queries), desc=f"Evaluating {model_name}"):
        query_data = queries[i]
        question = query_data['question']
        gold_answers = extract_gold_answers(query_data)

        try:
            # 执行查询
            prediction = model.execute(question)

            # 计算指标
            em = compute_exact_match(gold_answers, prediction)
            f1 = compute_f1(gold_answers, prediction)

            # 保存结果
            results['predictions'].append({
                'question': question,
                'gold_answer': gold_answers,
                'prediction': prediction,
                'em': em,
                'f1': f1
            })

            results['exact_matches'].append(em)
            results['f1_scores'].append(f1)

        except Exception as e:
            print(f"\n查询 {i+1} 出错: {e}")
            results['errors'].append({
                'index': i,
                'question': question,
                'error': str(e)
            })
            # 出错时记录为0分
            results['exact_matches'].append(0.0)
            results['f1_scores'].append(0.0)

    # 计算汇总统计
    results['avg_em'] = sum(results['exact_matches']) / len(results['exact_matches'])
    results['avg_f1'] = sum(results['f1_scores']) / len(results['f1_scores'])
    results['accuracy'] = results['avg_em']  # EM就是准确率
    results['num_errors'] = len(results['errors'])

    return results


def print_evaluation_results(results: Dict[str, Any]):
    """
    打印评测结果
    """
    print(f"\n{'='*80}")
    print(f"评测结果: {results['model_name']}")
    print(f"{'='*80}")
    print(f"查询数量: {len(results['exact_matches'])}")
    print(f"错误数量: {results['num_errors']}")
    print(f"平均EM (Exact Match): {results['avg_em']:.4f}")
    print(f"平均F1 Score: {results['avg_f1']:.4f}")
    print(f"准确率 (Accuracy): {results['accuracy']:.4f}")
    print(f"{'='*80}\n")


def save_results(results: Dict[str, Any], output_file: str):
    """
    保存评测结果到文件
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存到: {output_file}")


def compare_results(results_list: List[Dict[str, Any]]):
    """
    比较多个模型的评测结果
    """
    print(f"\n{'='*80}")
    print(f"模型对比")
    print(f"{'='*80}")
    print(f"{'模型名称':<20} {'EM':<10} {'F1':<10} {'准确率':<10}")
    print(f"{'-'*80}")

    for results in results_list:
        print(f"{results['model_name']:<20} {results['avg_em']:<10.4f} {results['avg_f1']:<10.4f} {results['accuracy']:<10.4f}")

    print(f"{'='*80}\n")


def main():
    """
    主评测流程
    """
    # 配置参数
    HOTPOTQA_FILE = r"D:\Develop\all_RAG\routing_rag\HotpotQA\hotpot_1000_samples.jsonl"
    NUM_SAMPLES = 50  # 评测样本数，设为None表示全部
    OUTPUT_DIR = r"D:\Develop\all_RAG\routing_rag\evaluation_results"

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("="*80)
    print("HotpotQA评测 - NoRAG vs NaiveRAG")
    print("="*80)

    # 1. 提取文档和查询
    documents, queries = extract_documents_from_hotpotqa(HOTPOTQA_FILE, NUM_SAMPLES)

    if not documents or not queries:
        print("未能提取到数据，退出。")
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

    print(f"NaiveRAG索引构建成功，包含 {len(naive_rag.documents)} 个文档。")

    # 4. 评测NoRAG
    no_rag_results = evaluate_model(no_rag, queries, "NoRAG", max_samples=NUM_SAMPLES)
    print_evaluation_results(no_rag_results)
    save_results(no_rag_results, os.path.join(OUTPUT_DIR, "no_rag_results.json"))

    # 5. 评测NaiveRAG
    naive_rag_results = evaluate_model(naive_rag, queries, "NaiveRAG", max_samples=NUM_SAMPLES)
    print_evaluation_results(naive_rag_results)
    save_results(naive_rag_results, os.path.join(OUTPUT_DIR, "naive_rag_results.json"))

    # 6. 对比结果
    compare_results([no_rag_results, naive_rag_results])

    # 7. 保存对比结果
    comparison = {
        'models': [no_rag_results, naive_rag_results],
        'improvement': {
            'em': naive_rag_results['avg_em'] - no_rag_results['avg_em'],
            'f1': naive_rag_results['avg_f1'] - no_rag_results['avg_f1']
        }
    }
    save_results(comparison, os.path.join(OUTPUT_DIR, "comparison_results.json"))

    print("\n评测完成！")


if __name__ == "__main__":
    # 需要导入string模块用于normalize_answer函数
    import string

    main()