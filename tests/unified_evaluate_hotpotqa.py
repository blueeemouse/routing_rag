"""
统一的HotpotQA评测脚本
支持评测NoRAG、NaiveRAG、GraphRAG中任意组合的性能
可通过参数指定要评测的模型

使用示例:
    # 评测所有三种模型
    python unified_evaluate_hotpotqa.py --models all

    # 只评测NoRAG
    python unified_evaluate_hotpotqa.py --models no_rag

    # 评测NaiveRAG和GraphRAG
    python unified_evaluate_hotpotqa.py --models naive_rag,graph_rag

    # 自定义样本数量
    python unified_evaluate_hotpotqa.py --models all --num_samples 100
参数说明：
    total_time是一个rag方法处理所有query的时间（即执行execute方法的时间之和）（所以包含了检索和生成）
    total_retrieval_time是一个rag方法在所有query上检索的时间的综合（支持NaiveRAG和GraphRAG）
    - NaiveRAG: retriever.retrieve() 的执行时间（包含嵌入生成 + 向量搜索）
    - GraphRAG: map_query_to_entities() 的执行时间（包含嵌入生成 + 向量搜索 + 实体匹配后处理）


"""

import json
import sys
import os
import argparse
from typing import List, Dict, Any, Set
from collections import Counter
import re
import string
import time

# Add routing_rag path (使用相对路径，自动获取项目根目录)
ROUTING_RAG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

from rag_implementations.naive_rag.naive_rag_impl import NaiveRAG
from rag_implementations.no_rag.no_rag_impl import NoRAG
from rag_implementations.graph_rag.graph_rag_impl import GraphRAG


def normalize_answer(s):
    """标准化答案文本 - 与官方实现保持一致"""
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


def compute_f1(gold_answers: List[str], prediction: str) -> tuple[float, float, float]:
    """
    计算F1分数、Precision和Recall - 与官方实现保持一致

    Returns:
        (f1, precision, recall) 元组
    """
    normalized_prediction = normalize_answer(prediction)
    prediction_tokens = normalized_prediction.split()

    ZERO_METRIC = (0.0, 0.0, 0.0)

    max_f1 = 0.0
    max_precision = 0.0
    max_recall = 0.0

    for gold_answer in gold_answers:
        normalized_gold = normalize_answer(gold_answer)

        # 特殊处理：yes/no/noanswer
        if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_gold:
            continue
        if normalized_gold in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_gold:
            continue

        gold_tokens = normalized_gold.split()

        common = Counter(prediction_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            continue

        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(gold_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        if f1 > max_f1:
            max_f1 = f1
            max_precision = precision
            max_recall = recall

    return (max_f1, max_precision, max_recall)


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


def load_hotpotqa_samples(jsonl_file_path: str, num_samples: int = None) -> tuple[List[str], List[Dict[str, Any]]]:
    """
    加载HotpotQA样本

    Returns:
        documents: 文档文本列表（用于NaiveRAG）
        queries: 查询数据列表（用于所有模型）
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

                # 提取文档（用于NaiveRAG）
                context = data.get('context', [])
                for title, sentence_list in context:
                    doc_text = "\n\n".join(sentence_list)
                    documents.append(doc_text)

                # 提取查询
                queries.append(data)
                count += 1

                if count % 100 == 0:
                    print(f"  已处理 {count} 个样本...")

            except json.JSONDecodeError as e:
                print(f"警告: 解析第 {count+1} 行 JSON 时出错: {e}")
                continue

    print(f"数据加载完成，共处理了 {count} 个样本，获得 {len(documents)} 个文档块。")
    return documents, queries


def evaluate_model(
    model,
    queries: List[Dict[str, Any]],
    model_name: str,
    record_retrieval_time: bool = False,
    graphrag_context: Dict[str, Any] = None,
    progress_callback=None,
    query_delay: float = 0.0
) -> Dict[str, Any]:
    """
    评测单个模型

    Args:
        model: RAG模型实例
        queries: 查询数据列表
        model_name: 模型名称
        record_retrieval_time: 是否记录检索时间
        graphrag_context: GraphRAG所需的上下文信息
        progress_callback: 进度回调函数
        query_delay: 每条查询之间的延迟（秒）

    Returns:
        评测结果字典
    """
    results = {
        'model_name': model_name,
        'predictions': [],
        'exact_matches': [],
        'f1_scores': [],
        'precisions': [],
        'recalls': [],
        'total_time': 0.0,
        'avg_time': 0.0,
        'total_retrieval_time': 0.0,
        'avg_retrieval_time': 0.0,
        'total_generation_time': 0.0,
        'avg_generation_time': 0.0,
        'total_generation_tokens': 0,
        'avg_generation_tokens': 0,
        'errors': []
    }

    num_queries = len(queries)

    for i, query_data in enumerate(queries):
        question = query_data['question']
        gold_answers = extract_gold_answers(query_data)

        try:
            # 调用进度回调
            if progress_callback:
                progress_callback(model_name, i, num_queries, question)

            # 记录开始时间
            start_time = time.time()

            # 执行查询（这里很关键，要确保能给GraphRAG传进去context才行。不过这里感觉莫名其妙多了一个model_name的判断，正常用
            # 原来的名字不就好了吗……）
            if model_name == 'Graphrag' and graphrag_context:
                prediction = model.execute(question, context=graphrag_context)
            else:
                prediction = model.execute(question)

            # 记录结束时间
            end_time = time.time()

            # 计算总时间
            total_time = end_time - start_time
            results['total_time'] += total_time

            # 获取生成时间（现在的处理还是有一些问题的，如果没有专门的成员变量记录上一次query的generation time
            # 则会直接用total_time近似）（主要是还不支持GraphRAG）
            if hasattr(model, 'last_generation_time'):
                generation_time = model.last_generation_time
                generation_tokens = getattr(model, 'last_generation_tokens', 0)
            else:
                generation_time = total_time
                generation_tokens = 0

            # 获取检索时间（现在仅支持NaiveRAG统计检索时间）
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
            f1, precision, recall = compute_f1(gold_answers, prediction)

            # 保存结果
            results['predictions'].append({
                'question': question,
                'gold_answer': gold_answers,
                'prediction': prediction,
                'em': em,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'total_time': total_time,
                'retrieval_time': retrieval_time,
                'generation_time': generation_time,
                'generation_tokens': generation_tokens
            })

            results['exact_matches'].append(em)
            results['f1_scores'].append(f1)
            results['precisions'].append(precision)
            results['recalls'].append(recall)

        except Exception as e:
            print(f"\n{model_name} 查询 {i+1} 出错: {e}")
            results['errors'].append({
                'index': i,
                'question': question,
                'error': str(e)
            })
            # 出错时记录为0分
            results['exact_matches'].append(0.0)
            results['f1_scores'].append(0.0)
            results['precisions'].append(0.0)
            results['recalls'].append(0.0)

        # 在每条查询后添加延迟（无论成功还是失败，最后一条不需要延迟）
        if query_delay > 0 and i < num_queries - 1:
            time.sleep(query_delay)

    # 计算汇总统计
    num_queries = len(results['exact_matches'])
    results['avg_em'] = sum(results['exact_matches']) / num_queries
    results['avg_f1'] = sum(results['f1_scores']) / num_queries
    results['avg_precision'] = sum(results['precisions']) / num_queries
    results['avg_recall'] = sum(results['recalls']) / num_queries
    results['accuracy'] = results['avg_em']
    results['num_errors'] = len(results['errors'])
    results['avg_time'] = results['total_time'] / num_queries
    results['avg_generation_time'] = results['total_generation_time'] / num_queries
    if results['total_retrieval_time'] > 0:
        results['avg_retrieval_time'] = results['total_retrieval_time'] / num_queries
    if results['total_generation_tokens'] > 0:
        results['avg_generation_tokens'] = results['total_generation_tokens'] / num_queries

    return results


def print_evaluation_results(results: Dict[str, Any]):
    """打印评测结果"""
    print(f"\n{'='*80}")
    print(f"评测结果: {results['model_name']}")
    print(f"{'='*80}")
    print(f"查询数量: {len(results['exact_matches'])}")
    print(f"错误数量: {results['num_errors']}")
    print(f"平均EM (Exact Match): {results['avg_em']:.4f}")
    print(f"平均F1 Score: {results['avg_f1']:.4f}")
    print(f"平均Precision: {results['avg_precision']:.4f}")
    print(f"平均Recall: {results['avg_recall']:.4f}")
    print(f"准确率 (Accuracy): {results['accuracy']:.4f}")

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


def compare_results(results_list: List[Dict[str, Any]]):
    """比较多个模型的评测结果"""
    print(f"\n{'='*80}")
    print(f"模型对比")
    print(f"{'='*80}")
    print(f"{'模型名称':<20} {'EM':<10} {'F1':<10} {'Precision':<10} {'Recall':<10} {'平均时间(s)':<12}")
    print(f"{'-'*80}")

    for results in results_list:
        avg_time = results.get('avg_time', 0)
        print(f"{results['model_name']:<20} {results['avg_em']:<10.4f} {results['avg_f1']:<10.4f} {results['avg_precision']:<10.4f} {results['avg_recall']:<10.4f} {avg_time:<12.2f}")

    print(f"{'='*80}\n")


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


def save_results(results: Dict[str, Any], output_dir: str, timestamp: str, settings_config: Dict[str, Any] = None, args: argparse.Namespace = None):
    """
    保存评测结果到JSON文件（包含配置信息、命令行参数等）

    Args:
        results: 评测结果字典
        output_dir: 输出目录
        timestamp: 时间戳（用于文件名）
        settings_config: 配置信息（可选）
        args: 命令行参数（可选）
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{results['model_name']}_results_{timestamp}.json")

    # 准备保存的数据
    save_data = {
        'model_name': results['model_name'],
        'settings': settings_config,  # 配置信息
        'arguments': {},  # 命令行参数
        'results': results
    }

    # 添加命令行参数
    if args:
        save_data['arguments'] = {
            'models': args.models,
            'num_samples': args.num_samples,
            'hotpotqa_file': args.hotpotqa_file,
            'output_dir': args.output_dir,
            'settings_file': args.settings_file,
            'graphrag_work_dir': args.graphrag_work_dir,
            'graphrag_config_file': args.graphrag_config_file,
            'skip_graphrag_index': args.skip_graphrag_index,
            'delay': args.delay,
            'naive_rag_index_path': args.naive_rag_index_path
        }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)

    print(f"结果已保存到: {output_file}")


def save_comparison(results_list: List[Dict[str, Any]], output_dir: str, timestamp: str, args: argparse.Namespace = None):
    """
    保存对比结果

    Args:
        results_list: 评测结果列表
        output_dir: 输出目录
        timestamp: 时间戳
        args: 命令行参数（可选）
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"comparison_results_{timestamp}.json")

    comparison = {
        'timestamp': timestamp,
        'models': results_list,
        'arguments': {},  # 命令行参数
        'summary': {}
    }

    # 添加命令行参数
    if args:
        comparison['arguments'] = {
            'models': args.models,
            'num_samples': args.num_samples,
            'hotpotqa_file': args.hotpotqa_file,
            'output_dir': args.output_dir,
            'settings_file': args.settings_file,
            'graphrag_work_dir': args.graphrag_work_dir,
            'graphrag_config_file': args.graphrag_config_file,
            'skip_graphrag_index': args.skip_graphrag_index,
            'delay': args.delay,
            'naive_rag_index_path': args.naive_rag_index_path
        }

    # 计算对比摘要
    if len(results_list) >= 2:
        # 以第一个模型为基准
        baseline = results_list[0]
        for i in range(1, len(results_list)):
            current = results_list[i]
            comparison['summary'][f"{current['model_name']}_vs_{baseline['model_name']}"] = {
                'em_improvement': current['avg_em'] - baseline['avg_em'],
                'f1_improvement': current['avg_f1'] - baseline['avg_f1'],
                'time_diff': current['avg_time'] - baseline['avg_time']
            }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    print(f"对比结果已保存到: {output_file}")


def prepare_hotpotqa_data_for_graphrag(hotpotqa_file: str, output_dir: str, num_samples: int = 5):
    """
    准备HotpotQA数据用于GraphRAG索引构建
    将HotpotQA的文档提取为文本文件

    Args:
        hotpotqa_file: HotpotQA数据文件路径
        output_dir: 输出目录
        num_samples: 处理的样本数量
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n正在从HotpotQA数据中提取文档...")
    print(f"输入文件: {hotpotqa_file}")
    print(f"输出目录: {output_dir}")
    if num_samples:
        print(f"样本数量: {num_samples}")

    count = 0
    doc_count = 0

    with open(hotpotqa_file, 'r', encoding='utf-8') as f:
        for line in f:
            if num_samples and count >= num_samples:
                break

            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                context = data.get('context', [])

                # 提取每个文档
                for title, sentences in context:
                    doc_text = "\n\n".join(sentences)

                    # 保存为文本文件
                    # 使用安全的文件名
                    safe_title = title.replace('/', '_').replace('\\', '_').replace(':', '_')
                    # 这一段是添加针对Windows不允许字符的处理（Linux上应该不会有这个问题的）
                    for char in '<>:"\\|?*':
                        safe_title = safe_title.replace(char, '_')
                    doc_file = os.path.join(output_dir, f"{safe_title}.txt")

                    with open(doc_file, 'w', encoding='utf-8') as f:
                        f.write(f"{title}\n\n{doc_text}")

                    doc_count += 1

                count += 1
                if count % 100 == 0:
                    print(f"  已处理 {count}/{num_samples if num_samples else 'all'} 个样本，提取 {doc_count} 个文档...")

            except json.JSONDecodeError as e:
                print(f"警告: 解析第 {count+1} 行 JSON 时出错: {e}")
                continue

    print(f"文档提取完成！共处理 {count} 个样本，提取 {doc_count} 个文档。")
    return doc_count


def create_graphrag_config(config_file: str, work_dir: str):
    """创建GraphRAG配置文件"""
    import yaml

    config = {
        'root_dir': work_dir,
        'models': {
            'default_chat_model': {
                'type': 'chat',
                'api_key': '${GRAPHRAG_API_KEY}',
                'model': 'gpt-4o',  # 使用与 settings.yaml 一致的模型
                'model_provider': 'openai',
                'api_base': '${API_BASE_URL}',
                'temperature': 0,  # 降低温度，提高确定性
                'max_tokens': 2000,  # 减少 max_tokens，强制生成简洁答案（默认200）
                'request_timeout': 180.0,
                'concurrent_requests': 25
            },
            'default_embedding_model': {
                'type': 'embedding',
                'api_key': '${GRAPHRAG_API_KEY}',
                'model': 'text-embedding-ada-002',
                'model_provider': 'openai',
                'api_base': '${API_BASE_URL}',
                'request_timeout': 180.0,
                'concurrent_requests': 25
            }
        },
        'input': {
            'type': 'file',
            'file_type': 'text',
            'base_dir': os.path.join(work_dir, 'input'),
            'encoding': 'utf-8',
            'text_column': 'text'
        },
        'output': {
            'type': 'file',
            'base_dir': os.path.join(work_dir, 'output')
        },
        'cache': {
            'type': 'file',
            'base_dir': os.path.join(work_dir, 'cache')
        },
        'reporting': {
            'type': 'file',
            'base_dir': os.path.join(work_dir, 'logs')
        },
        'chunks': {
            'size': 300,
            'overlap': 50,
            'group_by_columns': ['id']
        },
        'extract_graph': {
            'prompt': None,
            'entity_types': ['person', 'organization', 'event', 'location', 'gpe'],
            'max_gleanings': 1,
            'model': 'default_chat_model'
        },
        'embed_graph': {
            'enabled': True,
            'dimensions': 1536,
            'num_walks': 10,
            'walk_length': 40
        },
        'cluster_graph': {
            'max_cluster_size': 10,
            'use_lcc': True
        },
        'community_reports': {
            'max_length': 500,
            'max_input_length': 2000,
            'model_id': 'default_chat_model'
        },
        'summarize_descriptions': {
            'max_length': 200,
            'max_input_tokens': 1000,
            'model_id': 'default_chat_model'
        },
        'prune_graph': {
            'min_node_degree': 1,
            'min_edge_weight_pct': 40.0
        },
        'embeddings': {
            'model': 'text-embedding-ada-002',
            'batch_size': 16,
            'batch_max_tokens': 8191,
            'model_id': 'default_embedding_model'
        },
        'local_search': {
            'chat_model_id': 'default_chat_model',
            'embedding_model_id': 'default_embedding_model',
            # 添加优化的 prompt 配置
            'prompt': """Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.

Guidelines:
1. Provide ONLY the answer, no explanations or reasoning
2. Keep the answer as short as possible - typically 1-5 words
3. For yes/no questions, answer only "yes" or "no"
4. For dates, use the exact format (e.g., "December 31, 2015")
5. For numbers, provide just the number (e.g., "1522")
6. For names, provide just the name (e.g., "Terry Crews")
7. DO NOT include phrases like "The answer is", "According to", "Based on", etc.
8. DO NOT add any additional context or information
9. If the answer is not in the context, say "I don't know"

Query: {query_str}
Answer:"""
        },
        'global_search': {
            'chat_model_id': 'default_chat_model',
            'embedding_model_id': 'default_embedding_model'
        },
        'snapshots': {
            'embeddings': False,
            'graphml': False,
            'raw_graph': False
        }
    }

    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    print(f"配置文件已创建: {config_file}")


def save_run_config(args: argparse.Namespace, graphrag_work_dir: str):
    """
    保存运行配置到 graphrag_work_dir（包括命令行参数和设置）

    Args:
        args: 命令行参数
        graphrag_work_dir: GraphRAG工作目录
    """
    os.makedirs(graphrag_work_dir, exist_ok=True)
    config_file = os.path.join(graphrag_work_dir, 'run_config.json')

    # 准备保存的数据
    save_data = {
        'timestamp': time.strftime("%Y%m%d_%H%M%S"),
        'command_line_args': {
            'models': args.models,
            'num_samples': args.num_samples,
            'hotpotqa_file': args.hotpotqa_file,
            'output_dir': args.output_dir,
            'settings_file': args.settings_file,
            'graphrag_work_dir': args.graphrag_work_dir,
            'graphrag_config_file': args.graphrag_config_file,
            'skip_graphrag_index': args.skip_graphrag_index,
            'delay': args.delay,
            'naive_rag_index_path': args.naive_rag_index_path
        },
        'description': '此文件记录了评估运行的参数配置'
    }

    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)

    print(f"运行配置已保存到: {config_file}")


def main():
    """主评测流程"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='统一的HotpotQA评测脚本，支持评测NoRAG、NaiveRAG、GraphRAG中任意组合的性能',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    # 评测所有三种模型
    python unified_evaluate_hotpotqa.py --models all

    # 只评测NoRAG
    python unified_evaluate_hotpotqa.py --models no_rag

    # 评测NaiveRAG和GraphRAG
    python unified_evaluate_hotpotqa.py --models naive_rag,graph_rag

    # 自定义样本数量
    python unified_evaluate_hotpotqa.py --models all --num_samples 100

    # 指定GraphRAG工作目录
    python unified_evaluate_hotpotqa.py --models graph_rag --graphrag_work_dir ./my_graphrag_data
        """
    )

    parser.add_argument(
        '--models',
        type=str,
        default='all',
        help='要评测的模型，逗号分隔。可选: no_rag, naive_rag, graph_rag, all (默认: all)'
    )

    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='评测样本数量 (默认: 5)'
    )

    parser.add_argument(
        '--hotpotqa_file',
        type=str,
        default=r"D:\Develop\all_RAG\routing_rag\HotpotQA\hotpot_dev_distractor_1000_samples.jsonl",
        help='HotpotQA数据文件路径'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='输出目录（默认: 项目根目录/evaluation_results）'
    )

    parser.add_argument(
        '--settings_file',
        type=str,
        default=None,
        help='配置文件路径（默认: 项目根目录/config/settings.yaml）'
    )

    parser.add_argument(
        '--graphrag_work_dir',
        type=str,
        default=None,
        help='GraphRAG工作目录'
    )

    parser.add_argument(
        '--graphrag_config_file',
        type=str,
        default=None,
        help='GraphRAG配置文件路径（默认: graphrag_work_dir/graphrag_hotpotqa_config.yml）'
    )

    parser.add_argument(
        '--skip_graphrag_index',
        action='store_true',
        help='跳过GraphRAG索引构建（如果索引已存在）'
    )

    parser.add_argument(
        '--delay',
        type=float,
        default=2.5,
        help='每次查询之间的延迟（秒）'
    )

    parser.add_argument(
        '--naive_rag_index_path',
        type=str,
        default=None,
        help='NaiveRAG索引存储路径（默认: naive_rag_index_storage_{num_samples}_samples）'
    )

    args = parser.parse_args()

    # 设置默认路径（基于项目根目录）
    if args.settings_file is None:
        args.settings_file = os.path.join(ROUTING_RAG_ROOT, 'config', 'settings.yaml')
    if args.output_dir is None:
        args.output_dir = os.path.join(ROUTING_RAG_ROOT, 'evaluation_results')

    # 解析要评测的模型
    models_to_eval: Set[str] = set()
    if args.models.lower() == 'all':
        models_to_eval = {'no_rag', 'naive_rag', 'graph_rag'}
    else:
        models_to_eval = set(m.strip().lower() for m in args.models.split(','))

    # 验证模型名称
    valid_models = {'no_rag', 'naive_rag', 'graph_rag'}
    invalid_models = models_to_eval - valid_models
    if invalid_models:
        print(f"错误: 无效的模型名称: {', '.join(invalid_models)}")
        print(f"有效的模型名称: {', '.join(valid_models)}")
        sys.exit(1)

    print("="*80)
    print("统一的HotpotQA评测")
    print("="*80)
    print(f"评测模型: {', '.join(sorted(models_to_eval))}")
    print(f"样本数量: {args.num_samples}")
    print(f"输出目录: {args.output_dir}")
    print("="*80)

    # 加载并过滤配置
    print("\n加载配置文件...")
    filtered_settings = load_and_filter_settings(args.settings_file)
    print(f"配置已加载（敏感信息已过滤，prompt已保留）")

    # 1. 加载数据
    print("\n" + "="*80)
    print("步骤1: 加载HotpotQA数据")
    print("="*80)
    documents, queries = load_hotpotqa_samples(args.hotpotqa_file, args.num_samples)

    if not documents or not queries:
        print("未能加载数据，退出。")
        return

    # 2. 初始化模型
    print("\n" + "="*80)
    print("步骤2: 初始化模型")
    print("="*80)

    models = {}
    graphrag_context = None

    # 初始化NoRAG
    if 'no_rag' in models_to_eval:
        print("初始化NoRAG...")
        models['no_rag'] = NoRAG()
        print("NoRAG初始化完成")

    # 初始化NaiveRAG
    if 'naive_rag' in models_to_eval:
        print("初始化NaiveRAG...")
        models['naive_rag'] = NaiveRAG()

        # 处理NaiveRAG索引路径
        naive_rag_index_loaded = False
        if args.naive_rag_index_path:
            print(f"检查NaiveRAG索引路径: {args.naive_rag_index_path}")
            # 检查路径是否存在且包含索引文件
            if os.path.exists(args.naive_rag_index_path) and os.listdir(args.naive_rag_index_path):
                print("尝试加载已存在的NaiveRAG索引...")
                naive_rag_index_loaded = models['naive_rag'].load_index(args.naive_rag_index_path)
                if naive_rag_index_loaded:
                    print("NaiveRAG索引加载成功！")
                else:
                    print("NaiveRAG索引加载失败，将重新构建索引...")
            else:
                print("NaiveRAG索引路径不存在或为空，将构建新索引并保存...")

        # 如果未加载索引，构建新索引
        if not naive_rag_index_loaded:
            print("构建NaiveRAG索引...")
            success = models['naive_rag'].build_index_from_data(documents)

            if not success:
                print("NaiveRAG索引构建失败，退出。")
                return

            print(f"NaiveRAG索引构建成功，包含 {len(models['naive_rag'].documents)} 个文档")

            # 如果指定了索引路径，保存索引
            if args.naive_rag_index_path:
                print(f"保存NaiveRAG索引到: {args.naive_rag_index_path}")
                save_success = models['naive_rag'].save_index(args.naive_rag_index_path)
                if save_success:
                    print("NaiveRAG索引保存成功！")
                else:
                    print("警告：NaiveRAG索引保存失败")

    # 初始化GraphRAG
    if 'graph_rag' in models_to_eval:
        print("初始化GraphRAG...")
        graphrag = GraphRAG()

        # 设置GraphRAG配置文件路径
        if args.graphrag_config_file is None:
            args.graphrag_config_file = os.path.join(args.graphrag_work_dir, 'graphrag_hotpotqa_config.yml')

        # 检查GraphRAG索引
        output_dir = os.path.join(args.graphrag_work_dir, "output")
        # entities_path = os.path.join(output_dir, "entities.parquet")

        if not args.skip_graphrag_index:
            print("\nGraphRAG索引尚未构建，开始准备数据...")

            # 保存运行配置（只在构建索引时保存）
            print("保存运行配置到GraphRAG工作目录...")
            save_run_config(args, args.graphrag_work_dir)

            # 准备数据
            input_dir = os.path.join(args.graphrag_work_dir, "input")
            prepare_hotpotqa_data_for_graphrag(args.hotpotqa_file, input_dir, args.num_samples)

            # 创建配置文件
            if not os.path.exists(args.graphrag_config_file):
                print(f"创建GraphRAG配置文件: {args.graphrag_config_file}")
                create_graphrag_config(args.graphrag_config_file, args.graphrag_work_dir)
            else:
                print(f"GraphRAG配置文件已存在: {args.graphrag_config_file}")

            # 构建索引
            print("\n开始构建GraphRAG索引...")
            print(f"工作目录: {args.graphrag_work_dir}")
            print(f"配置文件: {args.graphrag_config_file}")
            print("注意：这可能需要较长时间...")

            # graphrag = GraphRAG()
            success = graphrag.build_index_from_path(
                root_dir=args.graphrag_work_dir,
                config_filepath=args.graphrag_config_file
            )

            if not success:
                print("GraphRAG索引构建失败，退出。")
                return

            print("GraphRAG索引构建成功！")
        else:
            if args.skip_graphrag_index:
                print("跳过GraphRAG索引构建（--skip_graphrag_index）")
            else:
                print("GraphRAG索引已存在，跳过构建")
            print(f"索引路径: {output_dir}")

        # 初始化GraphRAG用于查询
        # models['graph_rag'] = GraphRAG()
        models['graph_rag'] = graphrag  # 应该用统一的对象吧

        # 准备GraphRAG查询上下文
        config_filename = os.path.basename(args.graphrag_config_file)
        graphrag_context = {
            'search_mode': 'local',
            'data_path': args.graphrag_work_dir,
            'config_filename': config_filename
        }

        print("GraphRAG初始化完成")

    # 3. 评测模型
    print("\n" + "="*80)
    print("步骤3: 评测模型")
    print("="*80)

    results_list = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # 定义进度回调
    def progress_callback(model_name, current, total, question):
        print(f"[{model_name}] {current}/{total}: {question[:50]}...")

    # 评测每个模型
    for model_name in sorted(models_to_eval):
        if model_name not in models:
            continue
        # 这display_name是结果记录文件的方法的名字（不过看着怪怪的，感觉其实没必要……）
        model_display_name = model_name.replace('_', '').title()
        print(f"\n开始评测 {model_display_name}...")

        record_retrieval = (model_name in ['naive_rag', 'graph_rag'])

        # print('query:', queries)
        # print('graphrag_context:', graphrag_context)
        results = evaluate_model(
            models[model_name],
            queries,
            model_display_name,
            record_retrieval_time=record_retrieval,
            graphrag_context=graphrag_context,
            progress_callback=progress_callback,
            query_delay=args.delay
        )

        print_evaluation_results(results)
        save_results(results, args.output_dir, timestamp, filtered_settings, args)
        results_list.append(results)

    # 4. 对比结果
    if len(results_list) > 1:
        print("\n" + "="*80)
        print("步骤4: 对比结果")
        print("="*80)
        compare_results(results_list)
        save_comparison(results_list, args.output_dir, timestamp, args)

    # 5. 完成
    print("\n" + "="*80)
    print("评测完成！")
    print("="*80)
    print(f"结果已保存到: {args.output_dir}")
    print(f"时间戳: {timestamp}")


if __name__ == "__main__":
    main()