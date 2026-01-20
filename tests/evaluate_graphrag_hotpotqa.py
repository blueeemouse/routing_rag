"""
GraphRAG HotpotQA评测脚本
评测GraphRAG在HotpotQA数据集上的性能（不分离检索和生成时间）
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

from rag_implementations.graph_rag.graph_rag_impl import GraphRAG


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


def load_hotpotqa_samples(jsonl_file_path: str, num_samples: int = 10) -> List[Dict[str, Any]]:
    """加载HotpotQA样本（仅查询数据）"""
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
                queries.append(data)
                count += 1

            except json.JSONDecodeError as e:
                print(f"警告: 解析第 {count+1} 行 JSON 时出错: {e}")
                continue

    print(f"已加载 {count} 个样本。")
    return queries


def quick_evaluate_graphrag(
    graphrag: GraphRAG,
    queries: List[Dict[str, Any]],
    data_path: str,
    config_filename: str = None,
    delay_seconds: float = 2.5,
) -> Dict[str, Any]:
    """快速评测GraphRAG"""
    results = {
        'model_name': 'GraphRAG',
        'predictions': [],
        'exact_matches': [],
        'f1_scores': [],
        'total_time': 0.0,
        'avg_time': 0.0,
        'total_generation_time': 0.0,
        'avg_generation_time': 0.0,
        'total_generation_tokens': 0,
        'avg_generation_tokens': 0
    }

    print(f"\n开始评测 GraphRAG...")

    for i, query_data in enumerate(queries):
        question = query_data['question']
        gold_answers = extract_gold_answers(query_data)

        try:
            print(f"\n[{i+1}/{len(queries)}] 问题: {question[:50]}...")

            # 准备上下文
            context = {
                'search_mode': 'local',
                'data_path': data_path
            }
            
            # 如果指定了配置文件名，添加到context中
            if config_filename:
                context['config_filename'] = config_filename

            # 记录开始时间
            start_time = time.time()

            # 执行查询
            prediction = graphrag.execute(question, context=context)

            # 记录结束时间
            end_time = time.time()

            # 计算总时间（不分离检索和生成）
            total_time = end_time - start_time
            generation_time = total_time  # 全部算作生成时间
            generation_tokens = 0  # GraphRAG暂时不返回token信息

            # 累计时间
            results['total_time'] += total_time
            results['total_generation_time'] += generation_time
            results['total_generation_tokens'] += generation_tokens

            # 计算指标
            em = compute_exact_match(gold_answers, prediction)
            f1 = compute_f1(gold_answers, prediction)

            print(f"EM: {em:.2f}, F1: {f1:.2f}")
            print(f"总时间: {total_time:.2f}s")
            print(f"预测答案: {prediction[:100]}...")
            print(f"标准答案: {gold_answers}")

            results['predictions'].append({
                'question': question,
                'gold_answer': gold_answers,
                'prediction': prediction,
                'em': em,
                'f1': f1,
                'total_time': total_time,
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
            import traceback
            traceback.print_exc()
            results['exact_matches'].append(0.0)
            results['f1_scores'].append(0.0)

    # 计算平均值
    num_queries = len(results['exact_matches'])
    results['avg_em'] = sum(results['exact_matches']) / num_queries
    results['avg_f1'] = sum(results['f1_scores']) / num_queries
    results['avg_time'] = results['total_time'] / num_queries
    results['avg_generation_time'] = results['total_generation_time'] / num_queries
    if results['total_generation_tokens'] > 0:
        results['avg_generation_tokens'] = results['total_generation_tokens'] / num_queries

    return results


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

    if 'total_generation_tokens' in results and results['total_generation_tokens'] > 0:
        print(f"总生成tokens: {results['total_generation_tokens']}")
        print(f"平均生成tokens: {results['avg_generation_tokens']:.1f}")

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


def prepare_hotpotqa_data_for_graphrag(hotpotqa_file: str, output_dir: str, num_samples: int = 1000):
    """
    准备HotpotQA数据用于GraphRAG索引构建
    将HotpotQA的文档提取为文本文件
    
    Args:
        hotpotqa_file: HotpotQA数据文件路径
        output_dir: 输出目录
        num_samples: 处理的样本数量
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"正在从HotpotQA数据中提取文档...")
    print(f"输入文件: {hotpotqa_file}")
    print(f"输出目录: {output_dir}")
    print(f"样本数量: {num_samples}")
    
    count = 0
    doc_count = 0
    
    with open(hotpotqa_file, 'r', encoding='utf-8') as f:
        for line in f:
            if count >= num_samples:
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
                    doc_file = os.path.join(output_dir, f"{safe_title}.txt")
                    
                    with open(doc_file, 'w', encoding='utf-8') as f:
                        f.write(f"{title}\n\n{doc_text}")
                    
                    doc_count += 1
                
                count += 1
                if count % 100 == 0:
                    print(f"  已处理 {count}/{num_samples} 个样本，提取 {doc_count} 个文档...")
                    
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
                'model': 'gpt-3.5-turbo',
                'model_provider': 'openai',
                'api_base': '${API_BASE_URL}',
                'temperature': 0,
                'max_tokens': 4096,
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
            'embedding_model_id': 'default_embedding_model'
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


def main():
    """主流程"""
    # 配置
    HOTPOTQA_FILE = r"D:\Develop\all_RAG\routing_rag\HotpotQA\hotpot_1000_samples.jsonl"
    NUM_SAMPLES = 5  # 使用全部1000条数据构建索引
    NUM_TEST_SAMPLES = 5  # 评测时只用10条
    GRAPH_RAG_WORK_DIR = r"D:\Develop\all_RAG\routing_rag\graphrag_hotpotqa_data"  # GraphRAG工作目录
    GRAPH_RAG_CONFIG_FILE = r"D:\Develop\all_RAG\routing_rag\graphrag_hotpotqa_data\graphrag_hotpotqa_config.yml"  # GraphRAG配置文件
    OUTPUT_DIR = r"D:\Develop\all_RAG\routing_rag\evaluation_results"
    SETTINGS_FILE = r"D:\Develop\all_RAG\routing_rag\config\settings.yaml"

    print("="*80)
    print("GraphRAG HotpotQA评测")
    print("="*80)

    # 加载并过滤配置
    print("\n加载配置文件...")
    filtered_settings = load_and_filter_settings(SETTINGS_FILE)
    print(f"配置已加载（敏感信息已过滤，prompt已保留）")

    # 1. 准备HotpotQA数据
    print("\n" + "="*80)
    print("步骤1: 准备HotpotQA数据")
    print("="*80)
    input_dir = os.path.join(GRAPH_RAG_WORK_DIR, "input")
    prepare_hotpotqa_data_for_graphrag(HOTPOTQA_FILE, input_dir, NUM_SAMPLES)

    # 2. 创建GraphRAG配置文件（如果不存在）
    print("\n" + "="*80)
    print("步骤2: 创建GraphRAG配置文件")
    print("="*80)
    if not os.path.exists(GRAPH_RAG_CONFIG_FILE):
        print(f"创建GraphRAG配置文件: {GRAPH_RAG_CONFIG_FILE}")
        create_graphrag_config(GRAPH_RAG_CONFIG_FILE, GRAPH_RAG_WORK_DIR)
    else:
        print(f"GraphRAG配置文件已存在: {GRAPH_RAG_CONFIG_FILE}")

    # 3. 检查GraphRAG索引是否已构建
    print("\n" + "="*80)
    print("步骤3: 检查GraphRAG索引")
    print("="*80)
    output_dir = os.path.join(GRAPH_RAG_WORK_DIR, "output")
    entities_path = os.path.join(output_dir, "entities.parquet")
    
    if not os.path.exists(entities_path):
        print("GraphRAG索引尚未构建，开始构建索引...")
        print(f"工作目录: {GRAPH_RAG_WORK_DIR}")
        print(f"配置文件: {GRAPH_RAG_CONFIG_FILE}")
        print("注意：这可能需要较长时间...")
        
        # 初始化GraphRAG
        graphrag = GraphRAG()
        
        # 构建索引
        success = graphrag.build_index_from_path(
            root_dir=GRAPH_RAG_WORK_DIR,
            config_filepath=GRAPH_RAG_CONFIG_FILE
        )
        
        if not success:
            print("GraphRAG索引构建失败，退出。")
            return
        
        print("GraphRAG索引构建成功！")
    else:
        print("GraphRAG索引已存在，跳过构建。")
        print(f"索引路径: {output_dir}")

    # 4. 初始化GraphRAG（用于查询）
    print("\n" + "="*80)
    print("步骤4: 初始化GraphRAG")
    print("="*80)
    graphrag = GraphRAG()

    # 5. 加载评测数据
    print("\n" + "="*80)
    print("步骤5: 加载评测数据")
    print("="*80)
    queries = load_hotpotqa_samples(HOTPOTQA_FILE, NUM_TEST_SAMPLES)

    if not queries:
        print("未能加载数据，退出。")
        return

    # 6. 评测GraphRAG
    print("\n" + "="*80)
    print("步骤6: 评测GraphRAG")
    print("="*80)
    # 提取配置文件名（相对于data_path）
    config_filename = os.path.basename(GRAPH_RAG_CONFIG_FILE)
    graphrag_results = quick_evaluate_graphrag(graphrag, queries, GRAPH_RAG_WORK_DIR, config_filename)
    print_results(graphrag_results)

    # 7. 保存结果
    print("\n" + "="*80)
    print("步骤7: 保存结果")
    print("="*80)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_results(graphrag_results, OUTPUT_DIR, timestamp, filtered_settings)

    print("\n" + "="*80)
    print("GraphRAG评测完成！")
    print("="*80)


if __name__ == "__main__":
    main()