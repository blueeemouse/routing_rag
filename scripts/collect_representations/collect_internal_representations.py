"""
从 LLM 内部提取表征，用于辅助路由决策。

使用 transformers 加载 fp16 Qwen2.5-3B-Instruct，对训练数据中的 query 做 prefill，
提取各层 hidden states（均值池化 + 最后一个 token），以及 next token logits。

用法:
    conda activate rag_routing
    python scripts/collect_representations/collect_internal_representations.py
    python scripts/collect_representations/collect_internal_representations.py --config config/collect_representations.yaml --num_samples 10
"""

import argparse
import json
import os
import signal
import sys
from typing import Dict, List, Any

import numpy as np
import torch
import yaml
from tqdm import tqdm

# 项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)


def load_config(config_path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    if not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_training_data(data_path: str, num_samples: int = None) -> List[Dict]:
    """加载训练数据"""
    if not os.path.isabs(data_path):
        data_path = os.path.join(PROJECT_ROOT, data_path)
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    samples = data.get('samples', data) if isinstance(data, dict) else data
    if num_samples is not None:
        samples = samples[:num_samples]
    return samples


def load_model_and_tokenizer(config: Dict[str, Any]):
    """加载模型和分词器"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = config['model']['name']
    dtype_str = config['model'].get('dtype', 'float16')
    device = config['model'].get('device', 'cpu')
    attn_impl = config['model'].get('attn_implementation', 'eager')

    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }
    dtype = dtype_map.get(dtype_str, torch.float16)

    print(f"Loading model: {model_name}")
    print(f"  dtype: {dtype_str}, device: {device}, attn: {attn_impl}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()

    print(f"  Model loaded. Layers: {model.config.num_hidden_layers}, Hidden size: {model.config.hidden_size}")
    return model, tokenizer


def extract_representations_batch(
    model,
    tokenizer,
    queries: List[str],
    layer_ids: List[int],
    collect_logits: bool = True,
    device: str = 'cpu',
) -> Dict[str, np.ndarray]:
    """
    对一批 query 提取内部表征。

    Args:
        model: CausalLM 模型
        tokenizer: 分词器
        queries: query 列表
        layer_ids: 要提取的层号列表 (0-indexed)
        collect_logits: 是否收集 next token logits
        device: 运行设备

    Returns:
        dict: 每个 key 对应 shape=(batch, dim) 的 numpy 数组
    """
    # 构造输入：每个 query 作为单条 user message
    messages_list = [[{"role": "user", "content": q}] for q in queries]

    # 用 chat template tokenize（不 padding，手动处理）
    all_input_ids = []
    all_attention_masks = []

    for messages in messages_list:
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors=None,
        )
        all_input_ids.append(input_ids)
        all_attention_masks.append([1] * len(input_ids))

    # 右 padding
    max_len = max(len(ids) for ids in all_input_ids)
    padded_input_ids = []
    padded_attention_masks = []
    for ids, mask in zip(all_input_ids, all_attention_masks):
        pad_len = max_len - len(ids)
        padded_input_ids.append(ids + [tokenizer.pad_token_id] * pad_len)
        padded_attention_masks.append(mask + [0] * pad_len)

    input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
    attention_mask = torch.tensor(padded_attention_masks, dtype=torch.long)

    if device != 'cpu':
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    hidden_states = outputs.hidden_states  # tuple: (embedding + num_layers) 个 (batch, seq, hidden_dim)
    batch_size = input_ids.shape[0]
    hidden_dim = model.config.hidden_size

    result = {}
    layer_names = {layer_ids[0]: 'shallow', layer_ids[1]: 'middle', layer_ids[2]: 'deep'}
    # 确保按层号排序
    sorted_layers = sorted(layer_ids)

    for layer_id in sorted_layers:
        name = layer_names[layer_id]
        # hidden_states[0] = embedding, hidden_states[i+1] = layer i
        layer_hidden = hidden_states[layer_id + 1].cpu().to(torch.float32)  # (batch, seq, hidden_dim)

        # Mean pooling (masked)
        mask_expanded = attention_mask.unsqueeze(-1).to(torch.float32)  # (batch, seq, 1)
        masked_sum = (layer_hidden * mask_expanded).sum(dim=1)  # (batch, hidden_dim)
        mask_sum = mask_expanded.sum(dim=1).clamp(min=1e-9)  # (batch, 1)
        mean_pooled = (masked_sum / mask_sum).numpy().astype(np.float16)

        # Last token hidden state
        last_token_indices = attention_mask.sum(dim=1) - 1  # (batch,)
        last_token_indices = last_token_indices.clamp(min=0)
        last_token_hidden = layer_hidden[
            torch.arange(batch_size), last_token_indices
        ].numpy().astype(np.float16)

        result[f'{name}_mean'] = mean_pooled
        result[f'{name}_last_token'] = last_token_hidden

    # Next token logits
    if collect_logits:
        logits = outputs.logits  # (batch, seq, vocab_size)
        last_token_indices = attention_mask.sum(dim=1) - 1
        last_token_indices = last_token_indices.clamp(min=0)
        next_token_logits = logits[
            torch.arange(batch_size), last_token_indices
        ].cpu().numpy().astype(np.float16)  # (batch, vocab_size)
        result['next_token_logits'] = next_token_logits

    return result


def main():
    parser = argparse.ArgumentParser(description='收集 LLM 内部表征用于路由分析')
    parser.add_argument('--config', type=str, default='config/collect_representations.yaml',
                        help='配置文件路径')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='处理的样本数量（默认全部）')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录（覆盖配置）')
    parser.add_argument('--input_file', type=str, default=None,
                        help='输入数据文件路径（覆盖配置）')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 加载数据
    data_path = args.input_file or config['data']['input_file']
    samples = load_training_data(data_path, args.num_samples)
    total_samples = len(samples)
    print(f"Loaded {total_samples} samples from {data_path}")

    # 加载模型
    model, tokenizer = load_model_and_tokenizer(config)
    layer_ids = [config['layers']['shallow'], config['layers']['middle'], config['layers']['deep']]
    collect_logits = config.get('collect_logits', True)
    batch_size = config['data']['batch_size']
    shard_size = config['data']['shard_size']

    # 输出目录
    output_dir = args.output_dir or config['output']['dir']
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(PROJECT_ROOT, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 断点续跑：检查已有 shard
    start_idx = 0
    shard_idx = 0
    while True:
        shard_path = os.path.join(output_dir, f'shard_{shard_idx:04d}.npz')
        if os.path.exists(shard_path):
            shard_data = np.load(shard_path, allow_pickle=True)
            shard_len = shard_data['shallow_mean'].shape[0]
            start_idx += shard_len
            shard_idx += 1
            print(f"  Existing shard {shard_idx}: {shard_len} samples, cumulative: {start_idx}")
        else:
            break

    if start_idx > 0:
        print(f"Resuming from sample {start_idx}")

    # 信号处理：Ctrl+C 优雅退出
    interrupted = False
    def signal_handler(sig, frame):
        nonlocal interrupted
        print("\n\nInterrupted! Saving current progress...")
        interrupted = True

    original_handler = signal.signal(signal.SIGINT, signal_handler)

    try:
        # 按分片处理
        pbar = tqdm(total=total_samples, initial=start_idx, desc="Collecting representations")

        while start_idx < total_samples and not interrupted:
            end_idx = min(start_idx + shard_size, total_samples)
            batch_queries = [samples[i]['question'] for i in range(start_idx, end_idx)]

            # 批量提取
            all_representations = {
                'shallow_mean': [],
                'shallow_last_token': [],
                'middle_mean': [],
                'middle_last_token': [],
                'deep_mean': [],
                'deep_last_token': [],
            }
            if collect_logits:
                all_representations['next_token_logits'] = []

            for batch_start in range(0, len(batch_queries), batch_size):
                if interrupted:
                    break
                batch_end = min(batch_start + batch_size, len(batch_queries))
                batch = batch_queries[batch_start:batch_end]

                try:
                    result = extract_representations_batch(
                        model, tokenizer, batch, layer_ids,
                        collect_logits=collect_logits,
                        device=config['model'].get('device', 'cpu'),
                    )
                    for key in result:
                        all_representations[key].append(result[key])
                except Exception as e:
                    print(f"\n  Error at samples {start_idx + batch_start}-{start_idx + batch_end}: {e}")
                    # 填充零向量
                    hidden_dim = model.config.hidden_size
                    vocab_size = model.config.vocab_size
                    actual_batch = len(batch)
                    for key in all_representations:
                        if 'logits' in key:
                            all_representations[key].append(
                                np.zeros((actual_batch, vocab_size), dtype=np.float16))
                        else:
                            all_representations[key].append(
                                np.zeros((actual_batch, hidden_dim), dtype=np.float16))

                pbar.update(batch_end - batch_start)

            if interrupted:
                break

            # 合并批次
            shard_data = {}
            for key, arr_list in all_representations.items():
                shard_data[key] = np.concatenate(arr_list, axis=0)

            # 保存 shard
            shard_path = os.path.join(output_dir, f'shard_{shard_idx:04d}.npz')
            np.savez_compressed(shard_path, **shard_data)
            shard_size_actual = shard_data['shallow_mean'].shape[0]
            print(f"\n  Saved shard {shard_idx}: {shard_size_actual} samples -> {shard_path}")

            start_idx = end_idx
            shard_idx += 1

        pbar.close()

        # 保存 metadata
        metadata_path = os.path.join(output_dir, 'metadata.json')
        metadata = []
        for i in range(total_samples):
            meta = {
                'question': samples[i]['question'],
            }
            for key in ['optimal_strategy', 'no_rag_score', 'naive_rag_score', 'source',
                        'no_rag_em', 'no_rag_f1', 'naive_rag_em', 'naive_rag_f1', 'score_diff']:
                if key in samples[i]:
                    meta[key] = samples[i][key]
            metadata.append(meta)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"\nMetadata saved to {metadata_path}")

        # 输出统计
        print("\n=== Representation Statistics ===")
        for key in sorted(os.listdir(output_dir)):
            if key.endswith('.npz'):
                shard_path = os.path.join(output_dir, key)
                data = np.load(shard_path)
                print(f"\n  {key}:")
                for arr_name in sorted(data.files):
                    arr = data[arr_name]
                    print(f"    {arr_name}: shape={arr.shape}, dtype={arr.dtype}, "
                          f"mean={arr.mean():.6f}, std={arr.std():.6f}, "
                          f"min={arr.min():.6f}, max={arr.max():.6f}")

        print(f"\nDone! Total shards: {shard_idx}, samples: {start_idx}")

    except Exception as e:
        print(f"\nFatal error: {e}")
        raise
    finally:
        signal.signal(signal.SIGINT, original_handler)


if __name__ == '__main__':
    main()
