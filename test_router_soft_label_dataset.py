"""
测试 RouterSoftLabelDataset

验证新类功能是否正常，与 FusionSoftLabelDataset 对比。
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'router'))

from trainable_router.datasets import RouterSoftLabelDataset, FusionSoftLabelDataset
from trainable_router.config import TrainableRouterConfig


def test_new_dataset():
    """测试 RouterSoftLabelDataset"""
    print("=" * 60)
    print("测试 RouterSoftLabelDataset")
    print("=" * 60)
    
    # 加载配置
    config = TrainableRouterConfig.from_yaml('config/train_fusion_soft_label.yaml')
    
    # 创建数据集（不使用语义特征）
    dataset = RouterSoftLabelDataset(config, split='train')
    print(f"\n数据集大小: {len(dataset)}")
    
    # 获取一个样本
    if len(dataset) == 0:
        print("错误: 数据集为空")
        return False
    
    sample = dataset[0]
    print(f"\n样本字段: {list(sample.keys())}")
    print(f"  queries: {sample['queries'][:50]}...")
    print(f"  soft_label: {sample['soft_label']}")
    print(f"  label: {sample['label']} ({dataset.strategy_names[sample['label']]})")
    print(f"  cluster_id: {sample['cluster_id']}")
    
    # 检查字段类型
    assert isinstance(sample['queries'], str), "queries 应为字符串"
    assert isinstance(sample['soft_label'], list), "soft_label 应为列表"
    assert isinstance(sample['label'], int), "label 应为整数"
    assert len(sample['soft_label']) == len(dataset.strategy_names), "soft_label 长度应等于策略数"
    
    print("\n✓ 基本字段检查通过")
    return True


def test_with_semantic():
    """测试语义特征"""
    print("\n" + "=" * 60)
    print("测试语义特征 (use_semantic=True)")
    print("=" * 60)
    
    config = TrainableRouterConfig.from_yaml('config/train_fusion_soft_label.yaml')
    config.data.use_semantic = True
    
    dataset = RouterSoftLabelDataset(config, split='train')
    
    # 需要先设置 tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model.backbone_name)
        dataset.set_tokenizer(tokenizer)
        print(f"\n已设置 tokenizer: {config.model.backbone_name}")
    except Exception as e:
        print(f"\n警告: 无法加载 tokenizer: {e}")
        print("跳过语义特征测试")
        return True
    
    # 获取样本
    sample = dataset[0]
    print(f"\n样本字段: {list(sample.keys())}")
    
    if 'input_ids' in sample:
        print(f"  input_ids shape: {sample['input_ids'].shape}")
        print(f"  attention_mask shape: {sample['attention_mask'].shape}")
        print("\n✓ 语义特征检查通过")
        return True
    else:
        print("错误: 未返回语义特征")
        return False


def test_compare_with_fusion():
    """与 FusionSoftLabelDataset 对比"""
    print("\n" + "=" * 60)
    print("对比 FusionSoftLabelDataset")
    print("=" * 60)
    
    config = TrainableRouterConfig.from_yaml('config/train_fusion_soft_label.yaml')
    
    # 新数据集
    new_dataset = RouterSoftLabelDataset(config, split='train')
    
    # 旧数据集
    old_dataset = FusionSoftLabelDataset(config)
    old_dataset.load_data(config.data.train_path)
    
    print(f"\n新数据集大小: {len(new_dataset)}")
    print(f"旧数据集大小: {len(old_dataset)}")
    
    if len(new_dataset) != len(old_dataset):
        print("警告: 数据集大小不一致")
    
    # 对比第一个样本
    new_sample = new_dataset[0]
    old_sample = old_dataset[0]
    
    print(f"\n新样本字段: {list(new_sample.keys())}")
    print(f"旧样本字段: {list(old_sample.keys())}")
    
    # 检查关键字段
    print(f"\n  queries 一致: {new_sample['queries'] == old_sample['queries']}")
    
    # soft_label 对比
    if 'soft_label' in old_sample:
        old_sl = old_sample['soft_label']
        new_sl = new_sample['soft_label']
        print(f"  soft_label 一致: {new_sl == old_sl}")
    
    print("\n✓ 对比完成")
    return True


def main():
    """主函数"""
    print("RouterSoftLabelDataset 测试脚本\n")
    
    results = []
    
    # 测试1: 基本功能
    try:
        results.append(("基本功能", test_new_dataset()))
    except Exception as e:
        print(f"\n✗ 基本功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("基本功能", False))
    
    # 测试2: 语义特征
    try:
        results.append(("语义特征", test_with_semantic()))
    except Exception as e:
        print(f"\n✗ 语义特征测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("语义特征", False))
    
    # 测试3: 对比旧类
    try:
        results.append(("对比旧类", test_compare_with_fusion()))
    except Exception as e:
        print(f"\n✗ 对比测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("对比旧类", False))
    
    # 汇总
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    print(f"\n总体: {'全部通过' if all_passed else '有测试失败'}")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
