"""
Filter out samples where no_rag is strictly better than naive_rag.
Keep:
1. Tie samples (source == "tie_converted") - marked as no_rag for cost reasons
2. Samples where naive_rag is strictly better than no_rag (optimal_strategy == "naive_rag" and source == "original")
"""
import json
from pathlib import Path


def filter_training_data(input_path: str, output_path: str):
    """Filter training data to remove no_rag strictly better samples."""
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = data['samples']
    
    # Statistics
    total_original = len(samples)
    no_rag_strictly_better = 0
    naive_rag_strictly_better = 0
    tie_samples = 0
    
    filtered_samples = []
    
    for sample in samples:
        source = sample.get('source', 'original')
        optimal_strategy = sample.get('optimal_strategy', '')
        
        if source == 'tie_converted':
            # Tie samples - keep
            filtered_samples.append(sample)
            tie_samples += 1
        elif optimal_strategy == 'naive_rag':
            # naive_rag strictly better - keep
            filtered_samples.append(sample)
            naive_rag_strictly_better += 1
        else:
            # no_rag strictly better - remove
            no_rag_strictly_better += 1
    
    # Create output data
    output_data = {
        'samples': filtered_samples,
        'statistics': {
            'original_total': total_original,
            'filtered_total': len(filtered_samples),
            'removed_no_rag_strictly_better': no_rag_strictly_better,
            'kept_naive_rag_strictly_better': naive_rag_strictly_better,
            'kept_tie_samples': tie_samples,
            'filter_description': 'Removed samples where no_rag is strictly better than naive_rag'
        }
    }
    
    # Save filtered data
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("=" * 60)
    print("Data Filtering Summary")
    print("=" * 60)
    print(f"Original total:     {total_original}")
    print(f"Removed (no_rag > naive_rag): {no_rag_strictly_better}")
    print(f"Kept (naive_rag > no_rag):    {naive_rag_strictly_better}")
    print(f"Kept (tie samples):           {tie_samples}")
    print("-" * 60)
    print(f"Filtered total:    {len(filtered_samples)}")
    print("=" * 60)
    print(f"Output saved to: {output_path}")
    
    return output_data


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Filter training data')
    parser.add_argument('--input', type=str, 
                        default='HotpotQA_train_data/label_analysis/all_labels_with_tie_converted.json',
                        help='Input JSON file path')
    parser.add_argument('--output', type=str,
                        default='HotpotQA_train_data/label_analysis/all_labels_no_rag_strictly_better_removed.json',
                        help='Output JSON file path')
    
    args = parser.parse_args()
    
    filter_training_data(args.input, args.output)
