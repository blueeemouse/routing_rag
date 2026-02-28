#!/usr/bin/env python3
"""
Create Curriculum Learning Stage 2 Dataset (Mixed: Tie + Non-Tie)

Purpose:
    Construct a mixed dataset for curriculum learning stage 2.
    Stage 1: Train on balanced non-tie samples (all_labels_balanced_1000.json)
    Stage 2: Continue training on mixed samples (tie + non-tie)

Design Rationale:
    - Tie samples: When no_rag and naive_rag have equal performance, we teach the model
      to prefer no_rag (faster, more efficient) as a prior knowledge.
    - Non-tie samples: Maintain balanced distribution to preserve learned patterns.
    - Ratio: Non-tie : Tie = 2:1 (2000 : 1000)

Dataset Composition:
    - Non-tie samples: 2000 total
        - no_rag: 1000 (oversampled from 361 original)
        - naive_rag: 1000 (undersampled from 2090 original)
    - Tie samples: 1000 total
        - All labeled as no_rag (efficiency prior)
    - Total: 3000 samples

Author: Auto-generated
Date: 2026-02-28
"""

import json
import random
import os
from datetime import datetime
from collections import Counter

# Set random seed for reproducibility
random.seed(42)

# Path config
TIE_FILE = "HotpotQA_train_data/label_analysis/tie_queries.json"
NON_TIE_FILE = "HotpotQA_train_data/label_analysis/all_labels_no_tie.json"
OUTPUT_DIR = "HotpotQA_train_data/label_analysis/curriculum_stage2"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "curriculum_stage2_mixed_3000.json")
INFO_FILE = os.path.join(OUTPUT_DIR, "stage2_data_info.txt")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # =========================================================================
    # Load source data
    # =========================================================================
    print("=" * 80)
    print("Loading source data...")
    print("=" * 80)
    
    # Load tie samples
    print(f"\nLoading tie samples: {TIE_FILE}")
    with open(TIE_FILE, 'r', encoding='utf-8') as f:
        tie_data = json.load(f)
    print(f"  Total tie samples: {len(tie_data)}")
    
    # Load non-tie samples
    print(f"\nLoading non-tie samples: {NON_TIE_FILE}")
    with open(NON_TIE_FILE, 'r', encoding='utf-8') as f:
        non_tie_data = json.load(f)
    
    non_tie_samples = non_tie_data['samples']
    no_rag_samples = [s for s in non_tie_samples if s['optimal_strategy'] == 'no_rag']
    naive_rag_samples = [s for s in non_tie_samples if s['optimal_strategy'] == 'naive_rag']
    
    print(f"  no_rag samples: {len(no_rag_samples)}")
    print(f"  naive_rag samples: {len(naive_rag_samples)}")
    
    # =========================================================================
    # Sample tie data (1000 samples, all labeled as no_rag)
    # =========================================================================
    print("\n" + "=" * 80)
    print("Processing tie samples...")
    print("=" * 80)
    
    # Random sample 1000 tie samples
    tie_sampled = random.sample(tie_data, 1000)
    
    # Convert to standard format and label as no_rag
    tie_processed = []
    for item in tie_sampled:
        tie_processed.append({
            'question': item['question'],
            'optimal_strategy': 'no_rag',  # Efficiency prior: when equal, choose faster method
            'source': 'tie',  # Mark source for tracking
            'no_rag_score': item.get('no_rag_score', 0.0),
            'naive_rag_score': item.get('naive_rag_score', 0.0),
        })
    
    print(f"  Sampled: 1000 tie samples")
    print(f"  Label: all -> no_rag (efficiency prior)")
    
    # =========================================================================
    # Sample non-tie data (2000 samples, balanced)
    # =========================================================================
    print("\n" + "=" * 80)
    print("Processing non-tie samples...")
    print("=" * 80)
    
    # Undersample naive_rag: 2090 -> 1000
    naive_rag_sampled = random.sample(naive_rag_samples, 1000)
    for s in naive_rag_sampled:
        s['source'] = 'non_tie'
    
    # Oversample no_rag: 361 -> 1000
    no_rag_all = no_rag_samples.copy()
    additional_needed = 1000 - len(no_rag_samples)
    oversampled = random.choices(no_rag_samples, k=additional_needed)
    no_rag_sampled = no_rag_all + oversampled
    for s in no_rag_sampled:
        s['source'] = 'non_tie'
    
    print(f"\n  no_rag:")
    print(f"    Original: {len(no_rag_samples)}")
    print(f"    Oversampled: {additional_needed}")
    print(f"    Final: {len(no_rag_sampled)}")
    
    print(f"\n  naive_rag:")
    print(f"    Original: {len(naive_rag_samples)}")
    print(f"    Undersampled to: {len(naive_rag_sampled)}")
    
    # =========================================================================
    # Combine and shuffle
    # =========================================================================
    print("\n" + "=" * 80)
    print("Combining dataset...")
    print("=" * 80)
    
    all_samples = tie_processed + no_rag_sampled + naive_rag_sampled
    random.shuffle(all_samples)
    
    # Verify distribution
    strategy_count = Counter(s['optimal_strategy'] for s in all_samples)
    source_count = Counter(s['source'] for s in all_samples)
    
    print(f"\nFinal dataset statistics:")
    print(f"  Total samples: {len(all_samples)}")
    print(f"  By strategy:")
    print(f"    no_rag: {strategy_count['no_rag']} (tie={1000}, non_tie={1000})")
    print(f"    naive_rag: {strategy_count['naive_rag']}")
    print(f"  By source:")
    print(f"    tie: {source_count['tie']}")
    print(f"    non_tie: {source_count['non_tie']}")
    print(f"  Ratio (non-tie : tie): {source_count['non_tie']} : {source_count['tie']}")
    
    # =========================================================================
    # Save dataset
    # =========================================================================
    print("\n" + "=" * 80)
    print("Saving dataset...")
    print("=" * 80)
    
    output_data = {
        'description': 'Curriculum Stage 2: Mixed tie and non-tie samples',
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_samples': len(all_samples),
        'statistics': {
            'strategy_distribution': dict(strategy_count),
            'source_distribution': dict(source_count),
        },
        'samples': all_samples
    }
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nDataset saved: {OUTPUT_FILE}")
    
    # =========================================================================
    # Create info file
    # =========================================================================
    info_content = f"""================================================================================
Curriculum Learning Stage 2 Dataset Info
================================================================================

Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

================================================================================
Purpose
================================================================================

This dataset is designed for Curriculum Learning Stage 2:
  - Stage 1: Train on balanced non-tie samples (all_labels_balanced_1000.json)
             Learn to distinguish when one strategy is clearly better
  - Stage 2: Continue training on mixed samples (this file)
             Learn to apply efficiency prior for tie cases

Key Concept - Efficiency Prior:
  When no_rag and naive_rag have equal performance, prefer no_rag because:
  - No retrieval overhead (faster)
  - No retrieval noise (cleaner context)
  - Lower cost (no embedding/search)

================================================================================
Dataset Composition
================================================================================

Total samples: 3000
Ratio: Non-tie : Tie = 2 : 1 (2000 : 1000)

Non-tie samples (2000 total):
  - Source: all_labels_no_tie.json
  - no_rag: 1000 samples
    * Original: 361 samples (all kept)
    * Oversampled: 639 samples (random duplication)
  - naive_rag: 1000 samples
    * Original: 2090 samples
    * Undersampled: 1000 samples (random selection)

Tie samples (1000 total):
  - Source: tie_queries.json (total 2549 available)
  - Sampled: 1000 samples (random selection)
  - Label: ALL labeled as no_rag (efficiency prior)
  - Note: Original tie samples have equal scores, we teach model to prefer faster method

================================================================================
Label Distribution
================================================================================

Final labels:
  - no_rag: 2000 (1000 tie + 1000 non-tie)
  - naive_rag: 1000 (non-tie only)

Note: The label distribution is intentionally imbalanced (2:1) because:
  1. We want to reinforce the efficiency prior
  2. Tie cases should always route to no_rag

================================================================================
Training Recommendations
================================================================================

1. Load Stage 1 best model as starting point
2. Use lower learning rate (e.g., 1e-5 or 2e-5)
3. Consider class weights if needed:
   - no_rag: 1.0
   - naive_rag: 2.0 (to balance the 2:1 label ratio)
4. Monitor validation accuracy on both tie and non-tie samples

================================================================================
File Locations
================================================================================

Dataset file: curriculum_stage2_mixed_3000.json
Source files:
  - tie_queries.json
  - all_labels_no_tie.json
Related:
  - Stage 1 data: balanced_samples/all_labels_balanced_1000.json

================================================================================
"""
    
    with open(INFO_FILE, 'w', encoding='utf-8') as f:
        f.write(info_content)
    print(f"Info file saved: {INFO_FILE}")
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
