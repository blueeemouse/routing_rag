#!/usr/bin/env python3
"""
Create balanced dataset
- Total samples: 1000
- no_rag: 500 (oversampling: 361 original + 139 duplicated)
- naive_rag: 500 (undersampling: randomly select from 2090)
"""

import json
import random
import os
from datetime import datetime

# Set random seed for reproducibility
random.seed(42)

# Path config
INPUT_FILE = "HotpotQA_train_data/label_analysis/all_labels_no_tie.json"
OUTPUT_DIR = "HotpotQA_train_data/label_analysis/balanced_samples"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "all_labels_balanced_1000.json")
INFO_FILE = os.path.join(OUTPUT_DIR, "balanced_1000_samples_info.txt")

def main():
    # Load original data
    print(f"Loading data: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = data['samples']
    
    # Separate by class
    no_rag_samples = [s for s in samples if s['optimal_strategy'] == 'no_rag']
    naive_rag_samples = [s for s in samples if s['optimal_strategy'] == 'naive_rag']
    
    print(f"\nOriginal distribution:")
    print(f"  no_rag: {len(no_rag_samples)}")
    print(f"  naive_rag: {len(naive_rag_samples)}")
    
    # Oversample no_rag: 361 -> 500
    # Strategy: take all 361, then randomly duplicate 139
    no_rag_all = no_rag_samples.copy()
    additional_needed = 500 - len(no_rag_samples)
    oversampled = random.choices(no_rag_samples, k=additional_needed)
    no_rag_balanced = no_rag_all + oversampled
    
    print(f"\nno_rag oversampling:")
    print(f"  Original: {len(no_rag_samples)}")
    print(f"  Duplicated: {additional_needed}")
    print(f"  Final: {len(no_rag_balanced)}")
    
    # Undersample naive_rag: 2090 -> 500
    naive_rag_balanced = random.sample(naive_rag_samples, 500)
    
    print(f"\nnaive_rag undersampling:")
    print(f"  Original: {len(naive_rag_samples)}")
    print(f"  Final: {len(naive_rag_balanced)}")
    
    # Combine and shuffle
    balanced_samples = no_rag_balanced + naive_rag_balanced
    random.shuffle(balanced_samples)
    
    print(f"\nFinal dataset:")
    print(f"  Total samples: {len(balanced_samples)}")
    print(f"  no_rag: {sum(1 for s in balanced_samples if s['optimal_strategy'] == 'no_rag')}")
    print(f"  naive_rag: {sum(1 for s in balanced_samples if s['optimal_strategy'] == 'naive_rag')}")
    
    # Save data
    output_data = {"samples": balanced_samples}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nData saved: {OUTPUT_FILE}")
    
    # Create info file
    info_content = f"""================================================================================
Balanced Dataset Info
================================================================================

Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

================================================================================
Data Source
================================================================================

Source file: all_labels_no_tie.json
  - No tie samples
  - Original distribution: no_rag=361, naive_rag=2090 (ratio 1:5.79)

================================================================================
Sampling Strategy
================================================================================

Total samples: 1000
Class ratio: 1:1 (balanced)

no_rag (500 samples):
  - Original: 361 samples (all kept)
  - Oversampled: 139 samples (random duplication)
  - Method: Random selection from original samples

naive_rag (500 samples):
  - Original: 2090 samples
  - Undersampled: 500 samples (random selection)

Random seed: 42 (for reproducibility)

================================================================================
Usage Suggestions
================================================================================

Training params:
  - class_weights: no_rag=1.0, naive_rag=1.0 (no weighting needed)
  - Other params: Can transfer from previous search (lr=7e-5, wd=10, temp=0.5)
  - Note: May need to re-search weight_decay for balanced data

================================================================================
File Locations
================================================================================

Data file: balanced_1000_samples.json
Training script: scripts/train_balanced_1000.ps1

================================================================================
"""
    
    with open(INFO_FILE, 'w', encoding='utf-8') as f:
        f.write(info_content)
    print(f"Info file saved: {INFO_FILE}")

if __name__ == "__main__":
    main()
