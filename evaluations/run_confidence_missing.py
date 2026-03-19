#!/usr/bin/env python3
"""
Run Confidence Evaluation for Missing Models Only

Completes the confidence evaluation for models that were missing from the initial run.
"""

import subprocess
import sys
from pathlib import Path
import time

# Only the models that were missing from the original run
MISSING_MODELS = [
    'claude-opus-4-1-20250805',
    'meta-llama/llama-4-maverick'
]

DATASETS = [
    'standard_datasets/truthfulqa_complete.json',
    'standard_datasets/simpleqa_complete.json',
    'standard_datasets/factscore_complete.json',
    'standard_datasets/mmlupro_complete.json'
]

SAMPLE_SIZE = 1000
THRESHOLDS = [0.5, 0.7, 0.9]


def main():
    """Run confidence evaluation for missing models only."""
    print("🔧 Running Confidence Evaluation for Missing Models")
    print(f"Models to complete: {MISSING_MODELS}")
    print(f"Datasets: {len(DATASETS)}")
    print(f"Sample size: {SAMPLE_SIZE}")
    print("="*60)
    
    start_time = time.time()
    completed = 0
    failed = 0
    
    for dataset in DATASETS:
        dataset_name = Path(dataset).stem
        print(f"\n📊 Dataset: {dataset_name}")
        print("-"*40)
        
        for model in MISSING_MODELS:
            # Check if already exists
            model_safe = model.replace('/', '_').replace(':', '_')
            output_file = f"results/{dataset_name}_{model_safe}_unilateral_confidence_results.json"
            
            if Path(output_file).exists():
                print(f"✓ Already exists: {model} on {dataset_name}")
                completed += 1
                continue
            
            print(f"🚀 Running: {model} on {dataset_name}")
            
            cmd = [
                'python', 'unilateral_confidence_evaluator.py',
                '--dataset', dataset,
                '--model', model,
                '--samples', str(SAMPLE_SIZE),
                '--thresholds'] + [str(t) for t in THRESHOLDS] + [
                '--output-dir', 'results'
            ]
            
            try:
                eval_start = time.time()
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                eval_time = time.time() - eval_start
                print(f"✅ Completed in {eval_time/60:.1f} minutes")
                completed += 1
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed: {e}")
                failed += 1
            except KeyboardInterrupt:
                print("\n⚠️  Interrupted by user")
                sys.exit(1)
    
    # Summary
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("MISSING MODELS EVALUATION COMPLETE")
    print("="*60)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    
    # Check what we have now
    print("\n📋 Final model coverage:")
    all_models = [
        'gpt-4.1-2025-04-14',
        'gpt-4.1-mini-2025-04-14', 
        'claude-opus-4-1-20250805',
        'claude-3-5-haiku-20241022',
        'meta-llama/llama-4-scout',
        'meta-llama/llama-4-maverick',
        'google/gemini-2.5-flash'
    ]
    
    for model in all_models:
        model_safe = model.replace('/', '_').replace(':', '_')
        count = 0
        for dataset in DATASETS:
            dataset_name = Path(dataset).stem
            if Path(f"results/{dataset_name}_{model_safe}_unilateral_confidence_results.json").exists():
                count += 1
        status = "✅" if count == 4 else f"⚠️  ({count}/4)"
        print(f"  {status} {model}")
    
    print("\n✅ You can now run the full comparison analysis!")


if __name__ == "__main__":
    main()