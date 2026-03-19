#!/usr/bin/env python3
"""
Run Confidence-Based Unilateral Evaluation Suite

Runs confidence-based evaluation across all models and benchmarks,
reusing the existing unilateral evaluation infrastructure.
"""

import subprocess
import sys
from pathlib import Path
import time

# Configuration matching existing unilateral suite
MODELS = [
    'gpt-4.1-2025-04-14',
    'gpt-4.1-mini-2025-04-14',
    'claude-opus-4-1-20250805',
    'claude-3-5-haiku-20241022',
    'meta-llama/llama-4-scout',
    'meta-llama/llama-4-maverick',
    'google/gemini-2.5-flash'
]

DATASETS = [
    'standard_datasets/truthfulqa_complete.json',
    'standard_datasets/simpleqa_complete.json',
    'standard_datasets/factscore_complete.json',
    'standard_datasets/mmlupro_complete.json'
]

SAMPLE_SIZE = 1000
THRESHOLDS = [0.5, 0.7, 0.9]  # Standard thresholds for comparison


def run_confidence_evaluation(model: str, dataset: str):
    """Run confidence evaluation for a model-dataset pair."""
    dataset_name = Path(dataset).stem
    
    print(f"\n{'='*60}")
    print(f"Running: {model} on {dataset_name}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print('='*60)
    
    cmd = [
        'python', 'unilateral_confidence_evaluator.py',
        '--dataset', dataset,
        '--model', model,
        '--samples', str(SAMPLE_SIZE),
        '--thresholds'] + [str(t) for t in THRESHOLDS] + [
        '--output-dir', 'results'
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        print(f"✅ Completed: {model} on {dataset_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {model} on {dataset_name}")
        print(f"Error: {e.stderr}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)


def main():
    """Run the complete confidence evaluation suite."""
    print("🚀 Starting Confidence-Based Unilateral Evaluation Suite")
    print(f"Models: {len(MODELS)}")
    print(f"Datasets: {len(DATASETS)}")
    print(f"Sample size: {SAMPLE_SIZE}")
    print(f"Thresholds: {THRESHOLDS}")
    print(f"Total evaluations: {len(MODELS) * len(DATASETS)}")
    
    start_time = time.time()
    completed = 0
    failed = 0
    
    # Track progress
    results_log = []
    
    for dataset in DATASETS:
        for model in MODELS:
            evaluation_start = time.time()
            
            # Check if already completed
            dataset_name = Path(dataset).stem
            model_safe = model.replace('/', '_').replace(':', '_')
            output_file = f"results/{dataset_name}_{model_safe}_unilateral_confidence_results.json"
            
            if Path(output_file).exists():
                print(f"\n✓ Skipping (already exists): {model} on {dataset_name}")
                completed += 1
                continue
            
            # Run evaluation
            success = run_confidence_evaluation(model, dataset)
            
            if success:
                completed += 1
                evaluation_time = time.time() - evaluation_start
                results_log.append({
                    'model': model,
                    'dataset': dataset_name,
                    'status': 'completed',
                    'time': evaluation_time
                })
                print(f"⏱️  Evaluation time: {evaluation_time:.1f}s")
            else:
                failed += 1
                results_log.append({
                    'model': model,
                    'dataset': dataset_name,
                    'status': 'failed',
                    'time': 0
                })
            
            # Progress update
            total_done = completed + failed
            total_needed = len(MODELS) * len(DATASETS)
            print(f"\n📊 Progress: {total_done}/{total_needed} ({total_done*100/total_needed:.1f}%)")
            
            elapsed = time.time() - start_time
            if completed > 0:
                avg_time = elapsed / total_done
                remaining = (total_needed - total_done) * avg_time
                print(f"⏱️  Elapsed: {elapsed/60:.1f} min | ETA: {remaining/60:.1f} min")
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("CONFIDENCE EVALUATION SUITE COMPLETE")
    print("="*60)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {completed/(completed+failed)*100:.1f}%")
    
    # Show any failures
    if failed > 0:
        print("\nFailed evaluations:")
        for entry in results_log:
            if entry['status'] == 'failed':
                print(f"  - {entry['model']} on {entry['dataset']}")
    
    print("\n✅ Confidence evaluation suite finished!")
    print(f"Results saved in: results/")


if __name__ == "__main__":
    main()