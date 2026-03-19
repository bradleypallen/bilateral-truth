#!/usr/bin/env python3
"""
Run Confidence-Based Unilateral Evaluation Suite in Parallel

Runs confidence-based evaluation across all models and benchmarks in parallel,
similar to the other evaluation suites.
"""

import subprocess
import sys
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Optional

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
MAX_WORKERS = 7  # Run all models in parallel for each dataset


def run_single_evaluation(model: str, dataset: str) -> Tuple[str, str, bool, float]:
    """Run a single confidence evaluation and return results."""
    dataset_name = Path(dataset).stem
    model_safe = model.replace('/', '_').replace(':', '_')
    
    # Check if already completed
    output_file = f"results/{dataset_name}_{model_safe}_unilateral_confidence_results.json"
    if Path(output_file).exists():
        print(f"✓ Skipping (exists): {model} on {dataset_name}")
        return model, dataset_name, True, 0.0
    
    print(f"🚀 Starting: {model} on {dataset_name}")
    start_time = time.time()
    
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
        elapsed = time.time() - start_time
        print(f"✅ Completed: {model} on {dataset_name} ({elapsed/60:.1f} min)")
        return model, dataset_name, True, elapsed
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ Failed: {model} on {dataset_name}")
        print(f"   Error: {str(e.stderr)[:200]}")
        return model, dataset_name, False, elapsed
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted: {model} on {dataset_name}")
        raise


def main():
    """Run the complete confidence evaluation suite in parallel."""
    print("🚀 Starting Parallel Confidence-Based Unilateral Evaluation Suite")
    print(f"Models: {len(MODELS)}")
    print(f"Datasets: {len(DATASETS)}")
    print(f"Sample size: {SAMPLE_SIZE}")
    print(f"Thresholds: {THRESHOLDS}")
    print(f"Total evaluations: {len(MODELS) * len(DATASETS)}")
    print(f"Max parallel workers: {MAX_WORKERS}")
    print("="*60)
    
    start_time = time.time()
    completed = 0
    failed = 0
    results_log = []
    
    try:
        # Process each dataset with all models in parallel
        for dataset in DATASETS:
            dataset_name = Path(dataset).stem
            print(f"\n📊 Processing dataset: {dataset_name}")
            print("-"*40)
            
            dataset_start = time.time()
            
            # Run all models for this dataset in parallel
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Submit all model evaluations for this dataset
                futures = {
                    executor.submit(run_single_evaluation, model, dataset): model
                    for model in MODELS
                }
                
                # Process completed evaluations
                for future in as_completed(futures):
                    model_name = futures[future]
                    try:
                        model, ds_name, success, elapsed = future.result()
                        
                        if success:
                            completed += 1
                        else:
                            failed += 1
                        
                        results_log.append({
                            'model': model,
                            'dataset': ds_name,
                            'status': 'completed' if success else 'failed',
                            'time': elapsed
                        })
                        
                    except Exception as e:
                        print(f"❌ Exception for {model_name}: {e}")
                        failed += 1
                        results_log.append({
                            'model': model_name,
                            'dataset': dataset_name,
                            'status': 'failed',
                            'time': 0
                        })
            
            dataset_elapsed = time.time() - dataset_start
            print(f"⏱️  Dataset {dataset_name} completed in {dataset_elapsed/60:.1f} minutes")
            
            # Progress update
            total_done = completed + failed
            total_needed = len(MODELS) * len(DATASETS)
            print(f"📊 Overall progress: {total_done}/{total_needed} ({total_done*100/total_needed:.1f}%)")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Suite interrupted by user")
        print("Note: Running evaluations will continue in background until complete")
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("PARALLEL CONFIDENCE EVALUATION SUITE COMPLETE")
    print("="*60)
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    
    if completed + failed > 0:
        print(f"Success rate: {completed/(completed+failed)*100:.1f}%")
    
    # Show any failures
    if failed > 0:
        print("\nFailed evaluations:")
        for entry in results_log:
            if entry['status'] == 'failed':
                print(f"  - {entry['model']} on {entry['dataset']}")
    
    # Calculate average time per evaluation
    successful_times = [r['time'] for r in results_log if r['status'] == 'completed' and r['time'] > 0]
    if successful_times:
        avg_time = sum(successful_times) / len(successful_times)
        print(f"\nAverage time per evaluation: {avg_time/60:.1f} minutes")
        
        # Time saved by parallelization
        sequential_time = sum(successful_times)
        print(f"Time saved by parallelization: {(sequential_time - total_time)/60:.1f} minutes")
        print(f"Speedup factor: {sequential_time/total_time:.1f}x")
    
    print("\n✅ Parallel confidence evaluation suite finished!")
    print(f"Results saved in: results/")


if __name__ == "__main__":
    main()