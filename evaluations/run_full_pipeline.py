#!/usr/bin/env python3
"""
Master Pipeline Script for Bilateral vs Unilateral Evaluation

Complete end-to-end workflow from data preparation to final results with CI.
Supports parallel execution and checkpoint recovery.
"""

import json
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import sys
from typing import List, Dict, Tuple
import logging

# Configuration
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
    'truthfulqa_complete',
    'simpleqa_complete',
    'factscore_complete',
    'mmlupro_complete'
]

EVALUATION_METHODS = {
    'bilateral': {
        'script': 'generic_evaluator.py',
        'args': ['--epistemic-policy', 'classical'],
        'output_suffix': '_classical_results.json'
    },
    'unilateral_forced': {
        'script': 'unilateral_evaluator.py',
        'args': ['--prompt-style', 'direct'],
        'output_suffix': '_unilateral_direct_results.json'
    },
    'unilateral_uncertain': {
        'script': 'unilateral_evaluator.py',
        'args': ['--prompt-style', 'uncertain'],
        'output_suffix': '_unilateral_uncertain_results.json'
    },
    'confidence': {
        'script': 'unilateral_confidence_evaluator.py',
        'args': ['--thresholds', '0.5', '0.7', '0.9'],
        'output_suffix': '_unilateral_confidence_results.json'
    }
}

SAMPLE_SIZE = 1000
MAX_WORKERS = 7
BOOTSTRAP_ITERATIONS = 1000
SUBSAMPLE_SIZE = 100

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_data_preparation():
    """Stage 1: Generate standard format datasets."""
    logger.info("="*70)
    logger.info("STAGE 1: DATA PREPARATION")
    logger.info("="*70)
    
    generators = [
        'data_generators/truthfulqa_generator.py',
        'data_generators/simpleqa_generator.py',
        'data_generators/factscore_generator.py',
        'data_generators/mmlupro_generator.py'
    ]
    
    for generator in generators:
        if Path(generator).exists():
            logger.info(f"Running {generator}")
            try:
                subprocess.run(['python', generator], check=True)
                logger.info(f"✅ Completed {generator}")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed {generator}: {e}")
        else:
            logger.warning(f"⚠️  Generator not found: {generator}")
    
    # Validate outputs
    for dataset in DATASETS:
        path = f"standard_datasets/{dataset}.json"
        if Path(path).exists():
            logger.info(f"✅ Dataset ready: {path}")
        else:
            logger.error(f"❌ Dataset missing: {path}")


def run_single_evaluation(method: str, model: str, dataset: str) -> Tuple[str, bool]:
    """Run a single evaluation task."""
    model_safe = model.replace('/', '_').replace(':', '_')
    output_file = f"results/{dataset}_{model_safe}{EVALUATION_METHODS[method]['output_suffix']}"
    
    # Check if already exists
    if Path(output_file).exists():
        logger.info(f"✓ Skipping (exists): {method} - {model} - {dataset}")
        return output_file, True
    
    # Build command
    config = EVALUATION_METHODS[method]
    cmd = [
        'python', config['script'],
        '--dataset', f'standard_datasets/{dataset}.json',
        '--model', model,
        '--samples', str(SAMPLE_SIZE)
    ] + config['args'] + ['--output-dir', 'results']
    
    logger.info(f"🚀 Running: {method} - {model} - {dataset}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        elapsed = time.time() - start_time
        logger.info(f"✅ Completed in {elapsed/60:.1f}min: {method} - {model} - {dataset}")
        return output_file, True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed: {method} - {model} - {dataset}")
        logger.error(f"   Error: {e.stderr[:200]}")
        return output_file, False


def run_evaluations_parallel(methods: List[str] = None):
    """Stage 2: Run all evaluations in parallel."""
    logger.info("="*70)
    logger.info("STAGE 2: EVALUATION EXECUTION (PARALLEL)")
    logger.info("="*70)
    
    if methods is None:
        methods = list(EVALUATION_METHODS.keys())
    
    # Create all tasks
    tasks = []
    for method in methods:
        for dataset in DATASETS:
            for model in MODELS:
                tasks.append((method, model, dataset))
    
    logger.info(f"Total evaluation tasks: {len(tasks)}")
    
    completed = 0
    failed = 0
    
    # Process by dataset to maximize parallelism
    for dataset in DATASETS:
        dataset_tasks = [(m, mod, d) for m, mod, d in tasks if d == dataset]
        logger.info(f"\n📊 Processing dataset: {dataset} ({len(dataset_tasks)} tasks)")
        
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(run_single_evaluation, *task): task
                for task in dataset_tasks
            }
            
            for future in as_completed(futures):
                task = futures[future]
                try:
                    output_file, success = future.result()
                    if success:
                        completed += 1
                    else:
                        failed += 1
                except Exception as e:
                    logger.error(f"Exception in task {task}: {e}")
                    failed += 1
                
                # Progress update
                total_done = completed + failed
                logger.info(f"Progress: {total_done}/{len(tasks)} ({total_done*100/len(tasks):.1f}%)")
    
    logger.info(f"\nEvaluation complete: {completed} succeeded, {failed} failed")


def run_confidence_interval_generation():
    """Stage 3: Generate confidence intervals using subsampling bootstrap."""
    logger.info("="*70)
    logger.info("STAGE 3: CONFIDENCE INTERVAL GENERATION")
    logger.info("="*70)
    
    ci_script = """
import json
import numpy as np
from pathlib import Path
import random
from typing import Dict, List, Tuple

MODELS = {models}
DATASETS = {datasets}
METHODS = {methods}
BOOTSTRAP_ITERATIONS = {bootstrap}
SUBSAMPLE_SIZE = {subsample}

def bootstrap_subsample(detailed_results: List[Dict], metric_key: str) -> List[float]:
    '''Generate bootstrap samples for a metric.'''
    n = len(detailed_results)
    if n == 0:
        return []
    
    metrics = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        # Subsample with replacement
        indices = [random.randint(0, n-1) for _ in range(min(SUBSAMPLE_SIZE, n))]
        sample = [detailed_results[i] for i in indices]
        
        # Calculate metric on subsample
        if metric_key == 'accuracy':
            correct = sum(1 for r in sample if r.get('is_correct', False))
            metrics.append(correct / len(sample))
        elif metric_key == 'coverage':
            answered = sum(1 for r in sample if not r.get('abstained', False))
            metrics.append(answered / len(sample))
        elif metric_key == 'f1_macro':
            # Simplified F1 calculation
            tp = sum(1 for r in sample if r.get('prediction') == 'correct' and r.get('ground_truth') == 'correct')
            fp = sum(1 for r in sample if r.get('prediction') == 'correct' and r.get('ground_truth') == 'incorrect')
            tn = sum(1 for r in sample if r.get('prediction') == 'incorrect' and r.get('ground_truth') == 'incorrect')
            fn = sum(1 for r in sample if r.get('prediction') == 'incorrect' and r.get('ground_truth') == 'correct')
            
            if (tp + fp) > 0:
                precision = tp / (tp + fp)
            else:
                precision = 0
            if (tp + fn) > 0:
                recall = tp / (tp + fn)
            else:
                recall = 0
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0
            metrics.append(f1)
    
    return metrics

def calculate_ci(values: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    '''Calculate confidence interval.'''
    if not values:
        return (0, 0, 0)
    
    alpha = (1 - confidence) / 2
    lower = np.percentile(values, alpha * 100)
    upper = np.percentile(values, (1 - alpha) * 100)
    mean = np.mean(values)
    return (mean, lower, upper)

# Process all results
all_cis = {{}}

for method in METHODS:
    all_cis[method] = {{}}
    
    for dataset in DATASETS:
        all_cis[method][dataset] = {{}}
        
        for model in MODELS:
            model_safe = model.replace('/', '_').replace(':', '_')
            
            # Determine file path based on method
            if method == 'bilateral':
                file_path = f'results/{{dataset}}_{{model_safe}}_classical_results.json'
            elif method == 'unilateral_forced':
                file_path = f'results/{{dataset}}_{{model_safe}}_unilateral_direct_results.json'
            elif method == 'unilateral_uncertain':
                file_path = f'results/{{dataset}}_{{model_safe}}_unilateral_uncertain_results.json'
            elif method == 'confidence':
                file_path = f'results/{{dataset}}_{{model_safe}}_unilateral_confidence_results.json'
            
            if Path(file_path).exists():
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                detailed = data.get('detailed_results', [])
                if detailed:
                    # Generate CIs for each metric
                    acc_samples = bootstrap_subsample(detailed, 'accuracy')
                    cov_samples = bootstrap_subsample(detailed, 'coverage')
                    f1_samples = bootstrap_subsample(detailed, 'f1_macro')
                    
                    all_cis[method][dataset][model] = {{
                        'accuracy': calculate_ci(acc_samples),
                        'coverage': calculate_ci(cov_samples),
                        'f1_macro': calculate_ci(f1_samples)
                    }}
                    
                    print(f'✅ Generated CI for {{method}} - {{dataset}} - {{model}}')

# Save results
with open('results/confidence_intervals_complete.json', 'w') as f:
    json.dump(all_cis, f, indent=2)

print('\\n💾 Saved confidence intervals to results/confidence_intervals_complete.json')
""".format(
        models=MODELS,
        datasets=DATASETS,
        methods=list(EVALUATION_METHODS.keys()),
        bootstrap=BOOTSTRAP_ITERATIONS,
        subsample=SUBSAMPLE_SIZE
    )
    
    # Write and execute CI script
    ci_script_path = 'temp_ci_generator.py'
    with open(ci_script_path, 'w') as f:
        f.write(ci_script)
    
    try:
        subprocess.run(['python', ci_script_path], check=True)
        logger.info("✅ Confidence intervals generated")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ CI generation failed: {e}")
    finally:
        Path(ci_script_path).unlink(missing_ok=True)


def run_final_analysis():
    """Stage 4: Generate final comparison tables and visualizations."""
    logger.info("="*70)
    logger.info("STAGE 4: FINAL ANALYSIS & TABLES")
    logger.info("="*70)
    
    # Generate comparison tables
    scripts = [
        'generate_table1_with_confidence.py',
        'generate_confidence_ci.py',
        'analyze_statistical_significance.py'  # Would need to create this
    ]
    
    for script in scripts:
        if Path(script).exists():
            logger.info(f"Running {script}")
            try:
                subprocess.run(['python', script], check=True)
                logger.info(f"✅ Completed {script}")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed {script}: {e}")


def validate_results():
    """Validate all expected results exist."""
    logger.info("="*70)
    logger.info("VALIDATION")
    logger.info("="*70)
    
    missing = []
    corrupted = []
    
    for method in EVALUATION_METHODS:
        for dataset in DATASETS:
            for model in MODELS:
                model_safe = model.replace('/', '_').replace(':', '_')
                output_file = f"results/{dataset}_{model_safe}{EVALUATION_METHODS[method]['output_suffix']}"
                
                if not Path(output_file).exists():
                    missing.append(output_file)
                else:
                    # Check if valid JSON
                    try:
                        with open(output_file, 'r') as f:
                            data = json.load(f)
                            if 'total_samples' not in data:
                                corrupted.append(output_file)
                    except:
                        corrupted.append(output_file)
    
    if missing:
        logger.warning(f"Missing {len(missing)} files:")
        for f in missing[:5]:
            logger.warning(f"  - {f}")
    
    if corrupted:
        logger.error(f"Corrupted {len(corrupted)} files:")
        for f in corrupted[:5]:
            logger.error(f"  - {f}")
    
    if not missing and not corrupted:
        logger.info("✅ All results validated successfully!")
    
    return len(missing) == 0 and len(corrupted) == 0


def main():
    """Main pipeline orchestration."""
    parser = argparse.ArgumentParser(description='Run full evaluation pipeline')
    parser.add_argument('--stage', choices=['all', 'data', 'eval', 'ci', 'analysis'],
                       default='all', help='Which stage to run')
    parser.add_argument('--parallel', action='store_true',
                       help='Run evaluations in parallel')
    parser.add_argument('--methods', nargs='+',
                       choices=list(EVALUATION_METHODS.keys()),
                       help='Specific methods to evaluate')
    parser.add_argument('--models', nargs='+', help='Specific models to evaluate')
    parser.add_argument('--datasets', nargs='+', help='Specific datasets to evaluate')
    parser.add_argument('--workers', type=int, default=MAX_WORKERS,
                       help='Number of parallel workers')
    
    args = parser.parse_args()
    
    # Override globals if specified
    if args.models:
        global MODELS
        MODELS = args.models
    if args.datasets:
        global DATASETS
        DATASETS = args.datasets
    if args.workers:
        global MAX_WORKERS
        MAX_WORKERS = args.workers
    
    start_time = time.time()
    
    logger.info("🚀 STARTING FULL EVALUATION PIPELINE")
    logger.info(f"Models: {len(MODELS)}")
    logger.info(f"Datasets: {len(DATASETS)}")
    logger.info(f"Methods: {args.methods or 'all'}")
    logger.info(f"Parallel: {args.parallel}")
    
    # Run stages
    if args.stage in ['all', 'data']:
        run_data_preparation()
    
    if args.stage in ['all', 'eval']:
        if args.parallel:
            run_evaluations_parallel(args.methods)
        else:
            # Sequential execution
            for method in (args.methods or EVALUATION_METHODS.keys()):
                for dataset in DATASETS:
                    for model in MODELS:
                        run_single_evaluation(method, model, dataset)
    
    if args.stage in ['all', 'ci']:
        run_confidence_interval_generation()
    
    if args.stage in ['all', 'analysis']:
        run_final_analysis()
    
    # Validate
    if args.stage == 'all':
        validate_results()
    
    elapsed = time.time() - start_time
    logger.info(f"\n✅ PIPELINE COMPLETE in {elapsed/3600:.1f} hours")


if __name__ == "__main__":
    main()