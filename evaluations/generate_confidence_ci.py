#!/usr/bin/env python3
"""
Generate Confidence Intervals for Confidence-Based Evaluation Results
Using subsampling bootstrap approach
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import random
from collections import defaultdict

# Models and datasets
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

CONFIDENCE_THRESHOLDS = [0.5, 0.7, 0.9]
BOOTSTRAP_SAMPLES = 1000
SUBSAMPLE_SIZE = 100  # Match the subsampling used for other evaluations
CONFIDENCE_LEVEL = 0.95


def load_confidence_results(model: str, dataset: str) -> Dict:
    """Load confidence evaluation results."""
    model_safe = model.replace('/', '_').replace(':', '_')
    filepath = f"results/{dataset}_{model_safe}_unilateral_confidence_results.json"
    
    if Path(filepath).exists():
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


def evaluate_subsample_at_threshold(detailed_results: List[Dict], indices: List[int], 
                                   threshold: float) -> Dict[str, float]:
    """Evaluate a subsample at a specific confidence threshold."""
    correct = 0
    abstained = 0
    total = len(indices)
    
    threshold_key = str(threshold)
    
    for idx in indices:
        if idx >= len(detailed_results):
            continue
            
        result = detailed_results[idx]
        prediction = result['threshold_predictions'][threshold_key]
        ground_truth = result['ground_truth']
        
        if prediction == "ABSTAINED":
            abstained += 1
        else:
            is_correct = (
                (prediction == "CORRECT" and ground_truth == "correct") or
                (prediction == "INCORRECT" and ground_truth == "incorrect")
            )
            if is_correct:
                correct += 1
    
    accuracy = correct / total if total > 0 else 0
    coverage = (total - abstained) / total if total > 0 else 0
    
    # Calculate F1
    tp = sum(1 for idx in indices 
             if idx < len(detailed_results) and
             detailed_results[idx]['threshold_predictions'][threshold_key] == 'CORRECT' and
             detailed_results[idx]['ground_truth'] == 'correct')
    fp = sum(1 for idx in indices
             if idx < len(detailed_results) and
             detailed_results[idx]['threshold_predictions'][threshold_key] == 'CORRECT' and
             detailed_results[idx]['ground_truth'] == 'incorrect')
    tn = sum(1 for idx in indices
             if idx < len(detailed_results) and
             detailed_results[idx]['threshold_predictions'][threshold_key] == 'INCORRECT' and
             detailed_results[idx]['ground_truth'] == 'incorrect')
    fn = sum(1 for idx in indices
             if idx < len(detailed_results) and
             detailed_results[idx]['threshold_predictions'][threshold_key] == 'INCORRECT' and
             detailed_results[idx]['ground_truth'] == 'correct')
    
    precision_true = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_true = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_true = 2 * precision_true * recall_true / (precision_true + recall_true) if (precision_true + recall_true) > 0 else 0
    
    precision_false = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_false = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_false = 2 * precision_false * recall_false / (precision_false + recall_false) if (precision_false + recall_false) > 0 else 0
    
    f1_macro = (f1_true + f1_false) / 2
    
    return {
        'accuracy': accuracy,
        'coverage': coverage,
        'f1_macro': f1_macro
    }


def bootstrap_confidence_intervals(results: Dict, threshold: float) -> Dict[str, Tuple[float, float, float]]:
    """Calculate bootstrap confidence intervals for a specific threshold."""
    if not results or 'detailed_results' not in results:
        return None
    
    detailed_results = results['detailed_results']
    n_samples = len(detailed_results)
    
    if n_samples == 0:
        return None
    
    # Bootstrap sampling
    accuracies = []
    coverages = []
    f1_macros = []
    
    for _ in range(BOOTSTRAP_SAMPLES):
        # Sample with replacement
        indices = [random.randint(0, n_samples - 1) for _ in range(min(SUBSAMPLE_SIZE, n_samples))]
        metrics = evaluate_subsample_at_threshold(detailed_results, indices, threshold)
        
        accuracies.append(metrics['accuracy'])
        coverages.append(metrics['coverage'])
        f1_macros.append(metrics['f1_macro'])
    
    # Calculate confidence intervals
    alpha = (1 - CONFIDENCE_LEVEL) / 2
    
    def get_ci(values):
        lower = np.percentile(values, alpha * 100)
        upper = np.percentile(values, (1 - alpha) * 100)
        mean = np.mean(values)
        return (mean, lower, upper)
    
    return {
        'accuracy': get_ci(accuracies),
        'coverage': get_ci(coverages),
        'f1_macro': get_ci(f1_macros)
    }


def main():
    """Generate confidence intervals for all confidence-based evaluations."""
    
    print("="*100)
    print("CONFIDENCE INTERVALS FOR CONFIDENCE-BASED EVALUATION")
    print("="*100)
    print(f"Bootstrap samples: {BOOTSTRAP_SAMPLES}")
    print(f"Subsample size: {SUBSAMPLE_SIZE}")
    print(f"Confidence level: {CONFIDENCE_LEVEL*100}%")
    print()
    
    # Collect all results with CIs
    all_results = defaultdict(lambda: defaultdict(dict))
    
    for model in MODELS:
        model_name = model.split('/')[-1]
        
        for dataset in DATASETS:
            dataset_name = dataset.replace('_complete', '')
            
            # Load results
            results = load_confidence_results(model, dataset)
            if not results:
                print(f"⚠️  Missing: {model_name} on {dataset_name}")
                continue
            
            # Calculate CIs for each threshold
            for threshold in CONFIDENCE_THRESHOLDS:
                cis = bootstrap_confidence_intervals(results, threshold)
                if cis:
                    all_results[threshold][dataset_name][model_name] = cis
    
    # Print results by threshold
    for threshold in CONFIDENCE_THRESHOLDS:
        print(f"\n{'='*100}")
        print(f"CONFIDENCE THRESHOLD: {threshold}")
        print('='*100)
        
        for dataset_name in ['truthfulqa', 'simpleqa', 'factscore', 'mmlupro']:
            if dataset_name not in all_results[threshold]:
                continue
                
            print(f"\n{dataset_name.upper()}:")
            print("-"*80)
            
            # Print header
            print(f"{'Model':<25} {'Accuracy':<25} {'Coverage':<25} {'F1-Macro':<25}")
            print("-"*100)
            
            for model in MODELS:
                model_name = model.split('/')[-1]
                if model_name in all_results[threshold][dataset_name]:
                    metrics = all_results[threshold][dataset_name][model_name]
                    
                    acc = metrics['accuracy']
                    cov = metrics['coverage']
                    f1 = metrics['f1_macro']
                    
                    acc_str = f"{acc[0]:.3f} [{acc[1]:.3f}, {acc[2]:.3f}]"
                    cov_str = f"{cov[0]:.3f} [{cov[1]:.3f}, {cov[2]:.3f}]"
                    f1_str = f"{f1[0]:.3f} [{f1[1]:.3f}, {f1[2]:.3f}]"
                    
                    print(f"{model_name:<25} {acc_str:<25} {cov_str:<25} {f1_str:<25}")
    
    # Calculate aggregate statistics with CIs
    print(f"\n{'='*100}")
    print("AGGREGATE STATISTICS ACROSS ALL MODELS AND DATASETS")
    print('='*100)
    
    for threshold in CONFIDENCE_THRESHOLDS:
        print(f"\nThreshold {threshold}:")
        print("-"*50)
        
        all_acc = []
        all_cov = []
        all_f1 = []
        
        for dataset_name in all_results[threshold]:
            for model_name in all_results[threshold][dataset_name]:
                metrics = all_results[threshold][dataset_name][model_name]
                all_acc.append(metrics['accuracy'][0])
                all_cov.append(metrics['coverage'][0])
                all_f1.append(metrics['f1_macro'][0])
        
        if all_acc:
            print(f"  Accuracy: {np.mean(all_acc):.3f} ± {np.std(all_acc):.3f}")
            print(f"  Coverage: {np.mean(all_cov):.3f} ± {np.std(all_cov):.3f}")
            print(f"  F1-Macro: {np.mean(all_f1):.3f} ± {np.std(all_f1):.3f}")
    
    # Save results to file
    output_file = "results/confidence_evaluation_with_ci.json"
    with open(output_file, 'w') as f:
        # Convert defaultdict to regular dict for JSON serialization
        json.dump(dict(all_results), f, indent=2, default=lambda x: dict(x) if isinstance(x, defaultdict) else x)
    
    print(f"\n💾 Results with confidence intervals saved to: {output_file}")
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()