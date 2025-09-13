#!/usr/bin/env python3
"""
Estimate standard errors of macro F1 scores using subsampling method from Politis & Romano (1994).

Since we only have one pass through benchmarks due to API cost constraints,
we use subsampling to estimate the variability of our performance metrics.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import f1_score, precision_recall_fscore_support
import glob
from collections import defaultdict


def calculate_f1_with_abstention(y_true: List[str], y_pred: List[str]) -> Tuple[float, float]:
    """
    Calculate macro F1 score accounting for abstentions.
    
    Returns:
        (f1_score, coverage) tuple
    """
    # Filter out abstentions
    answered_mask = [pred != 'abstain' for pred in y_pred]
    coverage = sum(answered_mask) / len(y_pred) if len(y_pred) > 0 else 0.0
    
    if coverage == 0:
        return 0.0, 0.0
    
    y_true_answered = [y for y, answered in zip(y_true, answered_mask) if answered]
    y_pred_answered = [y for y, answered in zip(y_pred, answered_mask) if answered]
    
    if len(y_true_answered) == 0:
        return 0.0, coverage
    
    # Calculate macro F1 on answered items
    f1 = f1_score(y_true_answered, y_pred_answered, average='macro', zero_division=0)
    
    return f1, coverage


def subsample_f1_scores(y_true: List[str], y_pred: List[str], 
                        subsample_size: int, n_subsamples: int = 1000,
                        seed: int = 42) -> List[float]:
    """
    Generate F1 scores from multiple subsamples using Politis & Romano method.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        subsample_size: Size of each subsample (typically n/2 to n/√2)
        n_subsamples: Number of subsamples to generate
        seed: Random seed for reproducibility
    
    Returns:
        List of F1 scores from subsamples
    """
    np.random.seed(seed)
    n = len(y_true)
    
    if subsample_size > n:
        subsample_size = n
    
    f1_scores = []
    
    for _ in range(n_subsamples):
        # Random subsample without replacement
        indices = np.random.choice(n, size=subsample_size, replace=False)
        
        y_true_sub = [y_true[i] for i in indices]
        y_pred_sub = [y_pred[i] for i in indices]
        
        f1_sub, _ = calculate_f1_with_abstention(y_true_sub, y_pred_sub)
        f1_scores.append(f1_sub)
    
    return f1_scores


def estimate_standard_error(y_true: List[str], y_pred: List[str],
                           subsample_ratio: float = 0.632) -> Dict[str, float]:
    """
    Estimate standard error using subsampling method.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels  
        subsample_ratio: Ratio of sample size for subsampling (default 0.632 ≈ 1-1/e)
    
    Returns:
        Dictionary with statistics
    """
    n = len(y_true)
    subsample_size = int(n * subsample_ratio)
    
    # Generate subsamples
    f1_subsamples = subsample_f1_scores(y_true, y_pred, subsample_size)
    
    # Calculate full sample F1
    f1_full, coverage = calculate_f1_with_abstention(y_true, y_pred)
    
    # Compute statistics
    f1_mean = np.mean(f1_subsamples)
    f1_std = np.std(f1_subsamples, ddof=1)
    
    # Adjust standard error for subsample size (Politis & Romano correction)
    # SE = σ_subsample * sqrt(m/n) where m is subsample size
    se_adjusted = f1_std * np.sqrt(subsample_size / n)
    
    # 95% confidence interval using normal approximation
    ci_lower = f1_full - 1.96 * se_adjusted
    ci_upper = f1_full + 1.96 * se_adjusted
    
    return {
        'f1_score': f1_full,
        'coverage': coverage,
        'se': se_adjusted,
        'se_unadjusted': f1_std,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'subsample_mean': f1_mean,
        'subsample_std': f1_std,
        'n_samples': n,
        'subsample_size': subsample_size
    }


def load_evaluation_results(result_file: str) -> Tuple[List[str], List[str]]:
    """Load predictions and ground truth from a result file."""
    with open(result_file) as f:
        data = json.load(f)
    
    y_true = []
    y_pred = []
    
    if 'detailed_results' in data:
        for item in data['detailed_results']:
            # Get ground truth - handle different field names
            ground_truth = item.get('expected_label', '') or item.get('ground_truth', '')
            y_true.append(ground_truth)
            
            # Handle different result formats
            if 'bilateral_value' in item:
                # Bilateral results - apply classical policy
                bilateral = item['bilateral_value']
                if bilateral == '<t,f>':
                    y_pred.append('correct')
                elif bilateral == '<f,t>':
                    y_pred.append('incorrect')
                else:
                    y_pred.append('abstain')
            elif 'prediction' in item:
                # Unilateral results with raw prediction
                pred = item['prediction'].lower()
                if pred in ['correct', 'incorrect']:
                    y_pred.append(pred)
                elif pred == 'uncertain':
                    y_pred.append('abstain')
                else:
                    y_pred.append(pred)
            elif 'predicted_label' in item:
                # Alternative unilateral format
                pred = item['predicted_label']
                if pred in ['correct', 'incorrect', 'uncertain']:
                    y_pred.append(pred if pred != 'uncertain' else 'abstain')
                else:
                    y_pred.append(pred)
    
    return y_true, y_pred


def analyze_all_results_with_se():
    """Analyze all evaluation results and compute standard errors."""
    
    results = []
    
    # Find all result files by type
    bilateral_files = glob.glob('results/*_classical_results.json')
    unilateral_direct_files = glob.glob('results/*_unilateral_direct_results.json')
    unilateral_uncertain_files = glob.glob('results/*_unilateral_uncertain_results.json')
    
    # Process bilateral results
    for file in bilateral_files:
        try:
            with open(file) as f:
                data = json.load(f)
            
            model = data.get('model', '')
            filename = Path(file).stem
            
            # Extract dataset from filename (e.g., "factscore_complete" from "factscore_complete_model_classical_results")
            parts = filename.split('_')
            dataset = '_'.join(parts[:2]) if len(parts) >= 2 else ''  # e.g., factscore_complete
            
            # Load predictions and ground truth
            y_true, y_pred = load_evaluation_results(file)
            
            if len(y_true) == 0 or len(y_pred) == 0:
                continue
            
            # Estimate standard error
            stats = estimate_standard_error(y_true, y_pred)
            
            results.append({
                'model': model.split('/')[-1] if '/' in model else model,
                'dataset': dataset.replace('_complete', ''),
                'method': 'bilateral_classical',
                'f1_score': stats['f1_score'],
                'se': stats['se'],
                'ci_lower': stats['ci_lower'],
                'ci_upper': stats['ci_upper'],
                'coverage': stats['coverage'],
                'n_samples': stats['n_samples']
            })
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    
    # Process unilateral direct results
    for file in unilateral_direct_files:
        try:
            with open(file) as f:
                data = json.load(f)
            
            model = data.get('model', '')
            filename = Path(file).stem
            
            # Extract dataset from filename
            parts = filename.split('_')
            dataset = '_'.join(parts[:2]) if len(parts) >= 2 else ''
            
            # Load predictions and ground truth
            y_true, y_pred = load_evaluation_results(file)
            
            if len(y_true) == 0 or len(y_pred) == 0:
                continue
            
            # Estimate standard error
            stats = estimate_standard_error(y_true, y_pred)
            
            results.append({
                'model': model.split('/')[-1] if '/' in model else model,
                'dataset': dataset.replace('_complete', ''),
                'method': 'unilateral_direct',
                'f1_score': stats['f1_score'],
                'se': stats['se'],
                'ci_lower': stats['ci_lower'],
                'ci_upper': stats['ci_upper'],
                'coverage': stats['coverage'],
                'n_samples': stats['n_samples']
            })
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    
    # Process unilateral uncertain results
    for file in unilateral_uncertain_files:
        try:
            with open(file) as f:
                data = json.load(f)
            
            model = data.get('model', '')
            filename = Path(file).stem
            
            # Extract dataset from filename
            parts = filename.split('_')
            dataset = '_'.join(parts[:2]) if len(parts) >= 2 else ''
            
            # Load predictions and ground truth
            y_true, y_pred = load_evaluation_results(file)
            
            if len(y_true) == 0 or len(y_pred) == 0:
                continue
            
            # Estimate standard error
            stats = estimate_standard_error(y_true, y_pred)
            
            results.append({
                'model': model.split('/')[-1] if '/' in model else model,
                'dataset': dataset.replace('_complete', ''),
                'method': 'unilateral_uncertain',
                'f1_score': stats['f1_score'],
                'se': stats['se'],
                'ci_lower': stats['ci_lower'],
                'ci_upper': stats['ci_upper'],
                'coverage': stats['coverage'],
                'n_samples': stats['n_samples']
            })
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    
    return pd.DataFrame(results)


def print_results_with_confidence_intervals(df: pd.DataFrame):
    """Print results with confidence intervals."""
    
    print("\n" + "="*120)
    print("MACRO F1 SCORES WITH 95% CONFIDENCE INTERVALS")
    print("(Standard errors estimated using Politis & Romano (1994) subsampling method)")
    print("="*120)
    
    # Group by method
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        
        print(f"\n{method.upper().replace('_', ' ')}:")
        print("-"*120)
        print(f"{'Model':<25} {'Dataset':<15} {'F1 Score':>12} {'SE':>8} {'95% CI':>20} {'Coverage':>10}")
        print("-"*120)
        
        for _, row in method_df.sort_values(['dataset', 'model']).iterrows():
            ci = f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
            print(f"{row['model']:<25} {row['dataset']:<15} "
                  f"{row['f1_score']:>12.3f} {row['se']:>8.4f} {ci:>20} {row['coverage']:>10.1%}")
        
        # Method summary
        print(f"\nMethod Average: {method_df['f1_score'].mean():.3f} ± {method_df['se'].mean():.4f}")
    
    # Pairwise comparisons with significance testing
    print("\n" + "="*120)
    print("PAIRWISE METHOD COMPARISONS")
    print("="*120)
    
    bilateral = df[df['method'] == 'bilateral_classical']
    unilateral = df[df['method'] == 'unilateral_direct']
    uncertain = df[df['method'] == 'unilateral_uncertain']
    
    # Merge for pairwise comparison
    comparison = pd.merge(
        bilateral[['model', 'dataset', 'f1_score', 'se', 'ci_lower', 'ci_upper']],
        unilateral[['model', 'dataset', 'f1_score', 'se', 'ci_lower', 'ci_upper']],
        on=['model', 'dataset'],
        suffixes=('_bi', '_uni')
    )
    
    if len(comparison) > 0:
        print("\nBilateral vs Forced Unilateral:")
        
        # Calculate pooled standard error for difference
        comparison['se_diff'] = np.sqrt(comparison['se_bi']**2 + comparison['se_uni']**2)
        comparison['f1_diff'] = comparison['f1_score_bi'] - comparison['f1_score_uni']
        
        # Test if confidence intervals overlap
        comparison['significant'] = ~((comparison['ci_lower_bi'] <= comparison['ci_upper_uni']) & 
                                     (comparison['ci_lower_uni'] <= comparison['ci_upper_bi']))
        
        print(f"  Average F1 difference: {comparison['f1_diff'].mean():.3f} ± {comparison['se_diff'].mean():.4f}")
        print(f"  Significant differences: {comparison['significant'].sum()}/{len(comparison)} pairs")
        print(f"  Bilateral wins: {(comparison['f1_diff'] > 0).sum()}/{len(comparison)}")
        
        # Show significant differences
        sig_diffs = comparison[comparison['significant']]
        if len(sig_diffs) > 0:
            print("\n  Statistically significant differences (non-overlapping 95% CIs):")
            for _, row in sig_diffs.iterrows():
                winner = "Bilateral" if row['f1_diff'] > 0 else "Unilateral"
                print(f"    {row['model']}-{row['dataset']}: {winner} wins by {abs(row['f1_diff']):.3f}")


def save_results_with_se(df: pd.DataFrame, output_file: str = 'results/evaluation_results_with_se.csv'):
    """Save results with standard errors to CSV."""
    df.to_csv(output_file, index=False)
    print(f"\n💾 Results with standard errors saved to {output_file}")


def main():
    """Main analysis pipeline."""
    print("Analyzing evaluation results with standard error estimation...")
    
    # Load and analyze all results
    df = analyze_all_results_with_se()
    
    if len(df) == 0:
        print("No evaluation results found")
        return
    
    # Print results with confidence intervals
    print_results_with_confidence_intervals(df)
    
    # Save results
    save_results_with_se(df)
    
    print("\n" + "="*120)
    print("SUMMARY")
    print("="*120)
    print(f"Total evaluations analyzed: {len(df)}")
    print(f"Models: {df['model'].nunique()}")
    print(f"Datasets: {df['dataset'].nunique()}")
    print(f"Methods: {df['method'].nunique()}")
    print(f"\nAverage standard error: {df['se'].mean():.4f}")
    print(f"Average 95% CI width: {(df['ci_upper'] - df['ci_lower']).mean():.3f}")


if __name__ == "__main__":
    main()