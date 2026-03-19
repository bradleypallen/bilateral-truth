#!/usr/bin/env python3
"""
Generate Table 1: Comprehensive Performance Comparison
Including bilateral, unilateral (forced/uncertain), and confidence-based approaches
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
import numpy as np

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


def load_result(filepath: str) -> Dict:
    """Load a result file if it exists."""
    if Path(filepath).exists():
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


def get_metrics(result: Dict, approach: str, threshold: float = None) -> Dict:
    """Extract key metrics from result."""
    if not result:
        return {'accuracy': None, 'coverage': None, 'f1_macro': None}
    
    if approach == 'confidence' and threshold is not None:
        # Get metrics for specific threshold
        threshold_key = str(threshold)
        if threshold_key in result.get('threshold_analysis', {}):
            stats = result['threshold_analysis'][threshold_key]
            return {
                'accuracy': stats.get('accuracy', 0),
                'coverage': stats.get('coverage', 0),
                'f1_macro': stats.get('f1_macro', 0)
            }
    else:
        # Get overall metrics
        return {
            'accuracy': result.get('accuracy', 0),
            'coverage': result.get('coverage', 0),
            'f1_macro': result.get('f1_macro', 0)
        }


def create_comprehensive_table():
    """Create comprehensive performance table."""
    
    # Collect all results
    all_results = []
    
    for model in MODELS:
        model_safe = model.replace('/', '_').replace(':', '_')
        
        for dataset in DATASETS:
            # Bilateral results
            bilateral_file = f"results/{dataset}_{model_safe}_classical_results.json"
            bilateral_result = load_result(bilateral_file)
            if bilateral_result:
                metrics = get_metrics(bilateral_result, 'bilateral')
                all_results.append({
                    'Model': model.split('/')[-1],  # Simplify model name
                    'Dataset': dataset.replace('_complete', ''),
                    'Approach': 'Bilateral',
                    'Accuracy': metrics['accuracy'],
                    'Coverage': metrics['coverage'],
                    'F1-Macro': metrics['f1_macro']
                })
            
            # Unilateral Forced
            forced_file = f"results/{dataset}_{model_safe}_unilateral_direct_results.json"
            forced_result = load_result(forced_file)
            if forced_result:
                metrics = get_metrics(forced_result, 'unilateral_forced')
                all_results.append({
                    'Model': model.split('/')[-1],
                    'Dataset': dataset.replace('_complete', ''),
                    'Approach': 'Unilateral-Forced',
                    'Accuracy': metrics['accuracy'],
                    'Coverage': metrics['coverage'],
                    'F1-Macro': metrics['f1_macro']
                })
            
            # Unilateral Uncertain
            uncertain_file = f"results/{dataset}_{model_safe}_unilateral_uncertain_results.json"
            uncertain_result = load_result(uncertain_file)
            if uncertain_result:
                metrics = get_metrics(uncertain_result, 'unilateral_uncertain')
                all_results.append({
                    'Model': model.split('/')[-1],
                    'Dataset': dataset.replace('_complete', ''),
                    'Approach': 'Unilateral-Uncertain',
                    'Accuracy': metrics['accuracy'],
                    'Coverage': metrics['coverage'],
                    'F1-Macro': metrics['f1_macro']
                })
            
            # Confidence-based (multiple thresholds)
            confidence_file = f"results/{dataset}_{model_safe}_unilateral_confidence_results.json"
            confidence_result = load_result(confidence_file)
            if confidence_result:
                for threshold in CONFIDENCE_THRESHOLDS:
                    metrics = get_metrics(confidence_result, 'confidence', threshold)
                    all_results.append({
                        'Model': model.split('/')[-1],
                        'Dataset': dataset.replace('_complete', ''),
                        'Approach': f'Confidence-{threshold}',
                        'Accuracy': metrics['accuracy'],
                        'Coverage': metrics['coverage'],
                        'F1-Macro': metrics['f1_macro']
                    })
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Create summary statistics
    print("\n" + "="*100)
    print("TABLE 1: COMPREHENSIVE PERFORMANCE COMPARISON")
    print("="*100)
    
    # Average by approach across all models and datasets
    print("\n📊 OVERALL AVERAGES BY APPROACH:")
    print("-"*50)
    approach_avg = df.groupby('Approach')[['Accuracy', 'Coverage', 'F1-Macro']].mean()
    approach_avg = approach_avg.round(3)
    print(approach_avg.to_string())
    
    # Average by model across all datasets and approaches
    print("\n📊 AVERAGE PERFORMANCE BY MODEL:")
    print("-"*50)
    model_avg = df.groupby('Model')[['Accuracy', 'Coverage', 'F1-Macro']].mean()
    model_avg = model_avg.round(3)
    print(model_avg.to_string())
    
    # Average by dataset across all models and approaches
    print("\n📊 AVERAGE PERFORMANCE BY DATASET:")
    print("-"*50)
    dataset_avg = df.groupby('Dataset')[['Accuracy', 'Coverage', 'F1-Macro']].mean()
    dataset_avg = dataset_avg.round(3)
    print(dataset_avg.to_string())
    
    # Detailed table by model and dataset
    print("\n📊 DETAILED RESULTS BY MODEL AND DATASET:")
    print("-"*100)
    
    for dataset in DATASETS:
        dataset_name = dataset.replace('_complete', '')
        print(f"\n{dataset_name.upper()}:")
        dataset_df = df[df['Dataset'] == dataset_name]
        
        # Pivot table for better readability
        pivot = dataset_df.pivot_table(
            index='Model',
            columns='Approach',
            values=['Accuracy', 'Coverage', 'F1-Macro'],
            aggfunc='first'
        )
        
        # Format for display
        print("\nAccuracy:")
        if 'Accuracy' in pivot:
            acc_table = pivot['Accuracy'].round(3)
            print(acc_table.to_string())
        
        print("\nCoverage:")
        if 'Coverage' in pivot:
            cov_table = pivot['Coverage'].round(3)
            print(cov_table.to_string())
        
        print("\nF1-Macro:")
        if 'F1-Macro' in pivot:
            f1_table = pivot['F1-Macro'].round(3)
            print(f1_table.to_string())
    
    # Save to CSV
    df.to_csv('results/table1_comprehensive_performance.csv', index=False)
    print("\n💾 Full results saved to: results/table1_comprehensive_performance.csv")
    
    # Key insights
    print("\n" + "="*100)
    print("KEY INSIGHTS:")
    print("="*100)
    
    # Best approach by accuracy
    best_acc = approach_avg['Accuracy'].idxmax()
    print(f"✓ Highest Accuracy: {best_acc} ({approach_avg.loc[best_acc, 'Accuracy']:.3f})")
    
    # Best approach by coverage
    best_cov = approach_avg['Coverage'].idxmax()
    print(f"✓ Highest Coverage: {best_cov} ({approach_avg.loc[best_cov, 'Coverage']:.3f})")
    
    # Best approach by F1
    best_f1 = approach_avg['F1-Macro'].idxmax()
    print(f"✓ Highest F1-Macro: {best_f1} ({approach_avg.loc[best_f1, 'F1-Macro']:.3f})")
    
    # Confidence threshold analysis
    print("\n📈 CONFIDENCE THRESHOLD ANALYSIS:")
    confidence_approaches = [f'Confidence-{t}' for t in CONFIDENCE_THRESHOLDS]
    conf_data = approach_avg.loc[confidence_approaches]
    
    print(f"  As threshold increases from 0.5 → 0.9:")
    print(f"    • Accuracy: {conf_data.loc['Confidence-0.5', 'Accuracy']:.3f} → {conf_data.loc['Confidence-0.9', 'Accuracy']:.3f}")
    print(f"    • Coverage: {conf_data.loc['Confidence-0.5', 'Coverage']:.3f} → {conf_data.loc['Confidence-0.9', 'Coverage']:.3f}")
    print(f"    • F1-Macro: {conf_data.loc['Confidence-0.5', 'F1-Macro']:.3f} → {conf_data.loc['Confidence-0.9', 'F1-Macro']:.3f}")
    
    # Bilateral vs best unilateral
    bilateral_f1 = approach_avg.loc['Bilateral', 'F1-Macro']
    best_unilateral = approach_avg.drop('Bilateral').loc[:, 'F1-Macro'].idxmax()
    best_unilateral_f1 = approach_avg.loc[best_unilateral, 'F1-Macro']
    
    print(f"\n✓ Bilateral vs Best Alternative:")
    print(f"    • Bilateral F1: {bilateral_f1:.3f}")
    print(f"    • Best Alternative ({best_unilateral}): {best_unilateral_f1:.3f}")
    print(f"    • Bilateral Advantage: {(bilateral_f1 - best_unilateral_f1):.3f}")
    
    return df


def main():
    """Generate comprehensive Table 1."""
    df = create_comprehensive_table()
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)
    print("✅ Table 1 with confidence-based evaluation has been generated")
    print("📊 Results include bilateral, unilateral (forced/uncertain), and confidence (0.5/0.7/0.9)")
    print("💾 Data saved to: results/table1_comprehensive_performance.csv")


if __name__ == "__main__":
    main()