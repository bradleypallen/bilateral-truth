#!/usr/bin/env python3
"""
Calculate mean and standard deviation for comparative performance table.
"""

import json
import glob
import numpy as np
from pathlib import Path
from collections import defaultdict

# Model mapping
models = [
    'gpt-4.1-2025-04-14',
    'gpt-4.1-mini-2025-04-14', 
    'claude-opus-4-1-20250805',
    'claude-3-5-haiku-20241022',
    'meta-llama_llama-4-scout',
    'meta-llama_llama-4-maverick',
    'google_gemini-2.5-flash'
]

datasets = ['truthfulqa_complete', 'simpleqa_complete', 'mmlupro_complete', 'factscore_complete']

# Initialize storage
results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

# Process files
for filepath in glob.glob('*_results.json'):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Parse filename
        filename = Path(filepath).stem
        
        # Find dataset
        dataset = None
        for d in datasets:
            if d in filename:
                dataset = d
                break
        if not dataset:
            continue
            
        # Find model
        model = None
        for m in models:
            model_safe = m.replace('/', '_')
            if model_safe in filename:
                model = m
                break
        if not model:
            continue
        
        # Determine approach
        approach = None
        if 'classical' in filename:
            approach = 'Bilateral'
        elif 'unilateral_direct' in filename:
            approach = 'Unilateral-Binary'
        elif 'unilateral_uncertain' in filename:
            approach = 'Unilateral-Ternary'
        elif 'unilateral_confidence' in filename:
            # Process confidence thresholds
            if 'threshold_analysis' in data:
                for threshold, metrics in data['threshold_analysis'].items():
                    thresh_approach = f'Confidence-{threshold}'
                    results[dataset][thresh_approach]['f1'].append(metrics.get('f1_macro', 0))
                    results[dataset][thresh_approach]['coverage'].append(metrics.get('coverage', 0))
                    if 'evaluation_time' in data:
                        results[dataset][thresh_approach]['time'].append(data['evaluation_time'] / 3)
            continue
        else:
            continue
        
        # Store metrics
        if approach:
            results[dataset][approach]['f1'].append(data.get('f1_macro', 0))
            results[dataset][approach]['coverage'].append(data.get('coverage', 0))
            if 'evaluation_time' in data:
                results[dataset][approach]['time'].append(data['evaluation_time'])
                
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

# Calculate statistics and generate LaTeX table
print("\\begin{table}[htbp]")
print("\\centering")
print("\\caption{Performance metrics for different evaluation approaches on four benchmark datasets (N=1000). Values shown as mean (std) across 7 models.}")
print("\\begin{tabular}{llcccc}")
print("\\toprule")
print("\\textbf{Dataset} & \\textbf{Approach} & \\textbf{Macro F1} & \\textbf{Coverage} & \\textbf{Time (s)} & \\textbf{Tokens} \\\\")
print("\\midrule")

dataset_display = {
    'truthfulqa_complete': 'TruthfulQA',
    'simpleqa_complete': 'SimpleQA', 
    'mmlupro_complete': 'MMLU-Pro',
    'factscore_complete': 'FACTScore'
}

approaches_order = ['Bilateral', 'Unilateral-Binary', 'Unilateral-Ternary', 
                   'Confidence-0.5', 'Confidence-0.7', 'Confidence-0.9']

for i, dataset in enumerate(datasets):
    dataset_name = dataset_display[dataset]
    print(f"\\multirow{{6}}{{*}}{{{dataset_name}}}")
    
    for j, approach in enumerate(approaches_order):
        if approach in results[dataset]:
            f1_vals = results[dataset][approach]['f1']
            cov_vals = results[dataset][approach]['coverage']
            time_vals = results[dataset][approach]['time']
            
            if f1_vals:
                f1_mean = np.mean(f1_vals)
                f1_std = np.std(f1_vals)
                f1_str = f"{f1_mean:.3f} ({f1_std:.3f})"
            else:
                f1_str = "--"
                
            if cov_vals:
                cov_mean = np.mean(cov_vals)
                cov_std = np.std(cov_vals)
                cov_str = f"{cov_mean:.3f} ({cov_std:.3f})"
            else:
                cov_str = "--"
                
            if time_vals:
                time_mean = np.mean(time_vals)
                time_std = np.std(time_vals)
                time_str = f"{time_mean:.1f} ({time_std:.1f})"
            else:
                time_str = "--"
            
            print(f" & {approach} & {f1_str} & {cov_str} & {time_str} & -- \\\\")
        else:
            print(f" & {approach} & -- & -- & -- & -- \\\\")
    
    if i < len(datasets) - 1:
        print("\\cline{1-6}")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

# Calculate overall means
print("\n% Overall means across all datasets and models:")
overall_stats = defaultdict(lambda: {'f1': [], 'coverage': [], 'time': []})

for dataset in results:
    for approach in results[dataset]:
        overall_stats[approach]['f1'].extend(results[dataset][approach]['f1'])
        overall_stats[approach]['coverage'].extend(results[dataset][approach]['coverage'])
        overall_stats[approach]['time'].extend(results[dataset][approach]['time'])

print("\n% Summary Statistics:")
for approach in approaches_order:
    if approach in overall_stats:
        f1_mean = np.mean(overall_stats[approach]['f1']) if overall_stats[approach]['f1'] else 0
        cov_mean = np.mean(overall_stats[approach]['coverage']) if overall_stats[approach]['coverage'] else 0
        time_mean = np.mean(overall_stats[approach]['time']) if overall_stats[approach]['time'] else 0
        print(f"% {approach}: F1={f1_mean:.3f}, Coverage={cov_mean:.3f}, Time={time_mean:.1f}s")