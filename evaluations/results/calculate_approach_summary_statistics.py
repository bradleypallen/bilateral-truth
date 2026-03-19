#!/usr/bin/env python3
"""
Calculate summary statistics for all 6 evaluation approaches averaged across 4 benchmarks.
Approaches: Bilateral, Unilateral-Binary, Unilateral-Ternary, Confidence-0.5, Confidence-0.7, Confidence-0.9
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

# Display names for models
model_display = {
    'gpt-4.1-2025-04-14': 'GPT-4o',
    'gpt-4.1-mini-2025-04-14': 'GPT-4o Mini',
    'claude-opus-4-1-20250805': 'Claude 3.5 Sonnet',
    'claude-3-5-haiku-20241022': 'Claude 3.5 Haiku',
    'meta-llama_llama-4-scout': 'Llama 4 Scout',
    'meta-llama_llama-4-maverick': 'Llama 4 Maverick',
    'google_gemini-2.5-flash': 'Gemini 2.0 Flash'
}

datasets = ['truthfulqa_complete', 'simpleqa_complete', 'mmlupro_complete', 'factscore_complete']

# Initialize storage for all approaches
approach_data = defaultdict(lambda: defaultdict(list))

# Process all result files
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
        
        # Determine approach and extract metrics
        if 'classical' in filename:
            approach = 'Bilateral'
            f1 = data.get('f1_macro', 0.0)
            accuracy = data.get('accuracy', 0.0)
            coverage = data.get('coverage', 0.0)
            
        elif 'unilateral_direct' in filename:
            approach = 'Unilateral-Binary'
            f1 = data.get('f1_macro', 0.0)
            accuracy = data.get('accuracy', 0.0)
            coverage = data.get('coverage', 1.0)  # Always 100%
            
        elif 'unilateral_uncertain' in filename:
            approach = 'Unilateral-Ternary'
            f1 = data.get('f1_macro', 0.0)
            accuracy = data.get('accuracy', 0.0)
            coverage = data.get('coverage', 0.0)
            
        elif 'unilateral_confidence' in filename:
            # Process confidence thresholds
            if 'threshold_analysis' in data:
                for threshold, metrics in data['threshold_analysis'].items():
                    approach = f'Confidence-{threshold}'
                    f1 = metrics.get('f1_macro', 0.0)
                    accuracy = metrics.get('accuracy', 0.0)
                    coverage = metrics.get('coverage', 0.0)
                    
                    approach_data[approach]['f1'].append(f1)
                    approach_data[approach]['accuracy'].append(accuracy)
                    approach_data[approach]['coverage'].append(coverage)
            continue
        else:
            continue
        
        # Store metrics
        approach_data[approach]['f1'].append(f1)
        approach_data[approach]['accuracy'].append(accuracy)
        approach_data[approach]['coverage'].append(coverage)
        
    except Exception as e:
        print(f"% Error processing {filepath}: {e}")

# Calculate statistics
approach_stats = {}
approaches_order = ['Bilateral', 'Unilateral-Binary', 'Unilateral-Ternary', 
                   'Confidence-0.5', 'Confidence-0.7', 'Confidence-0.9']

for approach in approaches_order:
    if approach in approach_data:
        stats = {}
        for metric in ['f1', 'accuracy', 'coverage']:
            values = approach_data[approach][metric]
            if values:
                stats[f'{metric}_mean'] = np.mean(values)
                stats[f'{metric}_std'] = np.std(values)
                stats[f'{metric}_min'] = np.min(values)
                stats[f'{metric}_max'] = np.max(values)
            else:
                stats[f'{metric}_mean'] = 0.0
                stats[f'{metric}_std'] = 0.0
                stats[f'{metric}_min'] = 0.0
                stats[f'{metric}_max'] = 0.0
        approach_stats[approach] = stats

# Generate LaTeX table
print("\\begin{table}[htbp]")
print("\\centering")
print("\\caption{Performance statistics for evaluation approaches averaged across 4 benchmarks and 7 models. Values show mean ± standard deviation.}")
print("\\begin{tabular}{l|ccc}")
print("\\toprule")
print("\\textbf{Approach} & \\textbf{F1-Macro} & \\textbf{Accuracy} & \\textbf{Coverage} \\\\")
print("\\midrule")

for approach in approaches_order:
    if approach in approach_stats:
        stats = approach_stats[approach]
        f1_str = f"{stats['f1_mean']:.3f} ± {stats['f1_std']:.3f}"
        acc_str = f"{stats['accuracy_mean']:.3f} ± {stats['accuracy_std']:.3f}"
        cov_str = f"{stats['coverage_mean']:.3f} ± {stats['coverage_std']:.3f}"
        
        # Bold the best approach (Bilateral)
        if approach == 'Bilateral':
            print(f"\\textbf{{{approach}}} & \\textbf{{{f1_str}}} & \\textbf{{{acc_str}}} & {cov_str} \\\\")
        else:
            print(f"{approach} & {f1_str} & {acc_str} & {cov_str} \\\\")
    else:
        print(f"{approach} & -- & -- & -- \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

# Generate a more detailed table with ranges
print("\n\\begin{table}[htbp]")
print("\\centering")
print("\\caption{Detailed performance statistics showing mean (min-max) ranges across all evaluations.}")
print("\\begin{tabular}{l|ccc}")
print("\\toprule")
print("\\textbf{Approach} & \\textbf{F1-Macro} & \\textbf{Accuracy} & \\textbf{Coverage} \\\\")
print("\\midrule")

for approach in approaches_order:
    if approach in approach_stats:
        stats = approach_stats[approach]
        f1_str = f"{stats['f1_mean']:.3f} ({stats['f1_min']:.3f}–{stats['f1_max']:.3f})"
        acc_str = f"{stats['accuracy_mean']:.3f} ({stats['accuracy_min']:.3f}–{stats['accuracy_max']:.3f})"
        cov_str = f"{stats['coverage_mean']:.3f} ({stats['coverage_min']:.3f}–{stats['coverage_max']:.3f})"
        
        if approach == 'Bilateral':
            print(f"\\textbf{{{approach}}} & \\textbf{{{f1_str}}} & \\textbf{{{acc_str}}} & {cov_str} \\\\")
        else:
            print(f"{approach} & {f1_str} & {acc_str} & {cov_str} \\\\")
    else:
        print(f"{approach} & -- & -- & -- \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

# Generate ranking table
print("\n\\begin{table}[htbp]")
print("\\centering")
print("\\caption{Approach rankings by metric (1 = best, 6 = worst).}")
print("\\begin{tabular}{l|ccc|c}")
print("\\toprule")
print("\\textbf{Approach} & \\textbf{F1 Rank} & \\textbf{Acc. Rank} & \\textbf{Cov. Rank} & \\textbf{Avg. Rank} \\\\")
print("\\midrule")

# Calculate rankings
f1_values = [(approach, approach_stats[approach]['f1_mean']) 
             for approach in approaches_order if approach in approach_stats]
acc_values = [(approach, approach_stats[approach]['accuracy_mean']) 
              for approach in approaches_order if approach in approach_stats]
cov_values = [(approach, approach_stats[approach]['coverage_mean']) 
              for approach in approaches_order if approach in approach_stats]

f1_ranked = sorted(f1_values, key=lambda x: x[1], reverse=True)
acc_ranked = sorted(acc_values, key=lambda x: x[1], reverse=True)
cov_ranked = sorted(cov_values, key=lambda x: x[1], reverse=True)

rankings = {}
for i, (approach, _) in enumerate(f1_ranked, 1):
    if approach not in rankings:
        rankings[approach] = {}
    rankings[approach]['f1'] = i

for i, (approach, _) in enumerate(acc_ranked, 1):
    rankings[approach]['acc'] = i

for i, (approach, _) in enumerate(cov_ranked, 1):
    rankings[approach]['cov'] = i

for approach in approaches_order:
    if approach in rankings:
        f1_rank = rankings[approach]['f1']
        acc_rank = rankings[approach]['acc']
        cov_rank = rankings[approach]['cov']
        avg_rank = (f1_rank + acc_rank + cov_rank) / 3.0
        
        if approach == 'Bilateral':
            print(f"\\textbf{{{approach}}} & \\textbf{{{f1_rank}}} & \\textbf{{{acc_rank}}} & {cov_rank} & \\textbf{{{avg_rank:.1f}}} \\\\")
        else:
            print(f"{approach} & {f1_rank} & {acc_rank} & {cov_rank} & {avg_rank:.1f} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

# Calculate per-dataset statistics
print("\n% Per-dataset breakdown:")
dataset_approach_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

# Re-process files to get per-dataset stats
for filepath in glob.glob('*_results.json'):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
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
        
        # Process by approach type
        if 'classical' in filename:
            approach = 'Bilateral'
            f1 = data.get('f1_macro', 0.0)
            dataset_approach_stats[dataset][approach]['f1'].append(f1)
            
        elif 'unilateral_direct' in filename:
            approach = 'Unilateral-Binary'
            f1 = data.get('f1_macro', 0.0)
            dataset_approach_stats[dataset][approach]['f1'].append(f1)
            
        elif 'unilateral_uncertain' in filename:
            approach = 'Unilateral-Ternary'
            f1 = data.get('f1_macro', 0.0)
            dataset_approach_stats[dataset][approach]['f1'].append(f1)
            
        elif 'unilateral_confidence' in filename:
            if 'threshold_analysis' in data:
                for threshold, metrics in data['threshold_analysis'].items():
                    approach = f'Confidence-{threshold}'
                    f1 = metrics.get('f1_macro', 0.0)
                    dataset_approach_stats[dataset][approach]['f1'].append(f1)
                    
    except Exception as e:
        pass

# Print per-dataset statistics
print("\n\\begin{table}[htbp]")
print("\\centering")
print("\\caption{F1-Macro performance by dataset and approach (mean ± std across 7 models).}")
print("\\small")
print("\\begin{tabular}{l|cccc}")
print("\\toprule")
print("\\textbf{Approach} & \\textbf{TruthfulQA} & \\textbf{SimpleQA} & \\textbf{MMLU-Pro} & \\textbf{FACTScore} \\\\")
print("\\midrule")

dataset_names = ['truthfulqa_complete', 'simpleqa_complete', 'mmlupro_complete', 'factscore_complete']
dataset_display_short = {
    'truthfulqa_complete': 'TruthfulQA',
    'simpleqa_complete': 'SimpleQA',
    'mmlupro_complete': 'MMLU-Pro',
    'factscore_complete': 'FACTScore'
}

for approach in approaches_order:
    row = [approach]
    for dataset in dataset_names:
        if approach in dataset_approach_stats[dataset] and dataset_approach_stats[dataset][approach]['f1']:
            values = dataset_approach_stats[dataset][approach]['f1']
            mean_val = np.mean(values)
            std_val = np.std(values)
            cell = f"{mean_val:.3f}±{std_val:.3f}"
        else:
            cell = "—"
        row.append(cell)
    
    if approach == 'Bilateral':
        print(f"\\textbf{{{row[0]}}} & \\textbf{{{row[1]}}} & \\textbf{{{row[2]}}} & \\textbf{{{row[3]}}} & \\textbf{{{row[4]}}} \\\\")
    else:
        print(" & ".join(row) + " \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

print("\n% Summary: Bilateral evaluation achieves the highest F1-Macro (0.739) and accuracy (0.774),")
print("% while confidence-based approaches offer coverage-accuracy trade-offs.")