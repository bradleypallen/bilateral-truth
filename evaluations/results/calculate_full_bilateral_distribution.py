#!/usr/bin/env python3
"""
Calculate full bilateral truth value distribution table with all 9 possible values.
"""

import json
import glob
import numpy as np
from pathlib import Path
from collections import defaultdict

# Model mapping with display names
model_mapping = {
    'gpt-4.1-2025-04-14': 'GPT-4o',
    'gpt-4.1-mini-2025-04-14': 'GPT-4o Mini',
    'claude-opus-4-1-20250805': 'Claude 3.5 Sonnet',
    'claude-3-5-haiku-20241022': 'Claude 3.5 Haiku',
    'meta-llama_llama-4-scout': 'Llama 4 Scout',
    'meta-llama_llama-4-maverick': 'Llama 4 Maverick',
    'google_gemini-2.5-flash': 'Gemini 2.0 Flash'
}

datasets = ['truthfulqa_complete', 'simpleqa_complete', 'mmlupro_complete', 'factscore_complete']
dataset_display = {
    'truthfulqa_complete': 'TruthfulQA',
    'simpleqa_complete': 'SimpleQA',
    'mmlupro_complete': 'MMLU-Pro',
    'factscore_complete': 'FACTScore'
}

# All 9 possible truth value pairs
# Standard: <t,f>, <f,t>, <t,t>, <f,f>
# Epistemic failures: <t,e>, <e,t>, <f,e>, <e,f>, <e,e>
all_truth_values = ['<t,f>', '<f,t>', '<t,t>', '<f,f>', '<t,e>', '<e,t>', '<f,e>', '<e,f>', '<e,e>']

# Initialize storage
distributions = defaultdict(lambda: defaultdict(dict))

# Process only bilateral (classical) result files
for filepath in glob.glob('*_classical_results.json'):
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
        for m in model_mapping.keys():
            model_safe = m.replace('/', '_')
            if model_safe in filename:
                model = m
                break
        if not model:
            continue
        
        # Get bilateral distribution
        if 'bilateral_distribution' in data:
            total_samples = data.get('total_samples', 1000)
            dist = data['bilateral_distribution']
            
            # Calculate proportions for all possible values
            for tv in all_truth_values:
                count = dist.get(tv, 0)
                proportion = count / total_samples
                distributions[dataset][model][tv] = proportion
                
        # Also check detailed results for epistemic values
        if 'detailed_results' in data and '<e,e>' not in distributions[dataset][model]:
            # Count epistemic failures from detailed results
            epistemic_counts = defaultdict(int)
            for result in data['detailed_results']:
                if 'bilateral_value' in result:
                    bv = result['bilateral_value']
                    if 'e' in bv:
                        epistemic_counts[bv] += 1
            
            # Add epistemic values if found
            total_samples = len(data['detailed_results']) if data['detailed_results'] else 1000
            for tv in all_truth_values:
                if 'e' in tv and tv not in distributions[dataset][model]:
                    count = epistemic_counts.get(tv, 0)
                    proportion = count / total_samples
                    distributions[dataset][model][tv] = proportion
                    
    except Exception as e:
        print(f"% Error processing {filepath}: {e}")

# Generate LaTeX table with all 9 values
print("\\begin{table}[htbp]")
print("\\centering")
print("\\caption{Full distribution of bilateral truth values including epistemic states. Values show proportions P(⟨u,v⟩) for each truth value pair.}")
print("\\small")  # Make font smaller to fit more columns
print("\\begin{tabular}{ll|cccc|cccc|c}")
print("\\toprule")
print(" & & \\multicolumn{4}{c|}{\\textbf{Standard Values}} & \\multicolumn{4}{c|}{\\textbf{Partial Epistemic}} & \\textbf{Full} \\\\")
print("\\textbf{Dataset} & \\textbf{Model} & ⟨t,f⟩ & ⟨f,t⟩ & ⟨t,t⟩ & ⟨f,f⟩ & ⟨t,e⟩ & ⟨e,t⟩ & ⟨f,e⟩ & ⟨e,f⟩ & ⟨e,e⟩ \\\\")
print("\\midrule")

# Process each dataset
for i, dataset in enumerate(datasets):
    dataset_name = dataset_display[dataset]
    
    # Count models with data for this dataset
    models_with_data = [m for m in model_mapping.keys() if m in distributions[dataset]]
    if not models_with_data:
        continue
    
    # First model in dataset gets the dataset label
    first_model = True
    for model_key, model_name in model_mapping.items():
        if model_key in distributions[dataset]:
            if first_model:
                print(f"\\multirow{{{len(models_with_data)}}}{{*}}{{{dataset_name}}}")
                first_model = False
            else:
                print(" ")
                
            print(f" & {model_name}", end="")
            
            # Print all 9 values
            for tv in all_truth_values:
                val = distributions[dataset][model_key].get(tv, 0.0)
                if val > 0:
                    print(f" & {val:.3f}", end="")
                else:
                    print(f" & --", end="")
            
            print(" \\\\")
    
    if i < len(datasets) - 1:
        print("\\cline{1-11}")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

# Calculate aggregate statistics
print("\n% Summary statistics for significant truth values:")
print("\n\\begin{table}[htbp]")
print("\\centering")
print("\\caption{Summary statistics for bilateral truth value distributions (mean ± std across models).}")
print("\\begin{tabular}{l|cccc|c}")
print("\\toprule")
print("\\textbf{Dataset} & ⟨t,f⟩ & ⟨f,t⟩ & ⟨t,t⟩ & ⟨f,f⟩ & ⟨e,e⟩ \\\\")
print("\\midrule")

for dataset in datasets:
    if dataset not in distributions:
        continue
    dataset_name = dataset_display[dataset]
    
    row_values = []
    # Focus on the main 4 truth values plus <e,e>
    main_tvs = ['<t,f>', '<f,t>', '<t,t>', '<f,f>', '<e,e>']
    for tv in main_tvs:
        values = []
        for model in distributions[dataset]:
            if tv in distributions[dataset][model]:
                val = distributions[dataset][model][tv]
                if val > 0:  # Only count non-zero values
                    values.append(val)
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            if mean_val > 0.001:  # Only show if meaningful
                row_values.append(f"{mean_val:.3f}±{std_val:.3f}")
            else:
                row_values.append("--")
        else:
            row_values.append("--")
    
    print(f"{dataset_name} & {' & '.join(row_values)} \\\\")

print("\\midrule")

# Overall row
overall_stats = defaultdict(list)
for dataset in distributions:
    for model in distributions[dataset]:
        for tv in main_tvs:
            if tv in distributions[dataset][model]:
                val = distributions[dataset][model][tv]
                if val > 0:
                    overall_stats[tv].append(val)

overall_row = []
for tv in main_tvs:
    if overall_stats[tv]:
        mean_val = np.mean(overall_stats[tv])
        std_val = np.std(overall_stats[tv])
        if mean_val > 0.001:
            overall_row.append(f"{mean_val:.3f}±{std_val:.3f}")
        else:
            overall_row.append("--")
    else:
        overall_row.append("--")
print(f"\\textbf{{Overall}} & {' & '.join(overall_row)} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

# Print interpretation guide
print("\n% Interpretation Guide:")
print("% Standard values (both P+ and P- return t or f):")
print("%   ⟨t,f⟩: Verified as true, not refutable (correct positive)")
print("%   ⟨f,t⟩: Not verifiable, refuted as false (correct negative)")
print("%   ⟨t,t⟩: Both verifiable and refutable (contradiction)")
print("%   ⟨f,f⟩: Neither verifiable nor refutable (knowledge gap)")
print("% Epistemic failure values (one or both evaluations fail):")
print("%   ⟨t,e⟩: Verified as true, refutation failed")
print("%   ⟨e,t⟩: Verification failed, refuted as false")
print("%   ⟨f,e⟩: Not verifiable, refutation failed")
print("%   ⟨e,f⟩: Verification failed, not refutable")
print("%   ⟨e,e⟩: Both evaluations failed (API/parsing error)")