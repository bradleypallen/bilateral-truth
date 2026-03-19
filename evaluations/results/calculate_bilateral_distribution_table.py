#!/usr/bin/env python3
"""
Calculate bilateral truth value distribution table.
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

# Truth value pairs
truth_values = ['<t,f>', '<f,t>', '<t,t>', '<f,f>']

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
            
            # Calculate proportions
            for tv in truth_values:
                count = dist.get(tv, 0)
                proportion = count / total_samples
                distributions[dataset][model][tv] = proportion
                
    except Exception as e:
        print(f"% Error processing {filepath}: {e}")

# Generate LaTeX table
print("\\begin{table}[htbp]")
print("\\centering")
print("\\caption{Distribution of bilateral truth values across benchmark datasets. Values show proportions P(⟨u,v⟩) for each truth value pair.}")
print("\\begin{tabular}{llcccc}")
print("\\toprule")
print("\\textbf{Dataset} & \\textbf{Model} & \\textbf{P(⟨t,f⟩)} & \\textbf{P(⟨f,t⟩)} & \\textbf{P(⟨t,t⟩)} & \\textbf{P(⟨f,f⟩)} \\\\")
print("\\midrule")

# Process each dataset
for i, dataset in enumerate(datasets):
    dataset_name = dataset_display[dataset]
    
    # Count models with data for this dataset
    models_with_data = [m for m in model_mapping.keys() if m in distributions[dataset]]
    if not models_with_data:
        continue
        
    print(f"\\multirow{{{len(models_with_data)}}}{{*}}{{{dataset_name}}}")
    
    for j, (model_key, model_name) in enumerate(model_mapping.items()):
        if model_key in distributions[dataset]:
            values = []
            for tv in truth_values:
                val = distributions[dataset][model_key].get(tv, 0)
                values.append(f"{val:.3f}")
            
            print(f" & {model_name} & {' & '.join(values)} \\\\")
    
    if i < len(datasets) - 1:
        print("\\cline{1-6}")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

# Calculate aggregate statistics
print("\n% Aggregate statistics across all models:")
for dataset in datasets:
    if dataset not in distributions:
        continue
    dataset_name = dataset_display[dataset]
    print(f"\n% {dataset_name}:")
    
    for tv in truth_values:
        values = []
        for model in distributions[dataset]:
            if tv in distributions[dataset][model]:
                values.append(distributions[dataset][model][tv])
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"%   {tv}: {mean_val:.3f} ± {std_val:.3f}")

# Overall statistics
print("\n% Overall statistics across all datasets and models:")
overall_stats = defaultdict(list)
for dataset in distributions:
    for model in distributions[dataset]:
        for tv in truth_values:
            if tv in distributions[dataset][model]:
                overall_stats[tv].append(distributions[dataset][model][tv])

for tv in truth_values:
    if overall_stats[tv]:
        mean_val = np.mean(overall_stats[tv])
        std_val = np.std(overall_stats[tv])
        print(f"% {tv}: {mean_val:.3f} ± {std_val:.3f}")

# Create a summary table with means and standard deviations
print("\n")
print("\\begin{table}[htbp]")
print("\\centering")
print("\\caption{Summary statistics for bilateral truth value distributions. Values show mean ± std across models.}")
print("\\begin{tabular}{lcccc}")
print("\\toprule")
print("\\textbf{Dataset} & \\textbf{P(⟨t,f⟩)} & \\textbf{P(⟨f,t⟩)} & \\textbf{P(⟨t,t⟩)} & \\textbf{P(⟨f,f⟩)} \\\\")
print("\\midrule")

for dataset in datasets:
    if dataset not in distributions:
        continue
    dataset_name = dataset_display[dataset]
    
    row_values = []
    for tv in truth_values:
        values = []
        for model in distributions[dataset]:
            if tv in distributions[dataset][model]:
                values.append(distributions[dataset][model][tv])
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            row_values.append(f"{mean_val:.3f} ± {std_val:.3f}")
        else:
            row_values.append("--")
    
    print(f"{dataset_name} & {' & '.join(row_values)} \\\\")

print("\\midrule")

# Overall row
overall_row = []
for tv in truth_values:
    if overall_stats[tv]:
        mean_val = np.mean(overall_stats[tv])
        std_val = np.std(overall_stats[tv])
        overall_row.append(f"{mean_val:.3f} ± {std_val:.3f}")
    else:
        overall_row.append("--")
print(f"\\textbf{{Overall}} & {' & '.join(overall_row)} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")