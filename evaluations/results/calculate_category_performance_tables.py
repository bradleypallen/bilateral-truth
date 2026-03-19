#!/usr/bin/env python3
"""
Calculate model performance by category for each benchmark dataset.
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

# Initialize storage for category performance
category_performance = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

# Process bilateral (classical) result files to get category performance
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
        
        # Extract category analysis
        if 'category_analysis' in data:
            for category, metrics in data['category_analysis'].items():
                # Use accuracy as proxy for F1 if F1 not available
                f1 = metrics.get('f1_macro', metrics.get('f1', metrics.get('accuracy', 0.0)))
                accuracy = metrics.get('accuracy', 0.0)
                coverage = metrics.get('coverage', 0.0)
                samples = metrics.get('total', metrics.get('total_samples', 0))
                
                category_performance[dataset][category][model] = {
                    'f1': f1,
                    'accuracy': accuracy,
                    'coverage': coverage,
                    'samples': samples
                }
                
    except Exception as e:
        print(f"% Error processing {filepath}: {e}")

# Generate LaTeX tables for each dataset
for dataset in datasets:
    if dataset not in category_performance:
        continue
    
    dataset_name = dataset_display[dataset]
    
    # Get all categories for this dataset
    all_categories = sorted(category_performance[dataset].keys())
    
    # Limit categories for display (top 10 by sample size)
    category_samples = {}
    for cat in all_categories:
        total_samples = sum(category_performance[dataset][cat][m].get('samples', 0) 
                          for m in category_performance[dataset][cat])
        category_samples[cat] = total_samples
    
    # Sort categories by sample size and take top ones
    top_categories = sorted(category_samples.keys(), 
                           key=lambda x: category_samples[x], 
                           reverse=True)[:12]  # Top 12 categories
    
    print(f"\\begin{{table}}[htbp]")
    print(f"\\centering")
    print(f"\\caption{{Model performance (F1-Macro) by category on {dataset_name}. Values show F1 scores for bilateral evaluation.}}")
    print(f"\\small")
    
    # Create header with model names
    header = "\\begin{tabular}{l|" + "c" * len(model_mapping) + "}"
    print(header)
    print("\\toprule")
    
    model_headers = " & ".join([f"\\rotatebox{{45}}{{{model_mapping[m]}}}" 
                                for m in model_mapping.keys()])
    print(f"\\textbf{{Category}} & {model_headers} \\\\")
    print("\\midrule")
    
    # Print category rows
    for category in top_categories:
        cat_display = category.replace('_', ' ').title()
        # Truncate long category names
        if len(cat_display) > 20:
            cat_display = cat_display[:17] + "..."
        
        row = [cat_display]
        
        for model in model_mapping.keys():
            if model in category_performance[dataset][category]:
                f1 = category_performance[dataset][category][model]['f1']
                if f1 > 0:
                    # Color code based on performance
                    if f1 >= 0.9:
                        row.append(f"\\textbf{{{f1:.2f}}}")  # Bold for excellent
                    elif f1 >= 0.7:
                        row.append(f"{f1:.2f}")  # Normal for good
                    elif f1 >= 0.5:
                        row.append(f"\\textit{{{f1:.2f}}}")  # Italic for moderate
                    else:
                        row.append(f"\\textcolor{{gray}}{{{f1:.2f}}}")  # Gray for poor
                else:
                    row.append("--")
            else:
                row.append("--")
        
        print(" & ".join(row) + " \\\\")
    
    # Add summary statistics row
    print("\\midrule")
    print("\\textbf{Mean}", end="")
    for model in model_mapping.keys():
        f1_values = []
        for cat in top_categories:
            if model in category_performance[dataset][cat]:
                f1 = category_performance[dataset][cat][model]['f1']
                if f1 > 0:
                    f1_values.append(f1)
        if f1_values:
            mean_f1 = np.mean(f1_values)
            print(f" & \\textbf{{{mean_f1:.2f}}}", end="")
        else:
            print(" & --", end="")
    print(" \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print(f"\\end{{table}}")
    print()

# Create a summary table showing best/worst categories per model
print("\\begin{table}[htbp]")
print("\\centering")
print("\\caption{Best and worst performing categories for each model across all datasets}")
print("\\begin{tabular}{ll|ll}")
print("\\toprule")
print("\\textbf{Model} & \\textbf{Dataset} & \\textbf{Best Category (F1)} & \\textbf{Worst Category (F1)} \\\\")
print("\\midrule")

for model_key, model_name in model_mapping.items():
    first_row = True
    for dataset in datasets:
        if dataset not in category_performance:
            continue
        
        dataset_name = dataset_display[dataset]
        
        # Find best and worst categories for this model in this dataset
        model_categories = {}
        for cat in category_performance[dataset]:
            if model_key in category_performance[dataset][cat]:
                f1 = category_performance[dataset][cat][model_key]['f1']
                if f1 > 0:
                    model_categories[cat] = f1
        
        if model_categories:
            best_cat = max(model_categories.keys(), key=lambda x: model_categories[x])
            worst_cat = min(model_categories.keys(), key=lambda x: model_categories[x])
            
            best_f1 = model_categories[best_cat]
            worst_f1 = model_categories[worst_cat]
            
            # Truncate category names
            best_display = best_cat.replace('_', ' ').title()[:15]
            worst_display = worst_cat.replace('_', ' ').title()[:15]
            
            if first_row:
                print(f"\\multirow{{4}}{{*}}{{{model_name}}} & {dataset_name} & "
                      f"{best_display} ({best_f1:.2f}) & {worst_display} ({worst_f1:.2f}) \\\\")
                first_row = False
            else:
                print(f" & {dataset_name} & "
                      f"{best_display} ({best_f1:.2f}) & {worst_display} ({worst_f1:.2f}) \\\\")
    
    if not first_row:  # Only add line if we printed something
        print("\\cline{1-4}")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

# Create a heatmap-style summary for major categories across all models and datasets
print("\n% Category Performance Heatmap Data")
print("% This can be used to create a heatmap visualization")

# Identify common categories across datasets
common_categories = set()
for dataset in category_performance:
    common_categories.update(category_performance[dataset].keys())

# Filter to meaningful categories (present in multiple datasets)
category_counts = defaultdict(int)
for cat in common_categories:
    for dataset in category_performance:
        if cat in category_performance[dataset]:
            category_counts[cat] += 1

# Categories present in at least 2 datasets
multi_dataset_categories = [cat for cat, count in category_counts.items() if count >= 2]

if multi_dataset_categories:
    print("\n\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Cross-dataset category performance (F1-Macro). Categories appearing in multiple datasets.}")
    print("\\tiny")  # Very small font for this large table
    print("\\begin{tabular}{l|" + "c" * (len(datasets) * len(model_mapping)) + "}")
    print("\\toprule")
    
    # Header row with dataset groups
    header1 = "\\textbf{Category}"
    for dataset in datasets:
        dataset_name = dataset_display[dataset]
        header1 += f" & \\multicolumn{{{len(model_mapping)}}}{{c|}}{{{dataset_name}}}"
    print(header1 + " \\\\")
    
    # Sub-header with model names
    header2 = ""
    for dataset in datasets:
        for model in model_mapping.keys():
            model_short = model_mapping[model].split()[0]  # First word only
            header2 += f" & {model_short}"
    print(header2 + " \\\\")
    print("\\midrule")
    
    # Data rows
    for category in sorted(multi_dataset_categories)[:10]:  # Limit to 10 categories
        cat_display = category.replace('_', ' ').title()[:20]
        row = [cat_display]
        
        for dataset in datasets:
            for model in model_mapping.keys():
                if (dataset in category_performance and 
                    category in category_performance[dataset] and
                    model in category_performance[dataset][category]):
                    f1 = category_performance[dataset][category][model]['f1']
                    if f1 > 0:
                        # Use color intensity for heatmap effect
                        if f1 >= 0.8:
                            row.append(f"\\cellcolor{{green!30}}{f1:.2f}")
                        elif f1 >= 0.6:
                            row.append(f"\\cellcolor{{yellow!30}}{f1:.2f}")
                        elif f1 >= 0.4:
                            row.append(f"\\cellcolor{{orange!30}}{f1:.2f}")
                        else:
                            row.append(f"\\cellcolor{{red!30}}{f1:.2f}")
                    else:
                        row.append("--")
                else:
                    row.append("--")
        
        print(" & ".join(row) + " \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")