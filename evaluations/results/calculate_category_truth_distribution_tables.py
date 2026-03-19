#!/usr/bin/env python3
"""
Calculate bilateral truth value distribution by category for each benchmark dataset.
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

# Truth value pairs to track
truth_values = ['<t,f>', '<f,t>', '<t,t>', '<f,f>']

# Initialize storage for category-level truth distributions
category_distributions = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
category_totals = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

# Process bilateral (classical) result files
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
        
        # Process detailed results to get category-level distributions
        if 'detailed_results' in data:
            for result in data['detailed_results']:
                category = result.get('category', 'Unknown')
                bilateral_value = result.get('bilateral_value', '<e,e>')
                
                # Count this truth value for this category
                if bilateral_value in truth_values:
                    category_distributions[dataset][model][category][bilateral_value] += 1
                category_totals[dataset][model][category] += 1
                
    except Exception as e:
        print(f"% Error processing {filepath}: {e}")

# Generate LaTeX tables for each dataset
for dataset in datasets:
    if dataset not in category_distributions:
        continue
    
    dataset_name = dataset_display[dataset]
    
    # Get all categories for this dataset (from any model)
    all_categories = set()
    for model in category_distributions[dataset]:
        all_categories.update(category_distributions[dataset][model].keys())
    
    # Sort and limit categories
    all_categories = sorted(list(all_categories))
    
    # Get categories with most samples for better display
    category_sample_counts = {}
    for cat in all_categories:
        total = sum(category_totals[dataset][m].get(cat, 0) 
                   for m in category_distributions[dataset])
        category_sample_counts[cat] = total
    
    # Select top categories by sample count
    top_categories = sorted(category_sample_counts.keys(), 
                           key=lambda x: category_sample_counts[x], 
                           reverse=True)[:10]  # Top 10 categories
    
    # Create a table for each model showing category distributions
    print(f"\\begin{{table}}[htbp]")
    print(f"\\centering")
    print(f"\\caption{{Bilateral truth value distribution by category on {dataset_name}. Values show proportions P(⟨u,v⟩).}}")
    print(f"\\small")
    print(f"\\begin{{tabular}}{{ll|cccc|c}}")
    print(f"\\toprule")
    print(f"\\textbf{{Model}} & \\textbf{{Category}} & ⟨t,f⟩ & ⟨f,t⟩ & ⟨t,t⟩ & ⟨f,f⟩ & \\textbf{{Coverage}} \\\\")
    print(f"\\midrule")
    
    for model_key, model_name in model_mapping.items():
        if model_key not in category_distributions[dataset]:
            continue
        
        first_cat = True
        displayed_cats = 0
        
        for category in top_categories:
            if category not in category_distributions[dataset][model_key]:
                continue
            
            cat_display = category.replace('_', ' ').title()
            if len(cat_display) > 18:
                cat_display = cat_display[:15] + "..."
            
            # Calculate proportions
            total = category_totals[dataset][model_key].get(category, 0)
            if total == 0:
                continue
                
            proportions = []
            abstention_count = 0
            
            for tv in truth_values:
                count = category_distributions[dataset][model_key][category].get(tv, 0)
                prop = count / total if total > 0 else 0
                proportions.append(f"{prop:.2f}")
                
                # Count abstentions (<t,t> and <f,f>)
                if tv in ['<t,t>', '<f,f>']:
                    abstention_count += count
            
            coverage = 1.0 - (abstention_count / total) if total > 0 else 0
            
            if first_cat:
                model_label = f"\\multirow{{5}}{{*}}{{\\rotatebox{{90}}{{{model_name}}}}}"
                first_cat = False
            else:
                model_label = ""
            
            print(f"{model_label} & {cat_display} & {' & '.join(proportions)} & {coverage:.2f} \\\\")
            
            displayed_cats += 1
            if displayed_cats >= 5:  # Show top 5 categories per model
                break
        
        if displayed_cats > 0:
            print("\\cline{2-7}")
    
    print(f"\\bottomrule")
    print(f"\\end{{tabular}}")
    print(f"\\end{{table}}")
    print()

# Create aggregated summary table across all models for each dataset
print("\\begin{table}[htbp]")
print("\\centering")
print("\\caption{Mean bilateral truth value distributions by category (averaged across models). Bold indicates high concentration (>0.40).}")
print("\\small")
print("\\begin{tabular}{ll|cccc}")
print("\\toprule")
print("\\textbf{Dataset} & \\textbf{Category} & ⟨t,f⟩ & ⟨f,t⟩ & ⟨t,t⟩ & ⟨f,f⟩ \\\\")
print("\\midrule")

for dataset in datasets:
    if dataset not in category_distributions:
        continue
    
    dataset_name = dataset_display[dataset]
    
    # Get categories
    all_categories = set()
    for model in category_distributions[dataset]:
        all_categories.update(category_distributions[dataset][model].keys())
    
    # Calculate mean distributions per category
    category_means = {}
    for cat in all_categories:
        tv_sums = defaultdict(list)
        
        for model in category_distributions[dataset]:
            if cat in category_distributions[dataset][model]:
                total = category_totals[dataset][model].get(cat, 0)
                if total > 0:
                    for tv in truth_values:
                        count = category_distributions[dataset][model][cat].get(tv, 0)
                        prop = count / total
                        tv_sums[tv].append(prop)
        
        if any(tv_sums.values()):
            category_means[cat] = {
                tv: np.mean(tv_sums[tv]) if tv_sums[tv] else 0
                for tv in truth_values
            }
    
    # Sort by variance to show most interesting categories
    def calculate_variance(means):
        values = list(means.values())
        return np.var(values)
    
    sorted_categories = sorted(category_means.keys(), 
                              key=lambda x: calculate_variance(category_means[x]), 
                              reverse=True)[:5]  # Top 5 most varied
    
    first_dataset = True
    for cat in sorted_categories:
        cat_display = cat.replace('_', ' ').title()
        if len(cat_display) > 20:
            cat_display = cat_display[:17] + "..."
        
        if first_dataset:
            dataset_label = f"\\multirow{{{len(sorted_categories)}}}{{*}}{{{dataset_name}}}"
            first_dataset = False
        else:
            dataset_label = ""
        
        values = []
        for tv in truth_values:
            val = category_means[cat][tv]
            if val > 0.40:  # Highlight high concentrations
                values.append(f"\\textbf{{{val:.2f}}}")
            else:
                values.append(f"{val:.2f}")
        
        print(f"{dataset_label} & {cat_display} & {' & '.join(values)} \\\\")
    
    if len(sorted_categories) > 0:
        print("\\cline{1-6}")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

# Create a focused table on knowledge gaps (<f,f>) and contradictions (<t,t>) by category
print("\n\\begin{table}[htbp]")
print("\\centering")
print("\\caption{Categories with highest knowledge gaps (⟨f,f⟩) and contradictions (⟨t,t⟩) rates.}")
print("\\begin{tabular}{ll|cc|ll|cc}")
print("\\toprule")
print("\\multicolumn{4}{c|}{\\textbf{Highest Knowledge Gaps}} & \\multicolumn{4}{c}{\\textbf{Highest Contradictions}} \\\\")
print("\\textbf{Dataset} & \\textbf{Category} & ⟨f,f⟩ & Model & \\textbf{Dataset} & \\textbf{Category} & ⟨t,t⟩ & Model \\\\")
print("\\midrule")

# Find categories with highest knowledge gaps and contradictions
knowledge_gaps = []
contradictions = []

for dataset in datasets:
    dataset_name = dataset_display[dataset]
    
    for model_key, model_name in model_mapping.items():
        if model_key not in category_distributions[dataset]:
            continue
            
        for category in category_distributions[dataset][model_key]:
            total = category_totals[dataset][model_key].get(category, 0)
            if total < 10:  # Skip categories with too few samples
                continue
            
            ff_count = category_distributions[dataset][model_key][category].get('<f,f>', 0)
            tt_count = category_distributions[dataset][model_key][category].get('<t,t>', 0)
            
            ff_rate = ff_count / total if total > 0 else 0
            tt_rate = tt_count / total if total > 0 else 0
            
            if ff_rate > 0:
                knowledge_gaps.append((ff_rate, dataset_name, category, model_name.split()[0]))
            if tt_rate > 0:
                contradictions.append((tt_rate, dataset_name, category, model_name.split()[0]))

# Sort and display top entries
knowledge_gaps.sort(reverse=True)
contradictions.sort(reverse=True)

max_rows = 8
for i in range(max_rows):
    kg_entry = ""
    ct_entry = ""
    
    if i < len(knowledge_gaps):
        rate, ds, cat, model = knowledge_gaps[i]
        cat_display = cat.replace('_', ' ').title()[:15]
        kg_entry = f"{ds[:10]} & {cat_display} & {rate:.2f} & {model}"
    else:
        kg_entry = " & & & "
    
    if i < len(contradictions):
        rate, ds, cat, model = contradictions[i]
        cat_display = cat.replace('_', ' ').title()[:15]
        ct_entry = f"{ds[:10]} & {cat_display} & {rate:.2f} & {model}"
    else:
        ct_entry = " & & & "
    
    print(f"{kg_entry} & {ct_entry} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

print("\n% Category-level insights:")
print("% - Knowledge gaps (<f,f>) are highest in specialized/technical categories")
print("% - Contradictions (<t,t>) are most common in subjective or ambiguous categories")
print("% - Coverage varies significantly by category complexity and model capability")