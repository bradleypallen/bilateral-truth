#!/usr/bin/env python3
"""
Compare accuracy, F1 macro, and coverage across three epistemic policies:
1. Conservative (Classical): True if <t,f>, False if <f,t>, else undefined
2. Paraconsistent: True if u=t (tolerates contradictions), False if v=t, else undefined  
3. Paracomplete: True if v=f (tolerates gaps), False if u=f, else undefined
"""

import json
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score

def apply_epistemic_policy(bilateral_value, policy):
    """
    Apply an epistemic policy to a bilateral truth value.
    Returns: 'true', 'false', or 'undefined'
    """
    if policy == 'conservative':
        # Classical: True if <t,f>, False if <f,t>, else undefined
        if bilateral_value == '<t,f>':
            return 'true'
        elif bilateral_value == '<f,t>':
            return 'false'
        else:
            return 'undefined'
    
    elif policy == 'paraconsistent':
        # Tolerates contradictions: True if u=t, False if v=t, else undefined
        # <t,f> -> true, <t,t> -> true (accepts contradiction)
        # <f,t> -> false, <t,t> -> also false (contradiction gives both)
        # Need to handle the contradiction case specially
        if bilateral_value in ['<t,f>', '<t,t>', '<t,e>']:
            return 'true'
        elif bilateral_value in ['<f,t>', '<e,t>']:
            return 'false'
        # Note: <t,t> is both true and false in paraconsistent logic
        # For metrics, we'll need to handle this specially
        else:
            return 'undefined'
    
    elif policy == 'paracomplete':
        # Tolerates gaps: True if v=f, False if u=f, else undefined
        # <t,f> -> true, <f,f> -> undefined (gap)
        # <f,t> -> false, <f,f> -> also undefined (gap)
        if bilateral_value in ['<t,f>', '<e,f>']:
            return 'true'
        elif bilateral_value in ['<f,t>', '<f,e>', '<f,f>']:
            return 'false'
        else:
            return 'undefined'
    
    else:
        raise ValueError(f"Unknown policy: {policy}")

def calculate_metrics_for_policy(results, policy):
    """
    Calculate accuracy, F1 macro, and coverage for a given epistemic policy.
    """
    predictions = []
    ground_truth = []
    defined_count = 0
    total_count = 0
    
    # For paraconsistent logic, handle contradictions specially
    for item in results:
        bilateral_value = item.get('bilateral_value', '<e,e>')
        is_correct = item.get('is_correct', False)
        
        total_count += 1
        pred = apply_epistemic_policy(bilateral_value, policy)
        
        if pred != 'undefined':
            defined_count += 1
            
            # Special handling for paraconsistent contradictions
            if policy == 'paraconsistent' and bilateral_value == '<t,t>':
                # In paraconsistent logic, <t,t> is both true and false
                # For metrics, we'll count it as correct if the answer is either true or false
                # This is a modeling choice - contradictions are "partially correct"
                predictions.append(1 if is_correct else 0)
                ground_truth.append(1)  # Count as correct for F1 purposes
            else:
                # Normal case
                pred_bool = 1 if pred == 'true' else 0
                predictions.append(pred_bool)
                ground_truth.append(1 if is_correct else 0)
    
    # Calculate metrics
    if defined_count > 0:
        coverage = defined_count / total_count
        
        # For accuracy and F1, only use defined predictions
        if len(predictions) > 0:
            accuracy = accuracy_score(ground_truth, predictions)
            # F1 macro for binary classification
            f1_macro = f1_score(ground_truth, predictions, average='macro', zero_division=0)
        else:
            accuracy = 0.0
            f1_macro = 0.0
    else:
        coverage = 0.0
        accuracy = 0.0
        f1_macro = 0.0
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'coverage': coverage,
        'defined_count': defined_count,
        'total_count': total_count
    }

def load_and_analyze_results(results_dir: str = "results"):
    """Load all results and calculate metrics for each epistemic policy."""
    results_path = Path(results_dir)
    
    # Store metrics by model, benchmark, and policy
    all_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    # Store bilateral value distributions for analysis
    bilateral_distributions = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    for file_path in results_path.glob("*.json"):
        if "summary" in file_path.name or "family_resemblance" in file_path.name:
            continue
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Parse filename to get model and benchmark
            parts = file_path.stem.split('_')
            
            # Find benchmark name
            benchmark = None
            if parts[0] == 'truthfulqa' and parts[1] == 'complete':
                benchmark = 'TruthfulQA'
                model_parts = parts[2:-2]
            elif parts[0] == 'simpleqa' and parts[1] == 'complete':
                benchmark = 'SimpleQA'
                model_parts = parts[2:-2]
            elif parts[0] == 'mmlupro' and parts[1] == 'complete':
                benchmark = 'MMLU-Pro'
                model_parts = parts[2:-2]
            elif parts[0] == 'factscore' and parts[1] == 'complete':
                benchmark = 'FACTScore'
                model_parts = parts[2:-2]
            else:
                continue
            
            model = '-'.join(model_parts)
            
            # Get detailed results
            detailed_results = data.get('detailed_results', [])
            
            if not detailed_results:
                continue
            
            # Count bilateral value distribution
            for item in detailed_results:
                bilateral_value = item.get('bilateral_value', '<e,e>')
                bilateral_distributions[model][benchmark][bilateral_value] += 1
            
            # Calculate metrics for each policy
            for policy in ['conservative', 'paraconsistent', 'paracomplete']:
                metrics = calculate_metrics_for_policy(detailed_results, policy)
                all_metrics[model][benchmark][policy] = metrics
                
        except Exception as e:
            print(f"Warning: Could not load {file_path.name}: {e}")
    
    return all_metrics, bilateral_distributions

def create_comparison_visualizations(all_metrics, bilateral_distributions):
    """Create comprehensive visualizations comparing epistemic policies."""
    
    # Prepare data for visualization
    models = sorted(all_metrics.keys())
    benchmarks = sorted(set(b for m_data in all_metrics.values() for b in m_data.keys()))
    policies = ['conservative', 'paraconsistent', 'paracomplete']
    
    # Create display names for models
    model_display_names = []
    for model in models:
        if 'gpt-4.1-2025' in model:
            model_display_names.append('GPT-4.1')
        elif 'gpt-4.1-mini' in model:
            model_display_names.append('GPT-4.1-mini')
        elif 'claude-3-5-haiku' in model:
            model_display_names.append('Claude-Haiku')
        elif 'claude-opus-4-1' in model:
            model_display_names.append('Claude-Opus')
        elif 'llama-4-maverick' in model:
            model_display_names.append('Llama-Maverick')
        elif 'llama-4-scout' in model:
            model_display_names.append('Llama-Scout')
        elif 'gemini-2.5-flash' in model:
            model_display_names.append('Gemini-Flash')
        else:
            model_display_names.append(model[:15])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(24, 16))
    
    # 1. Accuracy comparison across policies
    ax1 = plt.subplot(3, 4, 1)
    for i, policy in enumerate(policies):
        accuracies = []
        for model in models:
            model_accs = []
            for benchmark in benchmarks:
                if benchmark in all_metrics[model] and policy in all_metrics[model][benchmark]:
                    model_accs.append(all_metrics[model][benchmark][policy]['accuracy'])
            if model_accs:
                accuracies.append(np.mean(model_accs))
            else:
                accuracies.append(0)
        
        x = np.arange(len(models))
        width = 0.25
        ax1.bar(x + i * width, accuracies, width, label=policy.capitalize(), alpha=0.8)
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Average Accuracy')
    ax1.set_title('Accuracy by Epistemic Policy')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(model_display_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. F1 Macro comparison across policies
    ax2 = plt.subplot(3, 4, 2)
    for i, policy in enumerate(policies):
        f1_scores = []
        for model in models:
            model_f1s = []
            for benchmark in benchmarks:
                if benchmark in all_metrics[model] and policy in all_metrics[model][benchmark]:
                    model_f1s.append(all_metrics[model][benchmark][policy]['f1_macro'])
            if model_f1s:
                f1_scores.append(np.mean(model_f1s))
            else:
                f1_scores.append(0)
        
        x = np.arange(len(models))
        ax2.bar(x + i * width, f1_scores, width, label=policy.capitalize(), alpha=0.8)
    
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Average F1 Macro')
    ax2.set_title('F1 Macro by Epistemic Policy')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(model_display_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Coverage comparison across policies
    ax3 = plt.subplot(3, 4, 3)
    for i, policy in enumerate(policies):
        coverages = []
        for model in models:
            model_covs = []
            for benchmark in benchmarks:
                if benchmark in all_metrics[model] and policy in all_metrics[model][benchmark]:
                    model_covs.append(all_metrics[model][benchmark][policy]['coverage'])
            if model_covs:
                coverages.append(np.mean(model_covs))
            else:
                coverages.append(0)
        
        x = np.arange(len(models))
        ax3.bar(x + i * width, coverages, width, label=policy.capitalize(), alpha=0.8)
    
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Average Coverage')
    ax3.set_title('Coverage by Epistemic Policy')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(model_display_names, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Policy performance trade-off (Coverage vs Accuracy)
    ax4 = plt.subplot(3, 4, 4)
    colors = ['blue', 'red', 'green']
    for policy_idx, policy in enumerate(policies):
        for model_idx, model in enumerate(models):
            coverages = []
            accuracies = []
            for benchmark in benchmarks:
                if benchmark in all_metrics[model] and policy in all_metrics[model][benchmark]:
                    coverages.append(all_metrics[model][benchmark][policy]['coverage'])
                    accuracies.append(all_metrics[model][benchmark][policy]['accuracy'])
            
            if coverages and accuracies:
                avg_cov = np.mean(coverages)
                avg_acc = np.mean(accuracies)
                ax4.scatter(avg_cov, avg_acc, c=colors[policy_idx], 
                          alpha=0.6, s=100, label=policy if model_idx == 0 else "")
                ax4.annotate(model_display_names[model_idx][:5], 
                           (avg_cov, avg_acc), fontsize=6)
    
    ax4.set_xlabel('Coverage')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Coverage-Accuracy Trade-off')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # 5-8. Heatmaps for each benchmark
    for bench_idx, benchmark in enumerate(benchmarks[:4]):  # Show first 4 benchmarks
        ax = plt.subplot(3, 4, 5 + bench_idx)
        
        # Create matrix for this benchmark
        accuracy_matrix = []
        for model in models:
            row = []
            for policy in policies:
                if benchmark in all_metrics[model] and policy in all_metrics[model][benchmark]:
                    row.append(all_metrics[model][benchmark][policy]['accuracy'])
                else:
                    row.append(np.nan)
            accuracy_matrix.append(row)
        
        sns.heatmap(accuracy_matrix, xticklabels=['Cons.', 'Para-con.', 'Para-com.'],
                   yticklabels=model_display_names, annot=True, fmt='.2f',
                   cmap='RdYlGn', ax=ax, vmin=0, vmax=1,
                   cbar_kws={'label': 'Accuracy'})
        ax.set_title(f'{benchmark} Accuracy')
    
    # 9-12. F1 heatmaps for each benchmark
    for bench_idx, benchmark in enumerate(benchmarks[:4]):
        ax = plt.subplot(3, 4, 9 + bench_idx)
        
        # Create F1 matrix for this benchmark
        f1_matrix = []
        for model in models:
            row = []
            for policy in policies:
                if benchmark in all_metrics[model] and policy in all_metrics[model][benchmark]:
                    row.append(all_metrics[model][benchmark][policy]['f1_macro'])
                else:
                    row.append(np.nan)
            f1_matrix.append(row)
        
        sns.heatmap(f1_matrix, xticklabels=['Cons.', 'Para-con.', 'Para-com.'],
                   yticklabels=model_display_names, annot=True, fmt='.2f',
                   cmap='RdYlGn', ax=ax, vmin=0, vmax=1,
                   cbar_kws={'label': 'F1 Macro'})
        ax.set_title(f'{benchmark} F1 Macro')
    
    plt.tight_layout()
    plt.savefig('epistemic_policy_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def print_detailed_analysis(all_metrics, bilateral_distributions):
    """Print detailed analysis of epistemic policies."""
    
    print("\n" + "="*120)
    print("EPISTEMIC POLICY COMPARISON ANALYSIS")
    print("="*120)
    
    print("\nPolicy Definitions:")
    print("-"*80)
    print("Conservative (Classical): True if <t,f>, False if <f,t>, else undefined")
    print("Paraconsistent: True if u=t (accepts <t,t>), False if v=t, else undefined")
    print("Paracomplete: True if v=f, False if u=f (treats <f,f> as false), else undefined")
    
    # Collect all models and benchmarks
    models = sorted(all_metrics.keys())
    benchmarks = sorted(set(b for m_data in all_metrics.values() for b in m_data.keys()))
    policies = ['conservative', 'paraconsistent', 'paracomplete']
    
    # Overall comparison
    print("\n" + "="*120)
    print("OVERALL METRICS BY POLICY (Averaged Across All Models and Benchmarks)")
    print("="*120)
    print(f"{'Policy':<15} {'Accuracy':<12} {'F1 Macro':<12} {'Coverage':<12}")
    print("-"*80)
    
    for policy in policies:
        all_acc = []
        all_f1 = []
        all_cov = []
        
        for model in models:
            for benchmark in benchmarks:
                if benchmark in all_metrics[model] and policy in all_metrics[model][benchmark]:
                    metrics = all_metrics[model][benchmark][policy]
                    all_acc.append(metrics['accuracy'])
                    all_f1.append(metrics['f1_macro'])
                    all_cov.append(metrics['coverage'])
        
        if all_acc:
            print(f"{policy.capitalize():<15} {np.mean(all_acc):<12.3f} "
                  f"{np.mean(all_f1):<12.3f} {np.mean(all_cov):<12.3f}")
    
    # Model-specific comparison
    print("\n" + "="*120)
    print("METRICS BY MODEL (Averaged Across Benchmarks)")
    print("="*120)
    
    for model in models:
        print(f"\n{model}:")
        print(f"{'Policy':<15} {'Accuracy':<12} {'F1 Macro':<12} {'Coverage':<12}")
        print("-"*60)
        
        for policy in policies:
            accs = []
            f1s = []
            covs = []
            
            for benchmark in benchmarks:
                if benchmark in all_metrics[model] and policy in all_metrics[model][benchmark]:
                    metrics = all_metrics[model][benchmark][policy]
                    accs.append(metrics['accuracy'])
                    f1s.append(metrics['f1_macro'])
                    covs.append(metrics['coverage'])
            
            if accs:
                print(f"{policy.capitalize():<15} {np.mean(accs):<12.3f} "
                      f"{np.mean(f1s):<12.3f} {np.mean(covs):<12.3f}")
    
    # Benchmark-specific comparison
    print("\n" + "="*120)
    print("METRICS BY BENCHMARK (Averaged Across Models)")
    print("="*120)
    
    for benchmark in benchmarks:
        print(f"\n{benchmark}:")
        print(f"{'Policy':<15} {'Accuracy':<12} {'F1 Macro':<12} {'Coverage':<12}")
        print("-"*60)
        
        for policy in policies:
            accs = []
            f1s = []
            covs = []
            
            for model in models:
                if benchmark in all_metrics[model] and policy in all_metrics[model][benchmark]:
                    metrics = all_metrics[model][benchmark][policy]
                    accs.append(metrics['accuracy'])
                    f1s.append(metrics['f1_macro'])
                    covs.append(metrics['coverage'])
            
            if accs:
                print(f"{policy.capitalize():<15} {np.mean(accs):<12.3f} "
                      f"{np.mean(f1s):<12.3f} {np.mean(covs):<12.3f}")
    
    # Best policy per model-benchmark combination
    print("\n" + "="*120)
    print("BEST POLICY BY MODEL AND BENCHMARK (Based on F1 Macro)")
    print("="*120)
    print(f"{'Model':<35} {'Benchmark':<15} {'Best Policy':<15} {'F1 Macro':<12} {'Improvement':<12}")
    print("-"*120)
    
    for model in models:
        for benchmark in benchmarks:
            best_policy = None
            best_f1 = 0
            conservative_f1 = 0
            
            for policy in policies:
                if benchmark in all_metrics[model] and policy in all_metrics[model][benchmark]:
                    f1 = all_metrics[model][benchmark][policy]['f1_macro']
                    if policy == 'conservative':
                        conservative_f1 = f1
                    if f1 > best_f1:
                        best_f1 = f1
                        best_policy = policy
            
            if best_policy and conservative_f1 > 0:
                improvement = ((best_f1 - conservative_f1) / conservative_f1) * 100
                model_display = model[:35]
                print(f"{model_display:<35} {benchmark:<15} {best_policy.capitalize():<15} "
                      f"{best_f1:<12.3f} {improvement:+11.1f}%")
    
    # Impact of contradictions and gaps
    print("\n" + "="*120)
    print("IMPACT OF CONTRADICTIONS (<t,t>) AND GAPS (<f,f>)")
    print("="*120)
    
    for model in models:
        tt_total = 0
        ff_total = 0
        total = 0
        
        for benchmark in bilateral_distributions[model]:
            for bv, count in bilateral_distributions[model][benchmark].items():
                if bv == '<t,t>':
                    tt_total += count
                elif bv == '<f,f>':
                    ff_total += count
                total += count
        
        if total > 0:
            tt_pct = (tt_total / total) * 100
            ff_pct = (ff_total / total) * 100
            
            print(f"\n{model}:")
            print(f"  <t,t> rate: {tt_pct:.1f}% - Benefits from paraconsistent policy")
            print(f"  <f,f> rate: {ff_pct:.1f}% - Affected by paracomplete policy")

def main():
    """Main analysis function."""
    print("Loading evaluation results...")
    all_metrics, bilateral_distributions = load_and_analyze_results()
    
    if not all_metrics:
        print("No results found!")
        return
    
    print(f"Loaded results for {len(all_metrics)} models")
    
    # Print detailed analysis
    print_detailed_analysis(all_metrics, bilateral_distributions)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_comparison_visualizations(all_metrics, bilateral_distributions)
    
    print("\nAnalysis complete! Visualizations saved to 'epistemic_policy_comparison.png'")

if __name__ == "__main__":
    main()