#!/usr/bin/env python3
"""
Compare epistemic policies against a unilateral baseline that only considers
the verification component (u) of bilateral truth values.

Unilateral policy: 
- True if u=t (verification succeeds)
- False if u=f (verification fails)  
- Undefined if u=e (verification evaluation fails)

This simulates standard factuality evaluation that only asks "Is this true?"
without the refutation dimension.
"""

import json
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.patches as mpatches

def apply_policy(bilateral_value, policy):
    """
    Apply an epistemic policy to a bilateral truth value.
    Returns: 'true', 'false', or 'undefined'
    """
    if policy == 'unilateral':
        # Only consider verification component (u)
        # <t,*> -> true, <f,*> -> false, <e,*> -> undefined
        if bilateral_value[1] == 't':  # u=t
            return 'true'
        elif bilateral_value[1] == 'f':  # u=f
            return 'false'
        elif bilateral_value[1] == 'e':  # u=e
            return 'undefined'
        else:
            return 'undefined'
    
    elif policy == 'conservative':
        # Classical: True if <t,f>, False if <f,t>, else undefined
        if bilateral_value == '<t,f>':
            return 'true'
        elif bilateral_value == '<f,t>':
            return 'false'
        else:
            return 'undefined'
    
    elif policy == 'paraconsistent':
        # Tolerates contradictions: True if u=t, False if v=t, else undefined
        if bilateral_value in ['<t,f>', '<t,t>', '<t,e>']:
            return 'true'
        elif bilateral_value in ['<f,t>', '<e,t>']:
            return 'false'
        else:
            return 'undefined'
    
    elif policy == 'paracomplete':
        # Tolerates gaps: True if v=f, False if u=f, else undefined
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
    Calculate accuracy, F1 macro, and coverage for a given policy.
    """
    predictions = []
    ground_truth = []
    defined_count = 0
    total_count = 0
    
    for item in results:
        bilateral_value = item.get('bilateral_value', '<e,e>')
        is_correct = item.get('is_correct', False)
        
        total_count += 1
        pred = apply_policy(bilateral_value, policy)
        
        if pred != 'undefined':
            defined_count += 1
            
            # Special handling for paraconsistent contradictions
            if policy == 'paraconsistent' and bilateral_value == '<t,t>':
                predictions.append(1 if is_correct else 0)
                ground_truth.append(1)
            else:
                pred_bool = 1 if pred == 'true' else 0
                predictions.append(pred_bool)
                ground_truth.append(1 if is_correct else 0)
    
    # Calculate metrics
    if defined_count > 0:
        coverage = defined_count / total_count
        
        if len(predictions) > 0:
            accuracy = accuracy_score(ground_truth, predictions)
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

def analyze_unilateral_impact(results):
    """
    Analyze what bilateral patterns are affected by unilateral evaluation.
    """
    unilateral_correct = 0
    bilateral_correct = 0
    unilateral_wrong_bilateral_right = []
    bilateral_wrong_unilateral_right = []
    both_wrong_different = []
    
    for item in results:
        bilateral_value = item.get('bilateral_value', '<e,e>')
        is_correct = item.get('is_correct', False)
        
        # Unilateral prediction
        uni_pred = apply_policy(bilateral_value, 'unilateral')
        # Conservative bilateral prediction
        bi_pred = apply_policy(bilateral_value, 'conservative')
        
        if uni_pred != 'undefined' and bi_pred != 'undefined':
            uni_correct_here = (uni_pred == 'true' and is_correct) or (uni_pred == 'false' and not is_correct)
            bi_correct_here = (bi_pred == 'true' and is_correct) or (bi_pred == 'false' and not is_correct)
            
            if uni_correct_here:
                unilateral_correct += 1
            if bi_correct_here:
                bilateral_correct += 1
                
            if not uni_correct_here and bi_correct_here:
                unilateral_wrong_bilateral_right.append(bilateral_value)
            elif uni_correct_here and not bi_correct_here:
                bilateral_wrong_unilateral_right.append(bilateral_value)
            elif not uni_correct_here and not bi_correct_here and uni_pred != bi_pred:
                both_wrong_different.append(bilateral_value)
    
    return {
        'unilateral_correct': unilateral_correct,
        'bilateral_correct': bilateral_correct,
        'uni_wrong_bi_right': unilateral_wrong_bilateral_right,
        'bi_wrong_uni_right': bilateral_wrong_unilateral_right,
        'both_wrong_different': both_wrong_different
    }

def load_and_analyze_results(results_dir: str = "results"):
    """Load all results and calculate metrics for each policy including unilateral baseline."""
    results_path = Path(results_dir)
    
    # Store metrics by model, benchmark, and policy
    all_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    # Store pattern analysis
    pattern_analysis = defaultdict(lambda: defaultdict(dict))
    
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
            
            # Calculate metrics for each policy
            for policy in ['unilateral', 'conservative', 'paraconsistent', 'paracomplete']:
                metrics = calculate_metrics_for_policy(detailed_results, policy)
                all_metrics[model][benchmark][policy] = metrics
            
            # Analyze patterns
            pattern_analysis[model][benchmark] = analyze_unilateral_impact(detailed_results)
                
        except Exception as e:
            print(f"Warning: Could not load {file_path.name}: {e}")
    
    return all_metrics, pattern_analysis

def create_comparison_visualizations(all_metrics, pattern_analysis):
    """Create visualizations comparing unilateral baseline with epistemic policies."""
    
    # Prepare data
    models = sorted(all_metrics.keys())
    benchmarks = sorted(set(b for m_data in all_metrics.values() for b in m_data.keys()))
    policies = ['unilateral', 'conservative', 'paraconsistent', 'paracomplete']
    
    # Create display names
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
    
    # Create figure
    fig = plt.figure(figsize=(24, 14))
    
    # Color scheme
    colors = {'unilateral': '#FF6B6B', 'conservative': '#4ECDC4', 
              'paraconsistent': '#95E77E', 'paracomplete': '#FFE66D'}
    
    # 1. F1 Macro comparison with unilateral baseline
    ax1 = plt.subplot(3, 4, 1)
    x = np.arange(len(models))
    width = 0.2
    
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
        
        ax1.bar(x + i * width, f1_scores, width, label=policy.capitalize(), 
                color=colors[policy], alpha=0.8)
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Average F1 Macro')
    ax1.set_title('F1 Macro: Unilateral Baseline vs Epistemic Policies')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(model_display_names, rotation=45, ha='right')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Coverage comparison
    ax2 = plt.subplot(3, 4, 2)
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
        
        ax2.bar(x + i * width, coverages, width, label=policy.capitalize(),
                color=colors[policy], alpha=0.8)
    
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Average Coverage')
    ax2.set_title('Coverage: Unilateral Baseline vs Epistemic Policies')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(model_display_names, rotation=45, ha='right')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Improvement over unilateral baseline
    ax3 = plt.subplot(3, 4, 3)
    improvements = defaultdict(list)
    
    for model in models:
        for policy in ['conservative', 'paraconsistent', 'paracomplete']:
            uni_f1s = []
            policy_f1s = []
            for benchmark in benchmarks:
                if benchmark in all_metrics[model]:
                    if 'unilateral' in all_metrics[model][benchmark]:
                        uni_f1s.append(all_metrics[model][benchmark]['unilateral']['f1_macro'])
                    if policy in all_metrics[model][benchmark]:
                        policy_f1s.append(all_metrics[model][benchmark][policy]['f1_macro'])
            
            if uni_f1s and policy_f1s:
                avg_uni = np.mean(uni_f1s)
                avg_policy = np.mean(policy_f1s)
                if avg_uni > 0:
                    improvement = ((avg_policy - avg_uni) / avg_uni) * 100
                else:
                    improvement = 0
                improvements[policy].append(improvement)
            else:
                improvements[policy].append(0)
    
    x = np.arange(len(models))
    width = 0.25
    for i, policy in enumerate(['conservative', 'paraconsistent', 'paracomplete']):
        ax3.bar(x + i * width, improvements[policy], width, 
                label=policy.capitalize(), alpha=0.8)
    
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.set_xlabel('Model')
    ax3.set_ylabel('F1 Improvement over Unilateral (%)')
    ax3.set_title('Bilateral Policies vs Unilateral Baseline')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(model_display_names, rotation=45, ha='right')
    ax3.legend(loc='lower left', fontsize=8)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Coverage-Accuracy trade-off with unilateral highlighted
    ax4 = plt.subplot(3, 4, 4)
    
    for policy in policies:
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
                
                # Highlight unilateral with different marker
                if policy == 'unilateral':
                    ax4.scatter(avg_cov, avg_acc, c=colors[policy], 
                              alpha=0.9, s=150, marker='s', edgecolors='black', linewidth=2,
                              label='Unilateral' if model_idx == 0 else "")
                else:
                    ax4.scatter(avg_cov, avg_acc, c=colors[policy], 
                              alpha=0.7, s=100, marker='o',
                              label=policy.capitalize() if model_idx == 0 else "")
                
                if model_idx % 2 == 0:  # Reduce label clutter
                    ax4.annotate(model_display_names[model_idx][:5], 
                               (avg_cov, avg_acc), fontsize=6)
    
    ax4.set_xlabel('Coverage')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Coverage-Accuracy Trade-off (Squares = Unilateral)')
    ax4.legend(loc='lower right', fontsize=8)
    ax4.grid(alpha=0.3)
    
    # 5-8. Benchmark-specific F1 comparisons
    for bench_idx, benchmark in enumerate(benchmarks[:4]):
        ax = plt.subplot(3, 4, 5 + bench_idx)
        
        f1_matrix = []
        for model in models:
            row = []
            for policy in policies:
                if benchmark in all_metrics[model] and policy in all_metrics[model][benchmark]:
                    row.append(all_metrics[model][benchmark][policy]['f1_macro'])
                else:
                    row.append(np.nan)
            f1_matrix.append(row)
        
        # Highlight unilateral column
        mask = np.zeros_like(f1_matrix)
        mask[:, 0] = True  # Unilateral is first column
        
        sns.heatmap(f1_matrix, xticklabels=['Uni.', 'Cons.', 'Para-con.', 'Para-com.'],
                   yticklabels=model_display_names, annot=True, fmt='.2f',
                   cmap='RdYlGn', ax=ax, vmin=0, vmax=1,
                   cbar_kws={'label': 'F1'}, mask=mask, linewidths=2, linecolor='red')
        
        # Re-draw unilateral column with different colormap
        uni_data = np.array(f1_matrix)[:, :1]
        mask_uni = np.ones_like(f1_matrix)
        mask_uni[:, 0] = False
        sns.heatmap(f1_matrix, xticklabels=['Uni.', 'Cons.', 'Para-con.', 'Para-com.'],
                   yticklabels=model_display_names, annot=True, fmt='.2f',
                   cmap='Reds', ax=ax, vmin=0, vmax=1, cbar=False,
                   mask=mask_uni, linewidths=2, linecolor='black')
        
        ax.set_title(f'{benchmark} F1 Scores')
    
    # 9-12. Benchmark-specific coverage comparisons
    for bench_idx, benchmark in enumerate(benchmarks[:4]):
        ax = plt.subplot(3, 4, 9 + bench_idx)
        
        cov_matrix = []
        for model in models:
            row = []
            for policy in policies:
                if benchmark in all_metrics[model] and policy in all_metrics[model][benchmark]:
                    row.append(all_metrics[model][benchmark][policy]['coverage'])
                else:
                    row.append(np.nan)
            cov_matrix.append(row)
        
        sns.heatmap(cov_matrix, xticklabels=['Uni.', 'Cons.', 'Para-con.', 'Para-com.'],
                   yticklabels=model_display_names, annot=True, fmt='.2f',
                   cmap='Blues', ax=ax, vmin=0, vmax=1,
                   cbar_kws={'label': 'Coverage'})
        ax.set_title(f'{benchmark} Coverage')
    
    plt.tight_layout()
    plt.savefig('unilateral_baseline_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def print_detailed_analysis(all_metrics, pattern_analysis):
    """Print detailed analysis comparing unilateral baseline with epistemic policies."""
    
    print("\n" + "="*120)
    print("UNILATERAL BASELINE VS BILATERAL EPISTEMIC POLICIES")
    print("="*120)
    
    print("\nUnilateral Policy (Standard Factuality Evaluation):")
    print("-"*80)
    print("Maps verification component only: u=t → True, u=f → False, u=e → Undefined")
    print("Ignores refutation dimension entirely")
    print("Simulates traditional 'Is this true?' evaluation")
    
    # Collect data
    models = sorted(all_metrics.keys())
    benchmarks = sorted(set(b for m_data in all_metrics.values() for b in m_data.keys()))
    policies = ['unilateral', 'conservative', 'paraconsistent', 'paracomplete']
    
    # Overall comparison
    print("\n" + "="*120)
    print("OVERALL METRICS (Averaged Across All Models and Benchmarks)")
    print("="*120)
    print(f"{'Policy':<15} {'Accuracy':<12} {'F1 Macro':<12} {'Coverage':<12} {'vs Unilateral':<15}")
    print("-"*100)
    
    overall_stats = {}
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
            avg_acc = np.mean(all_acc)
            avg_f1 = np.mean(all_f1)
            avg_cov = np.mean(all_cov)
            overall_stats[policy] = {'accuracy': avg_acc, 'f1': avg_f1, 'coverage': avg_cov}
            
            if policy == 'unilateral':
                print(f"{policy.capitalize():<15} {avg_acc:<12.3f} {avg_f1:<12.3f} {avg_cov:<12.3f} {'(baseline)':<15}")
            else:
                uni_f1 = overall_stats.get('unilateral', {}).get('f1', 0)
                if uni_f1 > 0:
                    improvement = ((avg_f1 - uni_f1) / uni_f1) * 100
                    print(f"{policy.capitalize():<15} {avg_acc:<12.3f} {avg_f1:<12.3f} {avg_cov:<12.3f} {improvement:+.1f}%")
                else:
                    print(f"{policy.capitalize():<15} {avg_acc:<12.3f} {avg_f1:<12.3f} {avg_cov:<12.3f} {'N/A':<15}")
    
    # Model-specific comparison
    print("\n" + "="*120)
    print("F1 PERFORMANCE BY MODEL")
    print("="*120)
    print(f"{'Model':<35} {'Unilateral':<12} {'Conservative':<12} {'Best Policy':<15} {'Improvement':<12}")
    print("-"*100)
    
    for model in models:
        uni_f1 = 0
        cons_f1 = 0
        best_f1 = 0
        best_policy = ''
        
        for policy in policies:
            f1s = []
            for benchmark in benchmarks:
                if benchmark in all_metrics[model] and policy in all_metrics[model][benchmark]:
                    f1s.append(all_metrics[model][benchmark][policy]['f1_macro'])
            
            if f1s:
                avg_f1 = np.mean(f1s)
                if policy == 'unilateral':
                    uni_f1 = avg_f1
                elif policy == 'conservative':
                    cons_f1 = avg_f1
                
                if avg_f1 > best_f1:
                    best_f1 = avg_f1
                    best_policy = policy
        
        if uni_f1 > 0:
            improvement = ((best_f1 - uni_f1) / uni_f1) * 100
            model_display = model[:35]
            print(f"{model_display:<35} {uni_f1:<12.3f} {cons_f1:<12.3f} {best_policy.capitalize():<15} {improvement:+11.1f}%")
    
    # Benchmark-specific insights
    print("\n" + "="*120)
    print("BENCHMARK-SPECIFIC INSIGHTS")
    print("="*120)
    
    for benchmark in benchmarks:
        print(f"\n{benchmark}:")
        print(f"{'Policy':<15} {'Accuracy':<12} {'F1 Macro':<12} {'Coverage':<12} {'vs Unilateral':<15}")
        print("-"*80)
        
        bench_stats = {}
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
                avg_acc = np.mean(accs)
                avg_f1 = np.mean(f1s)
                avg_cov = np.mean(covs)
                bench_stats[policy] = avg_f1
                
                if policy == 'unilateral':
                    print(f"{policy.capitalize():<15} {avg_acc:<12.3f} {avg_f1:<12.3f} {avg_cov:<12.3f} {'(baseline)':<15}")
                else:
                    uni_f1 = bench_stats.get('unilateral', 0)
                    if uni_f1 > 0:
                        improvement = ((avg_f1 - uni_f1) / uni_f1) * 100
                        print(f"{policy.capitalize():<15} {avg_acc:<12.3f} {avg_f1:<12.3f} {avg_cov:<12.3f} {improvement:+.1f}%")
    
    # Key bilateral patterns missed by unilateral
    print("\n" + "="*120)
    print("KEY BILATERAL PATTERNS MISSED BY UNILATERAL EVALUATION")
    print("="*120)
    
    print("\nPatterns where unilateral differs from conservative bilateral:")
    print("-"*80)
    print("1. <t,t> (contradictions): Unilateral sees as TRUE, Conservative as UNDEFINED")
    print("2. <f,f> (knowledge gaps): Unilateral sees as FALSE, Conservative as UNDEFINED")
    print("3. <t,e> (refutation failure): Unilateral sees as TRUE, Conservative as UNDEFINED")
    print("4. <f,e> (refutation failure): Unilateral sees as FALSE, Conservative as UNDEFINED")
    
    print("\nImplications:")
    print("-"*80)
    print("• Unilateral cannot distinguish contradictions from truth")
    print("• Unilateral cannot distinguish knowledge gaps from falsehood")
    print("• Unilateral misses epistemic nuance in evaluation failures")
    print("• Refutation dimension adds critical information for accuracy")

def main():
    """Main analysis function."""
    print("Loading evaluation results...")
    all_metrics, pattern_analysis = load_and_analyze_results()
    
    if not all_metrics:
        print("No results found!")
        return
    
    print(f"Loaded results for {len(all_metrics)} models")
    
    # Print detailed analysis
    print_detailed_analysis(all_metrics, pattern_analysis)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_comparison_visualizations(all_metrics, pattern_analysis)
    
    print("\nAnalysis complete! Visualizations saved to 'unilateral_baseline_comparison.png'")

if __name__ == "__main__":
    main()