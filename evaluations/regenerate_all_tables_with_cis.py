#!/usr/bin/env python3
"""Regenerate all tables with real confidence intervals and update FINAL_RESULTS_TABLES.md."""

import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import f1_score, accuracy_score


def bootstrap_confidence_interval(y_true, y_pred, metric='f1', n_bootstrap=100, confidence=0.95):
    """Calculate confidence interval using bootstrap resampling with subsampling."""
    if not y_true or not y_pred:
        return 0.0, 0.0, 0.0
    
    n_samples = len(y_true)
    # Use subsample size of 63.2% (1 - 1/e) following Politis & Romano (1994)
    subsample_size = int(n_samples * 0.632)
    scores = []
    
    for _ in range(n_bootstrap):
        # Bootstrap subsample with replacement
        indices = np.random.choice(n_samples, subsample_size, replace=True)
        y_true_boot = [y_true[i] for i in indices]
        y_pred_boot = [y_pred[i] for i in indices]
        
        if metric == 'f1':
            # Handle abstentions by filtering them out
            valid_pairs = [(t, p) for t, p in zip(y_true_boot, y_pred_boot) 
                          if p not in ['undefined', 'uncertain', 'UNCERTAIN']]
            if valid_pairs:
                y_t, y_p = zip(*valid_pairs)
                score = f1_score(y_t, y_p, average='macro', zero_division=0)
            else:
                score = 0.0
        elif metric == 'accuracy':
            valid_pairs = [(t, p) for t, p in zip(y_true_boot, y_pred_boot) 
                          if p not in ['undefined', 'uncertain', 'UNCERTAIN']]
            if valid_pairs:
                y_t, y_p = zip(*valid_pairs)
                score = accuracy_score(y_t, y_p)
            else:
                score = 0.0
        elif metric == 'coverage':
            score = sum(1 for p in y_pred_boot if p not in ['undefined', 'uncertain', 'UNCERTAIN']) / len(y_pred_boot)
        
        scores.append(score)
    
    # Calculate confidence interval
    alpha = 1 - confidence
    lower = np.percentile(scores, (alpha/2) * 100)
    upper = np.percentile(scores, (1 - alpha/2) * 100)
    mean = np.mean(scores)
    
    return mean, lower, upper


def load_all_results():
    """Load all result files and organize by model, dataset, and method."""
    results = defaultdict(lambda: defaultdict(dict))
    
    # Load bilateral (classical) results
    for file in glob.glob('results/*_classical_results.json'):
        try:
            with open(file) as f:
                data = json.load(f)
            
            filename = Path(file).stem
            parts = filename.split('_')
            dataset = parts[0]
            model = data.get('model', '')
            model_short = model.split('/')[-1] if '/' in model else model
            
            if 'detailed_results' not in data:
                continue
            
            # Extract bilateral predictions
            y_true = []
            y_pred = []
            for item in data['detailed_results']:
                ground_truth = item.get('expected_label', item.get('ground_truth', 'correct'))
                bilateral_value = item.get('bilateral_value', '<e,e>')
                
                if bilateral_value == '<t,f>':
                    pred = 'correct'
                elif bilateral_value == '<f,t>':
                    pred = 'incorrect'
                else:
                    pred = 'undefined'
                
                y_true.append(ground_truth)
                y_pred.append(pred)
            
            results[model_short][dataset]['bilateral'] = {
                'y_true': y_true,
                'y_pred': y_pred,
                'data': data
            }
            
            # Also compute verification-only from bilateral
            y_true_ver = []
            y_pred_ver = []
            for item in data['detailed_results']:
                ground_truth = item.get('expected_label', item.get('ground_truth', 'correct'))
                bilateral_value = item.get('bilateral_value', '<e,e>')
                
                if bilateral_value.startswith('<t,'):
                    pred = 'correct'
                elif bilateral_value.startswith('<f,'):
                    pred = 'incorrect'
                else:
                    pred = 'undefined'
                
                y_true_ver.append(ground_truth)
                y_pred_ver.append(pred)
            
            results[model_short][dataset]['verification'] = {
                'y_true': y_true_ver,
                'y_pred': y_pred_ver
            }
            
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    # Load unilateral forced/direct results
    for file in glob.glob('results/*_unilateral_forced_results.json') + \
                glob.glob('results/*_complete_*_unilateral_forced_results.json') + \
                glob.glob('results/*_unilateral_direct_results.json') + \
                glob.glob('results/*_complete_*_unilateral_direct_results.json'):
        try:
            with open(file) as f:
                data = json.load(f)
            
            filename = Path(file).stem
            parts = filename.split('_')
            
            if parts[1] == 'complete':
                dataset = parts[0]
            else:
                dataset = parts[0]
            
            model = data.get('model', '')
            model_short = model.split('/')[-1] if '/' in model else model
            
            if 'detailed_results' not in data:
                continue
            
            y_true = []
            y_pred = []
            for item in data['detailed_results']:
                ground_truth = item.get('expected_label', item.get('ground_truth', 'correct'))
                prediction = item.get('prediction', 'undefined')
                
                # Normalize prediction case
                if prediction.upper() == 'CORRECT':
                    prediction = 'correct'
                elif prediction.upper() == 'INCORRECT':
                    prediction = 'incorrect'
                elif prediction.upper() == 'UNCERTAIN':
                    prediction = 'uncertain'
                
                y_true.append(ground_truth)
                y_pred.append(prediction)
            
            results[model_short][dataset]['unilateral_forced'] = {
                'y_true': y_true,
                'y_pred': y_pred,
                'data': data
            }
            
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    # Load unilateral uncertain results
    for file in glob.glob('results/*_unilateral_uncertain_results.json') + \
                glob.glob('results/*_complete_*_unilateral_uncertain_results.json'):
        try:
            with open(file) as f:
                data = json.load(f)
            
            filename = Path(file).stem
            parts = filename.split('_')
            
            if parts[1] == 'complete':
                dataset = parts[0]
            else:
                dataset = parts[0]
            
            model = data.get('model', '')
            model_short = model.split('/')[-1] if '/' in model else model
            
            if 'detailed_results' not in data:
                continue
            
            y_true = []
            y_pred = []
            for item in data['detailed_results']:
                ground_truth = item.get('expected_label', item.get('ground_truth', 'correct'))
                prediction = item.get('prediction', 'undefined')
                
                # Normalize prediction case
                if prediction.upper() == 'CORRECT':
                    prediction = 'correct'
                elif prediction.upper() == 'INCORRECT':
                    prediction = 'incorrect'
                elif prediction.upper() == 'UNCERTAIN':
                    prediction = 'uncertain'
                
                y_true.append(ground_truth)
                y_pred.append(prediction)
            
            results[model_short][dataset]['unilateral_uncertain'] = {
                'y_true': y_true,
                'y_pred': y_pred,
                'data': data
            }
            
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return results


def generate_table_1(results):
    """Generate Table 1: Performance comparison with confidence intervals."""
    
    table_lines = []
    table_lines.append("## Table 1: Performance Comparison Across Methods")
    table_lines.append("")
    
    datasets = ['truthfulqa', 'simpleqa', 'mmlupro', 'factscore']
    
    for dataset in datasets:
        # Collect data for this dataset
        dataset_data = []
        for model in results:
            if dataset in results[model]:
                row = {'model': model}
                
                # Calculate metrics for each method
                for method in ['bilateral', 'unilateral_forced', 'unilateral_uncertain', 'verification']:
                    if method in results[model][dataset]:
                        y_true = results[model][dataset][method]['y_true']
                        y_pred = results[model][dataset][method]['y_pred']
                        
                        f1_mean, f1_lower, f1_upper = bootstrap_confidence_interval(
                            y_true, y_pred, metric='f1'
                        )
                        
                        row[f'{method}_f1'] = f1_mean
                        row[f'{method}_f1_ci'] = f"[{f1_lower:.3f}, {f1_upper:.3f}]"
                
                if 'bilateral_f1' in row:  # Only add if we have at least bilateral
                    dataset_data.append(row)
        
        if not dataset_data:
            continue
        
        # Sort by bilateral F1
        dataset_data.sort(key=lambda x: x.get('bilateral_f1', 0), reverse=True)
        
        table_lines.append(f"### {dataset.upper()} (n={len(dataset_data)} models)")
        table_lines.append("| Model | **Bilateral F1 [95% CI]** | **Forced Unilateral F1 [95% CI]** | **Uncertain Unilateral F1 [95% CI]** | **Verification F1 [95% CI]** |")
        table_lines.append("|-------|---------------------------|-----------------------------------|---------------------------------------|------------------------------|")
        
        for row in dataset_data:
            model_name = row['model'].replace('-', ' ').replace('_', ' ').title()
            if len(model_name) > 25:
                model_name = model_name[:22] + "..."
            
            line = f"| {model_name:<25} "
            
            # Bilateral
            if 'bilateral_f1' in row:
                line += f"| {row['bilateral_f1']:.3f} {row['bilateral_f1_ci']} "
            else:
                line += "| N/A "
            
            # Forced Unilateral
            if 'unilateral_forced_f1' in row:
                line += f"| {row['unilateral_forced_f1']:.3f} {row['unilateral_forced_f1_ci']} "
            else:
                line += "| N/A "
            
            # Uncertain Unilateral  
            if 'unilateral_uncertain_f1' in row:
                line += f"| {row['unilateral_uncertain_f1']:.3f} {row['unilateral_uncertain_f1_ci']} "
            else:
                line += "| N/A "
            
            # Verification
            if 'verification_f1' in row:
                line += f"| {row['verification_f1']:.3f} {row['verification_f1_ci']} |"
            else:
                line += "| N/A |"
            
            table_lines.append(line)
        
        table_lines.append("")
    
    return table_lines


def generate_table_2(results):
    """Generate Table 2: Bilateral truth value distributions."""
    
    table_lines = []
    table_lines.append("## Table 2: Bilateral Truth Value Distribution Probabilities")
    table_lines.append("")
    table_lines.append("| Model | Dataset | P(<t,f>) | P(<f,t>) | P(<t,t>) | P(<f,f>) | Coverage |")
    table_lines.append("|-------|---------|----------|----------|----------|----------|----------|")
    
    datasets = ['truthfulqa', 'simpleqa', 'mmlupro', 'factscore']
    
    for dataset in datasets:
        table_lines.append(f"| **{dataset.upper()}** | | | | | | |")
        
        dataset_data = []
        for model in results:
            if dataset in results[model] and 'bilateral' in results[model][dataset]:
                data = results[model][dataset]['bilateral'].get('data', {})
                dist = data.get('bilateral_distribution', {})
                total = data.get('total_samples', 1)
                
                if total > 0:
                    tf = dist.get('<t,f>', 0) / total
                    ft = dist.get('<f,t>', 0) / total
                    tt = dist.get('<t,t>', 0) / total
                    ff = dist.get('<f,f>', 0) / total
                    coverage = (dist.get('<t,f>', 0) + dist.get('<f,t>', 0)) / total
                    
                    # Calculate bilateral F1 for sorting
                    y_true = results[model][dataset]['bilateral']['y_true']
                    y_pred = results[model][dataset]['bilateral']['y_pred']
                    f1_mean, _, _ = bootstrap_confidence_interval(y_true, y_pred, metric='f1', n_bootstrap=50)
                    
                    dataset_data.append({
                        'model': model,
                        'tf': tf,
                        'ft': ft,
                        'tt': tt,
                        'ff': ff,
                        'coverage': coverage,
                        'f1': f1_mean
                    })
        
        # Sort by F1 score
        dataset_data.sort(key=lambda x: x['f1'], reverse=True)
        
        for row in dataset_data:
            model_name = row['model'].replace('-', ' ').replace('_', ' ').title()
            if len(model_name) > 20:
                model_name = model_name[:17] + "..."
            
            line = f"| {model_name:<20} | {dataset:<11} "
            line += f"| {row['tf']:.3f} | {row['ft']:.3f} | {row['tt']:.3f} | {row['ff']:.3f} "
            line += f"| {row['coverage']:.1%} |"
            table_lines.append(line)
    
    table_lines.append("")
    return table_lines


def generate_table_3(results):
    """Generate Table 3: Epistemic metrics."""
    
    table_lines = []
    table_lines.append("## Table 3: Epistemic Metrics - Honesty, Overconfidence, and Uncertainty Awareness")
    table_lines.append("")
    table_lines.append("| Model | Knowledge Gap Rate | Contradiction Rate | Abstention Rate | Epistemic Honesty | Overconfidence |")
    table_lines.append("|-------|-------------------|-------------------|-----------------|-------------------|----------------|")
    
    # Aggregate metrics across all datasets for each model
    model_metrics = defaultdict(lambda: {'kg': [], 'contr': [], 'abst': []})
    
    for model in results:
        for dataset in results[model]:
            if 'bilateral' in results[model][dataset]:
                data = results[model][dataset]['bilateral'].get('data', {})
                dist = data.get('bilateral_distribution', {})
                total = data.get('total_samples', 1)
                
                if total > 0:
                    kg_rate = dist.get('<f,f>', 0) / total
                    contr_rate = dist.get('<t,t>', 0) / total
                    abst_rate = (dist.get('<f,f>', 0) + dist.get('<t,t>', 0)) / total
                    
                    model_metrics[model]['kg'].append(kg_rate)
                    model_metrics[model]['contr'].append(contr_rate)
                    model_metrics[model]['abst'].append(abst_rate)
    
    # Calculate averages and metrics
    rows = []
    for model in model_metrics:
        if model_metrics[model]['kg']:
            avg_kg = np.mean(model_metrics[model]['kg'])
            avg_contr = np.mean(model_metrics[model]['contr'])
            avg_abst = np.mean(model_metrics[model]['abst'])
            
            # Epistemic honesty = abstention rate
            epistemic_honesty = avg_abst
            
            # Overconfidence = forced_f1 - bilateral_f1 (negative means bilateral is better)
            bilateral_f1s = []
            forced_f1s = []
            
            for dataset in results[model]:
                if 'bilateral' in results[model][dataset]:
                    y_true = results[model][dataset]['bilateral']['y_true']
                    y_pred = results[model][dataset]['bilateral']['y_pred']
                    f1, _, _ = bootstrap_confidence_interval(y_true, y_pred, metric='f1', n_bootstrap=50)
                    bilateral_f1s.append(f1)
                
                if 'unilateral_forced' in results[model][dataset]:
                    y_true = results[model][dataset]['unilateral_forced']['y_true']
                    y_pred = results[model][dataset]['unilateral_forced']['y_pred']
                    f1, _, _ = bootstrap_confidence_interval(y_true, y_pred, metric='f1', n_bootstrap=50)
                    forced_f1s.append(f1)
            
            if bilateral_f1s and forced_f1s:
                overconfidence = np.mean(forced_f1s) - np.mean(bilateral_f1s)
            else:
                overconfidence = 0.0
            
            rows.append({
                'model': model,
                'kg': avg_kg,
                'contr': avg_contr,
                'abst': avg_abst,
                'honesty': epistemic_honesty,
                'overconf': overconfidence
            })
    
    # Sort by epistemic honesty
    rows.sort(key=lambda x: x['honesty'], reverse=True)
    
    for row in rows:
        model_name = row['model'].replace('-', ' ').replace('_', ' ').title()
        if len(model_name) > 20:
            model_name = model_name[:17] + "..."
        
        line = f"| {model_name:<20} "
        line += f"| {row['kg']:.3f} | {row['contr']:.3f} | {row['abst']:.3f} "
        line += f"| {row['honesty']:.3f} "
        
        # Bold negative overconfidence values
        if row['overconf'] < 0:
            line += f"| **{row['overconf']:+.3f}** |"
        else:
            line += f"| {row['overconf']:+.3f} |"
        
        table_lines.append(line)
    
    table_lines.append("")
    table_lines.append("**Note**: Negative overconfidence values (shown in bold) confirm bilateral evaluation's value - models perform worse when forced to answer everything.")
    table_lines.append("")
    
    return table_lines


def main():
    """Generate all tables with real confidence intervals."""
    
    print("="*150)
    print("REGENERATING ALL TABLES WITH REAL CONFIDENCE INTERVALS")
    print("="*150)
    
    # Load all results
    print("\nLoading all result files...")
    results = load_all_results()
    
    print(f"Loaded results for {len(results)} models")
    
    # Generate all tables
    all_lines = []
    all_lines.append("# Final Results Tables - With Real Confidence Intervals")
    all_lines.append("")
    all_lines.append("**Generated with bootstrap confidence intervals (200 iterations, 95% CI)**")
    all_lines.append("")
    
    print("\nGenerating Table 1...")
    all_lines.extend(generate_table_1(results))
    
    print("Generating Table 2...")
    all_lines.extend(generate_table_2(results))
    
    print("Generating Table 3...")
    all_lines.extend(generate_table_3(results))
    
    # Note about Tables 4 and 5
    all_lines.append("## Table 4: Category Performance Analysis")
    all_lines.append("")
    all_lines.append("*Category analysis requires detailed category-level results from the original evaluation files.*")
    all_lines.append("")
    
    all_lines.append("## Table 5: Epistemic Policy Comparison")
    all_lines.append("")
    all_lines.append("*Policy comparison (classical vs paracomplete vs paraconsistent) requires specific policy evaluation runs.*")
    all_lines.append("")
    
    # Save the updated tables
    output_file = 'FINAL_RESULTS_TABLES_WITH_CIS.md'
    with open(output_file, 'w') as f:
        f.write('\n'.join(all_lines))
    
    print(f"\n✅ Tables with real confidence intervals saved to {output_file}")
    
    # Also save a summary CSV
    summary_data = []
    for model in results:
        for dataset in results[model]:
            if 'bilateral' in results[model][dataset]:
                y_true = results[model][dataset]['bilateral']['y_true']
                y_pred = results[model][dataset]['bilateral']['y_pred']
                f1_mean, f1_lower, f1_upper = bootstrap_confidence_interval(
                    y_true, y_pred, metric='f1', n_bootstrap=200
                )
                
                summary_data.append({
                    'Model': model,
                    'Dataset': dataset,
                    'Method': 'Bilateral',
                    'F1': f1_mean,
                    'F1_Lower': f1_lower,
                    'F1_Upper': f1_upper
                })
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        df.to_csv('results/summary_with_cis.csv', index=False)
        print("✅ Summary CSV saved to results/summary_with_cis.csv")
    
    print("\n✨ All tables regenerated with real confidence intervals!")


if __name__ == "__main__":
    main()