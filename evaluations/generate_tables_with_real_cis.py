#!/usr/bin/env python3
"""Generate enhanced tables with REAL confidence intervals using bootstrap from original result files."""

import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import f1_score, accuracy_score


def bootstrap_confidence_interval(y_true, y_pred, metric='f1', n_bootstrap=100, confidence=0.95):
    """Calculate confidence interval using bootstrap resampling."""
    n_samples = len(y_true)
    scores = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = [y_true[i] for i in indices]
        y_pred_boot = [y_pred[i] for i in indices]
        
        if metric == 'f1':
            # Handle abstentions by filtering them out
            valid_pairs = [(t, p) for t, p in zip(y_true_boot, y_pred_boot) if p != 'undefined']
            if valid_pairs:
                y_t, y_p = zip(*valid_pairs)
                score = f1_score(y_t, y_p, average='macro', zero_division=0)
            else:
                score = 0.0
        elif metric == 'accuracy':
            valid_pairs = [(t, p) for t, p in zip(y_true_boot, y_pred_boot) if p != 'undefined']
            if valid_pairs:
                y_t, y_p = zip(*valid_pairs)
                score = accuracy_score(y_t, y_p)
            else:
                score = 0.0
        elif metric == 'coverage':
            score = sum(1 for p in y_pred_boot if p != 'undefined') / len(y_pred_boot)
        
        scores.append(score)
    
    # Calculate confidence interval
    alpha = 1 - confidence
    lower = np.percentile(scores, (alpha/2) * 100)
    upper = np.percentile(scores, (1 - alpha/2) * 100)
    mean = np.mean(scores)
    
    return mean, lower, upper


def extract_predictions_from_bilateral(results):
    """Extract predictions from bilateral results."""
    y_true = []
    y_pred = []
    
    for item in results:
        ground_truth = item.get('expected_label', item.get('ground_truth', 'correct'))
        bilateral_value = item.get('bilateral_value', '<e,e>')
        
        # Map bilateral value to prediction
        if bilateral_value == '<t,f>':
            pred = 'correct'
        elif bilateral_value == '<f,t>':
            pred = 'incorrect'
        else:
            pred = 'undefined'
        
        y_true.append(ground_truth)
        y_pred.append(pred)
    
    return y_true, y_pred


def extract_predictions_from_unilateral(results):
    """Extract predictions from unilateral results."""
    y_true = []
    y_pred = []
    
    for item in results:
        ground_truth = item.get('expected_label', item.get('ground_truth', 'correct'))
        prediction = item.get('prediction', 'undefined')
        
        y_true.append(ground_truth)
        y_pred.append(prediction)
    
    return y_true, y_pred


def compute_verification_only(bilateral_results):
    """Compute verification-only predictions from bilateral results."""
    y_true = []
    y_pred = []
    
    for item in bilateral_results:
        ground_truth = item.get('expected_label', item.get('ground_truth', 'correct'))
        bilateral_value = item.get('bilateral_value', '<e,e>')
        
        # Use only the verification component (first element)
        if bilateral_value.startswith('<t,'):
            pred = 'correct'
        elif bilateral_value.startswith('<f,'):
            pred = 'incorrect'
        else:
            pred = 'undefined'
        
        y_true.append(ground_truth)
        y_pred.append(pred)
    
    return y_true, y_pred


def generate_table_1_with_real_cis():
    """Generate Table 1 with real confidence intervals from original result files."""
    
    print("\n" + "="*150)
    print("TABLE 1: PERFORMANCE COMPARISON WITH REAL CONFIDENCE INTERVALS")
    print("="*150)
    
    all_results = []
    
    # Process bilateral results (classical)
    bilateral_files = glob.glob('results/*_classical_results.json')
    
    for file in bilateral_files:
        try:
            with open(file) as f:
                data = json.load(f)
            
            # Extract metadata
            filename = Path(file).stem
            parts = filename.split('_')
            
            # Handle both naming patterns
            if parts[1] == 'complete':
                dataset = parts[0]  # e.g., 'truthfulqa'
            else:
                dataset = parts[0]  # e.g., 'truthfulqa'
            
            model = data.get('model', '')
            model_short = model.split('/')[-1] if '/' in model else model
            
            if 'detailed_results' not in data:
                continue
            
            row = {
                'Model': model_short,
                'Dataset': dataset
            }
            
            # Extract bilateral predictions
            y_true_bi, y_pred_bi = extract_predictions_from_bilateral(data['detailed_results'])
            
            # Calculate bilateral metrics with CIs
            f1_mean, f1_lower, f1_upper = bootstrap_confidence_interval(y_true_bi, y_pred_bi, metric='f1')
            acc_mean, acc_lower, acc_upper = bootstrap_confidence_interval(y_true_bi, y_pred_bi, metric='accuracy')
            cov_mean, cov_lower, cov_upper = bootstrap_confidence_interval(y_true_bi, y_pred_bi, metric='coverage')
            
            row['Bilateral_F1'] = f1_mean
            row['Bilateral_F1_Lower'] = f1_lower
            row['Bilateral_F1_Upper'] = f1_upper
            row['Bilateral_Acc'] = acc_mean
            row['Bilateral_Cov'] = cov_mean
            
            # Compute verification-only from bilateral
            y_true_ver, y_pred_ver = compute_verification_only(data['detailed_results'])
            
            f1_mean, f1_lower, f1_upper = bootstrap_confidence_interval(y_true_ver, y_pred_ver, metric='f1')
            acc_mean, acc_lower, acc_upper = bootstrap_confidence_interval(y_true_ver, y_pred_ver, metric='accuracy')
            cov_mean, cov_lower, cov_upper = bootstrap_confidence_interval(y_true_ver, y_pred_ver, metric='coverage')
            
            row['Verification_F1'] = f1_mean
            row['Verification_F1_Lower'] = f1_lower
            row['Verification_F1_Upper'] = f1_upper
            row['Verification_Acc'] = acc_mean
            row['Verification_Cov'] = cov_mean
            
            all_results.append(row)
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    
    # Process unilateral forced results
    unilateral_files = glob.glob('results/*_unilateral_forced_results.json') + \
                       glob.glob('results/*_complete_*_unilateral_forced_results.json')
    
    for file in unilateral_files:
        try:
            with open(file) as f:
                data = json.load(f)
            
            # Extract metadata
            filename = Path(file).stem
            parts = filename.split('_')
            
            # Handle both naming patterns
            if parts[1] == 'complete':
                dataset = parts[0]  # e.g., 'truthfulqa'
            else:
                dataset = parts[0]  # e.g., 'truthfulqa'
            
            model = data.get('model', '')
            model_short = model.split('/')[-1] if '/' in model else model
            
            if 'detailed_results' not in data:
                continue
            
            # Find corresponding row or create new one
            matching_row = None
            for row in all_results:
                if row['Model'] == model_short and row['Dataset'] == dataset:
                    matching_row = row
                    break
            
            if not matching_row:
                matching_row = {'Model': model_short, 'Dataset': dataset}
                all_results.append(matching_row)
            
            # Extract unilateral predictions
            y_true_uni, y_pred_uni = extract_predictions_from_unilateral(data['detailed_results'])
            
            # Calculate unilateral metrics with CIs
            f1_mean, f1_lower, f1_upper = bootstrap_confidence_interval(y_true_uni, y_pred_uni, metric='f1')
            acc_mean, acc_lower, acc_upper = bootstrap_confidence_interval(y_true_uni, y_pred_uni, metric='accuracy')
            cov_mean, cov_lower, cov_upper = bootstrap_confidence_interval(y_true_uni, y_pred_uni, metric='coverage')
            
            matching_row['Unilateral_F1'] = f1_mean
            matching_row['Unilateral_F1_Lower'] = f1_lower
            matching_row['Unilateral_F1_Upper'] = f1_upper
            matching_row['Unilateral_Acc'] = acc_mean
            matching_row['Unilateral_Cov'] = cov_mean
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    
    # Process unilateral uncertain results
    uncertain_files = glob.glob('results/*_unilateral_uncertain_results.json') + \
                      glob.glob('results/*_complete_*_unilateral_uncertain_results.json')
    
    for file in uncertain_files:
        try:
            with open(file) as f:
                data = json.load(f)
            
            # Extract metadata
            filename = Path(file).stem
            parts = filename.split('_')
            
            # Handle both naming patterns
            if parts[1] == 'complete':
                dataset = parts[0]  # e.g., 'truthfulqa'
            else:
                dataset = parts[0]  # e.g., 'truthfulqa'
            
            model = data.get('model', '')
            model_short = model.split('/')[-1] if '/' in model else model
            
            if 'detailed_results' not in data:
                continue
            
            # Find corresponding row
            matching_row = None
            for row in all_results:
                if row['Model'] == model_short and row['Dataset'] == dataset:
                    matching_row = row
                    break
            
            if not matching_row:
                matching_row = {'Model': model_short, 'Dataset': dataset}
                all_results.append(matching_row)
            
            # Extract uncertain predictions
            y_true_unc, y_pred_unc = extract_predictions_from_unilateral(data['detailed_results'])
            
            # Calculate uncertain metrics with CIs
            f1_mean, f1_lower, f1_upper = bootstrap_confidence_interval(y_true_unc, y_pred_unc, metric='f1')
            acc_mean, acc_lower, acc_upper = bootstrap_confidence_interval(y_true_unc, y_pred_unc, metric='accuracy')
            cov_mean, cov_lower, cov_upper = bootstrap_confidence_interval(y_true_unc, y_pred_unc, metric='coverage')
            
            matching_row['Uncertain_F1'] = f1_mean
            matching_row['Uncertain_F1_Lower'] = f1_lower
            matching_row['Uncertain_F1_Upper'] = f1_upper
            matching_row['Uncertain_Acc'] = acc_mean
            matching_row['Uncertain_Cov'] = cov_mean
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    if df.empty:
        print("No results found!")
        return
    
    # Group by dataset and display
    datasets = ['truthfulqa', 'simpleqa', 'mmlupro', 'factscore']
    
    markdown_output = []
    
    for dataset in datasets:
        subset = df[df['Dataset'] == dataset]
        
        if len(subset) == 0:
            continue
        
        # Sort by Bilateral F1
        subset = subset.sort_values('Bilateral_F1', ascending=False)
        
        print(f"\n### {dataset.upper()} (n={len(subset)} models)")
        markdown_output.append(f"\n### {dataset.upper()} (n={len(subset)} models)")
        
        header = "| Model | **Bilateral F1 [95% CI]** | **Forced Unilateral F1 [95% CI]** | **Uncertain Unilateral F1 [95% CI]** | **Verification F1 [95% CI]** |"
        separator = "|-------|---------------------------|-----------------------------------|---------------------------------------|------------------------------|"
        
        print(header)
        print(separator)
        markdown_output.append(header)
        markdown_output.append(separator)
        
        for _, row in subset.iterrows():
            model_name = row['Model'].replace('_', ' ').replace('-', ' ').title()
            if len(model_name) > 25:
                model_name = model_name[:22] + "..."
            
            # Format each method's results
            line_parts = [f"| {model_name:<25}"]
            
            # Bilateral
            if 'Bilateral_F1' in row and not pd.isna(row['Bilateral_F1']):
                line_parts.append(f" | {row['Bilateral_F1']:.3f} [{row['Bilateral_F1_Lower']:.3f}, {row['Bilateral_F1_Upper']:.3f}]")
            else:
                line_parts.append(" | N/A")
            
            # Forced Unilateral
            if 'Unilateral_F1' in row and not pd.isna(row['Unilateral_F1']):
                line_parts.append(f" | {row['Unilateral_F1']:.3f} [{row['Unilateral_F1_Lower']:.3f}, {row['Unilateral_F1_Upper']:.3f}]")
            else:
                line_parts.append(" | N/A")
            
            # Uncertain Unilateral
            if 'Uncertain_F1' in row and not pd.isna(row['Uncertain_F1']):
                line_parts.append(f" | {row['Uncertain_F1']:.3f} [{row['Uncertain_F1_Lower']:.3f}, {row['Uncertain_F1_Upper']:.3f}]")
            else:
                line_parts.append(" | N/A")
            
            # Verification
            if 'Verification_F1' in row and not pd.isna(row['Verification_F1']):
                line_parts.append(f" | {row['Verification_F1']:.3f} [{row['Verification_F1_Lower']:.3f}, {row['Verification_F1_Upper']:.3f}] |")
            else:
                line_parts.append(" | N/A |")
            
            line = ''.join(line_parts)
            print(line)
            markdown_output.append(line)
    
    # Save to CSV for further analysis
    df.to_csv('results/table1_with_real_cis.csv', index=False)
    print("\n✅ Table 1 with real confidence intervals saved to results/table1_with_real_cis.csv")
    
    # Save markdown output
    with open('results/table1_with_real_cis.md', 'w') as f:
        f.write('\n'.join(markdown_output))
    print("✅ Markdown table saved to results/table1_with_real_cis.md")
    
    return df


def main():
    """Generate all tables with real confidence intervals."""
    
    print("="*150)
    print("GENERATING TABLES WITH REAL CONFIDENCE INTERVALS")
    print("="*150)
    
    # Generate Table 1 with real CIs
    table1_df = generate_table_1_with_real_cis()
    
    print("\n✨ Tables with real confidence intervals generated successfully!")


if __name__ == "__main__":
    main()