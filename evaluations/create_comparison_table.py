#!/usr/bin/env python3
"""Create a comprehensive comparison table of unilateral vs bilateral results."""

import json
import glob
from pathlib import Path
import pandas as pd
from collections import defaultdict

def load_all_comparisons():
    """Load all comparison results."""
    data = []
    
    # Find all comparison files
    for file in glob.glob('results/comparison_*_*.json'):
        try:
            with open(file) as f:
                comp = json.load(f)
                
                if 'results' not in comp:
                    continue
                    
                model = comp['model']
                dataset = comp['dataset'].replace('_complete', '')
                
                # Get metrics for each evaluation type
                bilateral = comp['results'].get('bilateral_classical', {})
                unilateral = comp['results'].get('unilateral_direct', {})
                verification = comp['results'].get('verification_only', {})
                
                if bilateral and unilateral:
                    row = {
                        'Model': model.split('/')[-1] if '/' in model else model,
                        'Dataset': dataset,
                        
                        # Bilateral metrics
                        'Bi_Acc': bilateral.get('accuracy', 0),
                        'Bi_F1': bilateral.get('f1_macro', 0),
                        'Bi_Cov': bilateral.get('coverage', 0),
                        
                        # Unilateral metrics
                        'Uni_Acc': unilateral.get('accuracy', 0),
                        'Uni_F1': unilateral.get('f1_macro', 0),
                        'Uni_Cov': unilateral.get('coverage', 1.0),
                        
                        # Verification-only metrics
                        'Ver_Acc': verification.get('accuracy', 0) if verification else 0,
                        'Ver_F1': verification.get('f1_macro', 0) if verification else 0,
                        'Ver_Cov': verification.get('coverage', 0) if verification else 0,
                        
                        # Differences
                        'F1_Diff': unilateral.get('f1_macro', 0) - bilateral.get('f1_macro', 0),
                        'Acc_Diff': unilateral.get('accuracy', 0) - bilateral.get('accuracy', 0),
                        'Cov_Diff': unilateral.get('coverage', 1.0) - bilateral.get('coverage', 0),
                        
                        # Winner
                        'Winner': 'Unilateral' if unilateral.get('f1_macro', 0) > bilateral.get('f1_macro', 0) else 'Bilateral'
                    }
                    
                    # Add bilateral distribution if available
                    if 'bilateral_distribution' in bilateral:
                        dist = bilateral['bilateral_distribution']
                        total = bilateral.get('total_samples', 1)
                        row['TT_Rate'] = dist.get('<t,t>', 0) / total if total > 0 else 0
                        row['FF_Rate'] = dist.get('<f,f>', 0) / total if total > 0 else 0
                    
                    data.append(row)
                    
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    
    return pd.DataFrame(data)

def print_summary_tables(df):
    """Print formatted summary tables."""
    
    # Overall summary
    print("\n" + "="*120)
    print("COMPREHENSIVE UNILATERAL VS BILATERAL COMPARISON")
    print("="*120)
    
    # Group by dataset
    datasets = df['Dataset'].unique()
    
    for dataset in sorted(datasets):
        subset = df[df['Dataset'] == dataset].sort_values('Bi_F1', ascending=False)
        
        if len(subset) == 0:
            continue
            
        print(f"\n{dataset.upper()}")
        print("-"*120)
        print(f"{'Model':<25} | {'Bilateral':^28} | {'Unilateral':^28} | {'Diff':^15} | Winner")
        print(f"{'':25} | {'Acc':>8} {'F1':>8} {'Cov':>8} | {'Acc':>8} {'F1':>8} {'Cov':>8} | {'F1 Diff':>15} |")
        print("-"*120)
        
        for _, row in subset.iterrows():
            model_name = row['Model'][:25]
            print(f"{model_name:<25} | {row['Bi_Acc']:>8.3f} {row['Bi_F1']:>8.3f} {row['Bi_Cov']:>8.3f} | "
                  f"{row['Uni_Acc']:>8.3f} {row['Uni_F1']:>8.3f} {row['Uni_Cov']:>8.3f} | "
                  f"{row['F1_Diff']:>+15.3f} | {row['Winner']}")
        
        # Dataset statistics
        uni_wins = len(subset[subset['Winner'] == 'Unilateral'])
        total = len(subset)
        avg_f1_diff = subset['F1_Diff'].mean()
        avg_bi_cov = subset['Bi_Cov'].mean()
        
        print(f"\n  Unilateral wins: {uni_wins}/{total} ({100*uni_wins/total:.1f}%)")
        print(f"  Average F1 difference (Uni-Bi): {avg_f1_diff:+.3f}")
        print(f"  Average bilateral coverage: {avg_bi_cov:.3f}")
    
    # Overall statistics
    print("\n" + "="*120)
    print("OVERALL STATISTICS")
    print("="*120)
    
    total_comparisons = len(df)
    uni_wins = len(df[df['Winner'] == 'Unilateral'])
    
    print(f"Total comparisons: {total_comparisons}")
    print(f"Unilateral wins: {uni_wins} ({100*uni_wins/total_comparisons:.1f}%)")
    print(f"Bilateral wins: {total_comparisons - uni_wins} ({100*(total_comparisons - uni_wins)/total_comparisons:.1f}%)")
    
    print(f"\nAverage metrics across all evaluations:")
    print(f"  Bilateral:  Acc={df['Bi_Acc'].mean():.3f}, F1={df['Bi_F1'].mean():.3f}, Cov={df['Bi_Cov'].mean():.3f}")
    print(f"  Unilateral: Acc={df['Uni_Acc'].mean():.3f}, F1={df['Uni_F1'].mean():.3f}, Cov={df['Uni_Cov'].mean():.3f}")
    print(f"  Average F1 difference (Uni-Bi): {df['F1_Diff'].mean():+.3f}")
    
    # Model-specific summary
    print("\n" + "="*120)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*120)
    
    model_stats = df.groupby('Model').agg({
        'Bi_F1': 'mean',
        'Uni_F1': 'mean',
        'F1_Diff': 'mean',
        'Winner': lambda x: (x == 'Unilateral').sum()
    }).round(3)
    
    model_stats['Total'] = df.groupby('Model').size()
    model_stats['Win_Rate'] = (model_stats['Winner'] / model_stats['Total'] * 100).round(1)
    
    model_stats = model_stats.sort_values('Bi_F1', ascending=False)
    
    print(f"{'Model':<25} | {'Avg Bi F1':>10} | {'Avg Uni F1':>10} | {'Avg Diff':>10} | {'Uni Wins':>10} | {'Win Rate':>10}")
    print("-"*85)
    
    for model, row in model_stats.iterrows():
        print(f"{model[:25]:<25} | {row['Bi_F1']:>10.3f} | {row['Uni_F1']:>10.3f} | "
              f"{row['F1_Diff']:>+10.3f} | {int(row['Winner']):>10}/{int(row['Total'])} | {row['Win_Rate']:>9.1f}%")
    
    # Check if verification-only data exists
    if df['Ver_F1'].sum() > 0:
        print("\n" + "="*120)
        print("VERIFICATION-ONLY COMPARISON")
        print("="*120)
        
        ver_subset = df[df['Ver_F1'] > 0]
        if len(ver_subset) > 0:
            print(f"{'Model':<25} | {'Dataset':<15} | {'Verification-Only F1':>20} | {'Bilateral F1':>15} | {'Diff':>10}")
            print("-"*95)
            
            for _, row in ver_subset.iterrows():
                ver_diff = row['Ver_F1'] - row['Bi_F1']
                print(f"{row['Model'][:25]:<25} | {row['Dataset']:<15} | {row['Ver_F1']:>20.3f} | "
                      f"{row['Bi_F1']:>15.3f} | {ver_diff:>+10.3f}")

def main():
    """Main entry point."""
    df = load_all_comparisons()
    
    if len(df) == 0:
        print("No comparison data found")
        return
    
    print_summary_tables(df)
    
    # Save to CSV for further analysis
    output_file = 'results/comparison_summary.csv'
    df.to_csv(output_file, index=False)
    print(f"\n💾 Full data saved to {output_file}")

if __name__ == "__main__":
    main()