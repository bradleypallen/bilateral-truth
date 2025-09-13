#!/usr/bin/env python3
"""Generate comprehensive final comparison report with all completed evaluations."""

import json
import glob
from pathlib import Path
import pandas as pd
from collections import defaultdict

def load_all_results():
    """Load all bilateral and unilateral results."""
    data = []
    
    # Get all comparison files (both original and final)
    comparison_files = glob.glob('results/comparison_*.json')
    
    # Track what we've already loaded to avoid duplicates
    loaded = set()
    
    for file in comparison_files:
        try:
            with open(file) as f:
                comp = json.load(f)
                
            if 'results' not in comp:
                continue
                
            model = comp['model']
            dataset = comp['dataset'].replace('_complete', '')
            
            # Create unique key
            key = f"{model}_{dataset}"
            if key in loaded:
                continue
            loaded.add(key)
            
            # Get metrics for each evaluation type
            bilateral = comp['results'].get('bilateral_classical', {})
            unilateral = comp['results'].get('unilateral_direct', {})
            verification = comp['results'].get('verification_only', {})
            
            if bilateral and unilateral and 'f1_macro' in bilateral and 'f1_macro' in unilateral:
                row = {
                    'Model': model,
                    'Model_Short': model.split('/')[-1] if '/' in model else model,
                    'Dataset': dataset,
                    
                    # Bilateral metrics
                    'Bi_Acc': bilateral.get('accuracy', 0),
                    'Bi_F1': bilateral.get('f1_macro', 0),
                    'Bi_Cov': bilateral.get('coverage', 0),
                    'Bi_Samples': bilateral.get('total_samples', 0),
                    
                    # Unilateral metrics
                    'Uni_Acc': unilateral.get('accuracy', 0),
                    'Uni_F1': unilateral.get('f1_macro', 0),
                    'Uni_Cov': unilateral.get('coverage', 1.0),
                    'Uni_Samples': unilateral.get('total_samples', 0),
                    
                    # Verification-only metrics (if available)
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
                    if total > 0:
                        row['TT_Rate'] = dist.get('<t,t>', 0) / total
                        row['FF_Rate'] = dist.get('<f,f>', 0) / total
                        row['TF_Rate'] = dist.get('<t,f>', 0) / total
                        row['FT_Rate'] = dist.get('<f,t>', 0) / total
                
                data.append(row)
                
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    
    return pd.DataFrame(data)

def print_comprehensive_report(df):
    """Print comprehensive comparison report."""
    
    print("\n" + "="*140)
    print("COMPLETE BILATERAL VS UNILATERAL COMPARISON REPORT")
    print("="*140)
    print(f"Total evaluations analyzed: {len(df)}")
    print()
    
    # Overall statistics
    total = len(df)
    uni_wins = len(df[df['Winner'] == 'Unilateral'])
    
    print("OVERALL RESULTS")
    print("-"*140)
    print(f"Bilateral wins: {total - uni_wins}/{total} ({100*(total - uni_wins)/total:.1f}%)")
    print(f"Unilateral wins: {uni_wins}/{total} ({100*uni_wins/total:.1f}%)")
    print(f"Average F1 difference (Uni - Bi): {df['F1_Diff'].mean():+.4f}")
    print(f"Average coverage - Bilateral: {df['Bi_Cov'].mean():.3f}, Unilateral: {df['Uni_Cov'].mean():.3f}")
    
    # Per-benchmark analysis
    print("\n" + "="*140)
    print("RESULTS BY BENCHMARK")
    print("="*140)
    
    benchmarks = ['truthfulqa', 'simpleqa', 'mmlupro', 'factscore']
    
    for benchmark in benchmarks:
        subset = df[df['Dataset'] == benchmark].sort_values('Bi_F1', ascending=False)
        
        if len(subset) == 0:
            continue
            
        print(f"\n{benchmark.upper()} (n={len(subset)} models)")
        print("-"*140)
        print(f"{'Model':<30} | {'Bilateral':^35} | {'Unilateral':^35} | {'Diff (U-B)':^20} | Winner")
        print(f"{'':30} | {'Acc':>10} {'F1':>10} {'Cov':>10} | {'Acc':>10} {'F1':>10} {'Cov':>10} | {'F1':>10} {'Cov':>10} |")
        print("-"*140)
        
        for _, row in subset.iterrows():
            model_name = row['Model_Short'][:30]
            print(f"{model_name:<30} | {row['Bi_Acc']:>10.3f} {row['Bi_F1']:>10.3f} {row['Bi_Cov']:>10.3f} | "
                  f"{row['Uni_Acc']:>10.3f} {row['Uni_F1']:>10.3f} {row['Uni_Cov']:>10.3f} | "
                  f"{row['F1_Diff']:>+10.3f} {row['Cov_Diff']:>+10.3f} | {row['Winner']:>10}")
        
        # Benchmark statistics
        uni_wins_bench = len(subset[subset['Winner'] == 'Unilateral'])
        total_bench = len(subset)
        avg_f1_diff = subset['F1_Diff'].mean()
        avg_bi_f1 = subset['Bi_F1'].mean()
        avg_uni_f1 = subset['Uni_F1'].mean()
        
        print(f"\nBenchmark Summary:")
        print(f"  Bilateral wins: {total_bench - uni_wins_bench}/{total_bench} ({100*(total_bench - uni_wins_bench)/total_bench:.1f}%)")
        print(f"  Average F1 - Bilateral: {avg_bi_f1:.3f}, Unilateral: {avg_uni_f1:.3f}")
        print(f"  Average F1 difference: {avg_f1_diff:+.3f}")
    
    # Model-level analysis
    print("\n" + "="*140)
    print("MODEL PERFORMANCE ACROSS BENCHMARKS")
    print("="*140)
    
    model_stats = df.groupby('Model_Short').agg({
        'Bi_F1': 'mean',
        'Uni_F1': 'mean',
        'F1_Diff': 'mean',
        'Bi_Cov': 'mean',
        'Winner': lambda x: (x == 'Unilateral').sum()
    }).round(3)
    
    model_stats['Total'] = df.groupby('Model_Short').size()
    model_stats['Win_Rate'] = (model_stats['Winner'] / model_stats['Total'] * 100).round(1)
    model_stats = model_stats.sort_values('Bi_F1', ascending=False)
    
    print(f"{'Model':<30} | {'Benchmarks':>10} | {'Avg Bi F1':>10} | {'Avg Uni F1':>10} | {'Avg Diff':>10} | {'Bi Cov':>10} | {'Uni Win Rate':>12}")
    print("-"*110)
    
    for model, row in model_stats.iterrows():
        print(f"{model[:30]:<30} | {int(row['Total']):>10} | {row['Bi_F1']:>10.3f} | "
              f"{row['Uni_F1']:>10.3f} | {row['F1_Diff']:>+10.3f} | {row['Bi_Cov']:>10.3f} | "
              f"{row['Win_Rate']:>11.1f}%")
    
    # Epistemic distribution analysis (if available)
    if 'TT_Rate' in df.columns:
        print("\n" + "="*140)
        print("BILATERAL EPISTEMIC STATE DISTRIBUTIONS")
        print("="*140)
        
        print(f"{'Model':<30} | {'<t,f>':>10} | {'<f,t>':>10} | {'<t,t>':>10} | {'<f,f>':>10} | {'Coverage':>10}")
        print("-"*90)
        
        for model in model_stats.index:
            model_subset = df[df['Model_Short'] == model]
            if 'TT_Rate' in model_subset.columns:
                avg_tt = model_subset['TT_Rate'].mean()
                avg_ff = model_subset['FF_Rate'].mean()
                avg_tf = model_subset['TF_Rate'].mean() if 'TF_Rate' in model_subset.columns else 0
                avg_ft = model_subset['FT_Rate'].mean() if 'FT_Rate' in model_subset.columns else 0
                avg_cov = model_subset['Bi_Cov'].mean()
                
                print(f"{model[:30]:<30} | {avg_tf:>10.1%} | {avg_ft:>10.1%} | "
                      f"{avg_tt:>10.1%} | {avg_ff:>10.1%} | {avg_cov:>10.1%}")
    
    # Final conclusions
    print("\n" + "="*140)
    print("KEY FINDINGS")
    print("="*140)
    
    if df['F1_Diff'].mean() < 0:
        print("✓ BILATERAL EVALUATION OUTPERFORMS FORCED UNILATERAL")
        print(f"  - Average F1 advantage: {-df['F1_Diff'].mean():.3f}")
        print(f"  - Win rate: {100*(total - uni_wins)/total:.1f}%")
        print("  - This SUPPORTS the ArXiv paper's theoretical predictions")
    else:
        print("✗ FORCED UNILATERAL OUTPERFORMS BILATERAL")
        print(f"  - Average F1 advantage: {df['F1_Diff'].mean():.3f}")
        print(f"  - Win rate: {100*uni_wins/total:.1f}%")
        print("  - This CONTRADICTS the ArXiv paper's theoretical predictions")
    
    print(f"\n  Coverage trade-off:")
    print(f"  - Bilateral coverage: {df['Bi_Cov'].mean():.1%} (abstains when uncertain)")
    print(f"  - Unilateral coverage: {df['Uni_Cov'].mean():.1%} (forced choice)")
    
    # Best and worst performers
    print("\n  Best bilateral performers:")
    best = model_stats.nlargest(3, 'Bi_F1')
    for i, (model, row) in enumerate(best.iterrows(), 1):
        print(f"    {i}. {model}: {row['Bi_F1']:.3f} F1")
    
    print("\n  Largest bilateral advantages:")
    largest_adv = model_stats.nsmallest(3, 'F1_Diff')
    for i, (model, row) in enumerate(largest_adv.iterrows(), 1):
        print(f"    {i}. {model}: {-row['F1_Diff']:.3f} F1 advantage")

def main():
    """Main entry point."""
    df = load_all_results()
    
    if len(df) == 0:
        print("No comparison data found")
        return
    
    print_comprehensive_report(df)
    
    # Save detailed results
    output_file = 'results/final_comparison_report.csv'
    df.to_csv(output_file, index=False)
    print(f"\n💾 Detailed results saved to {output_file}")
    
    # Print completion status
    print(f"\n📊 Report includes {len(df)} model-benchmark combinations")
    print(f"   Models: {df['Model_Short'].nunique()}")
    print(f"   Benchmarks: {df['Dataset'].nunique()}")

if __name__ == "__main__":
    main()