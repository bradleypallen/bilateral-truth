#!/usr/bin/env python3
"""Analyze unilateral vs bilateral results across all models and benchmarks."""

import json
import glob
from pathlib import Path
from collections import defaultdict

def analyze_benchmark(benchmark: str, pattern: str):
    """Analyze results for a specific benchmark."""
    results = []
    
    for file in glob.glob(f'results/{pattern}*_{benchmark}_complete.json'):
        try:
            with open(file) as f:
                data = json.load(f)
                model = data['model']
                
                # Get metrics for each method
                bi_class = data['results'].get('bilateral_classical', {})
                uni_direct = data['results'].get('unilateral_direct', {})
                ver_only = data['results'].get('verification_only', {})
                
                if uni_direct and bi_class and 'f1_macro' in uni_direct and 'f1_macro' in bi_class:
                    results.append({
                        'model': model.split('/')[-1][:25],  # Shorten model name
                        'uni_f1': uni_direct.get('f1_macro', 0),
                        'uni_acc': uni_direct.get('accuracy', 0),
                        'bi_f1': bi_class.get('f1_macro', 0),
                        'bi_acc': bi_class.get('accuracy', 0),
                        'bi_cov': bi_class.get('coverage', 0),
                        'ver_f1': ver_only.get('f1_macro', 0),
                        'ver_acc': ver_only.get('accuracy', 0),
                        'uni_beats_bi': uni_direct.get('f1_macro', 0) > bi_class.get('f1_macro', 0)
                    })
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    return results

def print_benchmark_summary(benchmark: str, results: list):
    """Print summary for a benchmark."""
    if not results:
        print(f"No results for {benchmark}")
        return
        
    print(f"\n{benchmark.upper()} RESULTS")
    print("="*90)
    print(f"{'Model':<25} {'Uni F1':>8} {'Bi F1':>8} {'Ver F1':>8} {'Bi Cov':>8} {'Winner':>12}")
    print("-"*90)
    
    for r in sorted(results, key=lambda x: x['bi_f1'], reverse=True):
        winner = 'Unilateral' if r['uni_beats_bi'] else 'Bilateral'
        print(f"{r['model']:<25} {r['uni_f1']:>8.3f} {r['bi_f1']:>8.3f} {r['ver_f1']:>8.3f} {r['bi_cov']:>8.3f} {winner:>12}")
    
    # Count winners
    uni_wins = sum(1 for r in results if r['uni_beats_bi'])
    print(f"\nUnilateral wins: {uni_wins}/{len(results)}")
    print(f"Bilateral wins: {len(results) - uni_wins}/{len(results)}")
    
    # Average metrics
    avg_uni_f1 = sum(r['uni_f1'] for r in results) / len(results)
    avg_bi_f1 = sum(r['bi_f1'] for r in results) / len(results)
    avg_ver_f1 = sum(r['ver_f1'] for r in results) / len(results)
    avg_bi_cov = sum(r['bi_cov'] for r in results) / len(results)
    
    print(f"\nAverage F1 Scores:")
    print(f"  Unilateral (forced): {avg_uni_f1:.3f}")
    print(f"  Bilateral classical: {avg_bi_f1:.3f}")
    print(f"  Verification-only:   {avg_ver_f1:.3f}")
    print(f"  Bilateral coverage:  {avg_bi_cov:.3f}")

def main():
    # Find most recent comparison files
    comparison_files = glob.glob('results/comparison_*.json')
    if not comparison_files:
        print("No comparison files found")
        return
    
    # Extract run ID from most recent files
    latest_file = max(comparison_files)
    pattern = Path(latest_file).name.split('_')[0] + '_' + Path(latest_file).name.split('_')[1] + '_' + Path(latest_file).name.split('_')[2]
    
    print(f"Analyzing results from pattern: {pattern}")
    
    # Analyze each benchmark
    benchmarks = ['truthfulqa', 'simpleqa', 'mmlupro', 'factscore']
    overall_results = defaultdict(list)
    
    for benchmark in benchmarks:
        results = analyze_benchmark(benchmark, pattern)
        if results:
            print_benchmark_summary(benchmark, results)
            for r in results:
                overall_results['all'].append(r)
    
    # Overall summary
    if overall_results['all']:
        print("\n" + "="*90)
        print("OVERALL SUMMARY ACROSS ALL BENCHMARKS")
        print("="*90)
        
        all_results = overall_results['all']
        uni_wins = sum(1 for r in all_results if r['uni_beats_bi'])
        total = len(all_results)
        
        print(f"Total comparisons: {total}")
        print(f"Unilateral wins:   {uni_wins} ({100*uni_wins/total:.1f}%)")
        print(f"Bilateral wins:    {total-uni_wins} ({100*(total-uni_wins)/total:.1f}%)")
        
        # Average differences
        f1_diffs = [r['uni_f1'] - r['bi_f1'] for r in all_results]
        avg_f1_diff = sum(f1_diffs) / len(f1_diffs)
        
        print(f"\nAverage F1 difference (Uni - Bi): {avg_f1_diff:+.3f}")
        
        if avg_f1_diff < 0:
            print("\n✓ BILATERAL EVALUATION GENERALLY OUTPERFORMS FORCED UNILATERAL")
            print("  This SUPPORTS the ArXiv paper's theoretical predictions")
        else:
            print("\n✗ FORCED UNILATERAL GENERALLY OUTPERFORMS BILATERAL")
            print("  This CONTRADICTS the ArXiv paper's theoretical predictions")
        
        # Compare verification-only vs bilateral
        ver_vs_bi_diffs = [r['ver_f1'] - r['bi_f1'] for r in all_results]
        avg_ver_diff = sum(ver_vs_bi_diffs) / len(ver_vs_bi_diffs)
        
        print(f"\nAverage F1 difference (Verification-only - Bilateral): {avg_ver_diff:+.3f}")
        if avg_ver_diff < 0:
            print("  → Refutation dimension adds value beyond verification alone")
        else:
            print("  → Verification alone may be sufficient")

if __name__ == "__main__":
    main()