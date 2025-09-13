#!/usr/bin/env python3
"""
Generate evaluation summary report from existing results JSON files.
Reads all results files in the results/ directory and produces a comprehensive summary.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict

def load_results_files(results_dir: str = "results") -> List[Dict]:
    """Load all results JSON files from the directory."""
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Error: Results directory '{results_dir}' not found")
        return []
    
    all_results = []
    
    # Find all results files (excluding summary files)
    for file_path in results_path.glob("*.json"):
        if "summary" in file_path.name:
            continue
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Extract benchmark and model from filename
                # Format: {benchmark}_{model}_{epistemic_policy}_results.json
                parts = file_path.stem.split('_')
                
                # Find where "classical" or other epistemic policy appears
                policy_idx = -2  # Usually second to last
                for i, part in enumerate(parts):
                    if part in ['classical', 'paraconsistent', 'paracomplete']:
                        policy_idx = i
                        break
                
                # Everything before policy is benchmark_model
                benchmark_model_parts = parts[:policy_idx]
                
                # Common benchmark prefixes
                benchmark_prefixes = ['truthfulqa', 'simpleqa', 'mmlupro', 'factscore']
                
                benchmark = None
                model_parts = []
                
                for prefix in benchmark_prefixes:
                    if benchmark_model_parts[0] == prefix:
                        if prefix == 'truthfulqa' and len(benchmark_model_parts) > 1 and benchmark_model_parts[1] == 'complete':
                            benchmark = 'truthfulqa_complete'
                            model_parts = benchmark_model_parts[2:]
                        elif prefix == 'simpleqa' and len(benchmark_model_parts) > 1 and benchmark_model_parts[1] == 'complete':
                            benchmark = 'simpleqa_complete'
                            model_parts = benchmark_model_parts[2:]
                        elif prefix == 'mmlupro' and len(benchmark_model_parts) > 1 and benchmark_model_parts[1] == 'complete':
                            benchmark = 'mmlupro_complete'
                            model_parts = benchmark_model_parts[2:]
                        elif prefix == 'factscore' and len(benchmark_model_parts) > 1 and benchmark_model_parts[1] == 'complete':
                            benchmark = 'factscore_complete'
                            model_parts = benchmark_model_parts[2:]
                        else:
                            benchmark = prefix
                            model_parts = benchmark_model_parts[1:]
                        break
                
                if not benchmark:
                    benchmark = benchmark_model_parts[0]
                    model_parts = benchmark_model_parts[1:]
                
                model = '_'.join(model_parts).replace('_', '-')
                
                # Add metadata to results
                data['benchmark'] = benchmark
                data['model'] = model
                data['file'] = file_path.name
                
                all_results.append(data)
                
        except Exception as e:
            print(f"Warning: Could not load {file_path.name}: {e}")
    
    return all_results

def generate_summary_report(results: List[Dict]) -> None:
    """Generate and print a comprehensive summary report."""
    
    if not results:
        print("No results files found!")
        return
    
    # Organize results by benchmark and model
    by_benchmark = defaultdict(list)
    by_model = defaultdict(list)
    models = set()
    benchmarks = set()
    
    for result in results:
        benchmark = result['benchmark']
        model = result['model']
        by_benchmark[benchmark].append(result)
        by_model[model].append(result)
        models.add(model)
        benchmarks.add(benchmark)
    
    # Print header
    print("\n" + "="*120)
    print("BILATERAL TRUTH EVALUATION SUMMARY REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*120)
    
    # Print overview
    print(f"\nDatasets evaluated: {len(benchmarks)}")
    print(f"Models evaluated: {len(models)}")
    print(f"Total result files: {len(results)}")
    
    # Benchmark names mapping
    benchmark_names = {
        'truthfulqa_complete': 'TruthfulQA',
        'simpleqa_complete': 'SimpleQA',
        'mmlupro_complete': 'MMLU-Pro',
        'factscore_complete': 'FACTScore'
    }
    
    # Print detailed results table
    print("\n" + "="*120)
    print("DETAILED RESULTS BY BENCHMARK AND MODEL")
    print("="*120)
    print(f"{'Benchmark':<15} {'Model':<35} {'Accuracy':<10} {'Coverage':<10} {'F1 Macro':<10} {'Samples':<10} {'Time (s)':<10}")
    print("-"*120)
    
    for benchmark in sorted(benchmarks):
        benchmark_display = benchmark_names.get(benchmark, benchmark)
        for result in sorted(by_benchmark[benchmark], key=lambda x: x['model']):
            accuracy = result.get('accuracy', 0)
            coverage = result.get('coverage', 0)
            f1_macro = result.get('f1_macro', 0)
            samples = result.get('total_samples', 0)
            eval_time = result.get('evaluation_time', 0)
            
            model_display = result['model'][:35]
            print(f"{benchmark_display:<15} {model_display:<35} {accuracy:<10.3f} {coverage:<10.3f} {f1_macro:<10.3f} {samples:<10} {eval_time:<10.1f}")
    
    # Print bilateral truth value distributions
    print("\n" + "="*120)
    print("BILATERAL TRUTH VALUE DISTRIBUTIONS (%)")
    print("Key: <verifiability, refutability> where t=true, f=false, e=evaluation failure")
    print("Core values: <t,f>=true, <f,t>=false, <t,t>=contradiction, <f,f>=knowledge gap/uncertainty")
    print("With failures: Values containing 'e' indicate evaluation failure in one or both dimensions")
    print("="*120)
    print(f"{'Benchmark':<15} {'Model':<30} {'<t,f>':<8} {'<f,t>':<8} {'<t,t>':<8} {'<f,f>':<8} {'<t,e>':<8} {'<e,t>':<8} {'<f,e>':<8} {'<e,f>':<8} {'<e,e>':<8}")
    print("-"*120)
    
    for benchmark in sorted(benchmarks):
        benchmark_display = benchmark_names.get(benchmark, benchmark)
        for result in sorted(by_benchmark[benchmark], key=lambda x: x['model']):
            # Count bilateral truth values
            tv_counts = {
                '<t,f>': 0, '<f,t>': 0, '<t,t>': 0, '<f,f>': 0,
                '<t,e>': 0, '<e,t>': 0, '<f,e>': 0, '<e,f>': 0, '<e,e>': 0
            }
            
            detailed_results = result.get('detailed_results', [])
            total = len(detailed_results)
            
            for item in detailed_results:
                bilateral_value = item.get('bilateral_value', '<e,e>')
                if bilateral_value in tv_counts:
                    tv_counts[bilateral_value] += 1
            
            model_display = result['model'][:30]
            if total > 0:
                print(f"{benchmark_display:<15} {model_display:<30}", end="")
                for tv in ['<t,f>', '<f,t>', '<t,t>', '<f,f>', '<t,e>', '<e,t>', '<f,e>', '<e,f>', '<e,e>']:
                    pct = (tv_counts[tv] / total * 100) if total > 0 else 0
                    print(f" {pct:6.1f}%", end="")
                print()
    
    # Print model performance summary
    print("\n" + "="*120)
    print("MODEL PERFORMANCE SUMMARY (Averaged Across Benchmarks)")
    print("="*120)
    print(f"{'Model':<35} {'Benchmarks':<12} {'Avg Accuracy':<12} {'Avg Coverage':<12} {'Avg F1':<10}")
    print("-"*120)
    
    for model in sorted(models):
        model_results = by_model[model]
        
        if model_results:
            accuracies = [r.get('accuracy', 0) for r in model_results]
            coverages = [r.get('coverage', 0) for r in model_results]
            f1_scores = [r.get('f1_macro', 0) for r in model_results]
            
            num_benchmarks = len(model_results)
            avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
            avg_coverage = sum(coverages) / len(coverages) if coverages else 0
            avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
            
            model_display = model[:35]
            print(f"{model_display:<35} {num_benchmarks:<12} {avg_accuracy:<12.3f} {avg_coverage:<12.3f} {avg_f1:<10.3f}")
    
    # Print category analysis if available
    print("\n" + "="*120)
    print("CATEGORY PERFORMANCE (Where Available)")
    print("="*120)
    
    has_categories = False
    for result in results:
        if result.get('category_analysis'):
            has_categories = True
            benchmark_display = benchmark_names.get(result['benchmark'], result['benchmark'])
            model_display = result['model'][:30]
            print(f"\n{benchmark_display} - {model_display}:")
            print(f"  {'Category':<30} {'Total':<10} {'Accuracy':<10} {'Coverage':<10}")
            print("  " + "-"*60)
            
            for category, stats in sorted(result['category_analysis'].items()):
                if isinstance(stats, dict) and 'total' in stats:
                    total = stats.get('total', 0)
                    accuracy = stats.get('accuracy', 0)
                    coverage = stats.get('coverage', 0)
                    category_display = category[:30]
                    print(f"  {category_display:<30} {total:<10} {accuracy:<10.3f} {coverage:<10.3f}")
    
    if not has_categories:
        print("  No category-level analysis available in results.")
    
    print("\n" + "="*120)
    print("END OF REPORT")
    print("="*120)

def save_summary_json(results: List[Dict], output_file: str = None) -> None:
    """Save summary data to JSON file."""
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results/summary_report_{timestamp}.json"
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_results": len(results),
        "models": sorted(list(set(r['model'] for r in results))),
        "benchmarks": sorted(list(set(r['benchmark'] for r in results))),
        "results": results
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary data saved to: {output_file}")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate summary report from evaluation results")
    parser.add_argument("--results-dir", default="results", help="Directory containing results JSON files")
    parser.add_argument("--save-json", action="store_true", help="Save summary data to JSON file")
    parser.add_argument("--output", help="Output file for JSON summary")
    args = parser.parse_args()
    
    # Load all results
    print(f"Loading results from {args.results_dir}/...")
    results = load_results_files(args.results_dir)
    
    if not results:
        print("No valid results files found!")
        return
    
    print(f"Loaded {len(results)} result files")
    
    # Generate report
    generate_summary_report(results)
    
    # Save JSON if requested
    if args.save_json:
        save_summary_json(results, args.output)

if __name__ == "__main__":
    main()