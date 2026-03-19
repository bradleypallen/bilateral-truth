#!/usr/bin/env python3
"""
Run Comprehensive Confidence-Based Comparison Suite

This script runs bilateral and confidence-based unilateral evaluations
on multiple benchmarks and models, then compares the results.
"""

import json
import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import time

# Default evaluation configuration
DEFAULT_CONFIG = {
    'models': ['gpt-4o-mini', 'claude-3-5-haiku-20241022'],
    'datasets': [
        'standard_datasets/truthfulqa_complete.json',
        'standard_datasets/simpleqa_complete.json'
    ],
    'sample_size': 100,
    'thresholds': [0.5, 0.6, 0.7, 0.8, 0.9],
    'output_dir': 'results'
}


def run_bilateral_evaluation(model: str, dataset: str, samples: int, output_dir: str) -> str:
    """Run bilateral evaluation and return results file path."""
    print(f"\n🔄 Running bilateral evaluation: {model} on {Path(dataset).stem}")
    
    cmd = [
        'python', 'generic_evaluator.py',
        '--dataset', dataset,
        '--model', model,
        '--samples', str(samples),
        '--output-dir', output_dir
    ]
    
    try:
        subprocess.run(cmd, check=True)
        # Construct expected output filename - generic_evaluator adds "_classical" by default
        dataset_name = Path(dataset).stem
        model_safe = model.replace('/', '_').replace(':', '_')
        return f"{output_dir}/{dataset_name}_{model_safe}_classical_results.json"
    except subprocess.CalledProcessError as e:
        print(f"❌ Bilateral evaluation failed: {e}")
        return None


def run_confidence_evaluation(model: str, dataset: str, samples: int, 
                            thresholds: List[float], output_dir: str) -> str:
    """Run confidence-based unilateral evaluation and return results file path."""
    print(f"\n🔄 Running confidence evaluation: {model} on {Path(dataset).stem}")
    
    cmd = [
        'python', 'unilateral_confidence_evaluator.py',
        '--dataset', dataset,
        '--model', model,
        '--samples', str(samples),
        '--thresholds'] + [str(t) for t in thresholds] + [
        '--output-dir', output_dir
    ]
    
    try:
        subprocess.run(cmd, check=True)
        # Construct expected output filename
        dataset_name = Path(dataset).stem
        model_safe = model.replace('/', '_').replace(':', '_')
        return f"{output_dir}/{dataset_name}_{model_safe}_unilateral_confidence_results.json"
    except subprocess.CalledProcessError as e:
        print(f"❌ Confidence evaluation failed: {e}")
        return None


def run_comparison(bilateral_file: str, confidence_file: str, output_dir: str) -> Dict:
    """Run comparison between bilateral and confidence results."""
    print(f"\n📊 Comparing results...")
    
    # Create comparison output filename
    base_name = Path(bilateral_file).stem.replace('_results', '')
    output_file = f"{output_dir}/{base_name}_confidence_comparison.json"
    
    cmd = [
        'python', 'compare_bilateral_confidence.py',
        '--bilateral', bilateral_file,
        '--confidence', confidence_file,
        '--output', output_file
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        
        # Load and return comparison results
        with open(output_file, 'r') as f:
            return json.load(f)
    except subprocess.CalledProcessError as e:
        print(f"❌ Comparison failed: {e}")
        print(f"Error output: {e.stderr}")
        return None


def generate_summary_report(all_comparisons: List[Dict], output_dir: str):
    """Generate a summary report across all comparisons."""
    
    summary = {
        'generation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_comparisons': len(all_comparisons),
        'by_model': {},
        'by_dataset': {},
        'overall_insights': []
    }
    
    # Aggregate by model and dataset
    for comp in all_comparisons:
        if not comp:
            continue
            
        model = comp['model']
        dataset = comp['dataset']
        
        # Initialize structures
        if model not in summary['by_model']:
            summary['by_model'][model] = {
                'datasets_evaluated': [],
                'avg_bilateral_accuracy': [],
                'avg_bilateral_coverage': [],
                'best_threshold_by_f1': {},
                'confidence_insights': []
            }
        
        if dataset not in summary['by_dataset']:
            summary['by_dataset'][dataset] = {
                'models_evaluated': [],
                'avg_bilateral_accuracy': [],
                'avg_bilateral_coverage': [],
                'best_models': {}
            }
        
        # Extract key metrics
        bilateral_file = comp.get('bilateral_file')
        if bilateral_file and Path(bilateral_file).exists():
            with open(bilateral_file, 'r') as f:
                bilateral_results = json.load(f)
            
            # Update model summary
            summary['by_model'][model]['datasets_evaluated'].append(dataset)
            summary['by_model'][model]['avg_bilateral_accuracy'].append(
                bilateral_results.get('accuracy', 0)
            )
            summary['by_model'][model]['avg_bilateral_coverage'].append(
                bilateral_results.get('coverage', 0)
            )
            
            # Find best threshold for this model/dataset
            best_f1 = -1
            best_threshold = None
            for threshold_comp in comp.get('threshold_comparisons', []):
                f1 = threshold_comp['confidence']['f1_macro']
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold_comp['threshold']
            
            if best_threshold:
                summary['by_model'][model]['best_threshold_by_f1'][dataset] = {
                    'threshold': best_threshold,
                    'f1': best_f1
                }
            
            # Update dataset summary
            summary['by_dataset'][dataset]['models_evaluated'].append(model)
            summary['by_dataset'][dataset]['avg_bilateral_accuracy'].append(
                bilateral_results.get('accuracy', 0)
            )
            summary['by_dataset'][dataset]['avg_bilateral_coverage'].append(
                bilateral_results.get('coverage', 0)
            )
    
    # Calculate averages
    for model, data in summary['by_model'].items():
        if data['avg_bilateral_accuracy']:
            data['avg_bilateral_accuracy'] = sum(data['avg_bilateral_accuracy']) / len(data['avg_bilateral_accuracy'])
            data['avg_bilateral_coverage'] = sum(data['avg_bilateral_coverage']) / len(data['avg_bilateral_coverage'])
        else:
            data['avg_bilateral_accuracy'] = 0
            data['avg_bilateral_coverage'] = 0
    
    for dataset, data in summary['by_dataset'].items():
        if data['avg_bilateral_accuracy']:
            data['avg_bilateral_accuracy'] = sum(data['avg_bilateral_accuracy']) / len(data['avg_bilateral_accuracy'])
            data['avg_bilateral_coverage'] = sum(data['avg_bilateral_coverage']) / len(data['avg_bilateral_coverage'])
        else:
            data['avg_bilateral_accuracy'] = 0
            data['avg_bilateral_coverage'] = 0
    
    # Generate insights
    summary['overall_insights'] = [
        "Confidence-based evaluation provides flexible coverage-accuracy tradeoffs",
        "Higher thresholds generally reduce coverage but may improve accuracy on answered questions",
        "Bilateral evaluation typically provides a balanced approach without requiring threshold tuning"
    ]
    
    # Save summary
    summary_file = f"{output_dir}/confidence_comparison_summary_{int(time.time())}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📋 Summary report saved to {summary_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("OVERALL SUMMARY - BILATERAL VS CONFIDENCE COMPARISON")
    print("="*70)
    print(f"Total Comparisons: {summary['total_comparisons']}")
    
    print("\n📊 By Model:")
    for model, data in summary['by_model'].items():
        print(f"\n  {model}:")
        print(f"    Datasets: {', '.join(data['datasets_evaluated'])}")
        print(f"    Avg Bilateral Accuracy: {data['avg_bilateral_accuracy']:.3f}")
        print(f"    Avg Bilateral Coverage: {data['avg_bilateral_coverage']:.3f}")
        if data['best_threshold_by_f1']:
            print(f"    Best Thresholds by Dataset:")
            for ds, info in data['best_threshold_by_f1'].items():
                print(f"      {ds}: {info['threshold']} (F1={info['f1']:.3f})")
    
    print("\n📊 By Dataset:")
    for dataset, data in summary['by_dataset'].items():
        print(f"\n  {dataset}:")
        print(f"    Models: {', '.join(data['models_evaluated'])}")
        print(f"    Avg Bilateral Accuracy: {data['avg_bilateral_accuracy']:.3f}")
        print(f"    Avg Bilateral Coverage: {data['avg_bilateral_coverage']:.3f}")
    
    print("\n💡 Key Insights:")
    for insight in summary['overall_insights']:
        print(f"  • {insight}")
    
    print("="*70)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Run comprehensive confidence-based comparison suite'
    )
    parser.add_argument('--models', type=str, nargs='+',
                       default=DEFAULT_CONFIG['models'],
                       help='Models to evaluate')
    parser.add_argument('--datasets', type=str, nargs='+',
                       default=DEFAULT_CONFIG['datasets'],
                       help='Dataset paths to evaluate')
    parser.add_argument('--samples', type=int,
                       default=DEFAULT_CONFIG['sample_size'],
                       help='Number of samples per evaluation')
    parser.add_argument('--thresholds', type=float, nargs='+',
                       default=DEFAULT_CONFIG['thresholds'],
                       help='Confidence thresholds to test')
    parser.add_argument('--output-dir', type=str,
                       default=DEFAULT_CONFIG['output_dir'],
                       help='Output directory for results')
    parser.add_argument('--skip-bilateral', action='store_true',
                       help='Skip bilateral evaluation if results already exist')
    parser.add_argument('--skip-confidence', action='store_true',
                       help='Skip confidence evaluation if results already exist')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("🚀 Starting Confidence-Based Comparison Suite")
    print(f"Models: {args.models}")
    print(f"Datasets: {[Path(d).stem for d in args.datasets]}")
    print(f"Sample size: {args.samples}")
    print(f"Thresholds: {args.thresholds}")
    
    all_comparisons = []
    
    # Run evaluations for each model and dataset combination
    for model in args.models:
        for dataset in args.datasets:
            print(f"\n{'='*60}")
            print(f"Processing: {model} on {Path(dataset).stem}")
            print('='*60)
            
            # Run bilateral evaluation
            bilateral_file = None
            if not args.skip_bilateral:
                bilateral_file = run_bilateral_evaluation(
                    model, dataset, args.samples, args.output_dir
                )
            else:
                # Check if file exists
                dataset_name = Path(dataset).stem
                model_safe = model.replace('/', '_').replace(':', '_')
                expected_file = f"{args.output_dir}/{dataset_name}_{model_safe}_classical_results.json"
                if Path(expected_file).exists():
                    bilateral_file = expected_file
                    print(f"✓ Using existing bilateral results: {bilateral_file}")
            
            # Run confidence evaluation
            confidence_file = None
            if not args.skip_confidence:
                confidence_file = run_confidence_evaluation(
                    model, dataset, args.samples, args.thresholds, args.output_dir
                )
            else:
                # Check if file exists
                dataset_name = Path(dataset).stem
                model_safe = model.replace('/', '_').replace(':', '_')
                expected_file = f"{args.output_dir}/{dataset_name}_{model_safe}_unilateral_confidence_results.json"
                if Path(expected_file).exists():
                    confidence_file = expected_file
                    print(f"✓ Using existing confidence results: {confidence_file}")
            
            # Run comparison if both files exist
            if bilateral_file and confidence_file:
                comparison = run_comparison(bilateral_file, confidence_file, args.output_dir)
                if comparison:
                    all_comparisons.append(comparison)
            else:
                print(f"⚠️  Skipping comparison - missing results files")
    
    # Generate summary report
    if all_comparisons:
        generate_summary_report(all_comparisons, args.output_dir)
    
    print("\n✅ Confidence comparison suite completed!")


if __name__ == "__main__":
    main()