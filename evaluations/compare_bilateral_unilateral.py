#!/usr/bin/env python3
"""
Compare Bilateral vs Unilateral Evaluation Results

This script compares the results from bilateral evaluation (with different epistemic policies)
against true forced-choice unilateral evaluation to validate theoretical predictions from
ArXiv paper 2507.09751v2.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from collections import defaultdict


def load_results(results_dir: Path, model: str, dataset: str) -> Dict:
    """Load all evaluation results for a given model and dataset."""
    results = {}
    
    # Pattern for bilateral results: {dataset}_{model}_{policy}_results.json
    # Pattern for unilateral results: {dataset}_{model}_unilateral_{style}_results.json
    
    model_safe = model.replace('/', '_').replace(':', '_')
    
    # Load bilateral results
    for policy in ['classical', 'paracomplete', 'paraconsistent']:
        bilateral_file = results_dir / f"{dataset}_{model_safe}_{policy}_results.json"
        if bilateral_file.exists():
            with open(bilateral_file, 'r') as f:
                results[f'bilateral_{policy}'] = json.load(f)
    
    # Load unilateral results
    for style in ['direct', 'zero-shot', 'few-shot']:
        unilateral_file = results_dir / f"{dataset}_{model_safe}_unilateral_{style}_results.json"
        if unilateral_file.exists():
            with open(unilateral_file, 'r') as f:
                results[f'unilateral_{style}'] = json.load(f)
    
    # Load verification-only (what we previously called "unilateral")
    # This is actually bilateral classical with only verification used
    if f'bilateral_classical' in results:
        # Extract verification-only metrics from bilateral classical
        bilateral_classical = results['bilateral_classical']
        verification_only = calculate_verification_only_metrics(bilateral_classical)
        if verification_only:
            results['verification_only'] = verification_only
    
    return results


def calculate_verification_only_metrics(bilateral_results: Dict) -> Optional[Dict]:
    """Calculate metrics using only the verification component of bilateral evaluation."""
    if 'detailed_results' not in bilateral_results:
        return None
    
    verification_only = {
        'evaluation_type': 'verification_only',
        'total_samples': 0,
        'correct': 0,
        'incorrect': 0,
        'abstained': 0,
        'detailed_results': []
    }
    
    for item in bilateral_results['detailed_results']:
        bilateral_value = item.get('bilateral_value', '<e,e>')
        ground_truth = item.get('expected_label', item.get('ground_truth'))
        
        # Extract verification component (first element)
        if bilateral_value.startswith('<t,'):
            prediction = 'correct'  # Map to expected_label format
        elif bilateral_value.startswith('<f,'):
            prediction = 'incorrect'
        else:
            prediction = 'undefined'
        
        # Check correctness
        if prediction == 'undefined':
            verification_only['abstained'] += 1
            is_correct = False  # Count as incorrect for accuracy
        else:
            is_correct = (prediction == ground_truth)
        
        verification_only['total_samples'] += 1
        if is_correct:
            verification_only['correct'] += 1
        else:
            verification_only['incorrect'] += 1
        
        verification_only['detailed_results'].append({
            'ground_truth': ground_truth,
            'prediction': prediction,
            'is_correct': is_correct
        })
    
    # Calculate metrics
    total = verification_only['total_samples']
    if total > 0:
        verification_only['accuracy'] = verification_only['correct'] / total
        verification_only['coverage'] = (total - verification_only['abstained']) / total
        
        # Calculate F1
        tp = sum(1 for r in verification_only['detailed_results'] 
                 if r['prediction'] == 'correct' and r['ground_truth'] == 'correct')
        fp = sum(1 for r in verification_only['detailed_results']
                 if r['prediction'] == 'correct' and r['ground_truth'] == 'incorrect')
        tn = sum(1 for r in verification_only['detailed_results']
                 if r['prediction'] == 'incorrect' and r['ground_truth'] == 'incorrect')
        fn = sum(1 for r in verification_only['detailed_results']
                 if r['prediction'] == 'incorrect' and r['ground_truth'] == 'correct')
        
        precision_true = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_true = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_true = 2 * precision_true * recall_true / (precision_true + recall_true) \
                  if (precision_true + recall_true) > 0 else 0
        
        precision_false = tn / (tn + fn) if (tn + fn) > 0 else 0
        recall_false = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_false = 2 * precision_false * recall_false / (precision_false + recall_false) \
                   if (precision_false + recall_false) > 0 else 0
        
        verification_only['f1_macro'] = (f1_true + f1_false) / 2
    
    return verification_only


def create_comparison_table(results: Dict) -> pd.DataFrame:
    """Create a comparison table of all evaluation methods."""
    rows = []
    
    for method, data in results.items():
        if isinstance(data, dict) and 'accuracy' in data:
            row = {
                'Method': method,
                'Accuracy': data.get('accuracy', 0),
                'Coverage': data.get('coverage', 0),
                'F1 Macro': data.get('f1_macro', 0),
                'Total Samples': data.get('total_samples', 0)
            }
            
            # Add bilateral-specific metrics
            if 'bilateral_distribution' in data:
                dist = data['bilateral_distribution']
                row['<t,t> Rate'] = dist.get('<t,t>', 0) / data['total_samples'] if data['total_samples'] > 0 else 0
                row['<f,f> Rate'] = dist.get('<f,f>', 0) / data['total_samples'] if data['total_samples'] > 0 else 0
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('F1 Macro', ascending=False)
    
    return df


def analyze_performance_differences(results: Dict) -> Dict:
    """Analyze key performance differences between evaluation methods."""
    analysis = {}
    
    # Compare unilateral vs bilateral classical
    if 'unilateral_direct' in results and 'bilateral_classical' in results:
        uni = results['unilateral_direct']
        bi = results['bilateral_classical']
        
        analysis['unilateral_vs_bilateral_classical'] = {
            'accuracy_diff': uni['accuracy'] - bi['accuracy'],
            'coverage_diff': uni['coverage'] - bi['coverage'],
            'f1_diff': uni['f1_macro'] - bi['f1_macro'],
            'interpretation': 'Positive values indicate unilateral performs better'
        }
    
    # Compare unilateral vs verification-only
    if 'unilateral_direct' in results and 'verification_only' in results:
        uni = results['unilateral_direct']
        ver = results['verification_only']
        
        analysis['unilateral_vs_verification_only'] = {
            'accuracy_diff': uni['accuracy'] - ver['accuracy'],
            'coverage_diff': uni['coverage'] - ver['coverage'],
            'f1_diff': uni['f1_macro'] - ver['f1_macro'],
            'interpretation': 'Shows impact of forcing choice vs allowing "cannot verify"'
        }
    
    # Compare verification-only vs bilateral policies
    if 'verification_only' in results:
        ver = results['verification_only']
        for policy in ['classical', 'paracomplete', 'paraconsistent']:
            key = f'bilateral_{policy}'
            if key in results:
                bi = results[key]
                analysis[f'verification_only_vs_{policy}'] = {
                    'accuracy_diff': ver['accuracy'] - bi['accuracy'],
                    'coverage_diff': ver['coverage'] - bi['coverage'],
                    'f1_diff': ver['f1_macro'] - bi['f1_macro'],
                    'interpretation': 'Shows value of refutation dimension'
                }
    
    return analysis


def print_comparison_report(model: str, dataset: str, results: Dict, analysis: Dict):
    """Print a formatted comparison report."""
    print("\n" + "="*80)
    print(f"BILATERAL VS UNILATERAL COMPARISON REPORT")
    print("="*80)
    print(f"Model: {model}")
    print(f"Dataset: {dataset}")
    print(f"Available evaluations: {', '.join(results.keys())}")
    print()
    
    # Performance table
    df = create_comparison_table(results)
    if not df.empty:
        print("\nPERFORMANCE COMPARISON")
        print("-"*80)
        print(df.to_string(index=False, float_format='%.3f'))
    
    # Key findings
    print("\n\nKEY FINDINGS")
    print("-"*80)
    
    # 1. Does forced unilateral beat bilateral classical?
    if 'unilateral_vs_bilateral_classical' in analysis:
        diff = analysis['unilateral_vs_bilateral_classical']
        print(f"\n1. Forced Unilateral vs Bilateral Classical:")
        print(f"   - Accuracy difference: {diff['accuracy_diff']:+.3f}")
        print(f"   - Coverage difference: {diff['coverage_diff']:+.3f}")
        print(f"   - F1 difference: {diff['f1_diff']:+.3f}")
        
        if diff['f1_diff'] > 0:
            print("   ✓ Forced unilateral OUTPERFORMS bilateral classical")
            print("     This CONTRADICTS the ArXiv paper's findings")
        else:
            print("   ✓ Bilateral classical OUTPERFORMS forced unilateral")
            print("     This SUPPORTS the ArXiv paper's findings")
    
    # 2. Impact of allowing uncertainty
    if 'unilateral_vs_verification_only' in analysis:
        diff = analysis['unilateral_vs_verification_only']
        print(f"\n2. Impact of Allowing Uncertainty (Forced vs Verification-only):")
        print(f"   - Accuracy difference: {diff['accuracy_diff']:+.3f}")
        print(f"   - Coverage difference: {diff['coverage_diff']:+.3f}")
        print(f"   - F1 difference: {diff['f1_diff']:+.3f}")
        
        if diff['f1_diff'] < 0:
            print("   ✓ Allowing 'cannot verify' IMPROVES performance")
            print("     Epistemic uncertainty is valuable")
        else:
            print("   ✓ Forcing choice IMPROVES performance")
            print("     Epistemic uncertainty may be harmful")
    
    # 3. Value of refutation dimension
    if 'verification_only_vs_paracomplete' in analysis:
        diff = analysis['verification_only_vs_paracomplete']
        print(f"\n3. Value of Refutation Dimension (Verification-only vs Paracomplete):")
        print(f"   - Accuracy difference: {diff['accuracy_diff']:+.3f}")
        print(f"   - F1 difference: {diff['f1_diff']:+.3f}")
        
        if diff['f1_diff'] < 0:
            print("   ✓ Adding refutation IMPROVES performance")
            print("     Bilateral evaluation adds value")
        else:
            print("   ✓ Verification alone is SUFFICIENT")
            print("     Refutation dimension may be redundant")
    
    print("\n" + "="*80)


def main():
    """Main entry point for comparison analysis."""
    parser = argparse.ArgumentParser(description='Compare bilateral vs unilateral evaluation')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory containing evaluation results')
    parser.add_argument('--model', type=str, required=True,
                       help='Model name to analyze')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., truthfulqa_complete)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for comparison report (optional)')
    
    args = parser.parse_args()
    
    # Load results
    results_dir = Path(args.results_dir)
    results = load_results(results_dir, args.model, args.dataset)
    
    if not results:
        print(f"No results found for model={args.model}, dataset={args.dataset}")
        return
    
    # Analyze differences
    analysis = analyze_performance_differences(results)
    
    # Print report
    print_comparison_report(args.model, args.dataset, results, analysis)
    
    # Save report if requested
    if args.output:
        report = {
            'model': args.model,
            'dataset': args.dataset,
            'results': results,
            'analysis': analysis,
            'comparison_table': create_comparison_table(results).to_dict()
        }
        
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()