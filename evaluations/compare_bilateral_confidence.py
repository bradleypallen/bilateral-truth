#!/usr/bin/env python3
"""
Compare Bilateral vs Confidence-Based Unilateral Evaluation

This script compares bilateral evaluation results with confidence-based 
unilateral evaluation at multiple threshold levels.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_results(file_path: str) -> Dict:
    """Load evaluation results from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def compare_at_threshold(bilateral_results: Dict, confidence_results: Dict, 
                         threshold: float) -> Dict:
    """Compare bilateral and confidence results at a specific threshold."""
    
    # Get bilateral metrics
    bilateral_accuracy = bilateral_results.get('accuracy', 0)
    bilateral_coverage = bilateral_results.get('coverage', 0)
    bilateral_f1 = bilateral_results.get('f1_macro', 0)
    
    # Get confidence metrics at threshold
    threshold_key = str(threshold)
    if threshold_key not in confidence_results['threshold_analysis']:
        print(f"⚠️  Threshold {threshold} not found in confidence results")
        return {}
    
    confidence_stats = confidence_results['threshold_analysis'][threshold_key]
    confidence_accuracy = confidence_stats['accuracy']
    confidence_coverage = confidence_stats['coverage']
    confidence_f1 = confidence_stats['f1_macro']
    
    # Calculate differences
    accuracy_diff = bilateral_accuracy - confidence_accuracy
    coverage_diff = bilateral_coverage - confidence_coverage
    f1_diff = bilateral_f1 - confidence_f1
    
    # Calculate accuracy on answered (non-abstained) questions
    bilateral_acc_answered = bilateral_results.get('accuracy_on_answered', bilateral_accuracy)
    confidence_acc_answered = confidence_stats.get('accuracy_on_answered', confidence_accuracy)
    acc_answered_diff = bilateral_acc_answered - confidence_acc_answered
    
    return {
        'threshold': threshold,
        'bilateral': {
            'accuracy': bilateral_accuracy,
            'coverage': bilateral_coverage,
            'f1_macro': bilateral_f1,
            'accuracy_on_answered': bilateral_acc_answered
        },
        'confidence': {
            'accuracy': confidence_accuracy,
            'coverage': confidence_coverage,
            'f1_macro': confidence_f1,
            'accuracy_on_answered': confidence_acc_answered,
            'abstained': confidence_stats.get('abstained', 0)
        },
        'differences': {
            'accuracy': accuracy_diff,
            'coverage': coverage_diff,
            'f1_macro': f1_diff,
            'accuracy_on_answered': acc_answered_diff
        }
    }


def analyze_agreement(bilateral_results: Dict, confidence_results: Dict,
                      threshold: float) -> Dict:
    """Analyze agreement between bilateral and confidence evaluations."""
    
    threshold_key = str(threshold)
    
    # Get detailed results
    bilateral_detailed = bilateral_results.get('detailed_results', [])
    confidence_detailed = confidence_results.get('detailed_results', [])
    
    if not bilateral_detailed or not confidence_detailed:
        print("⚠️  Detailed results not available for agreement analysis")
        return {}
    
    # Match results by assertion
    agreement_stats = {
        'both_correct': 0,
        'both_incorrect': 0,
        'bilateral_correct_confidence_incorrect': 0,
        'bilateral_incorrect_confidence_correct': 0,
        'bilateral_abstained': 0,
        'confidence_abstained': 0,
        'both_abstained': 0,
        'total': 0
    }
    
    # Create assertion lookup for bilateral results
    bilateral_lookup = {r['assertion']: r for r in bilateral_detailed}
    
    for conf_result in confidence_detailed:
        assertion = conf_result['assertion']
        if assertion not in bilateral_lookup:
            continue
        
        bil_result = bilateral_lookup[assertion]
        agreement_stats['total'] += 1
        
        # Get predictions
        bil_prediction = bil_result.get('prediction', '')
        conf_prediction = conf_result['threshold_predictions'].get(threshold_key, '')
        
        # Check abstentions
        bil_abstained = bil_result.get('abstained', False)
        conf_abstained = (conf_prediction == 'ABSTAINED')
        
        if bil_abstained and conf_abstained:
            agreement_stats['both_abstained'] += 1
        elif bil_abstained:
            agreement_stats['bilateral_abstained'] += 1
        elif conf_abstained:
            agreement_stats['confidence_abstained'] += 1
        else:
            # Both made predictions - check agreement
            bil_correct = bil_result.get('is_correct', False)
            
            # Determine if confidence prediction is correct
            ground_truth = conf_result['ground_truth']
            conf_correct = (
                (conf_prediction == 'CORRECT' and ground_truth == 'correct') or
                (conf_prediction == 'INCORRECT' and ground_truth == 'incorrect')
            )
            
            if bil_correct and conf_correct:
                agreement_stats['both_correct'] += 1
            elif not bil_correct and not conf_correct:
                agreement_stats['both_incorrect'] += 1
            elif bil_correct and not conf_correct:
                agreement_stats['bilateral_correct_confidence_incorrect'] += 1
            else:
                agreement_stats['bilateral_incorrect_confidence_correct'] += 1
    
    # Calculate agreement rate
    if agreement_stats['total'] > 0:
        # Agreement when both make predictions
        non_abstained = (agreement_stats['both_correct'] + 
                        agreement_stats['both_incorrect'] +
                        agreement_stats['bilateral_correct_confidence_incorrect'] +
                        agreement_stats['bilateral_incorrect_confidence_correct'])
        
        if non_abstained > 0:
            agreement_rate = (agreement_stats['both_correct'] + 
                            agreement_stats['both_incorrect']) / non_abstained
        else:
            agreement_rate = 0
        
        agreement_stats['agreement_rate'] = agreement_rate
        agreement_stats['disagreement_rate'] = 1 - agreement_rate if non_abstained > 0 else 0
    
    return agreement_stats


def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(
        description='Compare bilateral vs confidence-based unilateral evaluation'
    )
    parser.add_argument('--bilateral', type=str, required=True,
                       help='Path to bilateral evaluation results JSON')
    parser.add_argument('--confidence', type=str, required=True,
                       help='Path to confidence evaluation results JSON')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for comparison results')
    
    args = parser.parse_args()
    
    # Load results
    print("📂 Loading evaluation results...")
    bilateral_results = load_results(args.bilateral)
    confidence_results = load_results(args.confidence)
    
    # Verify same dataset and model
    if bilateral_results['dataset'] != confidence_results['dataset']:
        print(f"⚠️  Warning: Different datasets - {bilateral_results['dataset']} vs {confidence_results['dataset']}")
    
    if bilateral_results['model'] != confidence_results['model']:
        print(f"⚠️  Warning: Different models - {bilateral_results['model']} vs {confidence_results['model']}")
    
    # Get thresholds from confidence results
    thresholds = confidence_results.get('thresholds', [0.5, 0.7, 0.9])
    
    # Compare at each threshold
    comparison_results = {
        'bilateral_file': args.bilateral,
        'confidence_file': args.confidence,
        'model': bilateral_results['model'],
        'dataset': bilateral_results['dataset'],
        'threshold_comparisons': []
    }
    
    print("\n" + "="*70)
    print("BILATERAL VS CONFIDENCE-BASED COMPARISON")
    print("="*70)
    print(f"Model: {bilateral_results['model']}")
    print(f"Dataset: {bilateral_results['dataset']}")
    print(f"Total Samples: {bilateral_results.get('total_samples', 'N/A')}")
    
    # Print bilateral baseline
    print("\n" + "-"*70)
    print("BILATERAL EVALUATION (Baseline):")
    print(f"  Accuracy: {bilateral_results.get('accuracy', 0):.3f}")
    print(f"  Coverage: {bilateral_results.get('coverage', 0):.3f}")
    print(f"  F1 Macro: {bilateral_results.get('f1_macro', 0):.3f}")
    
    acc_answered = bilateral_results.get('accuracy_on_answered', 
                                         bilateral_results.get('accuracy', 0))
    print(f"  Accuracy on Answered: {acc_answered:.3f}")
    
    # Compare at each threshold
    print("\n" + "-"*70)
    print("CONFIDENCE-BASED RESULTS BY THRESHOLD:")
    
    for threshold in thresholds:
        print(f"\n📊 Threshold: {threshold}")
        print("-"*35)
        
        # Get comparison metrics
        comparison = compare_at_threshold(bilateral_results, confidence_results, threshold)
        if not comparison:
            continue
        
        comparison_results['threshold_comparisons'].append(comparison)
        
        # Print confidence metrics
        conf = comparison['confidence']
        print(f"  Accuracy: {conf['accuracy']:.3f}")
        print(f"  Coverage: {conf['coverage']:.3f}")
        print(f"  F1 Macro: {conf['f1_macro']:.3f}")
        print(f"  Accuracy on Answered: {conf['accuracy_on_answered']:.3f}")
        print(f"  Abstained: {conf['abstained']}")
        
        # Print differences
        diff = comparison['differences']
        print(f"\n  Differences (Bilateral - Confidence):")
        print(f"    Accuracy: {diff['accuracy']:+.3f}")
        print(f"    Coverage: {diff['coverage']:+.3f}")
        print(f"    F1 Macro: {diff['f1_macro']:+.3f}")
        print(f"    Acc on Answered: {diff['accuracy_on_answered']:+.3f}")
        
        # Analyze agreement
        agreement = analyze_agreement(bilateral_results, confidence_results, threshold)
        if agreement:
            comparison['agreement'] = agreement
            if 'agreement_rate' in agreement:
                print(f"\n  Agreement Analysis:")
                print(f"    Agreement Rate: {agreement['agreement_rate']:.3f}")
                print(f"    Both Correct: {agreement['both_correct']}")
                print(f"    Both Incorrect: {agreement['both_incorrect']}")
                print(f"    Disagreements: {agreement['bilateral_correct_confidence_incorrect'] + agreement['bilateral_incorrect_confidence_correct']}")
    
    # Print confidence distribution
    if 'confidence_distribution' in confidence_results:
        print("\n" + "-"*70)
        print("CONFIDENCE DISTRIBUTION:")
        dist = confidence_results['confidence_distribution']
        total = confidence_results.get('total_samples', 1)
        
        for bucket in sorted(dist.keys()):
            count = dist[bucket]
            percentage = (count / total * 100) if total > 0 else 0
            bar = '█' * int(percentage / 2)
            print(f"  {bucket}: {bar} {count} ({percentage:.1f}%)")
    
    # Print average confidence
    if 'average_confidence' in confidence_results:
        print(f"\n  Average Confidence: {confidence_results['average_confidence']:.3f}")
    
    print("\n" + "="*70)
    
    # Summary recommendations
    print("\n📋 SUMMARY:")
    print("-"*70)
    
    # Find best threshold based on F1 score
    best_threshold = None
    best_f1 = -1
    for comp in comparison_results['threshold_comparisons']:
        if comp['confidence']['f1_macro'] > best_f1:
            best_f1 = comp['confidence']['f1_macro']
            best_threshold = comp['threshold']
    
    if best_threshold:
        print(f"✓ Best confidence threshold (by F1): {best_threshold} (F1={best_f1:.3f})")
    
    # Compare best confidence to bilateral
    bilateral_f1 = bilateral_results.get('f1_macro', 0)
    if bilateral_f1 > best_f1:
        print(f"✓ Bilateral evaluation outperforms best confidence threshold")
        print(f"  Bilateral F1: {bilateral_f1:.3f} vs Best Confidence F1: {best_f1:.3f}")
    else:
        print(f"✓ Confidence threshold {best_threshold} matches or exceeds bilateral")
        print(f"  Confidence F1: {best_f1:.3f} vs Bilateral F1: {bilateral_f1:.3f}")
    
    # Coverage comparison
    bilateral_coverage = bilateral_results.get('coverage', 0)
    print(f"\n✓ Coverage comparison:")
    for comp in comparison_results['threshold_comparisons']:
        threshold = comp['threshold']
        conf_coverage = comp['confidence']['coverage']
        print(f"  Threshold {threshold}: {conf_coverage:.3f} vs Bilateral: {bilateral_coverage:.3f}")
    
    # Save results if output specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        print(f"\n💾 Comparison results saved to {args.output}")


if __name__ == "__main__":
    main()