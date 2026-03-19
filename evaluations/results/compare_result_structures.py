#!/usr/bin/env python3
"""
Compare the structure of result files from different evaluation approaches.
"""

import json
from pathlib import Path

# Sample one file from each approach type
approaches = {
    'Bilateral (Classical)': 'truthfulqa_complete_gpt-4.1-2025-04-14_classical_results.json',
    'Unilateral-Binary (Direct)': 'truthfulqa_complete_gpt-4.1-2025-04-14_unilateral_direct_results.json', 
    'Unilateral-Ternary (Uncertain)': 'truthfulqa_complete_gpt-4.1-2025-04-14_unilateral_uncertain_results.json',
    'Confidence': 'truthfulqa_complete_gpt-4.1-2025-04-14_unilateral_confidence_results.json'
}

for approach, filepath in approaches.items():
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        print(f'\n{"="*60}')
        print(f'{approach}')
        print(f'File: {filepath}')
        print("="*60)
        
        # Show all top-level keys
        print('\nTop-level keys:')
        for key in data.keys():
            if key != 'detailed_results':  # Skip the large detailed results
                value_type = type(data[key]).__name__
                if isinstance(data[key], (str, int, float, bool)):
                    print(f'  {key}: {data[key]}')
                elif isinstance(data[key], list):
                    print(f'  {key}: list with {len(data[key])} items')
                elif isinstance(data[key], dict):
                    print(f'  {key}: dict with keys {list(data[key].keys())[:5]}...')
                else:
                    print(f'  {key}: {value_type}')
        
        # Check detailed results structure
        if 'detailed_results' in data:
            print(f'\ndetailed_results: list with {len(data["detailed_results"])} items')
            if len(data['detailed_results']) > 0:
                sample = data['detailed_results'][0]
                print(f'  Sample result structure:')
                for key, value in sample.items():
                    if isinstance(value, (str, int, float, bool, type(None))):
                        print(f'    {key}: {value}')
                    elif isinstance(value, dict):
                        print(f'    {key}: dict with keys {list(value.keys())}')
                    elif isinstance(value, list):
                        print(f'    {key}: list with {len(value)} items')
                    else:
                        print(f'    {key}: {type(value).__name__}')
        
        # Special structures for different approaches
        if approach == 'Bilateral (Classical)':
            if 'bilateral_distribution' in data:
                dist = data['bilateral_distribution']
                print(f'\nBilateral truth value distribution:')
                for tv, count in dist.items():
                    print(f'  {tv}: {count}')
                    
        elif approach == 'Confidence':
            if 'threshold_analysis' in data:
                print(f'\nThreshold analysis for: {list(data["threshold_analysis"].keys())}')
                for threshold, metrics in data['threshold_analysis'].items():
                    print(f'  Threshold {threshold}:')
                    print(f'    F1: {metrics.get("f1_macro", "N/A"):.3f}')
                    print(f'    Coverage: {metrics.get("coverage", "N/A"):.1%}')
                    print(f'    Accuracy: {metrics.get("accuracy", "N/A"):.1%}')
            
            if 'confidence_distribution' in data:
                dist = data['confidence_distribution']
                print(f'\nConfidence distribution:')
                print(f'  Mean: {dist.get("mean", "N/A"):.3f}')
                print(f'  Std: {dist.get("std", "N/A"):.3f}')
                print(f'  Min: {dist.get("min", "N/A"):.3f}')
                print(f'  Max: {dist.get("max", "N/A"):.3f}')
                
    except FileNotFoundError:
        print(f'\n{approach}: File not found - {filepath}')
    except Exception as e:
        print(f'\n{approach}: Error - {e}')

print("\n" + "="*60)
print("KEY DIFFERENCES SUMMARY")
print("="*60)

print("""
1. BILATERAL (Classical):
   - Has 'epistemic_policy' field (classical/paracomplete/paraconsistent)
   - Contains 'bilateral_distribution' with truth value pairs (<t,f>, <f,t>, etc.)
   - Contains 'projected_distribution' (P+ and P- separately)
   - Each result has 'bilateral_truth_value' field

2. UNILATERAL-BINARY (Direct):
   - Has 'prompt_style': 'direct'
   - Forces binary CORRECT/INCORRECT answers
   - Coverage always 100% (no abstention)
   - Each result has 'prediction' field (CORRECT/INCORRECT)

3. UNILATERAL-TERNARY (Uncertain):
   - Has 'prompt_style': 'uncertain'
   - Allows CORRECT/INCORRECT/UNCERTAIN answers
   - Variable coverage based on UNCERTAIN responses
   - Each result has 'prediction' field (CORRECT/INCORRECT/UNCERTAIN)

4. CONFIDENCE:
   - Has 'evaluation_type': 'confidence'
   - Contains 'thresholds' list (e.g., [0.5, 0.7, 0.9])
   - Has 'threshold_analysis' with metrics for each threshold
   - Contains 'confidence_distribution' statistics
   - Each result has 'confidence' score (0.0-1.0)
   - Coverage varies by threshold (higher threshold = lower coverage)
""")