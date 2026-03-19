#!/usr/bin/env python3
"""Check if all result files have exactly 1000 samples processed."""

import json
import glob
from collections import defaultdict

# Check all result files
results = defaultdict(lambda: defaultdict(dict))

for pattern in ['*_classical_results.json', '*_unilateral_direct_results.json', '*_unilateral_uncertain_results.json']:
    for file in glob.glob(f'results/{pattern}'):
        try:
            with open(file) as f:
                data = json.load(f)
            
            # Extract info
            model = data.get('model', 'unknown')
            model_short = model.split('/')[-1] if '/' in model else model
            total = data.get('total_samples', 0)
            
            # Determine dataset and method from filename
            filename = file.split('/')[-1]
            parts = filename.split('_')
            
            if 'classical' in filename:
                method = 'bilateral'
            elif 'direct' in filename:
                method = 'unilateral_forced'
            elif 'uncertain' in filename:
                method = 'unilateral_uncertain'
            else:
                method = 'unknown'
            
            # Extract dataset
            dataset = parts[0]
            if len(parts) > 1 and parts[1] == 'complete':
                dataset = parts[0]
            
            results[model_short][dataset][method] = total
            
        except Exception as e:
            print(f'Error with {file}: {e}')

# Print summary table
print('\nSAMPLE COUNTS BY MODEL, DATASET, AND METHOD')
print('=' * 100)

datasets = ['truthfulqa', 'simpleqa', 'mmlupro', 'factscore']
methods = ['bilateral', 'unilateral_forced', 'unilateral_uncertain']

# Track which have issues
issues = []

for model in sorted(results.keys()):
    print(f'\n{model}:')
    print('-' * 80)
    header = f"{'Dataset':<15} | {'Bilateral':<12} | {'Forced Uni':<12} | {'Uncertain Uni':<12}"
    print(header)
    print('-' * 80)
    
    for dataset in datasets:
        counts = []
        for method in methods:
            count = results[model].get(dataset, {}).get(method, 'N/A')
            counts.append(str(count))
        
        # Highlight if any count != 1000
        row = f'{dataset:<15} | {counts[0]:<12} | {counts[1]:<12} | {counts[2]:<12}'
        if any(c != '1000' and c != 'N/A' for c in counts):
            row += ' ⚠️'
            for i, (method, count) in enumerate(zip(methods, counts)):
                if count != '1000' and count != 'N/A':
                    issues.append(f"{model} - {dataset} - {method}: {count} samples")
        print(row)

print('\n' + '=' * 100)
print('ISSUES FOUND (not 1000 samples):')
print('=' * 100)

if issues:
    for issue in issues:
        print(f'  ⚠️ {issue}')
else:
    print('  ✅ All evaluations have exactly 1000 samples!')

# Also check for missing evaluations
print('\n' + '=' * 100)
print('MISSING EVALUATIONS:')
print('=' * 100)

missing = []
for model in results:
    for dataset in datasets:
        for method in methods:
            if results[model].get(dataset, {}).get(method, 'N/A') == 'N/A':
                missing.append(f"{model} - {dataset} - {method}")

if missing:
    for m in missing:
        print(f'  ❌ {m}')
else:
    print('  ✅ No missing evaluations!')