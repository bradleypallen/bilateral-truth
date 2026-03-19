#!/usr/bin/env python3
"""
Generate comprehensive comparison table with F1, coverage, and execution time.
"""

import json
import glob
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

def load_result_file(filepath: str) -> Dict:
    """Load a single result file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_metrics(data: Dict) -> Tuple[float, float, float]:
    """Extract F1, coverage, and execution time from result data."""
    f1 = data.get('f1_macro', 0.0)
    coverage = data.get('coverage', 0.0)
    exec_time = data.get('evaluation_time', None)  # None if not present
    return f1, coverage, exec_time

def get_approach_name(filename: str) -> str:
    """Determine approach name from filename."""
    if 'classical' in filename:
        return 'Bilateral'
    elif 'unilateral_direct' in filename:
        return 'Unilateral-Binary'
    elif 'unilateral_uncertain' in filename:
        return 'Unilateral-Ternary'
    elif 'unilateral_confidence' in filename:
        # Need to look at thresholds in the file
        return 'Confidence'
    else:
        return 'Unknown'

def process_confidence_results(filepath: str) -> List[Tuple[str, float, float, float]]:
    """Process confidence results which have multiple thresholds."""
    data = load_result_file(filepath)
    results = []
    
    # Check if it has threshold_analysis (the actual structure)
    if 'threshold_analysis' in data:
        exec_time = data.get('evaluation_time', 0.0)
        # Assuming time is roughly equal per threshold
        time_per_threshold = exec_time / len(data['threshold_analysis'])
        
        for threshold, metrics in data['threshold_analysis'].items():
            approach = f"Confidence-{threshold}"
            f1 = metrics.get('f1_macro', 0.0)
            coverage = metrics.get('coverage', 0.0)
            results.append((approach, f1, coverage, time_per_threshold))
    else:
        # Fallback to single result
        f1, coverage, exec_time = extract_metrics(data)
        results.append(('Confidence', f1, coverage, exec_time))
    
    return results

def main():
    # Model name mapping
    model_display_names = {
        'gpt-4.1-2025-04-14': 'GPT-4.1',
        'gpt-4.1-mini-2025-04-14': 'GPT-4.1-Mini',
        'claude-opus-4-1-20250805': 'Claude-Opus',
        'claude-3-5-haiku-20241022': 'Claude-Haiku',
        'meta-llama_llama-4-scout': 'Llama-Scout',
        'meta-llama_llama-4-maverick': 'Llama-Maverick',
        'google_gemini-2.5-flash': 'Gemini-Flash'
    }
    
    dataset_display_names = {
        'truthfulqa_complete': 'TruthfulQA',
        'simpleqa_complete': 'SimpleQA',
        'mmlupro_complete': 'MMLU-Pro',
        'factscore_complete': 'FACTScore'
    }
    
    # Collect all results
    all_results = {}
    
    # Process all result files
    for filepath in glob.glob('*_results.json'):
        filename = Path(filepath).name
        
        # Parse filename to get dataset and model
        parts = filename.replace('_results.json', '').split('_')
        
        # Find dataset (first part that's in our dataset list)
        dataset = None
        for i in range(len(parts)):
            potential_dataset = '_'.join(parts[:i+2])  # Try combining first i+2 parts
            if potential_dataset in dataset_display_names:
                dataset = potential_dataset
                remaining = '_'.join(parts[i+2:])
                break
        
        if not dataset:
            continue
            
        # Extract model from remaining part
        model = None
        for model_key in model_display_names:
            if model_key.replace('/', '_').replace('-', '-') in remaining:
                model = model_key
                break
        
        if not model:
            # Try alternative model name formats
            if 'claude-opus' in remaining:
                model = 'claude-opus-4-1-20250805'
            elif 'claude-3-5-haiku' in remaining:
                model = 'claude-3-5-haiku-20241022'
            elif 'gpt-4.1-mini' in remaining:
                model = 'gpt-4.1-mini-2025-04-14'
            elif 'gpt-4.1' in remaining and 'mini' not in remaining:
                model = 'gpt-4.1-2025-04-14'
            elif 'llama-4-scout' in remaining:
                model = 'meta-llama_llama-4-scout'
            elif 'llama-4-maverick' in remaining:
                model = 'meta-llama_llama-4-maverick'
            elif 'gemini-2.5-flash' in remaining or 'gemini-2_5-flash' in remaining:
                model = 'google_gemini-2.5-flash'
            else:
                continue
        
        # Process based on approach type
        if 'unilateral_confidence' in filename:
            results = process_confidence_results(filepath)
            for approach, f1, coverage, exec_time in results:
                key = (model, dataset, approach)
                all_results[key] = (f1, coverage, exec_time)
        else:
            data = load_result_file(filepath)
            approach = get_approach_name(filename)
            f1, coverage, exec_time = extract_metrics(data)
            key = (model, dataset, approach)
            all_results[key] = (f1, coverage, exec_time)
    
    # Note: Verification-Only results would need to be calculated separately
    # as they require P+ only evaluation which isn't in our current result files
    # For now, we'll skip Verification-Only since we don't have that data
    
    # Generate markdown table for each dataset
    print("# Comprehensive Model/Approach Comparison")
    print("\n**Metrics: F1-Macro | Coverage | Execution Time (seconds)**")
    print("\n*Note: Execution times are per 1000 samples*")
    
    approaches = ['Bilateral', 'Unilateral-Binary', 'Unilateral-Ternary', 
                  'Confidence-0.5', 'Confidence-0.7', 'Confidence-0.9']
    
    for dataset_key, dataset_name in dataset_display_names.items():
        print(f"\n## {dataset_name}")
        print("\n| Model | " + " | ".join([f"**{a}**" for a in approaches]) + " |")
        print("|" + "-" * 15 + "|" + "|".join(["-" * 30] * len(approaches)) + "|")
        
        for model_key, model_name in model_display_names.items():
            row = [model_name]
            for approach in approaches:
                key = (model_key, dataset_key, approach)
                if key in all_results:
                    f1, cov, time = all_results[key]
                    # Format each metric on its own line for readability
                    if time is not None:
                        cell = f"F1: {f1:.3f}<br>Cov: {cov:.1%}<br>Time: {time:.1f}s"
                    else:
                        cell = f"F1: {f1:.3f}<br>Cov: {cov:.1%}"
                else:
                    cell = "—"
                row.append(cell)
            print("| " + " | ".join(row) + " |")
    
    # Summary statistics
    print("\n## Summary Statistics")
    print("\n### Average F1-Macro by Approach")
    approach_f1s = {}
    for approach in approaches:
        f1_values = [v[0] for k, v in all_results.items() if k[2] == approach]
        if f1_values:
            approach_f1s[approach] = np.mean(f1_values)
    
    for approach, avg_f1 in sorted(approach_f1s.items(), key=lambda x: -x[1]):
        print(f"- {approach}: {avg_f1:.3f}")
    
    print("\n### Average Coverage by Approach")
    approach_covs = {}
    for approach in approaches:
        cov_values = [v[1] for k, v in all_results.items() if k[2] == approach]
        if cov_values:
            approach_covs[approach] = np.mean(cov_values)
    
    for approach, avg_cov in sorted(approach_covs.items(), key=lambda x: -x[1]):
        print(f"- {approach}: {avg_cov:.1%}")
    
    print("\n### Average Execution Time by Approach (seconds per 1000 samples)")
    approach_times = {}
    for approach in approaches:
        time_values = [v[2] for k, v in all_results.items() if k[2] == approach and v[2] is not None]
        if time_values:
            approach_times[approach] = np.mean(time_values)
    
    for approach, avg_time in sorted(approach_times.items(), key=lambda x: x[1]):
        print(f"- {approach}: {avg_time:.1f}s")
    
    # Note if some approaches don't have timing data
    approaches_without_timing = [a for a in approaches if a not in approach_times]
    if approaches_without_timing:
        print(f"\n*Note: Timing data not available for: {', '.join(approaches_without_timing)}*")
    
    print("\n### Model Rankings (by average F1 across all approaches)")
    model_f1s = {}
    for model_key, model_name in model_display_names.items():
        f1_values = [v[0] for k, v in all_results.items() if k[0] == model_key]
        if f1_values:
            model_f1s[model_name] = np.mean(f1_values)
    
    for model, avg_f1 in sorted(model_f1s.items(), key=lambda x: -x[1]):
        print(f"- {model}: {avg_f1:.3f}")

if __name__ == "__main__":
    main()