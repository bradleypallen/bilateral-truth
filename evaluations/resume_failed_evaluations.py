#!/usr/bin/env python3
"""
Resume failed evaluations by running them individually with checkpointing.
This handles cases where models failed partway through due to rate limits or errors.
"""

import subprocess
import json
import sys
from pathlib import Path
import time

# Failed evaluations to resume
FAILED_RUNS = [
    {
        "model": "claude-3-5-haiku-20241022",
        "dataset": "standard_datasets/truthfulqa_complete.json",
        "samples": 1000
    },
    {
        "model": "meta-llama/llama-4-maverick", 
        "dataset": "standard_datasets/truthfulqa_complete.json",
        "samples": 1000
    },
    {
        "model": "claude-opus-4-1-20250805",
        "dataset": "standard_datasets/truthfulqa_complete.json", 
        "samples": 1000
    },
    {
        "model": "claude-opus-4-1-20250805",
        "dataset": "standard_datasets/simpleqa_complete.json",
        "samples": 1000
    }
]

def run_with_resume(model: str, dataset: str, samples: int, max_attempts: int = 5):
    """Run evaluation with automatic resume on failure."""
    
    dataset_name = Path(dataset).stem
    print(f"\n{'='*60}")
    print(f"Resuming: {model} on {dataset_name}")
    print(f"Target samples: {samples}")
    print(f"{'='*60}")
    
    for attempt in range(max_attempts):
        print(f"\nAttempt {attempt + 1}/{max_attempts}")
        
        cmd = [
            sys.executable,
            "unilateral_evaluator.py",
            "--dataset", dataset,
            "--model", model,
            "--samples", str(samples),
            "--prompt-style", "direct"
        ]
        
        try:
            # Run the evaluation
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            # Check if it completed successfully
            if result.returncode == 0:
                print("✓ Evaluation completed successfully!")
                
                # Check the results to see if we got all samples
                model_safe = model.replace('/', '_').replace(':', '_')
                results_file = f"results/{dataset_name}_{model_safe}_unilateral_direct_results.json"
                
                if Path(results_file).exists():
                    with open(results_file) as f:
                        data = json.load(f)
                        completed = data.get('total_samples', 0)
                        print(f"  Completed samples: {completed}/{samples}")
                        
                        if completed >= samples:
                            return True
                        else:
                            print(f"  Only {completed} samples completed, continuing...")
            else:
                print(f"✗ Evaluation failed with return code {result.returncode}")
                if result.stderr:
                    print(f"  Error: {result.stderr[:500]}")
                    
        except subprocess.TimeoutExpired:
            print("✗ Evaluation timed out after 1 hour")
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
        
        # Wait before retrying
        if attempt < max_attempts - 1:
            wait_time = min(60 * (2 ** attempt), 300)  # Exponential backoff, max 5 minutes
            print(f"  Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
    
    print(f"\n❌ Failed to complete {model} on {dataset_name} after {max_attempts} attempts")
    return False

def main():
    """Main entry point."""
    print("RESUMING FAILED EVALUATIONS")
    print("="*60)
    print(f"Failed runs to resume: {len(FAILED_RUNS)}")
    
    successful = 0
    failed = 0
    
    for run in FAILED_RUNS:
        if run_with_resume(run["model"], run["dataset"], run["samples"]):
            successful += 1
        else:
            failed += 1
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Successful completions: {successful}/{len(FAILED_RUNS)}")
    print(f"Failed: {failed}/{len(FAILED_RUNS)}")
    
    if failed > 0:
        print("\nFailed runs may need manual intervention or API quota increases.")

if __name__ == "__main__":
    main()