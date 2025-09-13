#!/usr/bin/env python3
"""
Rerun FACTScore evaluation for Anthropic models only.
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"✓ Loaded environment variables from {env_path}")

# Configuration - only Anthropic models
ANTHROPIC_MODELS = [
    "claude-opus-4-1-20250805",
    "claude-3-5-haiku-20241022"
]

FACTSCORE_BENCHMARK = {
    "name": "FACTScore",
    "dataset": "standard_datasets/factscore_complete.json",
    "samples": 1000,
    "description": "Atomic facts from LLM-generated biographies"
}

def run_evaluation(model: str) -> dict:
    """Run FACTScore evaluation for a single model."""
    start_time = time.time()
    
    cmd = [
        sys.executable,
        "generic_evaluator.py",
        "--dataset", FACTSCORE_BENCHMARK["dataset"],
        "--model", model,
        "--samples", str(FACTSCORE_BENCHMARK["samples"])
    ]
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting {FACTSCORE_BENCHMARK['name']} with {model}")
    print(f"  Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        elapsed = time.time() - start_time
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Completed {FACTSCORE_BENCHMARK['name']} with {model} in {elapsed:.1f}s")
        
        # Build the results file path
        model_safe = model.replace('/', '_').replace(':', '_')
        dataset_name = Path(FACTSCORE_BENCHMARK["dataset"]).stem
        results_file = f"results/{dataset_name}_{model_safe}_classical_results.json"
        
        return {
            "model": model,
            "status": "success",
            "elapsed_time": elapsed,
            "results_file": results_file
        }
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Failed {FACTSCORE_BENCHMARK['name']} with {model} after {elapsed:.1f}s")
        print(f"  Error: {e}")
        if e.stderr:
            print(f"  Stderr: {e.stderr[:500]}")
        
        return {
            "model": model,
            "status": "failed",
            "elapsed_time": elapsed,
            "error": str(e)
        }

def main():
    """Main function to rerun FACTScore for Anthropic models."""
    print(f"\n{'='*60}")
    print(f"Rerunning FACTScore Evaluation for Anthropic Models")
    print(f"{'='*60}")
    print(f"Models: {', '.join(ANTHROPIC_MODELS)}")
    print(f"Dataset: {FACTSCORE_BENCHMARK['dataset']}")
    print(f"Assertions to evaluate: {FACTSCORE_BENCHMARK['samples']}")
    print(f"{'='*60}\n")
    
    # Check for API key
    import os
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY not found in environment")
        print("   Please set it in your .env file or environment")
        return
    
    print("✓ Anthropic API key found\n")
    
    # Run evaluations sequentially (could parallelize if desired)
    results = []
    for model in ANTHROPIC_MODELS:
        result = run_evaluation(model)
        results.append(result)
        print()  # Blank line between models
    
    # Print summary
    print(f"{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for result in results:
        status = "✓" if result["status"] == "success" else "✗"
        print(f"{status} {result['model']}: {result['status']} ({result['elapsed_time']:.1f}s)")
        if result["status"] == "success":
            print(f"   Results: {result['results_file']}")
    
    print(f"{'='*60}")
    successful = sum(1 for r in results if r["status"] == "success")
    print(f"Completed: {successful}/{len(ANTHROPIC_MODELS)} models succeeded")

if __name__ == "__main__":
    main()