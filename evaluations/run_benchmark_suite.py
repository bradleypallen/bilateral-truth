#!/usr/bin/env python3
"""
Orchestrates parallel evaluation of benchmarks across multiple frontier models.
Runs each benchmark sequentially, but evaluates all models in parallel for each benchmark.
"""

import subprocess
import os
import sys
import time
import json
import threading
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
import re
from dotenv import load_dotenv

# Configuration
FRONTIER_MODELS = [
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini-2025-04-14",
    "claude-opus-4-1-20250805",
    "claude-3-5-haiku-20241022",
    "meta-llama/llama-4-scout",
    "meta-llama/llama-4-maverick",
    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash"
]

BENCHMARKS = [
    {
        "name": "TruthfulQA",
        "dataset": "standard_datasets/truthfulqa_complete.json",
        "samples": 1000,
        "description": "Tests factual accuracy and truthfulness"
    },
    {
        "name": "SimpleQA", 
        "dataset": "standard_datasets/simpleqa_complete.json",
        "samples": 1000,
        "description": "Simple fact verification tasks"
    },
    {
        "name": "MMLU-Pro",
        "dataset": "standard_datasets/mmlupro_complete.json",
        "samples": 1000,
        "description": "Comprehensive knowledge evaluation"
    },
    {
        "name": "FACTScore",
        "dataset": "standard_datasets/factscore_complete.json",
        "samples": 1000,
        "description": "Atomic facts from LLM-generated biographies"
    }
]

class ProgressMonitor:
    """Monitors and displays real-time progress of evaluations."""
    
    def __init__(self):
        self.progress = {}
        self.lock = threading.Lock()
        self.active = True
    
    def update(self, model: str, benchmark: str, current: int, total: int, status: str = "running"):
        """Update progress for a model-benchmark combination."""
        with self.lock:
            key = f"{benchmark}:{model}"
            self.progress[key] = {
                "current": current,
                "total": total,
                "percent": (current / total * 100) if total > 0 else 0,
                "status": status
            }
    
    def display(self):
        """Display current progress."""
        with self.lock:
            if not self.progress:
                return
            
            # Clear previous lines and display updated progress
            print("\r" + " " * 100, end="")  # Clear line
            print("\r", end="")
            
            for key, info in self.progress.items():
                benchmark, model = key.split(":")
                status_icon = "✓" if info["status"] == "completed" else "⏳"
                model_short = model.split("-")[0][:10]  # Shorten model names
                print(f"\r{status_icon} {benchmark[:10]:10} | {model_short:10} | [{info['current']:3}/{info['total']:3}] {info['percent']:5.1f}%", end="")
    
    def clear(self):
        """Clear progress for new benchmark."""
        with self.lock:
            self.progress.clear()

def monitor_subprocess_output(process, model: str, benchmark: str, monitor: ProgressMonitor):
    """Monitor subprocess output for progress updates."""
    total_assertions = 0
    current_assertion = 0
    
    for line in iter(process.stdout.readline, ''):
        if line:
            # Look for progress indicators in the output
            if "Processing assertion" in line:
                match = re.search(r'Processing assertion (\d+)/(\d+)', line)
                if match:
                    current_assertion = int(match.group(1))
                    total_assertions = int(match.group(2))
                    monitor.update(model, benchmark, current_assertion, total_assertions)
            elif "Evaluating" in line and "/" in line:
                # Alternative format: "Evaluating 5/100"
                match = re.search(r'Evaluating (\d+)/(\d+)', line)
                if match:
                    current_assertion = int(match.group(1))
                    total_assertions = int(match.group(2))
                    monitor.update(model, benchmark, current_assertion, total_assertions)
    
    # Mark as completed when done
    if total_assertions > 0:
        monitor.update(model, benchmark, total_assertions, total_assertions, "completed")

def run_evaluation(model: str, benchmark: Dict, run_id: str) -> Dict:
    """Run a single model evaluation (runs in subprocess)."""
    start_time = time.time()
    
    cmd = [
        sys.executable,
        "generic_evaluator.py",
        "--dataset", benchmark["dataset"],
        "--model", model,
        "--samples", str(benchmark["samples"])
    ]
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting {benchmark['name']} with {model}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        elapsed = time.time() - start_time
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Completed {benchmark['name']} with {model} in {elapsed:.1f}s")
        
        # Build the results file path based on generic_evaluator.py's naming convention
        # From generic_evaluator.py line 680-682:
        # output_file = f"{args.output_dir}/{dataset_name}_{model_safe}_{epistemic_policy.value}_results.json"
        model_safe = model.replace('/', '_').replace(':', '_')
        dataset_name = Path(benchmark["dataset"]).stem
        epistemic_policy = "classical"  # Default from generic_evaluator.py
        output_dir = "results"  # Default from generic_evaluator.py
        results_file = f"{output_dir}/{dataset_name}_{model_safe}_{epistemic_policy}_results.json"
        
        return {
            "model": model,
            "benchmark": benchmark["name"],
            "status": "success",
            "elapsed_time": elapsed,
            "results_file": results_file,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Failed {benchmark['name']} with {model} after {elapsed:.1f}s")
        print(f"  Error: {e}")
        if e.stderr:
            print(f"  Stderr: {e.stderr[:500]}")  # First 500 chars of error
        if e.stdout:
            print(f"  Stdout: {e.stdout[:500]}")  # First 500 chars of output
        
        return {
            "model": model,
            "benchmark": benchmark["name"],
            "status": "failed",
            "elapsed_time": elapsed,
            "error": str(e),
            "stdout": e.stdout if e.stdout else "",
            "stderr": e.stderr if e.stderr else ""
        }

def run_benchmark_parallel_with_progress(benchmark: Dict, run_id: str) -> List[Dict]:
    """Run a single benchmark across all models in parallel."""
    print(f"\n{'='*60}")
    print(f"Running {benchmark['name']}: {benchmark['description']}")
    print(f"Dataset: {benchmark['dataset']}")
    print(f"Assertions to evaluate: {benchmark['samples']}")
    print(f"Models: {', '.join(FRONTIER_MODELS)}")
    print(f"{'='*60}\n")
    
    results = []
    
    with ProcessPoolExecutor(max_workers=len(FRONTIER_MODELS)) as executor:
        # Submit all model evaluations for this benchmark
        futures = {
            executor.submit(run_evaluation, model, benchmark, run_id): model
            for model in FRONTIER_MODELS
        }
        
        # Collect results as they complete
        for future in as_completed(futures):
            model = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error running {model} on {benchmark['name']}: {e}")
                results.append({
                    "model": model,
                    "benchmark": benchmark["name"],
                    "status": "error",
                    "error": str(e)
                })
    
    return results

def save_summary(all_results: List[Dict], run_id: str):
    """Save a summary of all evaluation results."""
    summary_file = f"results/benchmark_suite_{run_id}_summary.json"
    
    # Create summary structure
    summary = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "models": FRONTIER_MODELS,
        "benchmarks": [b["name"] for b in BENCHMARKS],
        "results": all_results,
        "statistics": {}
    }
    
    # Calculate statistics
    for benchmark in BENCHMARKS:
        benchmark_results = [r for r in all_results if r["benchmark"] == benchmark["name"]]
        summary["statistics"][benchmark["name"]] = {
            "completed": len([r for r in benchmark_results if r["status"] == "success"]),
            "failed": len([r for r in benchmark_results if r["status"] == "failed"]),
            "total_time": sum(r.get("elapsed_time", 0) for r in benchmark_results)
        }
    
    # Save summary
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")
    
    # Print summary table
    print("\n" + "="*100)
    print("EVALUATION SUMMARY")
    print("="*100)
    print(f"{'Benchmark':<15} {'Model':<28} {'Status':<8} {'Time':<8} {'Accuracy':<10} {'Coverage':<10} {'F1':<8}")
    print("-"*100)
    
    for result in all_results:
        status_symbol = "✓" if result["status"] == "success" else "✗"
        elapsed = f"{result.get('elapsed_time', 0):.1f}s"
        
        # Extract performance metrics from results file if available
        accuracy = "-"
        coverage = "-"
        f1_score = "-"
        
        if result["status"] == "success" and result.get("results_file"):
            try:
                # Load the results file to get metrics
                with open(result["results_file"], 'r') as f:
                    eval_results = json.load(f)
                    accuracy = f"{eval_results.get('accuracy', 0):.3f}"
                    coverage = f"{eval_results.get('coverage', 0):.3f}"
                    f1_score = f"{eval_results.get('f1_macro', 0):.3f}"
            except Exception as e:
                # Debug: print what went wrong
                print(f"  Warning: Could not load results from {result.get('results_file')}: {e}")
        
        model_short = result['model'][:28]  # Truncate long model names
        print(f"{result['benchmark']:<15} {model_short:<28} {status_symbol:<8} {elapsed:<8} {accuracy:<10} {coverage:<10} {f1_score:<8}")
    
    print("="*100)
    
    # Print bilateral truth value distribution table
    print("\n" + "="*100)
    print("BILATERAL TRUTH VALUE DISTRIBUTIONS (%)")
    print("Key: <verifiability, refutability> where t=true, f=false, e=undefined")
    print("="*100)
    print(f"{'Benchmark':<15} {'Model':<28} {'<t,f>':<8} {'<f,t>':<8} {'<t,t>':<8} {'<f,f>':<8} {'<t,e>':<8} {'<e,t>':<8} {'<f,e>':<8} {'<e,f>':<8} {'<e,e>':<8}")
    print("-"*100)
    
    for result in all_results:
        if result["status"] == "success" and result.get("results_file"):
            try:
                with open(result["results_file"], 'r') as f:
                    eval_results = json.load(f)
                    
                    # Count bilateral truth values from detailed_results
                    tv_counts = {
                        '<t,f>': 0, '<f,t>': 0, '<t,t>': 0, '<f,f>': 0,
                        '<t,e>': 0, '<e,t>': 0, '<f,e>': 0, '<e,f>': 0, '<e,e>': 0
                    }
                    
                    total = len(eval_results.get('detailed_results', []))
                    for item in eval_results.get('detailed_results', []):
                        bilateral_value = item.get('bilateral_value', '<e,e>')
                        if bilateral_value in tv_counts:
                            tv_counts[bilateral_value] += 1
                    
                    # Convert to percentages
                    model_short = result['model'][:28]
                    if total > 0:
                        print(f"{result['benchmark']:<15} {model_short:<28}", end="")
                        for tv in ['<t,f>', '<f,t>', '<t,t>', '<f,f>', '<t,e>', '<e,t>', '<f,e>', '<e,f>', '<e,e>']:
                            pct = (tv_counts[tv] / total * 100) if total > 0 else 0
                            print(f" {pct:6.1f}%", end="")
                        print()
            except:
                pass
    
    print("="*100)
    
    # Print aggregate statistics by model
    print("\n" + "="*100)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*100)
    print(f"{'Model':<30} {'Benchmarks':<12} {'Avg Accuracy':<12} {'Avg Coverage':<12} {'Avg F1':<10}")
    print("-"*100)
    
    for model in FRONTIER_MODELS:
        model_results = [r for r in all_results if r["model"] == model and r["status"] == "success"]
        
        if model_results:
            accuracies = []
            coverages = []
            f1_scores = []
            
            for result in model_results:
                if result.get("results_file"):
                    try:
                        with open(result["results_file"], 'r') as f:
                            eval_results = json.load(f)
                            accuracies.append(eval_results.get('accuracy', 0))
                            coverages.append(eval_results.get('coverage', 0))
                            f1_scores.append(eval_results.get('f1_macro', 0))
                    except:
                        pass
            
            num_benchmarks = len(model_results)
            avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
            avg_coverage = sum(coverages) / len(coverages) if coverages else 0
            avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
            
            model_short = model[:30]
            print(f"{model_short:<30} {num_benchmarks:<12} {avg_accuracy:<12.3f} {avg_coverage:<12.3f} {avg_f1:<10.3f}")
        else:
            model_short = model[:30]
            print(f"{model_short:<30} {'0':<12} {'-':<12} {'-':<12} {'-':<10}")
    
    print("="*100)

def display_live_status(active_processes: Dict):
    """Display live status of running processes."""
    print("\n" + "="*60)
    print("LIVE STATUS")
    print("-"*60)
    
    for key, info in active_processes.items():
        status = "🟢 Running" if info["status"] == "running" else "✓ Complete"
        print(f"{info['benchmark']:<15} | {info['model']:<25} | {status}")
    
    print("="*60)

def main():
    """Main orchestration function."""
    # Load environment variables from .env file
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded environment variables from {env_path}")
    else:
        print(f"⚠️  No .env file found at {env_path}")
    
    # Generate unique run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Starting Benchmark Suite Evaluation")
    print(f"Run ID: {run_id}")
    print(f"Models: {', '.join(FRONTIER_MODELS)}")
    print(f"Benchmarks: {', '.join([b['name'] for b in BENCHMARKS])}")
    
    # Check for API keys
    missing_keys = []
    if not os.getenv("OPENAI_API_KEY"):
        missing_keys.append("OPENAI_API_KEY")
    if not os.getenv("ANTHROPIC_API_KEY"):
        missing_keys.append("ANTHROPIC_API_KEY")
    
    if missing_keys:
        print(f"\nWarning: Missing API keys: {', '.join(missing_keys)}")
        print("Some models may fail. Continue? (y/n): ", end="")
        if input().lower() != 'y':
            print("Aborted.")
            return
    
    # Create results directory
    Path("results").mkdir(exist_ok=True)
    
    # Track all results
    all_results = []
    
    
    # Run each benchmark sequentially
    for i, benchmark in enumerate(BENCHMARKS, 1):
        print(f"\n[BENCHMARK {i}/{len(BENCHMARKS)}]")
        benchmark_results = run_benchmark_parallel_with_progress(benchmark, run_id)
        all_results.extend(benchmark_results)
        
        # Brief pause between benchmarks
        if benchmark != BENCHMARKS[-1]:
            print(f"\n⏸️  Pausing 5 seconds before next benchmark...")
            time.sleep(5)
    
    # Save summary
    save_summary(all_results, run_id)
    
    print(f"\n🎉 Benchmark suite completed!")
    print(f"Total evaluations: {len(all_results)}")
    print(f"✓ Successful: {len([r for r in all_results if r['status'] == 'success'])}")
    print(f"✗ Failed: {len([r for r in all_results if r['status'] == 'failed'])}")

if __name__ == "__main__":
    main()