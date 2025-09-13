#!/usr/bin/env python3
"""
Run unilateral evaluations in parallel, matching the run_benchmark_suite.py framework.
This allows direct comparison with bilateral evaluation results.
"""

import subprocess
import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv

# Configuration - matching run_benchmark_suite.py
FRONTIER_MODELS = [
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini-2025-04-14",
    "claude-opus-4-1-20250805",
    "claude-3-5-haiku-20241022",
    "meta-llama/llama-4-scout",
    "meta-llama/llama-4-maverick",
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

def run_unilateral_evaluation(model: str, benchmark: Dict, prompt_style: str = "direct") -> Dict:
    """Run a single unilateral model evaluation (runs in subprocess)."""
    start_time = time.time()
    
    cmd = [
        sys.executable,
        "unilateral_evaluator.py",
        "--dataset", benchmark["dataset"],
        "--model", model,
        "--samples", str(benchmark["samples"]),
        "--prompt-style", prompt_style
    ]
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting unilateral {benchmark['name']} with {model}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        elapsed = time.time() - start_time
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Completed unilateral {benchmark['name']} with {model} in {elapsed:.1f}s")
        
        # Build the results file path
        model_safe = model.replace('/', '_').replace(':', '_')
        dataset_name = Path(benchmark["dataset"]).stem
        output_dir = "results"
        results_file = f"{output_dir}/{dataset_name}_{model_safe}_unilateral_{prompt_style}_results.json"
        
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
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Failed unilateral {benchmark['name']} with {model} after {elapsed:.1f}s")
        print(f"  Error: {e}")
        if e.stderr:
            print(f"  Stderr: {e.stderr[:500]}")
        
        return {
            "model": model,
            "benchmark": benchmark["name"],
            "status": "failed",
            "elapsed_time": elapsed,
            "error": str(e),
            "stdout": e.stdout if e.stdout else "",
            "stderr": e.stderr if e.stderr else ""
        }

def run_benchmark_parallel(benchmark: Dict, models: List[str], prompt_style: str = "direct") -> List[Dict]:
    """Run a single benchmark across all models in parallel."""
    print(f"\n{'='*60}")
    print(f"Running Unilateral Evaluation - {benchmark['name']}")
    print(f"Dataset: {benchmark['dataset']}")
    print(f"Prompt style: {prompt_style}")
    print(f"Assertions to evaluate: {benchmark['samples']}")
    print(f"Models: {', '.join(models)}")
    print(f"{'='*60}\n")
    
    results = []
    
    with ProcessPoolExecutor(max_workers=len(models)) as executor:
        # Submit all model evaluations for this benchmark
        futures = {
            executor.submit(run_unilateral_evaluation, model, benchmark, prompt_style): model
            for model in models
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

def run_comparison_analysis(all_results: List[Dict], run_id: str):
    """Run comparison analysis between unilateral and existing bilateral results."""
    print("\n" + "="*100)
    print("RUNNING COMPARISON ANALYSIS")
    print("="*100)
    
    comparison_reports = []
    
    # Group results by model and benchmark
    results_by_key = {}
    for result in all_results:
        if result["status"] == "success":
            key = (result["model"], result["benchmark"])
            results_by_key[key] = result
    
    # Run comparison for each successful evaluation
    for (model, benchmark), result in results_by_key.items():
        print(f"\nComparing {model} on {benchmark}...")
        
        # Find dataset name from benchmark
        dataset_name = None
        for bench in BENCHMARKS:
            if bench["name"] == benchmark:
                dataset_name = Path(bench["dataset"]).stem
                break
        
        if dataset_name:
            cmd = [
                sys.executable,
                "compare_bilateral_unilateral.py",
                "--model", model,
                "--dataset", dataset_name,
                "--output", f"results/comparison_{run_id}_{model.replace('/', '_')}_{dataset_name}.json"
            ]
            
            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                comparison_reports.append(f"comparison_{run_id}_{model.replace('/', '_')}_{dataset_name}.json")
                print(f"  ✓ Comparison report generated")
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Failed to generate comparison: {e}")
    
    return comparison_reports

def save_summary(all_results: List[Dict], run_id: str, prompt_style: str):
    """Save a summary of all unilateral evaluation results."""
    summary_file = f"results/unilateral_suite_{run_id}_summary.json"
    
    # Create summary structure
    summary = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "evaluation_type": "unilateral_forced_choice",
        "prompt_style": prompt_style,
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
    print("UNILATERAL EVALUATION SUMMARY")
    print("="*100)
    print(f"{'Benchmark':<15} {'Model':<28} {'Status':<8} {'Time':<8} {'Accuracy':<10} {'Coverage':<10} {'F1':<8}")
    print("-"*100)
    
    for result in all_results:
        status_symbol = "✓" if result["status"] == "success" else "✗"
        elapsed = f"{result.get('elapsed_time', 0):.1f}s"
        
        # Extract performance metrics
        accuracy = "-"
        coverage = "-"
        f1_score = "-"
        
        if result["status"] == "success" and result.get("results_file"):
            try:
                with open(result["results_file"], 'r') as f:
                    eval_results = json.load(f)
                    accuracy = f"{eval_results.get('accuracy', 0):.3f}"
                    coverage = f"{eval_results.get('coverage', 0):.3f}"
                    f1_score = f"{eval_results.get('f1_macro', 0):.3f}"
            except:
                pass
        
        model_short = result['model'][:28]
        print(f"{result['benchmark']:<15} {model_short:<28} {status_symbol:<8} {elapsed:<8} {accuracy:<10} {coverage:<10} {f1_score:<8}")
    
    print("="*100)

def main():
    """Main orchestration function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run unilateral evaluation suite')
    parser.add_argument('--models', nargs='+', default=None,
                       help='Specific models to evaluate (default: all frontier models)')
    parser.add_argument('--benchmarks', nargs='+', default=None,
                       help='Specific benchmarks to run (default: all)')
    parser.add_argument('--prompt-style', type=str, default='direct',
                       choices=['direct', 'zero-shot', 'few-shot', 'uncertain'],
                       help='Prompt style for unilateral evaluation')
    parser.add_argument('--samples', type=int, default=None,
                       help='Override number of samples per benchmark')
    parser.add_argument('--compare', action='store_true',
                       help='Run comparison analysis with bilateral results')
    
    args = parser.parse_args()
    
    # Load environment variables
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded environment variables from {env_path}")
    
    # Select models and benchmarks
    models = args.models if args.models else FRONTIER_MODELS
    
    if args.benchmarks:
        benchmarks = [b for b in BENCHMARKS if b["name"] in args.benchmarks]
    else:
        benchmarks = BENCHMARKS
    
    # Override sample size if specified
    if args.samples:
        for benchmark in benchmarks:
            benchmark["samples"] = args.samples
    
    # Generate unique run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Starting Unilateral Evaluation Suite")
    print(f"Run ID: {run_id}")
    print(f"Prompt Style: {args.prompt_style}")
    print(f"Models: {', '.join(models)}")
    print(f"Benchmarks: {', '.join([b['name'] for b in benchmarks])}")
    
    # Check for API keys
    missing_keys = []
    if not os.getenv("OPENAI_API_KEY"):
        missing_keys.append("OPENAI_API_KEY")
    if not os.getenv("ANTHROPIC_API_KEY"):
        missing_keys.append("ANTHROPIC_API_KEY")
    if not os.getenv("OPENROUTER_API_KEY"):
        missing_keys.append("OPENROUTER_API_KEY")
    
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
    
    # Run each benchmark sequentially, models in parallel
    for i, benchmark in enumerate(benchmarks, 1):
        print(f"\n[BENCHMARK {i}/{len(benchmarks)}]")
        benchmark_results = run_benchmark_parallel(benchmark, models, args.prompt_style)
        all_results.extend(benchmark_results)
        
        # Brief pause between benchmarks
        if benchmark != benchmarks[-1]:
            print(f"\n⏸️  Pausing 5 seconds before next benchmark...")
            time.sleep(5)
    
    # Save summary
    save_summary(all_results, run_id, args.prompt_style)
    
    # Run comparison analysis if requested
    if args.compare:
        comparison_reports = run_comparison_analysis(all_results, run_id)
        print(f"\n📊 Generated {len(comparison_reports)} comparison reports")
    
    print(f"\n🎉 Unilateral evaluation suite completed!")
    print(f"Total evaluations: {len(all_results)}")
    print(f"✓ Successful: {len([r for r in all_results if r['status'] == 'success'])}")
    print(f"✗ Failed: {len([r for r in all_results if r['status'] == 'failed'])}")
    
    if args.compare:
        print(f"\n💡 To view detailed comparisons, check the results/ directory")
        print(f"   Comparison files: comparison_{run_id}_*.json")

if __name__ == "__main__":
    main()