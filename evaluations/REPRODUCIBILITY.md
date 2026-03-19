# Reproducibility Guide

## Quick Start

### 1. Environment Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements_eval.txt
pip install -e ..  # Install bilateral-truth package

# Set up API keys
cp .env.example .env
# Edit .env with your API keys
```

### 2. Run Complete Pipeline
```bash
# Full sequential run (simple, ~60 hours)
python run_full_pipeline.py --stage all

# Full parallel run (fast, ~10 hours)
python run_full_pipeline.py --stage all --parallel --workers 7

# Specific stages only
python run_full_pipeline.py --stage data    # Prepare datasets
python run_full_pipeline.py --stage eval    # Run evaluations
python run_full_pipeline.py --stage ci      # Generate confidence intervals
python run_full_pipeline.py --stage analysis # Create final tables
```

### 3. Selective Execution
```bash
# Run specific methods only
python run_full_pipeline.py --stage eval --methods bilateral confidence

# Run specific models only
python run_full_pipeline.py --stage eval --models gpt-4.1-mini-2025-04-14

# Run specific datasets only
python run_full_pipeline.py --stage eval --datasets truthfulqa_complete
```

## Data Flow

```
1. RAW DATA
   ├── TruthfulQA (CSV)
   ├── SimpleQA (JSON)
   ├── FACTScore (JSON)
   └── MMLU-Pro (Parquet)
           ↓
2. STANDARD FORMAT (data_generators/)
   └── standard_datasets/*.json
           ↓
3. EVALUATIONS (parallel)
   ├── Bilateral (generic_evaluator.py)
   ├── Unilateral Forced (unilateral_evaluator.py)
   ├── Unilateral Uncertain (unilateral_evaluator.py)
   └── Confidence-based (unilateral_confidence_evaluator.py)
           ↓
4. RESULTS
   └── results/*_results.json
           ↓
5. CONFIDENCE INTERVALS (bootstrap subsampling)
   └── results/confidence_intervals_complete.json
           ↓
6. FINAL TABLES
   ├── FINAL_RESULTS_TABLES.md
   └── comparison_statistics.csv
```

## Checkpoint Recovery

All evaluators support automatic checkpoint recovery:
- Checkpoints saved every 10-20 samples
- Automatic resume on restart
- No work lost on interruption

To force fresh start:
```bash
rm checkpoints/*.json
```

## Validation

Verify complete results:
```bash
python run_full_pipeline.py --stage all --validate-only
```

This checks:
- All 112 evaluation files exist (7 models × 4 datasets × 4 methods)
- Files contain valid JSON
- Sample counts match expected (1000 per evaluation)

## Expected Output Structure

```
results/
├── {dataset}_{model}_classical_results.json              # Bilateral
├── {dataset}_{model}_unilateral_direct_results.json      # Forced
├── {dataset}_{model}_unilateral_uncertain_results.json   # Uncertain
├── {dataset}_{model}_unilateral_confidence_results.json  # Confidence
├── confidence_intervals_complete.json                    # All CIs
├── table1_comprehensive_performance.csv                  # Summary table
└── statistical_significance.json                         # Significance tests
```

## Configuration

Edit `run_full_pipeline.py` to modify:
- `MODELS`: List of models to evaluate
- `DATASETS`: List of datasets to use
- `SAMPLE_SIZE`: Number of samples per evaluation (default: 1000)
- `BOOTSTRAP_ITERATIONS`: Bootstrap iterations for CI (default: 1000)
- `SUBSAMPLE_SIZE`: Subsample size for bootstrap (default: 100)
- `MAX_WORKERS`: Parallel workers (default: 7)

## Time Estimates

### Per Evaluation (1000 samples)
- Bilateral: ~25 minutes
- Unilateral: ~15 minutes
- Confidence: ~15 minutes

### Total Runtime
- Sequential: ~60 hours
- Parallel (7 workers): ~10 hours

### By Stage
- Data preparation: ~5 minutes
- Evaluations: ~10 hours (parallel)
- Confidence intervals: ~10 minutes
- Analysis & tables: ~5 minutes

## Troubleshooting

### API Rate Limits
- Add delays: `--api-delay 1.0`
- Reduce workers: `--workers 3`

### Memory Issues
- Reduce batch size in evaluators
- Process datasets sequentially

### Checkpoint Recovery Failed
```bash
# Clear corrupted checkpoint
rm checkpoints/*{model}*.json
# Restart evaluation
```

### Missing Dependencies
```bash
pip install -r requirements_eval.txt --upgrade
```

## Reproducing Paper Results

To exactly reproduce the paper results:
1. Use the exact model versions specified
2. Set random seed: `--seed 42`
3. Use same sample size: `--samples 1000`
4. Use same bootstrap parameters: 1000 iterations, 100 subsample size

## Contact

For issues or questions about reproduction, please open an issue on the GitHub repository.