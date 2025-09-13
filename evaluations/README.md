# Bilateral Truth Evaluations

This directory contains evaluation frameworks for testing the bilateral-truth package against various benchmarks and datasets.

## Directory Structure

```
evaluations/
├── README.md                    # This file
└── truthfulqa/                  # TruthfulQA benchmark evaluation
    ├── truthfulqa_evaluation.py # Main evaluation script
    ├── explore_truthfulqa.py    # Dataset exploration utility
    ├── truthfulqa_data.csv      # TruthfulQA dataset
    ├── truthfulqa_results_*.json # Evaluation results (generated)
    ├── experimental_methodology.md # Scholarly methodology document
    ├── truthfulqa_evaluation_report.md # Evaluation report
    └── multi_model_comparison.md # Multi-model usage guide
```

## Available Evaluations

### TruthfulQA Benchmark

The TruthfulQA evaluation measures agreement between bilateral truth judgments and human factual assessments using the TruthfulQA benchmark dataset.

**Quick Start:**
```bash
cd evaluations/truthfulqa
python truthfulqa_evaluation.py --model mock --max-samples 5
```

**Features:**
- Multi-model support (GPT-4, Claude, OpenRouter, mock)
- Configurable system prompts and context
- Automatic checkpoint/recovery system
- Comprehensive statistical analysis
- Cohen's kappa inter-rater agreement measurement

**See:** `truthfulqa/multi_model_comparison.md` for detailed usage examples.

## Adding New Evaluations

To add a new benchmark evaluation:

1. Create a new subdirectory: `evaluations/[benchmark_name]/`
2. Include the main evaluation script
3. Add dataset files or download utilities
4. Provide documentation explaining the evaluation methodology
5. Update this README with the new evaluation

## Running Evaluations

All evaluation scripts should be run from their respective subdirectories to ensure correct relative paths for data files and outputs.

```bash
# Navigate to specific evaluation
cd evaluations/truthfulqa

# Run evaluation
python truthfulqa_evaluation.py --help
```

## Output Files

Each evaluation generates standardized outputs:
- **Results JSON**: Detailed evaluation results with metadata
- **Checkpoint files**: For recovery from interruptions (automatically cleaned up)
- **Reports**: Human-readable analysis and summaries

## Dependencies

Evaluations use the main bilateral-truth package and may have additional requirements:
- `datasets` - For loading benchmark datasets
- `scikit-learn` - For statistical analysis
- `pandas` - For data manipulation (optional)

Install additional dependencies as needed:
```bash
pip install datasets scikit-learn pandas
```