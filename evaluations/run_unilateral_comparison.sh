#!/bin/bash
# Run unilateral evaluations and compare with existing bilateral results

# Configuration
MODELS=("gpt-4.1-2025-04-14" "claude-opus-4-1-20250805" "meta-llama/llama-4-scout")
DATASETS=("truthfulqa_complete" "simpleqa_complete")
SAMPLES=100

echo "Starting Unilateral vs Bilateral Comparison Study"
echo "================================================"

# Step 1: Run unilateral evaluations
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo ""
        echo "Running unilateral evaluation: $model on $dataset"
        python unilateral_evaluator.py \
            --dataset "standard_datasets/${dataset}.json" \
            --model "$model" \
            --samples $SAMPLES \
            --prompt-style direct
    done
done

echo ""
echo "================================================"
echo "Step 2: Generate comparison reports"
echo "================================================"

# Step 2: Generate comparisons
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo ""
        echo "Comparing results for: $model on $dataset"
        python compare_bilateral_unilateral.py \
            --model "$model" \
            --dataset "$dataset" \
            --output "results/comparison_${model//\//_}_${dataset}.json"
    done
done

echo ""
echo "================================================"
echo "Comparison study complete!"
echo "Check results/ directory for detailed reports"