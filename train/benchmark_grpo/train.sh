#!/bin/bash
set -euo pipefail

# ============================================================================
# GRPO Training: All benchmark datasets merged
# Datasets:      Math500, GSM8K, AMC23, Minerva, OlympiadBench, AIME2024/2025
# Hardware:      8x A100 80GB
#
# Usage:
#   bash benchmark_grpo/train.sh [model_path]
#
# Args:
#   model_path  - HuggingFace model path (default: Qwen/Qwen3-4B-Base)
#
# This script will:
#   1. Prepare the dataset (download all benchmarks + merge) if not already done
#   2. Run GRPO training using the verl framework
#   3. Run evaluation on standard math benchmarks
# ============================================================================

MODEL_PATH="${1:-Qwen/Qwen3-4B-Base}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export VLLM_DISABLE_COMPILE_CACHE=1

# ======================== Step 1: Prepare Dataset ========================
if [ ! -f "$SCRIPT_DIR/data/train.parquet" ]; then
    echo "============================================"
    echo "Step 1: Downloading & merging all benchmark datasets"
    echo "============================================"
    python "$SCRIPT_DIR/prepare_data.py"
else
    echo "Dataset already prepared at $SCRIPT_DIR/data/"
fi

# ======================== Step 2: GRPO Training ========================
echo ""
echo "============================================"
echo "Step 2: Starting GRPO training"
echo "  Model:    $MODEL_PATH"
echo "  Datasets: All benchmarks (merged)"
echo "  GPUs:     8x A100 80GB"
echo "  Rollouts: 8 per question"
echo "  Config:   benchmark_grpo/config.yaml"
echo "============================================"
echo ""

python3 -m verl.trainer.main \
    config=benchmark_grpo/config.yaml \
    worker.actor.model.model_path="$MODEL_PATH"

echo ""
echo "============================================"
echo "Step 2: GRPO training complete"
echo "============================================"

# ======================== Step 3: Evaluate ========================
echo ""
echo "============================================"
echo "Step 3: Running evaluation on $MODEL_PATH"
echo "============================================"

bash evaluation/evaluate.bash "$MODEL_PATH"

echo ""
echo "============================================"
echo "All done!"
echo "  Model: $MODEL_PATH"
echo "============================================"
