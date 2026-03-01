#!/bin/bash

# Example usage script for balanced cluster generation

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0

# Run balanced generation with multiple questioner models.
# Replace the model names and centroids_dataset with your own HuggingFace paths.
python balanced_cluster_generation.py \
    --models \
        "${HUGGINGFACENAME}/questioner-v1" \
        "${HUGGINGFACENAME}/questioner-v2" \
    --centroids_dataset "${HUGGINGFACENAME}/math_clusters" \
    --centroids_file centroids.npy \
    --embedding_model Qwen/Qwen3-Embedding-0.6B \
    --output_file balanced_questions_2models.json \
    --max_per_cluster 20 \
    --questions_per_model 100 \
    --max_iterations 50 \
    --device cuda:0 \
    --embedding_device cuda:0

echo "Generation complete!"
echo "Check balanced_questions_2models.json for results"
