
Questioner_base_model=$1
Solver_base_model=$2
Model_abbr=$3
# Validate required arguments
if [ -z "$Questioner_base_model" ] || [ -z "$Solver_base_model" ] || [ -z "$Model_abbr" ]; then
    echo "ERROR: Missing required arguments"
    echo "Usage: bash scripts/main_rentropy.sh <questioner_base_model> <solver_base_model> <model_abbr>"
    echo "Example: bash scripts/main_rentropy.sh Qwen/Qwen3-4B-Base meta-llama/Llama-2-7b qwen3-4b"
    exit 1
fi

# Resolve the project root (two levels up from this script: scripts/ -> train/ -> project root)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# HUGGINGFACENAME must be set in the environment before calling this script
if [ -z "$HUGGINGFACENAME" ]; then
    echo "ERROR: HUGGINGFACENAME environment variable is not set."
    echo "  export HUGGINGFACENAME=your_hf_username"
    exit 1
fi

# Default STORAGE_PATH to <project_root>/train/storage if not already set
export STORAGE_PATH="${STORAGE_PATH:-${PROJECT_ROOT}/train/storage}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"


echo "Model_abbr: $Model_abbr"
echo "Questioner Base Model: $Questioner_base_model"
echo "Solver Base Model: $Solver_base_model"
echo "Running Rentropy experiment with cluster-entropy diversity reward"
echo "STORAGE_PATH: $STORAGE_PATH"
echo "HUGGINGFACENAME: $HUGGINGFACENAME"
mkdir -p \
  "$STORAGE_PATH/evaluation" \
  "$STORAGE_PATH/models" \
  "$STORAGE_PATH/generated_question" \
  "$STORAGE_PATH/temp_results"

# Set HuggingFace timeout environment variables to prevent hanging
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HUB_DOWNLOAD_TIMEOUT_STREAM=600

# Helper function to validate if model download is complete
validate_model_download() {
    local model_dir=$1
    
    # Check if config.json exists
    if [ ! -f "$model_dir/config.json" ]; then
        return 1
    fi
    
    # Check if model weight files exist (either .safetensors or .bin files)
    local weight_count=$(find "$model_dir" -maxdepth 1 \( -name "*.safetensors" -o -name "pytorch_model*.bin" \) 2>/dev/null | wc -l)
    
    if [ "$weight_count" -eq 0 ]; then
        # No weight files found, check if there's an index file indicating sharded model
        if [ -f "$model_dir/model.safetensors.index.json" ] || [ -f "$model_dir/pytorch_model.bin.index.json" ]; then
            echo "WARNING: Found index file but no weight files. Model download is incomplete." >&2
            return 1
        fi
        # For single-file models without index, should have at least one weight file
        return 1
    fi
    
    return 0
}

# Helper function to download and prepare a model
download_and_prepare_model() {
    local model_name=$1
    local model_type=$2
    
    echo "Pre-downloading $model_type model: $model_name" >&2
    
    local original_model="$model_name"
    local model_dir="$STORAGE_PATH/models/$(echo $model_name | tr '/' '_')"
    
    # Validate existing download
    if [ -d "$model_dir" ]; then
        if validate_model_download "$model_dir"; then
            echo "$model_type model already exists at $model_dir, skipping download." >&2
            local_model_path="$(realpath "$model_dir" 2>/dev/null || echo "$model_dir")"
            echo "Using local $model_type model path: $local_model_path" >&2
            echo "$local_model_path"
            return 0
        else
            echo "WARNING: Incomplete model download detected at $model_dir. Cleaning up..." >&2
            rm -rf "$model_dir"
        fi
    fi
    
    echo "$model_type model not found locally, downloading..." >&2
    python3 scripts/download_hf_model.py --repo-id "$original_model" --max-retries 5 --timeout 600 >&2 || {
        echo "ERROR: Failed to pre-download $model_type model after 5 retries." >&2
        echo "Please check your network connection and HuggingFace token (if required)." >&2
        echo "You can also manually download the model using:" >&2
        echo "  python3 scripts/download_hf_model.py --repo-id \"$original_model\" --max-retries 5 --timeout 600" >&2
        return 1
    }
    
    # Validate the download was successful
    if validate_model_download "$model_dir"; then
        local_model_path="$(realpath "$model_dir" 2>/dev/null || echo "$model_dir")"
        echo "Using local $model_type model path: $local_model_path" >&2
        echo "$local_model_path"
    else
        echo "WARNING: $model_type model download completed but validation failed. Using original path: $original_model" >&2
        echo "$original_model"
    fi
}

# Pre-download models to avoid hanging during training
echo "Pre-downloading models to avoid download issues during training..."

# Download questioner base model
Questioner_base_model=$(download_and_prepare_model "$Questioner_base_model" "Questioner") || exit 1

# Download solver base model
Solver_base_model=$(download_and_prepare_model "$Solver_base_model" "Solver") || exit 1

echo "Both models downloaded successfully."
echo "Questioner model: $Questioner_base_model"
echo "Solver model: $Solver_base_model"

CENTROIDS_PATH="${PROJECT_ROOT}/cluster_space/cluster_data/centroids.npy"
if [ ! -f "$CENTROIDS_PATH" ]; then
    echo "WARNING: Cluster centroids not found at $CENTROIDS_PATH"
    echo "Please run: python ${PROJECT_ROOT}/cluster_space/build_clusters.py first"
    echo "Continuing anyway (will fall back to mode 1 if diversity_mode > 1)"
fi

# Helper function to find latest checkpoint path
find_latest_checkpoint() {
    local checkpoint_dir="$1"
    if [ -f "${checkpoint_dir}/latest_global_step.txt" ]; then
        # Use tracker file if available
        local latest_step=$(cat "${checkpoint_dir}/latest_global_step.txt")
        echo "${checkpoint_dir}/global_step_${latest_step}/actor/huggingface"
    else
        # Fallback: find the highest global_step_* directory
        local latest_step=$(ls -d ${checkpoint_dir}/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n | tail -1)
        if [ -z "$latest_step" ]; then
            echo "ERROR: No checkpoint found in ${checkpoint_dir}" >&2
            return 1
        fi
        echo "${checkpoint_dir}/global_step_${latest_step}/actor/huggingface"
    fi
}

# Initialize first iteration with base models (separate for questioner and solver)
bash scripts/questioner_train_rentropy.sh $Solver_base_model $Questioner_base_model ${Model_abbr}_questioner_v1 
# QUESTIONER_V1_CHECKPOINT=$(find_latest_checkpoint "${STORAGE_PATH}/models/${Model_abbr}_questioner_v1")
# bash scripts/solver_train.sh $Solver_base_model "$QUESTIONER_V1_CHECKPOINT" ${Model_abbr}_solver_v1

# for i in {2..5}; do
#     prev=$((i-1))
    
#     # Train questioner with rentropy reward
#     SOLVER_PREV_CHECKPOINT=$(find_latest_checkpoint "${STORAGE_PATH}/models/${Model_abbr}_solver_v${prev}")
#     QUESTIONER_PREV_CHECKPOINT=$(find_latest_checkpoint "${STORAGE_PATH}/models/${Model_abbr}_questioner_v${prev}")
#     bash scripts/questioner_train_rentropy.sh \
#         "$SOLVER_PREV_CHECKPOINT" \
#         "$QUESTIONER_PREV_CHECKPOINT" \
#         ${Model_abbr}_questioner_v${i} 

#     # Train solver
#     SOLVER_PREV_CHECKPOINT=$(find_latest_checkpoint "${STORAGE_PATH}/models/${Model_abbr}_solver_v${prev}")
#     QUESTIONER_CURR_CHECKPOINT=$(find_latest_checkpoint "${STORAGE_PATH}/models/${Model_abbr}_questioner_v${i}")
#     bash scripts/solver_train.sh \
#         "$SOLVER_PREV_CHECKPOINT" \
#         "$QUESTIONER_CURR_CHECKPOINT" \
#         ${Model_abbr}_solver_v${i}
# done

# bash evaluation/evaluate.bash $Base_model
