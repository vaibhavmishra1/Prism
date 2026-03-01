
solver_model_path=$1
questioner_model_path=$2
save_path=$3
# Resolve project root (two levels up: scripts/ -> train/ -> project root) and add to PYTHONPATH
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

echo "========================================="
echo "Questioner Training Configuration"
echo "========================================="
echo "Solver model path: $solver_model_path"
echo "Questioner model path: $questioner_model_path"
echo "Save path: $save_path"
echo "========================================="

# Generate unique RUN_ID
RUN_ID=$(date +%s%N)
export RUN_ID
echo "RUN_ID=$RUN_ID"

# Start vLLM services (for majority voting)
bash vllm_service_init/start.sh $solver_model_path $RUN_ID
echo "vLLM services started with RUN_ID=$RUN_ID"

# Train Questioner with Rentropy reward function
echo "Start training questioner with Rentropy: $questioner_model_path -> $save_path"

# Training uses GPUs 0,1,2,3; vLLM uses 4,5,6; embedding uses 7
# Include GPU 7 in visible devices so embedding model can access it
CUDA_VISIBLE_DEVICES=0,1,2,3,7 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=4096 \
    worker.actor.model.model_path=$questioner_model_path \
    trainer.experiment_name=$save_path \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/$save_path \
    worker.reward.reward_function=./examples/reward_function/caller_rentropy.py:compute_score \
    trainer.val_freq=-1 \
    trainer.n_gpus_per_node=4 \
    data.format_prompt=./examples/format_prompt/questioner.jinja \
    data.rollout_batch_size=512 \
    data.val_batch_size=1024 \
    worker.rollout.n=4 \
    worker.actor.global_batch_size=16 \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.actor.micro_batch_size_per_device_for_experience=8 \
    trainer.total_epochs=6 \
    trainer.max_steps=6 \
    trainer.save_freq=5

sleep 5
echo "Stopping vLLM service (PID: $VLLM_PID)..."
kill $VLLM_PID 2>/dev/null
sleep 3
if ps -p $VLLM_PID > /dev/null 2>&1; then
    echo "Force killing vLLM..."
    kill -9 $VLLM_PID 2>/dev/null
fi
# Clean up any orphaned vLLM processes on port 5000
pkill -f "vllm_service_init.*port 5000" 2>/dev/null || true
sleep 2

# Merge model
echo "merging model"
# Find the latest checkpoint directory dynamically
CHECKPOINT_DIR="${STORAGE_PATH}/models/$save_path"
if [ -f "${CHECKPOINT_DIR}/latest_global_step.txt" ]; then
    # Use tracker file if available
    LATEST_STEP=$(cat "${CHECKPOINT_DIR}/latest_global_step.txt")
    ACTOR_PATH="${CHECKPOINT_DIR}/global_step_${LATEST_STEP}/actor"
else
    # Fallback: find the highest global_step_* directory
    LATEST_STEP=$(ls -d ${CHECKPOINT_DIR}/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n | tail -1)
    if [ -z "$LATEST_STEP" ]; then
        echo "ERROR: No checkpoint found in ${CHECKPOINT_DIR}"
        exit 1
    fi
    ACTOR_PATH="${CHECKPOINT_DIR}/global_step_${LATEST_STEP}/actor"
fi

if [ ! -d "$ACTOR_PATH" ]; then
    echo "ERROR: Checkpoint directory not found: $ACTOR_PATH"
    exit 1
fi

echo "Using checkpoint at: $ACTOR_PATH"
python scripts/model_merger.py --local_dir "$ACTOR_PATH"

sleep 10


echo "questioner training finished"
echo "Questioner training finished 