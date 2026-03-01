solver_model_path=$1
questioner_model_path=$2
experiment_name=$3
# Resolve project root (two levels up: scripts/ -> train/ -> project root)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export STORAGE_PATH="${STORAGE_PATH:-${PROJECT_ROOT}/train/storage}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# HUGGINGFACENAME must be set in the environment before calling this script
if [ -z "$HUGGINGFACENAME" ]; then
    echo "ERROR: HUGGINGFACENAME environment variable is not set."
    echo "  export HUGGINGFACENAME=your_hf_username"
    exit 1
fi

# Helper function to find latest checkpoint path
find_latest_checkpoint() {
    local checkpoint_dir="$1"
    # Find the highest global_step_* directory
    local latest_step=$(ls -d ${checkpoint_dir}/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n | tail -1)
    if [ -z "$latest_step" ]; then
        echo "ERROR: No checkpoint found in ${checkpoint_dir}" >&2
        return 1
    fi
    echo "${checkpoint_dir}/global_step_${latest_step}/actor"
}
mkdir -p \
  "$STORAGE_PATH/evaluation" \
  "$STORAGE_PATH/models" \
  "$STORAGE_PATH/generated_question" \
  "$STORAGE_PATH/temp_results"

echo $STORAGE_PATH

echo "start train solver $experiment_name $solver_model_path $questioner_model_path" 

export VLLM_DISABLE_COMPILE_CACHE=1


# echo 'start generate question'
# bash question_generate/question_generate.bash $questioner_model_path 2000 $experiment_name


# echo 'start compute diversity scores'
# CUDA_VISIBLE_DEVICES=7 python question_generate/compute_diversity_scores.py --experiment_name $experiment_name --num_shards 7 --device cuda:0


# echo 'start evaluate generated question'
# bash question_evaluate/evaluate.sh $solver_model_path $experiment_name


# echo 'start upload'
# python question_evaluate/upload.py --repo_name ${experiment_name} --max_score 0.81 --min_score 0.49 --min_diversity 0.00 --experiment_name ${experiment_name}
echo 'start train'

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=4096 \
    worker.actor.model.model_path=$solver_model_path \
    trainer.experiment_name=${experiment_name} \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/${experiment_name}/ \
    data.train_files=${HUGGINGFACENAME}/${experiment_name}@train \
    trainer.total_epochs=10 \
    trainer.max_steps=50 \
    data.format_prompt=./examples/format_prompt/solver.jinja \
    trainer.val_freq=2 \
    trainer.save_freq=5 \
    worker.rollout.n=8 \
    trainer.n_gpus_per_node=8 \
    worker.actor.global_batch_size=128 \
    worker.actor.micro_batch_size_per_device_for_update=4 \
    worker.actor.micro_batch_size_per_device_for_experience=16 \
    worker.actor.ppo_epochs=2 \
    algorithm.kl_coef=1.0e-4 \
    worker.actor.optim.lr_warmup_ratio=0.1 \
    worker.actor.optim.warmup_style=cosine \
    worker.actor.optim.min_lr_ratio=0.1 

echo "merging model"
# Find the latest checkpoint dynamically instead of hardcoding global_step_15
LATEST_CHECKPOINT=$(find_latest_checkpoint "${STORAGE_PATH}/models/${experiment_name}")
if [ $? -eq 0 ]; then
    echo "Found checkpoint at: $LATEST_CHECKPOINT"
    python scripts/model_merger.py --local_dir "$LATEST_CHECKPOINT"
else
    echo "ERROR: Could not find checkpoint for ${experiment_name}"
    exit 1
fi

sleep 10
sleep 5
echo "Stopping vLLM service (PID: $VLLM_PID)..."
kill $VLLM_PID 2>/dev/null
sleep 3
if ps -p $VLLM_PID > /dev/null 2>&1; then
    echo "Force killing vLLM..."
    kill -9 $VLLM_PID 2>/dev/null
fi
pkill -f "vllm_service_init.*port 5000" 2>/dev/null || true


sleep 10

echo "solver training finished"
echo "Solver training finished"
bash evaluation/evaluate.bash ${LATEST_CHECKPOINT}/huggingface
