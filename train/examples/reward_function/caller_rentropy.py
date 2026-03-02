
import regex as re
from typing import Dict, List
import json
import os
import time
import random
import requests
import yaml
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from collections import Counter
from mathruler.grader import extract_boxed_content, grade_answer

# Add cluster_space to path
import sys
RENTROPY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, RENTROPY_ROOT)

# Set CUDA_VISIBLE_DEVICES for embedding model BEFORE importing torch/ClusterAssigner
# GPU 7 is reserved for embedding model computation (GPUs 4,5,6 are used by vLLM servers)
# This must be set before any CUDA initialization
EMBEDDING_GPU = os.getenv("RENTROPY_EMBEDDING_GPU", "7")
os.environ['CUDA_VISIBLE_DEVICES'] = EMBEDDING_GPU

from cluster_space.cluster_assigner import ClusterAssigner

STORAGE_PATH = os.getenv(
    "STORAGE_PATH",
    os.path.join(RENTROPY_ROOT, "train", "storage"),
)

# ============================================================================
# Rentropy Configuration Loading
# ============================================================================

def load_rentropy_config() -> dict:
    """Load rentropy configuration from yaml file."""
    config_path = os.path.join(RENTROPY_ROOT, "rentropy_config.yaml")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"[Rentropy] Loaded config from {config_path}")
        return config
    else:
        print(f"[Rentropy] Config not found at {config_path}, using defaults (mode 1)")
        return {
            
            "centroids_path": None,
            "weights": {"rarity": 0.5, "batch_uniqueness": 0.2, "within_cluster_uniqueness": 0.2},
            "majority_vote_threshold": 0.5,
            "ema_decay": 0.99,
            "smoothing_alpha": 1.0,
            "scale_diversity_by_zpd": False,  # Changed default to False to avoid double scaling
            "lambda_weight": 5.0,  # Weight for diversity reward in final score
            "use_zpd_base_score": False,  # Use raw base_score instead of ZPD transformation
            "log_cluster_stats_freq": 1,  # Log cluster stats every N steps
        }

# Load config once at module import
RENTROPY_CONFIG = load_rentropy_config()

# Lazy-loaded cluster assigner
_cluster_assigner = None

# Global step counter for logging
_global_step_counter = 0

# Global variable to store last reward distribution
_last_reward_distribution = None

def get_cluster_assigner() -> ClusterAssigner:
    """Get or create the cluster assigner instance.
    
    Note: Creates cluster assigner for all modes (including mode 1) to enable
    logging of cluster statistics for comparison purposes.
    
    If init_cluster_counts_path is specified in config, cluster counts will be
    initialized from the previous iteration's log file instead of uniform.
    """
    global _cluster_assigner
    if _cluster_assigner is None:
        centroids_path = RENTROPY_CONFIG.get("centroids_path")
        if centroids_path and not os.path.isabs(centroids_path):
            centroids_path = os.path.join(RENTROPY_ROOT, centroids_path)
        
        # Get optional init counts path
        init_counts_path = RENTROPY_CONFIG.get("init_cluster_counts_path")
        if init_counts_path and not os.path.isabs(init_counts_path):
            init_counts_path = os.path.join(RENTROPY_ROOT, init_counts_path)
        
        if centroids_path and os.path.exists(centroids_path):
            _cluster_assigner = ClusterAssigner(
                centroids_path=centroids_path,
                embedding_model=RENTROPY_CONFIG.get("embedding_model", "Qwen/Qwen3-Embedding-0.6B"),
                ema_decay=RENTROPY_CONFIG.get("ema_decay", 0.99),
                smoothing_alpha=RENTROPY_CONFIG.get("smoothing_alpha", 1.0),
                init_counts_path=init_counts_path,
            )
            
            
        else:
            print(f"[Rentropy] WARNING: centroids not found at {centroids_path}")
    return _cluster_assigner

# ============================================================================
# Utility Functions
# ============================================================================

def generate_temp_filename(prefix="temp", suffix=".json"):
    timestamp = int(time.time() * 1000)
    rand_part = random.randint(0, 99999)
    return f"{STORAGE_PATH}/temp_results/{prefix}_{timestamp}_{rand_part}{suffix}"

def split_list(lst, n=4):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

os.environ["NO_PROXY"] = "0.0.0.0,127.0.0.1"

# Number of vLLM servers (set via env var for smoke test flexibility)
# Default is 3 because GPU 7 is reserved for embedding model
NUM_VLLM_SERVERS = int(os.getenv("RENTROPY_NUM_SERVERS", "3"))

def fetch(index, filepath):
    """Call vLLM server to process a batch."""
    port = 5000 + index
    try:
        response = requests.get(f"http://0.0.0.0:{port}/hello?name={filepath}", timeout=3000)  # 40 minutes
        print(f"[Server {port}] {response.status_code}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"[Server {port}] Error: {e}")
        return False

def generate_results(data):
    """Call vLLM servers to get majority voting scores.
    
    Supports 1-4 servers (configurable via RENTROPY_NUM_SERVERS env var).
    """
    num_servers = min(NUM_VLLM_SERVERS, len(data))
    if num_servers < 1:
        num_servers = 1
    
    datas = split_list(data, num_servers)
    random_names = [generate_temp_filename(prefix=f"temp_{i}", suffix=".json") for i in range(num_servers)]
    
    # Write data files
    for i in range(num_servers):
        with open(random_names[i], 'w') as f:
            json.dump(datas[i], f, indent=4)

    # Call servers in parallel
    final_results = []
    with ThreadPoolExecutor(max_workers=num_servers) as executor:
        futures = [executor.submit(fetch, i, random_names[i]) for i in range(num_servers)]
        for future in as_completed(futures):
            result = future.result()
            if not result:
                print("[Rentropy] WARNING: Some server calls failed")

    # Collect results
    for i in range(num_servers):
        results_file = random_names[i].replace('.json', '_results.json')
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                final_results.extend(json.load(f))
            os.remove(results_file)
        else:
            print(f"[Rentropy] WARNING: Results file not found: {results_file}")
            # Return empty results for this batch
            final_results.extend([{"question": d.get("question", ""), "answer": "", "score": -1} 
                                 for d in datas[i]])
    
    return final_results

def format_reward(predict: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, predict)
    return 1.0 if format_match else 0.0

def accuracy_reward(predict: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0

# ============================================================================
# Rentropy Diversity Reward
# ============================================================================

def compute_diversity_rewards(questions: List[str], base_scores: List[float]) -> List[float]:
   
    threshold = RENTROPY_CONFIG.get("majority_vote_threshold", 0.5)
    weights = RENTROPY_CONFIG.get("weights", {})
    
    n = len(questions)
    diversity_rewards = np.zeros(n)
    
    # Get cluster assigner (needed even for mode 1 for logging purposes)
    assigner = get_cluster_assigner()
    if assigner is None:
        print("[Rentropy] No cluster assigner available, returning zero diversity rewards")
        return diversity_rewards.tolist()
    
    # Filter to questions that pass majority vote threshold
    valid_indices = [i for i, s in enumerate(base_scores) if s >= threshold and questions[i]]
    valid_questions = [questions[i] for i in valid_indices]
    valid_base_scores = [base_scores[i] for i in valid_indices]
    
    if not valid_questions:
        return diversity_rewards.tolist()
    
    # Assign clusters (for all modes, including mode 1 for logging)
    cluster_ids = assigner.assign_clusters(valid_questions)
    
    zpd_scores = np.array([min(s, 1 - s) for s in valid_base_scores])
    
    # Normalize ZPD scores to [0, 1] range (max ZPD is 0.5 at score=0.5)
    # This makes scaling factor 1.0 at optimal difficulty
    zpd_scale_factors = np.ones(len(valid_indices))
 
    rarity_rewards = assigner.compute_rarity_reward(cluster_ids)
    for idx, valid_idx in enumerate(valid_indices):
        scaled_reward =  rarity_rewards[idx]
        diversity_rewards[valid_idx] += scaled_reward

    # Update cluster counts for valid questions
    assigner.update_counts(cluster_ids)
    
    # Log cluster statistics
    _log_cluster_stats(assigner, cluster_ids, valid_base_scores, valid_questions)
    
    return diversity_rewards.tolist()

def _log_cluster_stats(assigner: ClusterAssigner, cluster_ids: np.ndarray, 
                       base_scores: list, questions: list, reward_distribution: dict = None):
    """Log cluster statistics to file for later analysis."""
    global _global_step_counter, _last_reward_distribution
    _global_step_counter += 1
    
    # Use the global reward distribution if not provided
    if reward_distribution is None:
        reward_distribution = _last_reward_distribution
    
    # Only log periodically to avoid too many files
    log_freq = RENTROPY_CONFIG.get("log_cluster_stats_freq", 1)
    if _global_step_counter % log_freq != 0:
        return
    
    # Create logs directory
    log_dir = os.path.join(STORAGE_PATH, "cluster_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Get cluster statistics
    stats = assigner.get_stats()
    
    # Compute per-cluster scores
    cluster_score_sums = {}
    cluster_question_counts = {}
    
    for cid, score in zip(cluster_ids, base_scores):
        cid = int(cid)
        if cid not in cluster_score_sums:
            cluster_score_sums[cid] = 0.0
            cluster_question_counts[cid] = 0
        cluster_score_sums[cid] += score
        cluster_question_counts[cid] += 1
    
    # Compute averages
    cluster_avg_scores = {
        cid: cluster_score_sums[cid] / cluster_question_counts[cid]
        for cid in cluster_score_sums
    }
    
    # Build log entry
    log_entry = {
        "step": _global_step_counter,
        "timestamp": time.time(),
        "num_questions": len(cluster_ids),
        "cluster_stats": stats,
        "batch_cluster_distribution": cluster_question_counts,
        "batch_cluster_avg_scores": cluster_avg_scores,
    }
    
    # Add reward distribution if provided
    if reward_distribution:
        log_entry["reward_distribution"] = reward_distribution
    
    # Save to JSON file
    log_file = os.path.join(log_dir, f"cluster_stats_step_{_global_step_counter:06d}.json")
    with open(log_file, 'w') as f:
        json.dump(log_entry, f, indent=2)
    
    print(f"[Rentropy] Logged cluster stats to {log_file}")
    
    # Print reward distribution summary to console
    if reward_distribution:
        print(f"[Rentropy] Reward Distribution: "
              f"reward=1.0: {reward_distribution.get('reward_1', 0)}, "
              f"reward=0.0: {reward_distribution.get('reward_0', 0)}, "
              f"reward=-1.0: {reward_distribution.get('reward_neg1', 0)}, "
              f"other: {reward_distribution.get('reward_other', 0)}")

# ============================================================================
# Main Reward Function
# ============================================================================

def compute_zpd_reward(base_score: float, diversity_reward: float, lambda_weight: float) -> float:
    
    if base_score < 0.5 or base_score > 0.9:
        return 0.0
    
    # 2. ZPD with peak at 0.75 (optimal difficulty)
    # Linear interpolation: 1.0 at 0.75, 0.0 at boundaries (0.5 and 1.0)
    # But we cut off at 0.9, so effective range is [0.5, 0.9]
    zpd = max(0.0, 1.0 - abs(base_score - 0.75) / 0.4)
    
    # 3. Gated multiplicative reward
    # ZPD acts as a gate, diversity provides bonus
    final = zpd * (1.0 + lambda_weight * diversity_reward)
    
    return final

def compute_score(predicts: List[str], ground_truths: List[str], format_weight: float = 0.1, file_path: str = "") -> List[Dict[str, float]]:
    """
    Compute rewards with Rentropy diversity bonus.
    
    Returns dict with:
        - overall: final score (ZPD-gated multiplicative with diversity)
        - format: 1 if valid format, 0 otherwise
        - accuracy: diversity reward (for logging compatibility)
        - diversity: diversity reward bonus
        - base_score: original majority voting score
        - zpd: ZPD value (peaks at 0.7)
    """
    results = []
    
    # Extract questions and answers from predictions
    for i in range(len(predicts)):
        questions = re.findall(r"<question>(.*?)</question>", predicts[i], re.DOTALL)
        answers = extract_boxed_content(predicts[i])
        if questions and answers:
            try:
                question = questions[-1].strip()
                answer = answers[-1].strip()
                results.append({"question": question, "answer": answer})
            except:
                results.append({"question": "", "answer": ""})
        else:
            results.append({"question": "", "answer": ""})
    
    # Get majority voting scores from vLLM servers
    final_results = generate_results(results)
    
    # Extract questions and base scores
    questions = [r['question'] for r in final_results]
    base_scores = [r['score'] if r['question'] else -1 for r in final_results]
    
    # Compute diversity rewards
    diversity_rewards = compute_diversity_rewards(questions, base_scores)
    
    # Get lambda weight for diversity reward
    lambda_weight = RENTROPY_CONFIG.get("lambda_weight", 1.0)
    
    # Track reward distribution
    reward_counts = {
        "reward_1": 0,      # Questions with reward = 1.0
        "reward_0": 0,      # Questions with reward = 0.0
        "reward_neg1": 0,   # Questions with reward = -1.0
        "reward_other": 0,  # Questions with other rewards
        "total": len(final_results)
    }
    
    scores = []
    for i in range(len(final_results)):
        base_score = final_results[i]["score"]
        has_valid_question = bool(final_results[i]['question'])
        
        if has_valid_question and base_score >= 0:
            # Use new ZPD-gated multiplicative reward
            final_score = compute_zpd_reward(base_score, diversity_rewards[i], lambda_weight)
            # Compute ZPD for logging (matches the formula in compute_zpd_reward)
            
            zpd = min(base_score, 1.0 - base_score)
        else:
            final_score = -1
            zpd = 0.0
        
        # Count reward distribution
        if abs(final_score - 1.0) < 1e-6:
            reward_counts["reward_1"] += 1
        elif abs(final_score - 0.0) < 1e-6:
            reward_counts["reward_0"] += 1
        elif abs(final_score - (-1.0)) < 1e-6:
            reward_counts["reward_neg1"] += 1
        else:
            reward_counts["reward_other"] += 1
        
        scores.append({
            "overall": final_score,
            "format": 1 if has_valid_question else 0,
            "accuracy": diversity_rewards[i],  # For logging compatibility
            "diversity": diversity_rewards[i],
            "base_score": base_score,
            "zpd": zpd,
        })
    
    # Log reward distribution to console (every batch)
    print(f"[Rentropy] Reward Distribution - "
          f"Batch size: {reward_counts['total']}, "
          f"reward=1.0: {reward_counts['reward_1']} ({100*reward_counts['reward_1']/reward_counts['total']:.1f}%), "
          f"reward=0.0: {reward_counts['reward_0']} ({100*reward_counts['reward_0']/reward_counts['total']:.1f}%), "
          f"reward=-1.0: {reward_counts['reward_neg1']} ({100*reward_counts['reward_neg1']/reward_counts['total']:.1f}%), "
          f"other: {reward_counts['reward_other']} ({100*reward_counts['reward_other']/reward_counts['total']:.1f}%)")
    
    # Add reward distribution to next cluster stats log
    # We'll pass it through the logging mechanism by storing it globally
    global _last_reward_distribution
    _last_reward_distribution = reward_counts
    
    return scores
