import vllm
import torch
from transformers import AutoTokenizer
import argparse
from typing import List, TYPE_CHECKING
from vllm.outputs import RequestOutput
from evaluation.datasets_loader import get_dataset_handler
import json
import regex as re
import os
import sys
import numpy as np
import yaml

# Add project root to path so cluster_space can be imported.
# question_generate.py is at: <project_root>/train/question_generate/question_generate.py
# Going up 3 levels: question_generate/ -> train/ -> project_root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
# Keep RENTROPY_ROOT alias for backward compatibility with cluster_space imports
RENTROPY_ROOT = PROJECT_ROOT

def get_storage_path() -> str:
    """Resolve storage path, defaulting to <project_root>/train/storage if env var is missing."""
    default_path = os.path.join(PROJECT_ROOT, "train", "storage")
    return os.getenv("STORAGE_PATH", default_path)

STORAGE_PATH = get_storage_path()

# Note: CUDA_VISIBLE_DEVICES is set by the bash script for vLLM
# GPU 7 is reserved for embedding model - we'll use device="cuda:7" explicitly
# Don't override CUDA_VISIBLE_DEVICES here as it would break vLLM GPU assignment

def extract_boxed(text):
    results, i = [], 0
    prefix = r'\boxed{'
    plen = len(prefix)

    while True:
        start = text.find(prefix, i)
        if start == -1:
            break   # no more \boxed{…}

        j = start + plen
        depth = 1
        while j < len(text) and depth:
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
            j += 1

        results.append(text[start + plen : j - 1])
        i = j

    return results

def get_response_mask(response_ids, eos_token_id, dtype):
    batch_size, seq_len = response_ids.shape
    mask = torch.ones((batch_size, seq_len), dtype=dtype)
    for i in range(batch_size):
        for j in range(seq_len):
            if response_ids[i][j] == eos_token_id:
                mask[i][j:] = 0
                break
    return mask

def load_rentropy_config() -> dict:
    """Load rentropy configuration from yaml file."""
    config_path = os.path.join(RENTROPY_ROOT, "rentropy_config.yaml")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    else:
        return {
            "centroids_path": None,
            "weights": {"rarity": 0.5, "batch_uniqueness": 0.2, "within_cluster_uniqueness": 0.2},
        }

def compute_diversity_score_readonly(question: str, assigner, config: dict) -> float:
    """
    Compute diversity score for a single question WITHOUT updating cluster counts.
    This is read-only mode for evaluation purposes.
    
    Args:
        question: Question string
        assigner: ClusterAssigner instance (with current cluster counts)
        config: Rentropy configuration dict
    
    Returns:
        Diversity score (0.0 if mode=1 or no valid question)
    """
    if not question or not question.strip():
        return 0.0
    
    
    weights = config.get("weights", {})
    
    # Assign cluster
    cluster_ids = assigner.assign_clusters([question])
    if len(cluster_ids) == 0:
        return 0.0
    
    diversity_score = 0.0

    rarity_rewards = assigner.compute_rarity_reward(cluster_ids)
    rarity_weight = weights.get("rarity", 1.0)
    diversity_score += rarity_weight * rarity_rewards[0]

    return diversity_score

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = vllm.LLM(
        model=args.model,
        tokenizer=args.model,
        # gpu_memory_utilization=0.8,
        seed=int(args.suffix),
    )
    dataset_handler = get_dataset_handler("math")
    questions, answers = dataset_handler.load_data()
    question = questions[0]
    answer = answers[0]
    
    rentropy_config = None
    cluster_assigner = None
    if args.compute_diversity:
        # Load Rentropy config and cluster assigner for diversity calculation
        # Import ClusterAssigner here (after vLLM is initialized) to avoid GPU conflicts
        from cluster_space.cluster_assigner import ClusterAssigner

        rentropy_config = load_rentropy_config()
        if rentropy_config.get("centroids_path"):
            centroids_path = rentropy_config.get("centroids_path")
            if not os.path.isabs(centroids_path):
                centroids_path = os.path.join(RENTROPY_ROOT, centroids_path)
            
            init_counts_path = rentropy_config.get("init_cluster_counts_path")
            if init_counts_path and not os.path.isabs(init_counts_path):
                init_counts_path = os.path.join(RENTROPY_ROOT, init_counts_path)
            
            if os.path.exists(centroids_path):
                try:
                    cluster_assigner = ClusterAssigner(
                        centroids_path=centroids_path,
                        embedding_model=rentropy_config.get("embedding_model", "Qwen/Qwen3-Embedding-0.6B"),
                        ema_decay=rentropy_config.get("ema_decay", 0.99),
                        smoothing_alpha=rentropy_config.get("smoothing_alpha", 1.0),
                        init_counts_path=init_counts_path,
                        device=args.embedding_device,
                    )
                    print(f"[Question Generate {args.suffix}] Loaded cluster assigner on {args.embedding_device} (read-only mode)")
                except Exception as e:
                    print(f"[Question Generate {args.suffix}] WARNING: Failed to load cluster assigner: {e}")
                    print(f"[Question Generate {args.suffix}] Continuing without diversity scores")
            else:
                print(f"[Question Generate {args.suffix}] WARNING: Centroids not found at {centroids_path}")
                print(f"[Question Generate {args.suffix}] Continuing without diversity scores")
    chat = [
        {
            "role": "system",
            "content": (
                "You are an expert competition-math problem setter.\n"
                "FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial problem. "
                "The problem could come from any field of mathematics, including but not limited to algebra, geometry, number theory, combinatorics, prealgebra, probability, statistics, and calculus. "
                "Aim for a difficulty such that fewer than 30 % of advanced high-school students could solve it. "
                "Avoid re-using textbook clichés or famous contest problems.\n"
                "THEN, without revealing any of your private thoughts, output **exactly** the following two blocks:\n\n"
                "<question>\n"
                "{The full problem statement on one or more lines}\n"
                "</question>\n\n"
                r"\boxed{final_answer}"
                "\n\n"
                "Do NOT output anything else—no explanations, no extra markup."
            )
        },
        {
            "role": "user",
            "content": (
                "Generate one new, challenging reasoning question now. "
                "Remember to format the output exactly as instructed."
            )
        }
    ]

    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            chat, 
            tokenize=False,
            add_generation_prompt=True, 
            add_special_tokens=True
        )
    else:
        prompt = "system: " + chat[0]["content"] + '\n' + "user: " + chat[1]["content"]
    sample_params = vllm.SamplingParams(
        max_tokens=4096,
        temperature=1.0,
        top_p=0.95,
        n=1,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    completions: List[RequestOutput] = model.generate([prompt]*args.num_samples, sampling_params=sample_params)
    results=[]
    for completion in completions:
        response = completion.outputs[0].text
        try:
            questions = re.findall(r"<question>(.*?)</question>", response, re.DOTALL)
            answers = extract_boxed(response)

            if questions and answers:
                question = questions[-1].strip()
                answer = answers[-1].strip()
                
                # Calculate diversity score (read-only, no cluster count updates)
                diversity_score = 0.0
                if cluster_assigner is not None and rentropy_config is not None:
                    try:
                        diversity_score = compute_diversity_score_readonly(question, cluster_assigner, rentropy_config)
                    except Exception as e:
                        print(f"[Question Generate {args.suffix}] WARNING: Failed to compute diversity score: {e}")
                        diversity_score = 0.0
                
                results.append({
                    "question": question, 
                    "answer": answer, 
                    "score": 0,
                    "diversity_score": float(diversity_score)
                })
            else:
                results.append({"question": response, "answer": "", "score": -1, "diversity_score": 0.0})
        except:
            results.append({"question": response, "answer": "", "score": -1, "diversity_score": 0.0})
    
    output_dir = os.path.join(STORAGE_PATH, "generated_question")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{args.save_name}_{args.suffix}.json")
    print(f"[Question Generate {args.suffix}] Saving results to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    
    # Print summary
    valid_questions = [r for r in results if r.get("question") and r.get("score") == 0]
    if valid_questions and cluster_assigner:
        diversity_scores = [r.get("diversity_score", 0.0) for r in valid_questions]
        print(f"[Question Generate {args.suffix}] Generated {len(valid_questions)} valid questions")
        print(f"[Question Generate {args.suffix}] Diversity scores - min: {min(diversity_scores):.4f}, max: {max(diversity_scores):.4f}, mean: {np.mean(diversity_scores):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--num_samples", type=int, default=1250, help="Number of samples to generate")
    parser.add_argument("--suffix", type=str, default="", help="Suffix to add to the output file")
    parser.add_argument("--save_name", type=str, default="", help="")
    parser.add_argument("--compute_diversity", action="store_true", help="Compute diversity scores during generation")
    parser.add_argument("--embedding_device", type=str, default="cuda:0", help="Embedding device for diversity scoring")
    args = parser.parse_args()

    main(args) 