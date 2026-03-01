#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Description:
    This script evaluates generated questions from balanced_cluster_generation.py
    against a solver model using vLLM for efficient generation.
    
    Input file structure (balanced_questions_darwin_iter2.json):
    - problem: the question text
    - answer: the golden answer
    - source_model: which model generated it
    - cluster_id: which cluster it belongs to
    
Setup:
    pip install stopit transformers torch vllm

Example Usage:
    CUDA_VISIBLE_DEVICES=0 python evaluator.py \
        --input_file balanced_questions_darwin_iter2.json \
        --model "your_hf_username/solver-model" \
        --num_samples 8 \
        --output_file balanced_questions_darwin_iter2_evaluated.json
'''

import json
import vllm
from transformers import AutoTokenizer
import argparse
import os
import stopit
from mathruler.grader import extract_boxed_content, grade_answer

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Evaluate balanced cluster questions using vLLM.")
parser.add_argument("--input_file", type=str,
                   required=True,
                   help="Path to the input JSON file with questions.")
parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Base", 
                   help="Path to the solver model in Hugging Face format.")
parser.add_argument("--num_samples", type=int, default=8, 
                   help="Number of candidate answers to generate per question (n).")
parser.add_argument("--output_file", type=str,
                   default=None,
                   help="Path to save the evaluation results. Defaults to <input_file>_results.json.")
parser.add_argument("--batch_size", type=int, default=1000,
                   help="Process questions in batches of this size")
parser.add_argument("--start_idx", type=int, default=0,
                   help="Start processing from this index (for parallel processing)")
parser.add_argument("--end_idx", type=int, default=None,
                   help="End processing at this index (for parallel processing)")
args = parser.parse_args()

# Default output file to <input_file stem>_results.json if not specified
if args.output_file is None:
    base, ext = os.path.splitext(args.input_file)
    args.output_file = f"{base}_results{ext}"

# --- Timeout-Protected Grading Function ---
@stopit.threading_timeoutable(default='TIMED_OUT')
def grade_answer_with_timeout(res1, res2):
    """
    Wraps the mathruler 'grade_answer' function with a timeout.
    If the function takes too long, it returns 'TIMED_OUT' instead of hanging.
    """
    return grade_answer(res1, res2)

# --- Main Script Logic ---

# 1. Load and Prepare Data
print(f"Loading data from: {args.input_file}")
try:
    with open(args.input_file, "r") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"ERROR: Input file not found: {args.input_file}")
    exit(1)

print(f"Total questions in file: {len(data)}")

# Select subset based on start/end indices
if args.end_idx is None:
    args.end_idx = len(data)

data_subset = data[args.start_idx:args.end_idx]
print(f"Processing questions {args.start_idx} to {args.end_idx} ({len(data_subset)} questions)")

# Extract questions and golden answers
questions = [item["problem"] for item in data_subset]
golden_answers = [item["answer"] for item in data_subset]
source_models = [item.get("source_model", "unknown") for item in data_subset]
cluster_ids = [item.get("cluster_id", -1) for item in data_subset]

if not questions:
    print("No questions to process. Exiting.")
    exit(0)

# 2. Initialize Model and Tokenizer
print(f"Initializing vLLM for model: {args.model}")
tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
model = vllm.LLM(
    model=args.model,
    tokenizer=args.model,
    gpu_memory_utilization=0.85,
    trust_remote_code=True,
)
sample_params = vllm.SamplingParams(
    max_tokens=2048,
    temperature=1.0,
    top_p=1.0,
    top_k=40,
    stop_token_ids=[tokenizer.eos_token_id],
    n=args.num_samples,
)

# 3. Generate Responses
print(f"Generating {args.num_samples} samples for each question...")
chats = [
    [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": q}
    ] 
    for q in questions
]

if tokenizer.chat_template:
    prompts = [
        tokenizer.apply_chat_template(
            chat, 
            tokenize=False, 
            add_generation_prompt=True, 
            add_special_tokens=True
        ) 
        for chat in chats
    ]
else:
    prompts = [
        "system: " + chat[0]["content"] + '\n' + "user: " + chat[1]["content"] 
        for chat in chats
    ]

responses = model.generate(prompts, sampling_params=sample_params, use_tqdm=True)
print(f"Generation complete.")

# 4. Process and Grade Responses
results_all = []
print(f"Grading responses...")

for idx, (response, golden_answer, question, source_model, cluster_id) in enumerate(
    zip(responses, golden_answers, questions, source_models, cluster_ids)
):
    try:
        # Extract all generated samples (raw) and boxed content
        rollouts = [output.text for output in response.outputs]
        results = [extract_boxed_content(text) for text in rollouts]
        results = [res for res in results if res]  # Filter out None/empty results

        if not results:
            print(f"[{idx}] WARNING: No valid boxed answers found for question: '{question[:50]}...'")
            results_all.append({
                "question": question,
                "questioner_answer": golden_answer,
                "source_model": source_model,
                "cluster_id": cluster_id,
                "score": 0.0,
                "majority_answer": None,
                "majority_count": 0,
                "total_samples": 0,
                "questioner_agreement_count": 0,
                "questioner_agreement_score": 0.0,
                "answer": None,
                "answer_counts": {},
                "results": [],
                "rollouts": rollouts,
            })
            continue

        # Count answer frequencies using grader for matching
        answer_counts = {}
        
        for result in results:
            matched_existing = False
            
            # Try to match with existing answer groups using grader
            for existing_answer in answer_counts:
                # Cheap string comparison first
                if result.strip() == existing_answer.strip():
                    answer_counts[existing_answer] += 1
                    matched_existing = True
                    break
                
                # Use grader for semantic matching
                match_1 = grade_answer_with_timeout(result, existing_answer, timeout=10)
                if match_1 == 'TIMED_OUT':
                    continue
                elif match_1:
                    answer_counts[existing_answer] += 1
                    matched_existing = True
                    break
                
                # Try reverse
                match_2 = grade_answer_with_timeout(existing_answer, result, timeout=10)
                if match_2 == 'TIMED_OUT':
                    continue
                elif match_2:
                    answer_counts[existing_answer] += 1
                    matched_existing = True
                    break
            
            if not matched_existing:
                answer_counts[result] = 1
        
        # Determine majority answer and its count
        if not answer_counts:
            majority_answer = None
            majority_count = 0
        else:
            majority_answer = max(answer_counts, key=answer_counts.get)
            majority_count = answer_counts[majority_answer]
        
        # Calculate score based on majority consensus (not questioner's answer)
        score = majority_count / len(results) if results else 0.0
        
        # Also check agreement with questioner's answer for reference
        questioner_agreement_count = 0
        for result in results:
            if result.strip() == golden_answer.strip():
                questioner_agreement_count += 1
            else:
                match_1 = grade_answer_with_timeout(result, golden_answer, timeout=10)
                if match_1 == 'TIMED_OUT':
                    continue
                elif match_1:
                    questioner_agreement_count += 1
                else:
                    match_2 = grade_answer_with_timeout(golden_answer, result, timeout=10)
                    if match_2 != 'TIMED_OUT' and match_2:
                        questioner_agreement_count += 1
        
        # Skip certain question types that are hard to grade automatically
        skip_question = False
        if "证明" in question or 'prove' in question.lower() or 'show that' in question.lower():
            skip_question = True
        
        results_all.append({
            "question": question,
            "questioner_answer": golden_answer,
            "source_model": source_model,
            "cluster_id": cluster_id,
            "score": score,  # Consensus score (fraction agreeing with majority)
            "majority_answer": majority_answer,
            "majority_count": majority_count,
            "total_samples": len(results),
            "questioner_agreement_count": questioner_agreement_count,
            "questioner_agreement_score": questioner_agreement_count / len(results) if results else 0.0,
            "answer": majority_answer,  # Keep for backward compatibility
            "answer_counts": answer_counts,
            "results": results,
            "rollouts": rollouts,
        })
        
        if (idx + 1) % 10 == 0:
            print(f"[{idx + 1}/{len(questions)}] Processed. Current avg score: {sum(r['score'] for r in results_all) / len(results_all):.3f}")

    except Exception as e:
        print(f"[{idx}] CRITICAL ERROR processing question '{question[:50]}...': {e}")
        continue

# 5. Save Final Results
print(f"\nProcessed {len(results_all)} questions. Saving results to: {args.output_file}")
with open(args.output_file, "w") as f:
    json.dump(results_all, f, indent=4)

# 6. Print Summary Statistics
total_consensus_score = sum(r['score'] for r in results_all)
avg_consensus_score = total_consensus_score / len(results_all) if results_all else 0

total_questioner_score = sum(r.get('questioner_agreement_score', 0) for r in results_all)
avg_questioner_score = total_questioner_score / len(results_all) if results_all else 0

high_consensus_count = sum(1 for r in results_all if r.get('score', 0) >= 0.5)
high_questioner_agreement = sum(1 for r in results_all if r.get('questioner_agreement_score', 0) >= 0.5)

print("\n" + "="*70)
print("EVALUATION SUMMARY")
print("="*70)
print(f"Total questions evaluated: {len(results_all)}")
print(f"\nConsensus Metrics (solver agreement with majority):")
print(f"  Average consensus score: {avg_consensus_score:.3f}")
print(f"  Questions with consensus >= 0.5: {high_consensus_count} ({high_consensus_count/len(results_all)*100:.1f}%)")
print(f"\nQuestioner Agreement Metrics (solver agreement with questioner):")
print(f"  Average questioner agreement: {avg_questioner_score:.3f}")
print(f"  Questions with questioner agreement >= 0.5: {high_questioner_agreement} ({high_questioner_agreement/len(results_all)*100:.1f}%)")
print(f"\nOutput saved to: {args.output_file}")
print("="*70)

print("\nScript finished.")