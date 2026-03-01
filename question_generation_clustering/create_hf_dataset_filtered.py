#!/usr/bin/env python3
"""
Create HuggingFace dataset from OpenAI validated questions with score filtering.
"""

import os
import json
from datasets import Dataset
from huggingface_hub import login

# Configuration
INPUT_FILE = "balanced_questions_darwin_iter3_evaluated.json"
DATASET_NAME = "darwin_iter3_dataset"
HF_USERNAME = os.getenv("HUGGINGFACENAME", "")  # Set via: export HUGGINGFACENAME=your_hf_username
HF_TOKEN = os.getenv("HF_TOKEN", "")            # Set via: export HF_TOKEN=your_hf_token
MIN_SCORE = 0.5
MAX_SCORE = 0.9

def main():
    # Login to HuggingFace
    print("[Login] Authenticating with HuggingFace...")
    login(token=HF_TOKEN)
    print("✓ Logged in successfully")
    
    # Load the JSON file
    print(f"\n[Load] Reading from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data)} questions")
    
    # Transform and filter the data
    print(f"\n[Filter] Filtering for scores between {MIN_SCORE} and {MAX_SCORE}...")
    transformed_data = []
    for item in data:
        score = item.get("score", 0.0)
        if MIN_SCORE <= score <= MAX_SCORE:
            transformed_item = {
                "problem": item.get("question", ""),
                "answer": item.get("majority_answer", ""),
                "score": score
            }
            transformed_data.append(transformed_item)
    
    print(f"✓ Filtered to {len(transformed_data)} items (from {len(data)} total)")
    print(f"  Kept: {len(transformed_data)/len(data)*100:.1f}%")
    
    # Create dataset
    print("\n[Create] Creating HuggingFace dataset...")
    dataset = Dataset.from_list(transformed_data)
    print(f"✓ Dataset created with {len(dataset)} examples")
    
    # Print dataset info
    print("\n[Info] Dataset structure:")
    print(dataset)
    print(f"\nFirst example:")
    print(dataset[0])
    
    # Upload to HuggingFace
    repo_id = f"{HF_USERNAME}/{DATASET_NAME}"
    print(f"\n[Upload] Uploading to HuggingFace Hub: {repo_id}")
    dataset.push_to_hub(
        repo_id=repo_id,
        private=False,
        token=HF_TOKEN
    )
    print(f"✓ Successfully uploaded to: https://huggingface.co/datasets/{repo_id}")
    
    # Print statistics
    print("\n[Stats] Dataset statistics:")
    scores = [item["score"] for item in transformed_data]
    print(f"  Total problems: {len(transformed_data)}")
    print(f"  Score range: {MIN_SCORE} to {MAX_SCORE}")
    print(f"  Average score: {sum(scores)/len(scores):.3f}")
    print(f"  Min score: {min(scores):.3f}")
    print(f"  Max score: {max(scores):.3f}")
    
    # Score distribution
    from collections import Counter
    score_buckets = Counter()
    for score in scores:
        if score < 0.6:
            score_buckets["0.5-0.6"] += 1
        elif score < 0.7:
            score_buckets["0.6-0.7"] += 1
        elif score < 0.8:
            score_buckets["0.7-0.8"] += 1
        else:
            score_buckets["0.8-0.9"] += 1
    
    print(f"\n  Score distribution:")
    for bucket in ["0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9"]:
        count = score_buckets[bucket]
        print(f"    {bucket}: {count} ({count/len(scores)*100:.1f}%)")
    
    print("\n✅ All done!")

if __name__ == "__main__":
    main()
