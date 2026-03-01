#!/usr/bin/env python3
"""
Create HuggingFace dataset from matched OpenAI validated questions.
"""

import os
import json
from datasets import Dataset
from huggingface_hub import login

# Configuration
INPUT_FILE = "balanced_questions__darwin_iter2_openai_validated_matched.json"
DATASET_NAME = "darwin_iter2_dataset_verified_matched"
HF_USERNAME = os.getenv("HUGGINGFACENAME", "")  # Set via: export HUGGINGFACENAME=your_hf_username
HF_TOKEN = os.getenv("HF_TOKEN", "")            # Set via: export HF_TOKEN=your_hf_token

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
    
    # Transform the data
    print("\n[Transform] Converting to HuggingFace dataset format...")
    transformed_data = []
    for item in data:
        transformed_item = {
            "problem": item.get("question", ""),
            "answer": item.get("majority_answer", ""),
            "score": item.get("score", 0.0)
        }
        transformed_data.append(transformed_item)
    
    print(f"✓ Transformed {len(transformed_data)} items")
    
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
    print(f"  Average score: {sum(scores)/len(scores):.3f}")
    print(f"  Min score: {min(scores):.3f}")
    print(f"  Max score: {max(scores):.3f}")
    
    print("\n✅ All done!")

if __name__ == "__main__":
    main()
