#!/usr/bin/env python3
"""
Download math question corpus for cluster building.
Downloads EleutherAI/hendrycks_math train set and openai/gsm8k main subset train set.

Usage:
    python download_corpus.py --output_dir ./corpus
"""
import argparse
import json
import os
from pathlib import Path

try:
    from datasets import load_dataset, concatenate_datasets
except ImportError:
    print("Please install datasets: pip install datasets")
    raise


def download_hendrycks_math(output_dir: str):
    """
    Download EleutherAI/hendrycks_math train dataset from all configs.
    
    Fields preserved: source, question (from 'problem' field)
    """
    print("Downloading EleutherAI/hendrycks_math dataset...")
    try:
        # Load all configs and combine them
        configs = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 
                   'number_theory', 'prealgebra', 'precalculus']
        
        all_datasets = []
        from tqdm import tqdm
        
        print(f"Loading {len(configs)} configs from hendrycks_math...")
        for config in tqdm(configs, desc="Loading configs"):
            try:
                ds = load_dataset("EleutherAI/hendrycks_math", config, split="train")
                all_datasets.append(ds)
                print(f"  Loaded {len(ds)} examples from {config}")
            except Exception as e:
                print(f"  Warning: Failed to load config '{config}': {e}")
                continue
        
        if not all_datasets:
            print("No configs were successfully loaded!")
            return []
        
        # Combine all datasets
        combined_ds = concatenate_datasets(all_datasets)
        print(f"Combined dataset: {len(combined_ds)} total examples from hendrycks_math")
        
        questions = []
        for item in tqdm(combined_ds, desc="Processing hendrycks_math questions"):
            # Extract problem text (hendrycks_math uses 'problem' field)
            problem_text = item.get("problem", "")
            
            if problem_text and len(problem_text.strip()) > 10:
                question_data = {
                    "source": "hendrycks_math",
                    "question": problem_text,
                    "type": item.get("type", ""),
                    "level": item.get("level", ""),
                }
                questions.append(question_data)
        
        output_path = os.path.join(output_dir, "hendrycks_math_questions.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(questions)} hendrycks_math questions to {output_path}")
        
        # Print statistics
        if questions:
            from collections import Counter
            types = [q.get("type", "unknown") for q in questions]
            levels = [q.get("level", "unknown") for q in questions]
            
            print(f"\nhendrycks_math statistics:")
            print(f"  Questions by type:")
            for qtype, count in Counter(types).most_common(10):
                print(f"    {qtype}: {count}")
            print(f"  Questions by level:")
            for level, count in Counter(levels).most_common():
                print(f"    {level}: {count}")
        
        return questions
    except Exception as e:
        print(f"Failed to download hendrycks_math dataset: {e}")
        import traceback
        traceback.print_exc()
        return []


def download_gsm8k(output_dir: str):
    """
    Download openai/gsm8k main subset train dataset.
    
    Fields preserved: source, question
    """
    print("Downloading openai/gsm8k dataset...")
    try:
        ds = load_dataset("openai/gsm8k", "main", split="train")
        print(f"Loaded {len(ds)} total examples from gsm8k")
        
        questions = []
        from tqdm import tqdm
        for item in tqdm(ds, desc="Processing gsm8k questions"):
            # Extract question text (gsm8k uses 'question' field)
            question_text = item.get("question", "")
            
            if question_text and len(question_text.strip()) > 10:
                question_data = {
                    "source": "gsm8k",
                    "question": question_text,
                }
                questions.append(question_data)
        
        output_path = os.path.join(output_dir, "gsm8k_questions.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(questions)} gsm8k questions to {output_path}")
        
        return questions
    except Exception as e:
        print(f"Failed to download gsm8k dataset: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    parser = argparse.ArgumentParser(description="Download math corpus for cluster building")
    parser.add_argument("--output_dir", type=str, default="./corpus")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download both datasets
    hendrycks_questions = download_hendrycks_math(args.output_dir)
    gsm8k_questions = download_gsm8k(args.output_dir)
    
    # Combine all questions
    all_questions = hendrycks_questions + gsm8k_questions
    
    # Save combined corpus
    combined_path = os.path.join(args.output_dir, "all_questions.json")
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(all_questions, f, indent=2, ensure_ascii=False)
    print(f"\nTotal: {len(all_questions)} questions saved to {combined_path}")
    print(f"  - hendrycks_math: {len(hendrycks_questions)} questions")
    print(f"  - gsm8k: {len(gsm8k_questions)} questions")
    
    print("\nNext step: Run build_clusters.py to create cluster centroids")
    print(f"  python build_clusters.py --corpus_file {combined_path} --output_dir ./cluster_data")


if __name__ == "__main__":
    main()
