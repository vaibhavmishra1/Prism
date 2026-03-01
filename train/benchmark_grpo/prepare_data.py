"""
Prepare ALL evaluation datasets for GRPO training.

Loads every dataset from evaluation/datasets_loader.py, formats them
uniformly (question + answer strings), merges into a single training set,
and saves as parquet files with a 90/10 train/val split.

Datasets included:
  - Math 500             (math answers)
  - GSM8K                (math answers)
  - AMC23                (math answers)
  - Minerva Math         (math answers)
  - OlympiadBench        (math answers)
  - AIME 2024            (integer answers)
  - AIME 2025            (integer answers)
  - MMLU-Pro             (multiple choice)
  - BBEH                 (text answers)
  - SuperGPQA            (multiple choice)
  - GPQA Diamond         (multiple choice)

Usage:
    python benchmark_grpo/prepare_data.py
"""

import os
import json
import random
from datasets import load_dataset, Dataset
import pandas


def load_tokens():
    """Load HF token from tokens.json if available."""
    tokens_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tokens.json")
    if os.path.exists(tokens_path):
        with open(tokens_path, "r") as f:
            tokens = json.load(f)
        hf_token = tokens.get("huggingface")
        if hf_token and hf_token != "yourhuggingfacetoken":
            os.environ["HF_TOKEN"] = hf_token
            return hf_token
    return os.environ.get("HF_TOKEN")


def load_math500():
    """Math 500 test set — math answers."""
    print("  Loading Math 500...")
    df = pandas.read_csv(
        "https://openaipublic.blob.core.windows.net/simple-evals/math_500_test.csv"
    )
    examples = [row.to_dict() for _, row in df.iterrows()]
    return [
        {"question": ex["Question"], "answer": str(ex["Answer"]), "source": "math500"}
        for ex in examples
    ]


def load_gsm8k():
    """GSM8K test set — math answers."""
    print("  Loading GSM8K...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    return [
        {
            "question": row["question"],
            "answer": row["answer"].split("#### ")[-1],
            "source": "gsm8k",
        }
        for row in dataset
    ]


def load_amc23():
    """AMC23 — math answers (small dataset)."""
    print("  Loading AMC23...")
    dataset = load_dataset("zwhe99/amc23", split="test")
    return [
        {"question": row["question"], "answer": str(row["answer"]), "source": "amc23"}
        for row in dataset
    ]


def load_minerva():
    """Minerva Math — math answers."""
    print("  Loading Minerva Math...")
    dataset = load_dataset("zwhe99/simplerl-minerva-math", split="test")
    return [
        {"question": row["problem"], "answer": str(row["answer"]), "source": "minerva"}
        for row in dataset
    ]


def load_olympiad():
    """OlympiadBench — math answers (final_answer is a list, take first)."""
    print("  Loading OlympiadBench...")
    dataset = load_dataset("zwhe99/simplerl-OlympiadBench", split="test")
    results = []
    for row in dataset:
        fa = row["final_answer"]
        answer = str(fa[0]) if isinstance(fa, list) and len(fa) > 0 else str(fa)
        results.append({"question": row["question"], "answer": answer, "source": "olympiad"})
    return results


def load_aime2024():
    """AIME 2024 — integer answers (small dataset)."""
    print("  Loading AIME 2024...")
    dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
    return [
        {"question": row["problem"], "answer": str(row["answer"]), "source": "aime2024"}
        for row in dataset
    ]


def load_aime2025():
    """AIME 2025 — integer answers (small dataset)."""
    print("  Loading AIME 2025...")
    dataset = load_dataset("yentinglin/aime_2025", "default")["train"]
    return [
        {"question": row["problem"], "answer": str(row["answer"]), "source": "aime2025"}
        for row in dataset
    ]



# All dataset loaders in order
DATASET_LOADERS = [
    ("Math 500", load_math500),
    ("GSM8K", load_gsm8k),
    ("AMC23", load_amc23),
    ("Minerva Math", load_minerva),
    ("OlympiadBench", load_olympiad),
    ("AIME 2024", load_aime2024),
    ("AIME 2025", load_aime2025)
]

# Small datasets get repeated to avoid being drowned out during training
# These are datasets with fewer than ~100 examples
REPEAT_FACTOR = {
    "aime2024": 8,   # ~30 problems -> ~240
    "aime2025": 8,   # ~30 problems -> ~240
    "amc23": 4,      # ~40 problems -> ~160
}


def main():
    hf_token = load_tokens()
    if hf_token:
        print(f"HF token loaded (length={len(hf_token)})")
    else:
        print("WARNING: No HF token found. Some datasets (GPQA) may fail to load.")
        print("  Set HF_TOKEN env var or update tokens.json")

    random.seed(42)

    all_examples = []
    dataset_stats = {}

    print("\nLoading datasets:")
    print("=" * 60)

    for name, loader_fn in DATASET_LOADERS:
        try:
            examples = loader_fn()
            source = examples[0]["source"] if examples else "unknown"

            # Apply repeat factor for small datasets
            repeat = REPEAT_FACTOR.get(source, 1)
            if repeat > 1:
                examples = examples * repeat
                print(f"    -> Repeated {repeat}x for balance")

            dataset_stats[name] = len(examples)
            all_examples.extend(examples)
            print(f"    -> {len(examples)} examples")
        except Exception as e:
            print(f"    -> FAILED: {e}")
            dataset_stats[name] = 0

    print("=" * 60)
    print(f"\nTotal merged examples: {len(all_examples)}")
    print("\nDataset breakdown:")
    for name, count in dataset_stats.items():
        pct = (count / len(all_examples) * 100) if all_examples else 0
        print(f"  {name:20s}: {count:6d} ({pct:5.1f}%)")

    # Shuffle all examples
    random.shuffle(all_examples)

    # Remove the 'source' key before saving (verl doesn't need it)
    # Actually keep it for debugging — it won't affect training since
    # the data loader only reads prompt_key and answer_key
    processed_dataset = Dataset.from_list(all_examples)

    # Split into train (90%) and val (10%)
    split = processed_dataset.train_test_split(test_size=0.1, seed=42)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "val.parquet")

    split["train"].to_parquet(train_path)
    split["test"].to_parquet(val_path)

    print(f"\nTrain set: {len(split['train'])} examples -> {train_path}")
    print(f"Val set:   {len(split['test'])} examples -> {val_path}")

    # Show a few samples
    print("\n--- Sample entries ---")
    for i in range(min(3, len(split["train"]))):
        ex = split["train"][i]
        q_preview = ex["question"][:120].replace("\n", " ")
        print(f"  [{ex['source']}] Q: {q_preview}...")
        print(f"           A: {ex['answer']}")
        print()

    print("Done! Run training with: bash benchmark_grpo/train.sh")


if __name__ == "__main__":
    main()
