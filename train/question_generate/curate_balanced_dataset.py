#!/usr/bin/env python3
"""
Curate a balanced solver training dataset from multiple questioner iterations.

Given k datasets from different questioner models, this script:
1. Pools all questions from all input HuggingFace datasets
2. Loads pre-computed cluster centroids (from your existing cluster space)
3. Embeds all questions using a sentence transformer
4. Assigns each question to the nearest cluster centroid
5. For each cluster, selects top-quality questions closest to target score
   with within-cluster diversity (MMR-style selection)
6. Deduplicates near-duplicate questions (cosine similarity > threshold)
7. Uploads a balanced dataset to HuggingFace

Usage:
    python question_generate/curate_balanced_dataset.py \
        --datasets your_hf_username/questioner-iter1 your_hf_username/questioner-iter2 \
        --output_repo your_hf_username/balanced_solver_v3 \
        --centroids_dataset your_hf_username/math_clusters \
        --centroids_file centroids.npy \
        --max_per_cluster 100 \
        --target_score 0.75 \
        --min_score 0.5 \
        --max_score 0.9

    # You can also load from local JSON files:
    python question_generate/curate_balanced_dataset.py \
        --local_files /path/to/results_0.json /path/to/results_1.json \
        --output_repo your_hf_username/balanced_solver_v3 \
        --centroids_dataset your_hf_username/math_clusters

    # Dry run (no upload, just save locally):
    python question_generate/curate_balanced_dataset.py \
        --datasets your_hf_username/questioner-iter1 \
        --output_dir ./balanced_output \
        --dry_run
"""

import argparse
import json
import os
import sys
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter

# ---------------------------------------------------------------------------
# Optional imports — we check at runtime so the script prints a clear error
# ---------------------------------------------------------------------------
try:
    from datasets import load_dataset, Dataset, DatasetDict
except ImportError:
    print("ERROR: 'datasets' library not found. Install with: pip install datasets")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# Note: sklearn is not needed since we use pre-computed centroids
# try:
#     from sklearn.cluster import KMeans
# except ImportError:
#     KMeans = None

try:
    from huggingface_hub import login as hf_login
except ImportError:
    hf_login = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# ===========================================================================
# 1. DATA LOADING
# ===========================================================================

def load_hf_datasets(dataset_names: List[str], split: str = "train") -> List[Dict]:
    """
    Load and pool questions from multiple HuggingFace datasets.

    Each dataset is expected to have at least columns: problem, answer, score.
    Optionally: diversity_score.

    Args:
        dataset_names: List of HF dataset identifiers (e.g. "your_hf_username/variant2-iter3_solver_v1")
        split: Which split to load (default: "train")

    Returns:
        List of dicts with keys: problem, answer, score, diversity_score, source
    """
    all_questions: List[Dict] = []

    for ds_name in dataset_names:
        print(f"[Load] Loading dataset: {ds_name} (split={split})")
        try:
            ds = load_dataset(ds_name, split=split)
        except Exception as e:
            print(f"[Load] WARNING: Failed to load {ds_name} with default config: {e}")
            print(f"[Load] Trying to list configs...")
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                info = api.dataset_info(ds_name)
                # Try loading with the first available config
                configs = [c.config_name for c in info.card_data.get("configs", [])] if info.card_data else []
                if not configs:
                    # Try loading all configs by listing parquet files
                    ds = load_dataset(ds_name, split=split, trust_remote_code=True)
                else:
                    print(f"[Load] Available configs: {configs}")
                    for config in configs:
                        try:
                            ds = load_dataset(ds_name, name=config, split=split)
                            print(f"[Load] Loaded config: {config}")
                            break
                        except Exception:
                            continue
                    else:
                        print(f"[Load] ERROR: Could not load any config from {ds_name}. Skipping.")
                        continue
            except Exception as e2:
                print(f"[Load] ERROR: Could not load {ds_name}: {e2}. Skipping.")
                continue

        count = 0
        for row in ds:
            # Support both "problem" and "question" column names
            problem = row.get("problem", row.get("question", ""))
            answer = row.get("answer", row.get("majority_answer", ""))
            raw_score = row.get("score", row.get("max_voting_score", -1))
            raw_diversity = row.get("diversity_score", 0.0)

            if not problem or not str(problem).strip():
                continue
            if not answer or str(answer).strip() in ("", "None"):
                continue

            # Handle scores stored as strings
            try:
                score = float(raw_score)
            except (ValueError, TypeError):
                continue
            try:
                diversity_score = float(raw_diversity) if raw_diversity is not None else 0.0
            except (ValueError, TypeError):
                diversity_score = 0.0

            all_questions.append({
                "problem": str(problem).strip(),
                "answer": str(answer).strip(),
                "score": score,
                "diversity_score": diversity_score,
                "source": ds_name,
            })
            count += 1

        print(f"[Load] Loaded {count} valid questions from {ds_name}")

    print(f"[Load] Total pooled questions: {len(all_questions)}")
    return all_questions


def load_local_files(file_paths: List[str]) -> List[Dict]:
    """
    Load questions from local JSON files (the _results.json format from evaluate.py).

    Args:
        file_paths: List of paths to JSON files

    Returns:
        List of dicts with keys: problem, answer, score, diversity_score, source
    """
    all_questions: List[Dict] = []

    for fpath in file_paths:
        print(f"[Load] Loading local file: {fpath}")
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[Load] ERROR: Could not load {fpath}: {e}. Skipping.")
            continue

        count = 0
        for item in data:
            problem = item.get("question", item.get("problem", ""))
            answer = item.get("answer", item.get("majority_answer", ""))
            score = item.get("score", item.get("max_voting_score", -1))
            diversity_score = item.get("diversity_score", 0.0)

            if not problem or not problem.strip():
                continue
            if not answer or answer in ("", "None"):
                continue

            all_questions.append({
                "problem": problem.strip(),
                "answer": str(answer).strip(),
                "score": float(score),
                "diversity_score": float(diversity_score),
                "source": os.path.basename(fpath),
            })
            count += 1

        print(f"[Load] Loaded {count} valid questions from {fpath}")

    print(f"[Load] Total pooled questions: {len(all_questions)}")
    return all_questions


# ===========================================================================
# 2. EMBEDDING
# ===========================================================================

def embed_questions(
    questions: List[str],
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    batch_size: int = 256,
    device: str = "cuda:0",
) -> np.ndarray:
    """
    Embed a list of question strings using SentenceTransformer.

    Returns:
        numpy array of shape (N, embedding_dim), L2-normalised.
    """
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers not installed. pip install sentence-transformers")

    print(f"[Embed] Loading embedding model: {model_name} on {device}")
    model = SentenceTransformer(model_name, trust_remote_code=True, device=device)

    print(f"[Embed] Embedding {len(questions)} questions (batch_size={batch_size})...")
    embeddings = model.encode(
        questions,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"[Embed] Embedding shape: {embeddings.shape}")
    return embeddings


# ===========================================================================
# 3. CLUSTER ASSIGNMENT (using pre-computed centroids)
# ===========================================================================

def load_centroids_from_hf(centroids_dataset: str, centroids_file: str = "centroids.npy") -> np.ndarray:
    """
    Load pre-computed cluster centroids from a HuggingFace dataset.

    Args:
        centroids_dataset: HF dataset name (e.g. "your_hf_username/math_clusters")
        centroids_file: filename within the dataset (default: "centroids.npy")

    Returns:
        numpy array of shape (num_clusters, embedding_dim)
    """
    print(f"[Centroids] Loading centroids from {centroids_dataset}/{centroids_file}")
    
    try:
        # Load the dataset
        ds = load_dataset(centroids_dataset, split="train")
        
        # Try to find the centroids file in the dataset
        # HF datasets with numpy files typically store them in a 'file' column or similar
        if centroids_file in ds.column_names:
            centroids = np.array(ds[centroids_file][0])
        elif "centroids" in ds.column_names:
            centroids = np.array(ds["centroids"][0])
        else:
            # If not in columns, try downloading the file directly from the repo
            from huggingface_hub import hf_hub_download
            centroids_path = hf_hub_download(
                repo_id=centroids_dataset,
                filename=centroids_file,
                repo_type="dataset"
            )
            centroids = np.load(centroids_path)
    
    except Exception as e:
        print(f"[Centroids] ERROR: Could not load centroids from {centroids_dataset}: {e}")
        print(f"[Centroids] Trying direct file download...")
        
        # Fallback: try direct download
        try:
            from huggingface_hub import hf_hub_download
            centroids_path = hf_hub_download(
                repo_id=centroids_dataset,
                filename=centroids_file,
                repo_type="dataset"
            )
            centroids = np.load(centroids_path)
        except Exception as e2:
            raise RuntimeError(f"Failed to load centroids from {centroids_dataset}: {e2}")
    
    print(f"[Centroids] Loaded {centroids.shape[0]} centroids with dimension {centroids.shape[1]}")
    return centroids


def assign_to_clusters(
    embeddings: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """
    Assign question embeddings to nearest cluster centroids.

    Args:
        embeddings: Question embeddings (N, dim), L2-normalized
        centroids: Cluster centroids (K, dim), should be L2-normalized

    Returns:
        labels: (N,) int array of cluster IDs
    """
    print(f"[Cluster] Assigning {len(embeddings)} questions to {len(centroids)} clusters...")
    
    # Normalize centroids if not already (for cosine similarity)
    centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids_normalized = centroids / (centroid_norms + 1e-12)
    
    # Compute cosine similarity: higher = closer
    # embeddings are already normalized, so dot product = cosine similarity
    similarities = embeddings @ centroids_normalized.T  # (N, K)
    
    # Assign to nearest (most similar) centroid
    labels = np.argmax(similarities, axis=1)
    
    # Print cluster distribution stats
    unique, counts = np.unique(labels, return_counts=True)
    print(f"[Cluster] Cluster sizes — min: {counts.min()}, max: {counts.max()}, "
          f"mean: {counts.mean():.1f}, median: {np.median(counts):.1f}")
    print(f"[Cluster] Assigned clusters: {len(unique)}/{len(centroids)} "
          f"({len(centroids) - len(unique)} empty)")

    return labels


# ===========================================================================
# 4. DEDUPLICATION
# ===========================================================================

def deduplicate_within_cluster(
    indices: List[int],
    embeddings: np.ndarray,
    similarity_threshold: float = 0.95,
) -> List[int]:
    """
    Remove near-duplicates within a cluster. For every pair with
    cosine similarity > threshold, keep only the first one encountered.

    Args:
        indices: list of global indices for this cluster's questions
        embeddings: full embedding matrix (N, dim) — already L2-normalised
        similarity_threshold: above this → considered duplicate

    Returns:
        Filtered list of indices (deduplicated).
    """
    if len(indices) <= 1:
        return indices

    cluster_embs = embeddings[indices]  # (k, dim)
    # Cosine similarity matrix (embeddings already normalised)
    sim_matrix = cluster_embs @ cluster_embs.T  # (k, k)

    keep_mask = np.ones(len(indices), dtype=bool)
    for i in range(len(indices)):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, len(indices)):
            if not keep_mask[j]:
                continue
            if sim_matrix[i, j] > similarity_threshold:
                keep_mask[j] = False  # Drop j, keep i

    kept = [idx for idx, keep in zip(indices, keep_mask) if keep]
    return kept


# ===========================================================================
# 5. SELECTION (MMR-style within each cluster)
# ===========================================================================

def zpd_priority(score: float, target: float = 0.75) -> float:
    """
    Priority score: how close is this question's majority voting score
    to the target difficulty? Higher is better.
    """
    return -abs(score - target)


def mmr_select(
    indices: List[int],
    embeddings: np.ndarray,
    scores: List[float],
    sources: List[str],
    max_select: int = 100,
    target_score: float = 0.75,
    lambda_mmr: float = 0.7,
    source_bonus: float = 0.1,
) -> List[int]:
    """
    Maximal Marginal Relevance selection within a cluster.

    Balances:
    - Relevance: closeness of majority voting score to target_score
    - Diversity: dissimilarity (in embedding space) from already selected questions
    - Source diversity: small bonus for questions from underrepresented source models

    Args:
        indices:     global indices of candidate questions in this cluster
        embeddings:  full (N, dim) embedding matrix (L2-normalised)
        scores:      majority voting scores for each candidate (parallel to indices)
        sources:     source dataset name for each candidate (parallel to indices)
        max_select:  how many to keep
        target_score: optimal ZPD score
        lambda_mmr:  weight for relevance vs diversity (higher → more relevance)
        source_bonus: bonus weight for source diversity

    Returns:
        List of selected global indices.
    """
    if len(indices) <= max_select:
        return indices

    n = len(indices)
    cluster_embs = embeddings[indices]  # (n, dim)

    # Pre-compute relevance: normalise zpd_priority to [0, 1]
    # zpd_priority is in [-1, 0], so we shift by 1
    relevance = np.array([1.0 + zpd_priority(s, target_score) for s in scores])

    selected: List[int] = []        # local indices into `indices`
    selected_set = set()
    remaining = set(range(n))

    # Track source counts among selected
    source_counts: Dict[str, int] = defaultdict(int)

    for _ in range(min(max_select, n)):
        best_idx = -1
        best_score = -float("inf")

        for local_i in remaining:
            # Relevance component
            rel = relevance[local_i]

            # Diversity component: min similarity to already-selected
            if selected:
                sims = cluster_embs[local_i] @ cluster_embs[selected].T
                max_sim = float(np.max(sims))
            else:
                max_sim = 0.0

            div = 1.0 - max_sim  # higher = more diverse

            # Source diversity: bonus if this source is underrepresented
            src = sources[local_i]
            total_selected = max(len(selected), 1)
            src_freq = source_counts.get(src, 0) / total_selected
            src_div = 1.0 - src_freq  # higher if source is rare among selected

            # Combined MMR score
            mmr = lambda_mmr * rel + (1.0 - lambda_mmr) * div + source_bonus * src_div

            if mmr > best_score:
                best_score = mmr
                best_idx = local_i

        if best_idx < 0:
            break

        selected.append(best_idx)
        selected_set.add(best_idx)
        remaining.discard(best_idx)
        source_counts[sources[best_idx]] += 1

    # Map back to global indices
    return [indices[i] for i in selected]


# ===========================================================================
# 6. MAIN CURATION PIPELINE
# ===========================================================================

def curate_balanced_dataset(
    questions: List[Dict],
    centroids_dataset: str,
    centroids_file: str = "centroids.npy",
    max_per_cluster: int = 100,
    min_score: float = 0.5,
    max_score: float = 0.9,
    target_score: float = 0.75,
    dedup_threshold: float = 0.95,
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    embedding_batch_size: int = 256,
    embedding_device: str = "cuda:0",
    lambda_mmr: float = 0.7,
    source_bonus: float = 0.1,
) -> Tuple[List[Dict], Dict]:
    """
    Full curation pipeline: embed → assign to pre-computed clusters → dedup → select.

    Args:
        questions:       pooled list of question dicts
        centroids_dataset: HF dataset with pre-computed cluster centroids
        centroids_file:  filename of centroids within the dataset (default: "centroids.npy")
        max_per_cluster: max questions to keep per cluster
        min_score:       minimum majority voting score
        max_score:       maximum majority voting score
        target_score:    optimal ZPD score for selection ranking
        dedup_threshold: cosine similarity above which questions are duplicates
        embedding_model: model name for SentenceTransformer
        embedding_batch_size: batch size for embedding
        embedding_device: device for embedding model
        lambda_mmr:      MMR relevance weight
        source_bonus:    source diversity bonus weight

    Returns:
        (selected_questions, stats_dict)
    """

    # ---- Step 0: Score filtering ----
    print(f"\n{'='*60}")
    print(f"[Step 0] Score filtering: keeping [{min_score}, {max_score}]")
    pre_count = len(questions)
    questions = [
        q for q in questions
        if min_score <= q["score"] <= max_score
    ]
    print(f"[Step 0] {pre_count} → {len(questions)} after score filter")

    if len(questions) == 0:
        print("[ERROR] No questions passed the score filter!")
        return [], {"error": "No questions passed score filter"}

    # ---- Step 1: Load pre-computed centroids ----
    print(f"\n{'='*60}")
    print("[Step 1] Loading pre-computed cluster centroids")
    centroids = load_centroids_from_hf(centroids_dataset, centroids_file)
    num_clusters = len(centroids)

    # ---- Step 2: Embed ----
    print(f"\n{'='*60}")
    print("[Step 2] Embedding questions")
    problem_texts = [q["problem"] for q in questions]
    embeddings = embed_questions(
        problem_texts,
        model_name=embedding_model,
        batch_size=embedding_batch_size,
        device=embedding_device,
    )

    # ---- Step 3: Assign to clusters ----
    print(f"\n{'='*60}")
    print("[Step 3] Assigning to pre-computed clusters")
    labels = assign_to_clusters(embeddings, centroids)

    # Assign cluster IDs to questions
    for i, q in enumerate(questions):
        q["cluster_id"] = int(labels[i])

    # ---- Step 4: Per-cluster processing ----
    print(f"\n{'='*60}")
    print("[Step 4] Per-cluster deduplication + MMR selection")

    # Group indices by cluster
    cluster_to_indices: Dict[int, List[int]] = defaultdict(list)
    for i, cid in enumerate(labels):
        cluster_to_indices[int(cid)].append(i)

    all_selected_indices: List[int] = []
    cluster_stats: Dict[int, Dict] = {}

    for cid in tqdm(sorted(cluster_to_indices.keys()), desc="Processing clusters"):
        indices = cluster_to_indices[cid]
        original_count = len(indices)

        # 3a. Deduplicate
        deduped_indices = deduplicate_within_cluster(
            indices, embeddings, similarity_threshold=dedup_threshold
        )
        dedup_count = len(deduped_indices)

        # 3b. MMR selection
        cluster_scores = [questions[i]["score"] for i in deduped_indices]
        cluster_sources = [questions[i]["source"] for i in deduped_indices]

        selected_indices = mmr_select(
            deduped_indices,
            embeddings,
            cluster_scores,
            cluster_sources,
            max_select=max_per_cluster,
            target_score=target_score,
            lambda_mmr=lambda_mmr,
            source_bonus=source_bonus,
        )

        all_selected_indices.extend(selected_indices)

        # Compute stats for this cluster
        selected_scores = [questions[i]["score"] for i in selected_indices]
        selected_sources = [questions[i]["source"] for i in selected_indices]
        cluster_stats[cid] = {
            "original_count": original_count,
            "after_dedup": dedup_count,
            "selected": len(selected_indices),
            "mean_score": float(np.mean(selected_scores)) if selected_scores else 0.0,
            "std_score": float(np.std(selected_scores)) if selected_scores else 0.0,
            "source_distribution": dict(Counter(selected_sources)),
        }

    # ---- Build output ----
    selected_questions = [questions[i] for i in all_selected_indices]

    # Remove cluster_id from questions that weren't selected (clean up)
    # Keep cluster_id in selected questions for reference
    for q in selected_questions:
        q.pop("source", None)  # Remove source (internal tracking field)

    # ---- Compute overall stats ----
    print(f"\n{'='*60}")
    print("[Summary]")

    total_selected = len(selected_questions)
    num_actual_clusters = len(cluster_stats)
    cluster_sizes = [cs["selected"] for cs in cluster_stats.values()]
    cluster_means = [cs["mean_score"] for cs in cluster_stats.values() if cs["selected"] > 0]

    empty_clusters = sum(1 for s in cluster_sizes if s == 0)
    full_clusters = sum(1 for s in cluster_sizes if s >= max_per_cluster)

    overall_stats = {
        "total_input_questions": pre_count,
        "after_score_filter": len(questions),
        "total_selected": total_selected,
        "num_clusters": num_actual_clusters,
        "max_per_cluster": max_per_cluster,
        "empty_clusters": empty_clusters,
        "full_clusters": full_clusters,
        "cluster_size_min": int(min(cluster_sizes)) if cluster_sizes else 0,
        "cluster_size_max": int(max(cluster_sizes)) if cluster_sizes else 0,
        "cluster_size_mean": float(np.mean(cluster_sizes)) if cluster_sizes else 0.0,
        "cluster_size_std": float(np.std(cluster_sizes)) if cluster_sizes else 0.0,
        "mean_of_cluster_means": float(np.mean(cluster_means)) if cluster_means else 0.0,
        "std_of_cluster_means": float(np.std(cluster_means)) if cluster_means else 0.0,
        "score_range": [min_score, max_score],
        "target_score": target_score,
        "per_cluster": cluster_stats,
    }

    print(f"  Total selected: {total_selected}")
    print(f"  Clusters: {num_actual_clusters} (empty: {empty_clusters}, full: {full_clusters})")
    print(f"  Cluster sizes — min: {overall_stats['cluster_size_min']}, "
          f"max: {overall_stats['cluster_size_max']}, "
          f"mean: {overall_stats['cluster_size_mean']:.1f} ± {overall_stats['cluster_size_std']:.1f}")
    print(f"  Mean of cluster mean scores: {overall_stats['mean_of_cluster_means']:.4f}")
    print(f"  Std of cluster mean scores:  {overall_stats['std_of_cluster_means']:.4f}")
    if empty_clusters > 0:
        print(f"  ⚠ {empty_clusters} clusters have 0 questions — "
              f"consider generating targeted questions for these clusters")

    return selected_questions, overall_stats


# ===========================================================================
# 7. OUTPUT / UPLOAD
# ===========================================================================

def save_locally(
    selected_questions: List[Dict],
    stats: Dict,
    output_dir: str,
):
    """Save curated dataset and stats to local directory."""
    os.makedirs(output_dir, exist_ok=True)

    # Save dataset as JSON
    dataset_path = os.path.join(output_dir, "balanced_dataset.json")
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(selected_questions, f, indent=2, ensure_ascii=False)
    print(f"[Save] Dataset saved to {dataset_path} ({len(selected_questions)} questions)")

    # Save stats
    # Convert per_cluster stats keys to strings for JSON
    stats_copy = dict(stats)
    if "per_cluster" in stats_copy:
        stats_copy["per_cluster"] = {
            str(k): v for k, v in stats_copy["per_cluster"].items()
        }
    stats_path = os.path.join(output_dir, "curation_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_copy, f, indent=2)
    print(f"[Save] Stats saved to {stats_path}")


def upload_to_huggingface(
    selected_questions: List[Dict],
    output_repo: str,
    hf_username: str,
    config_name: str = "balanced",
    private: bool = True,
):
    """Upload curated dataset to HuggingFace."""
    # Build the dataset in the format expected by solver training:
    # columns: problem, answer, score, diversity_score
    hf_data = []
    for q in selected_questions:
        hf_data.append({
            "problem": q["problem"],
            "answer": q["answer"],
            "score": q["score"],
            "diversity_score": q.get("diversity_score", 0.0),
            "cluster_id": q.get("cluster_id", -1),
        })

    train_dataset = Dataset.from_list(hf_data)
    dataset_dict = DatasetDict({"train": train_dataset})

    repo_name = output_repo
    if "/" not in repo_name:
        repo_name = f"{hf_username}/{output_repo}"

    print(f"[Upload] Pushing to {repo_name} (config={config_name}, private={private})")
    dataset_dict.push_to_hub(repo_name, private=private, config_name=config_name)
    print(f"[Upload] Done! Dataset available at: https://huggingface.co/datasets/{repo_name}")


# ===========================================================================
# 8. CLI
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Curate a balanced solver training dataset from multiple questioner iterations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From HuggingFace datasets (using pre-computed centroids):
  python curate_balanced_dataset.py \\
      --datasets your_hf_username/questioner-iter1 your_hf_username/questioner-iter2 \\
      --output_repo your_hf_username/balanced_solver_v3 \\
      --centroids_dataset your_hf_username/math_clusters

  # From local JSON files:
  python curate_balanced_dataset.py \\
      --local_files results_0.json results_1.json \\
      --output_repo your_hf_username/balanced_solver_v3 \\
      --centroids_dataset your_hf_username/math_clusters

  # Dry run (save locally, no upload):
  python curate_balanced_dataset.py \\
      --datasets your_hf_username/questioner-iter1 \\
      --output_dir ./balanced_output --dry_run \\
      --centroids_dataset your_hf_username/math_clusters
        """,
    )

    # Input sources
    input_group = parser.add_argument_group("Input Sources (at least one required)")
    input_group.add_argument(
        "--datasets", nargs="+", default=[],
        help="HuggingFace dataset names to pool (e.g. your_hf_username/questioner-iter1)"
    )
    input_group.add_argument(
        "--local_files", nargs="+", default=[],
        help="Local JSON files to pool (evaluate.py _results.json format)"
    )
    input_group.add_argument(
        "--split", type=str, default="train",
        help="Which split to load from HF datasets (default: train)"
    )

    # Clustering
    cluster_group = parser.add_argument_group("Clustering")
    cluster_group.add_argument("--centroids_dataset", type=str, default="",
                               help="HuggingFace dataset with pre-computed centroids (e.g. your_hf_username/math_clusters)")
    cluster_group.add_argument("--centroids_file", type=str, default="centroids.npy",
                               help="Filename of centroids within the dataset (default: centroids.npy)")
    cluster_group.add_argument("--max_per_cluster", type=int, default=100,
                               help="Maximum questions per cluster (default: 100)")

    # Score filtering
    score_group = parser.add_argument_group("Score Filtering")
    score_group.add_argument("--min_score", type=float, default=0.5,
                             help="Minimum majority voting score (default: 0.5)")
    score_group.add_argument("--max_score", type=float, default=0.9,
                             help="Maximum majority voting score (default: 0.9)")
    score_group.add_argument("--target_score", type=float, default=0.75,
                             help="Optimal ZPD target score for selection ranking (default: 0.75)")

    # Deduplication
    dedup_group = parser.add_argument_group("Deduplication")
    dedup_group.add_argument("--dedup_threshold", type=float, default=0.95,
                             help="Cosine similarity threshold for deduplication (default: 0.95)")

    # Embedding
    embed_group = parser.add_argument_group("Embedding")
    embed_group.add_argument("--embedding_model", type=str, default="Qwen/Qwen3-Embedding-0.6B",
                             help="SentenceTransformer model for embedding (default: Qwen/Qwen3-Embedding-0.6B)")
    embed_group.add_argument("--embedding_batch_size", type=int, default=256,
                             help="Batch size for embedding (default: 256)")
    embed_group.add_argument("--embedding_device", type=str, default="cuda:0",
                             help="Device for embedding model (default: cuda:0)")

    # MMR selection
    mmr_group = parser.add_argument_group("MMR Selection")
    mmr_group.add_argument("--lambda_mmr", type=float, default=0.7,
                           help="MMR relevance weight (higher → prefer score closeness; default: 0.7)")
    mmr_group.add_argument("--source_bonus", type=float, default=0.1,
                           help="Bonus weight for source diversity (default: 0.1)")

    # Output
    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--output_repo", type=str, default="",
                              help="HuggingFace repo to upload (e.g. your_hf_username/balanced_solver_v3)")
    output_group.add_argument("--output_dir", type=str, default="./balanced_output",
                              help="Local directory to save output (default: ./balanced_output)")
    output_group.add_argument("--config_name", type=str, default="balanced",
                              help="HF dataset config name (default: balanced)")
    output_group.add_argument("--hf_username", type=str, default="",
                              help="HuggingFace username (default: from HUGGINGFACENAME env var)")
    output_group.add_argument("--hf_token_file", type=str, default="tokens.json",
                              help="Path to JSON file with 'huggingface' token (default: tokens.json)")
    output_group.add_argument("--private", action="store_true", default=True,
                              help="Upload as private dataset (default: True)")
    output_group.add_argument("--dry_run", action="store_true",
                              help="Save locally only, do not upload to HuggingFace")

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate inputs
    if not args.datasets and not args.local_files:
        print("ERROR: Must provide at least one of --datasets or --local_files")
        sys.exit(1)

    # ---- Load data ----
    all_questions: List[Dict] = []

    if args.datasets:
        all_questions.extend(load_hf_datasets(args.datasets, split=args.split))

    if args.local_files:
        all_questions.extend(load_local_files(args.local_files))

    if not all_questions:
        print("ERROR: No questions loaded from any source!")
        sys.exit(1)

    # Print source distribution
    source_counts = Counter(q["source"] for q in all_questions)
    print(f"\n[Input] Source distribution:")
    for src, cnt in source_counts.most_common():
        print(f"  {src}: {cnt}")

    # Print score distribution
    scores = [q["score"] for q in all_questions]
    print(f"\n[Input] Score distribution:")
    print(f"  min={min(scores):.3f}, max={max(scores):.3f}, "
          f"mean={np.mean(scores):.3f}, std={np.std(scores):.3f}")

    # ---- Curate ----
    selected_questions, stats = curate_balanced_dataset(
        all_questions,
        centroids_dataset=args.centroids_dataset,
        centroids_file=args.centroids_file,
        max_per_cluster=args.max_per_cluster,
        min_score=args.min_score,
        max_score=args.max_score,
        target_score=args.target_score,
        dedup_threshold=args.dedup_threshold,
        embedding_model=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
        embedding_device=args.embedding_device,
        lambda_mmr=args.lambda_mmr,
        source_bonus=args.source_bonus,
    )

    if not selected_questions:
        print("ERROR: No questions selected after curation!")
        sys.exit(1)

    # ---- Save locally ----
    save_locally(selected_questions, stats, args.output_dir)

    # ---- Upload to HuggingFace ----
    if not args.dry_run and args.output_repo:
        # Authenticate
        hf_username = args.hf_username or os.getenv("HUGGINGFACENAME", "")
        if not hf_username:
            print("WARNING: No HF username specified. Set --hf_username or HUGGINGFACENAME env var.")

        if os.path.exists(args.hf_token_file):
            with open(args.hf_token_file, "r") as f:
                token_data = json.load(f)
            token = token_data.get("huggingface", "")
            if token and token != "yourhuggingfacetoken":
                if hf_login is not None:
                    hf_login(token=token)
                    print("[Auth] Logged in to HuggingFace")
                else:
                    print("WARNING: huggingface_hub not installed, cannot login")
            else:
                print("WARNING: No valid HF token found in tokens.json")
        else:
            print(f"WARNING: Token file {args.hf_token_file} not found. "
                  f"Assuming already authenticated.")

        upload_to_huggingface(
            selected_questions,
            output_repo=args.output_repo,
            hf_username=hf_username,
            config_name=args.config_name,
            private=args.private,
        )
    elif not args.dry_run and not args.output_repo:
        print("\n[Info] No --output_repo specified. Results saved locally only.")
        print("[Info] To upload, re-run with --output_repo <repo_name>")

    print(f"\n{'='*60}")
    print("[Done] Curation complete!")
    print(f"  Selected {len(selected_questions)} questions across {stats.get('num_clusters', '?')} clusters")
    print(f"  Local output: {args.output_dir}")
    if not args.dry_run and args.output_repo:
        repo = args.output_repo
        print(f"  HuggingFace: {repo}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
