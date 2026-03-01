#!/usr/bin/env python3
"""
Cluster pre-computed embeddings using K-Means.
This script loads pre-existing embeddings and questions, then performs clustering.

Usage:
    # Use all data
    python cluster_from_embeddings.py --questions corpus_cleaned/all_questions_cleaned.json \
                                       --embeddings corpus_cleaned/embeddings_cleaned.npy \
                                       --output_dir ./cluster_data_cleaned \
                                       --num_clusters 2048
    
    # Sample subset for faster testing
    python cluster_from_embeddings.py --num_samples 100000 \
                                       --num_clusters 512 \
                                       --max_iter 50 \
                                       --output_dir ./cluster_test
"""
import argparse
import json
import os
import numpy as np
import hashlib
from pathlib import Path
from typing import List, Dict
from sklearn.cluster import KMeans


def generate_question_id(question: str, index: int) -> str:
    """Generate a unique ID for a question based on content hash and index."""
    content = f"{question}_{index}".encode('utf-8')
    hash_obj = hashlib.md5(content)
    return f"q_{hash_obj.hexdigest()[:12]}_{index}"


def load_questions_from_file(filepath: str, dataset_source: str = None) -> List[Dict]:
    """
    Load questions from a JSON file. Supports multiple formats.
    
    Returns:
        List of metadata dicts, each with at least "question" and "source" fields.
    """
    if dataset_source is None:
        dataset_source = Path(filepath).stem
    
    print(f"Loading questions from: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions = []
    if isinstance(data, list):
        for item in data:
            question_text = None
            metadata = {}
            
            if isinstance(item, str):
                question_text = item
                metadata = {
                    "question": question_text,
                    "source": dataset_source,
                }
            elif isinstance(item, dict):
                # Try to extract question text
                for key in ['question', 'problem', 'text', 'input']:
                    if key in item and item[key]:
                        question_text = item[key]
                        break
                
                # Preserve all metadata fields
                metadata = {
                    "question": question_text,
                    "source": item.get('source', item.get('dataset_source', dataset_source)),
                    "id": item.get('id', ''),
                    "subject": item.get('subject', ''),
                    "format": item.get('format', ''),
                    "difficulty": item.get('difficulty', ''),
                }
            
            if question_text and len(question_text.strip()) > 10:
                questions.append(metadata)
    elif isinstance(data, dict):
        if 'questions' in data:
            for q in data['questions']:
                if isinstance(q, str) and q and len(q.strip()) > 10:
                    questions.append({
                        "question": q,
                        "source": dataset_source,
                    })
                elif isinstance(q, dict) and ('question' in q or 'problem' in q):
                    question_text = q.get('question') or q.get('problem') or q.get('text')
                    if question_text and len(question_text.strip()) > 10:
                        metadata = {
                            "question": question_text,
                            "source": q.get('source', q.get('dataset_source', dataset_source)),
                            "id": q.get('id', ''),
                            "subject": q.get('subject', ''),
                            "format": q.get('format', ''),
                            "difficulty": q.get('difficulty', ''),
                        }
                        questions.append(metadata)
    
    print(f"Loaded {len(questions):,} questions")
    return questions


def load_embeddings(filepath: str) -> np.ndarray:
    """Load embeddings from numpy file."""
    print(f"Loading embeddings from: {filepath}")
    embeddings = np.load(filepath)
    print(f"Loaded embeddings: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Number of embeddings: {embeddings.shape[0]:,}")
    
    # Verify no zero embeddings
    zero_mask = np.all(embeddings == 0, axis=1)
    num_zeros = np.sum(zero_mask)
    if num_zeros > 0:
        print(f"⚠️  Warning: Found {num_zeros} zero embeddings")
    else:
        print("✓ No zero embeddings found")
    
    # Check normalization
    norms = np.linalg.norm(embeddings, axis=1)
    if np.allclose(norms, 1.0, atol=0.01):
        print("✓ Embeddings are normalized")
    else:
        print(f"⚠️  Embeddings may not be normalized (mean norm: {np.mean(norms):.4f})")
    
    return embeddings


def fit_kmeans(embeddings: np.ndarray, num_clusters: int, n_init: int = 10, 
               max_iter: int = 300, random_state: int = 42) -> KMeans:
    """Fit K-Means clustering on embeddings."""
    print(f"\nFitting K-Means with {num_clusters} clusters...")
    print(f"Parameters: n_init={n_init}, max_iter={max_iter}, random_state={random_state}")
    
    kmeans = KMeans(
        n_clusters=num_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
        verbose=1,
    )
    kmeans.fit(embeddings)
    print(f"K-Means inertia: {kmeans.inertia_:.4f}")
    return kmeans


def save_cluster_data(
    output_dir: str,
    centroids: np.ndarray,
    labels: np.ndarray,
    questions: List[Dict],
    embeddings: np.ndarray,
):
    """
    Save cluster centroids, labels, embeddings, and question metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_questions = len(questions)
    assert len(labels) == num_questions, f"Mismatch: {len(labels)} labels vs {num_questions} questions"
    assert len(embeddings) == num_questions, f"Mismatch: {len(embeddings)} embeddings vs {num_questions} questions"
    
    print(f"\nSaving cluster data to: {output_dir}")
    
    # Save centroids
    centroids_path = os.path.join(output_dir, "centroids.npy")
    np.save(centroids_path, centroids)
    print(f"✓ Saved centroids to {centroids_path} (shape: {centroids.shape})")
    
    # Save labels (cluster assignments)
    labels_path = os.path.join(output_dir, "labels.npy")
    np.save(labels_path, labels)
    print(f"✓ Saved labels to {labels_path} (shape: {labels.shape})")
    
    # Save embeddings
    embeddings_path = os.path.join(output_dir, "embeddings.npy")
    np.save(embeddings_path, embeddings)
    print(f"✓ Saved embeddings to {embeddings_path} (shape: {embeddings.shape})")
    
    # Generate question IDs and create metadata
    question_metadata = []
    for index, q_meta in enumerate(questions):
        question_text = q_meta.get("question", "")
        # Use existing ID if present, otherwise generate one
        existing_id = q_meta.get("id", "")
        if existing_id:
            question_id = existing_id
        else:
            question_id = generate_question_id(question_text, index)
        
        # Build metadata preserving all original fields
        metadata = {
            "question_id": question_id,
            "index": index,
            "question": question_text,
            "cluster_id": int(labels[index]),
            "dataset_source": q_meta.get("source", q_meta.get("dataset_source", "unknown")),
        }
        
        # Preserve additional fields if present
        if "id" in q_meta and q_meta["id"]:
            metadata["original_id"] = q_meta["id"]
        if "subject" in q_meta:
            metadata["subject"] = q_meta.get("subject", "")
        if "format" in q_meta:
            metadata["format"] = q_meta.get("format", "")
        if "difficulty" in q_meta:
            metadata["difficulty"] = q_meta.get("difficulty", "")
        
        question_metadata.append(metadata)
    
    # Save question metadata
    metadata_path = os.path.join(output_dir, "question_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(question_metadata, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved question metadata to {metadata_path} ({len(question_metadata):,} questions)")
    
    # Save index to question_id mapping
    index_to_id = {item["index"]: item["question_id"] for item in question_metadata}
    index_mapping_path = os.path.join(output_dir, "index_to_question_id.json")
    with open(index_mapping_path, 'w') as f:
        json.dump(index_to_id, f, indent=2)
    print(f"✓ Saved index mapping to {index_mapping_path}")
    
    # Save cluster stats
    unique, counts = np.unique(labels, return_counts=True)
    stats = {
        "num_clusters": len(unique),
        "num_questions": num_questions,
        "cluster_sizes": {int(k): int(v) for k, v in zip(unique, counts)},
        "embedding_dim": centroids.shape[1],
        "dataset_sources": list(set([q.get("source", q.get("dataset_source", "unknown")) for q in questions])),
        "avg_cluster_size": float(np.mean(counts)),
        "min_cluster_size": int(np.min(counts)),
        "max_cluster_size": int(np.max(counts)),
    }
    
    # Add field-specific stats if available
    if questions and "subject" in questions[0]:
        from collections import Counter
        subjects = [q.get("subject", "") for q in questions if q.get("subject")]
        difficulties = [q.get("difficulty", "") for q in questions if q.get("difficulty")]
        formats = [q.get("format", "") for q in questions if q.get("format")]
        
        stats["subjects"] = dict(Counter(subjects))
        stats["difficulties"] = dict(Counter(difficulties))
        stats["formats"] = dict(Counter(formats))
    
    stats_path = os.path.join(output_dir, "cluster_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved cluster stats to {stats_path}")
    
    # Print some statistics
    print(f"\nCluster Statistics:")
    print(f"  Number of clusters: {stats['num_clusters']}")
    print(f"  Average cluster size: {stats['avg_cluster_size']:.1f}")
    print(f"  Min cluster size: {stats['min_cluster_size']}")
    print(f"  Max cluster size: {stats['max_cluster_size']}")


def main():
    parser = argparse.ArgumentParser(description="Cluster pre-computed embeddings using K-Means")
    parser.add_argument("--questions", type=str, 
                       default="/root/euclid/rentropy/cluster_space/corpus_cleaned/all_questions_cleaned.json",
                       help="Path to questions JSON file")
    parser.add_argument("--embeddings", type=str,
                       default="/root/euclid/rentropy/cluster_space/corpus_cleaned/embeddings_cleaned.npy",
                       help="Path to embeddings numpy file")
    parser.add_argument("--output_dir", type=str, default="./cluster_data_cleaned",
                       help="Output directory for cluster data")
    parser.add_argument("--num_clusters", type=int, default=2048,
                       help="Number of clusters for K-Means")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to use (default: use all data)")
    parser.add_argument("--n_init", type=int, default=10,
                       help="Number of K-Means initializations")
    parser.add_argument("--max_iter", type=int, default=300,
                       help="Maximum number of K-Means iterations")
    parser.add_argument("--random_state", type=int, default=42,
                       help="Random state for reproducibility")
    args = parser.parse_args()
    
    print("="*70)
    print("CLUSTERING PRE-COMPUTED EMBEDDINGS")
    print("="*70)
    
    # Load data
    questions_metadata = load_questions_from_file(args.questions)
    embeddings = load_embeddings(args.embeddings)
    
    # Verify counts match
    if len(questions_metadata) != len(embeddings):
        print(f"\n❌ ERROR: Count mismatch!")
        print(f"   Questions: {len(questions_metadata):,}")
        print(f"   Embeddings: {len(embeddings):,}")
        return
    
    print(f"\n✓ Questions and embeddings match: {len(questions_metadata):,} items")
    
    # Sample data if num_samples is specified
    if args.num_samples is not None and args.num_samples < len(questions_metadata):
        print(f"\n📊 Sampling {args.num_samples:,} items from {len(questions_metadata):,} total...")
        
        # Set random seed for reproducibility
        np.random.seed(args.random_state)
        
        # Generate random indices
        sampled_indices = np.random.choice(
            len(questions_metadata), 
            size=args.num_samples, 
            replace=False
        )
        sampled_indices = np.sort(sampled_indices)  # Sort for better cache locality
        
        print(f"Sampled indices range: [{sampled_indices[0]}, {sampled_indices[-1]}]")
        
        # Sample questions and embeddings
        questions_metadata = [questions_metadata[i] for i in sampled_indices]
        embeddings = embeddings[sampled_indices]
        
        print(f"✓ Sampled {len(questions_metadata):,} questions and embeddings")
    elif args.num_samples is not None and args.num_samples >= len(questions_metadata):
        print(f"\n⚠️  Requested samples ({args.num_samples:,}) >= total data ({len(questions_metadata):,})")
        print(f"   Using all {len(questions_metadata):,} items")
    
    # Check if we need to adjust num_clusters
    if len(questions_metadata) < args.num_clusters:
        print(f"\n⚠️  Warning: fewer questions ({len(questions_metadata):,}) than clusters ({args.num_clusters})")
        args.num_clusters = max(10, len(questions_metadata) // 5)
        print(f"   Reducing to {args.num_clusters} clusters")
    
    # Perform clustering
    kmeans = fit_kmeans(
        embeddings, 
        args.num_clusters,
        n_init=args.n_init,
        max_iter=args.max_iter,
        random_state=args.random_state
    )
    
    # Save results
    save_cluster_data(
        args.output_dir,
        kmeans.cluster_centers_,
        kmeans.labels_,
        questions_metadata,
        embeddings,
    )
    
    print("\n" + "="*70)
    print("CLUSTERING COMPLETE!")
    print("="*70)
    print(f"Output directory: {args.output_dir}")
    if args.num_samples is not None:
        print(f"Data used: {len(questions_metadata):,} sampled items")
    else:
        print(f"Data used: {len(questions_metadata):,} items (all data)")
    print(f"\nFiles saved:")
    print(f"  - centroids.npy: Cluster centroids ({kmeans.cluster_centers_.shape})")
    print(f"  - labels.npy: Cluster assignments ({kmeans.labels_.shape})")
    print(f"  - embeddings.npy: Question embeddings ({embeddings.shape})")
    print(f"  - question_metadata.json: Full metadata with question_id mapping")
    print(f"  - index_to_question_id.json: Quick index -> question_id lookup")
    print(f"  - cluster_stats.json: Summary statistics")
    print("="*70)


if __name__ == "__main__":
    main()
