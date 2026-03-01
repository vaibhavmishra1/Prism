#!/usr/bin/env python3
"""
Build cluster space from a corpus of math questions.
This is run ONCE offline before training.

Usage:
    python build_clusters.py --corpus_file all_questions.json --output_dir ./cluster_data --num_clusters 2048
"""
import argparse
import json
import os
import tempfile
import numpy as np
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from vllm import LLM
except ImportError:
    LLM = None

from sklearn.cluster import KMeans


def generate_question_id(question: str, index: int) -> str:
    """Generate a unique ID for a question based on content hash and index."""
    # Use hash of question + index for uniqueness
    content = f"{question}_{index}".encode('utf-8')
    hash_obj = hashlib.md5(content)
    return f"q_{hash_obj.hexdigest()[:12]}_{index}"


def load_questions_from_file(filepath: str, dataset_source: str = None) -> List[Dict]:
    """
    Load questions from a JSON file. Supports multiple formats.
    
    Supports:
    1. MathLake format: [{"question": "...", "source": "...", "id": "...", "subject": "...", "format": "...", "difficulty": "..."}, ...]
    2. Simple format: [{"question": "...", "source": "..."}, ...]
    3. Old format: ["question1", "question2", ...] (uses filename as source)
    4. Dict format: {"questions": [...]}
    
    Returns:
        List of metadata dicts, each with at least "question" and "source" fields.
        For MathLake format, preserves: source, id, question, subject, format, difficulty
    """
    if dataset_source is None:
        dataset_source = Path(filepath).stem  # Use filename as source (fallback)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions = []
    if isinstance(data, list):
        for item in data:
            question_text = None
            metadata = {}
            
            if isinstance(item, str):
                # Old format: plain string
                question_text = item
                metadata = {
                    "question": question_text,
                    "source": dataset_source,
                }
            elif isinstance(item, dict):
                # New format: dict with question and potentially full metadata
                # Try to extract question text
                for key in ['question', 'problem', 'text', 'input']:
                    if key in item and item[key]:
                        question_text = item[key]
                        break
                
                # Preserve all metadata fields from MathLake format
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
        # If it's a dict with a questions key
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
    
    return questions


def load_questions_from_dir(corpus_dir: str) -> List[Dict]:
    """
    Load questions from all JSON files in a directory.
    
    Returns:
        List of metadata dicts (each with "question" and "source" fields, plus MathLake fields if present)
    
    Note: If files contain source info in the data, that takes precedence.
    Otherwise, filename stem is used as source.
    """
    questions = []
    corpus_path = Path(corpus_dir)
    
    for json_file in corpus_path.glob("*.json"):
        dataset_source = json_file.stem  # Use filename as source (fallback)
        print(f"Loading from {json_file} (default source: {dataset_source})...")
        file_questions = load_questions_from_file(str(json_file), dataset_source)
        questions.extend(file_questions)
        
        # Show actual sources found in this file
        if file_questions:
            sources_found = set(q.get('source', dataset_source) for q in file_questions)
            if len(sources_found) > 1 or (len(sources_found) == 1 and list(sources_found)[0] != dataset_source):
                print(f"  Found sources: {sources_found}")
    
    return questions


def embed_questions(questions: List[str], model_name: str, batch_size: int = 1024 * 10, normalize: bool = True, use_vllm: bool = False, timeout: int = 300, min_batch_size: int = 100) -> np.ndarray:
    """Embed questions using a sentence transformer model or vLLM."""
    print(f"Loading embedding model: {model_name} (use_vllm={use_vllm})")
    
    if use_vllm:
        if LLM is None:
            raise ImportError("vllm is not installed. Please install it with: pip install vllm")
        
        # vLLM implementation for embedding
        print(f"Using vLLM for embedding...")
        max_tokens = 32768
        model = LLM(
            model=model_name,
            runner="pooling",
            trust_remote_code=True,
            gpu_memory_utilization=0.85, # Leave some room for other things
            enforce_eager=True,
            max_model_len=max_tokens,
        )
        
        # Get tokenizer to properly check/truncate questions
        tokenizer = model.get_tokenizer()
        
        # Truncate questions that are too long using actual tokenization
        truncated_questions = []
        num_truncated = 0
        print(f"Checking and truncating questions longer than {max_tokens} tokens...")
        for q in tqdm(questions, desc="Processing questions"):
            # First do a quick character-based filter (4 chars ≈ 1 token) to avoid tokenizing everything
            if len(q) > max_tokens * 2:
                # Likely too long, tokenize to check
                tokens = tokenizer.encode(q)
                if len(tokens) > max_tokens:
                    # Truncate by decoding only the first max_tokens
                    q = tokenizer.decode(tokens[:max_tokens -1 ])
                    num_truncated += 1
            truncated_questions.append(q)
        
        if num_truncated > 0:
            print(f"Warning: Truncated {num_truncated} questions that exceeded {max_tokens} tokens")
        
        print(f"Embedding {len(truncated_questions)} questions in batches of {batch_size}...")
        
        embedding_dim = 1024
        print(f"Embedding dimension: {embedding_dim}")
        
        # Use memory-mapped array to avoid OOM with large datasets
        temp_dir = tempfile.gettempdir()
        
        # Check for existing memmap file from previous run
        existing_files = [f for f in os.listdir(temp_dir) if f.startswith('embeddings_') and f.endswith('.npy')]
        embeddings_file = None
        embeddings = None
        resume_mode = False
        
        if existing_files:
            # Found existing file(s), use the most recent one
            existing_files.sort(key=lambda f: os.path.getmtime(os.path.join(temp_dir, f)), reverse=True)
            embeddings_file = os.path.join(temp_dir, existing_files[0])
            
            # Check if it matches our expected size
            expected_size = len(truncated_questions) * embedding_dim * 4  # 4 bytes per float32
            actual_size = os.path.getsize(embeddings_file)
            
            if actual_size == expected_size:
                print(f"Found existing embeddings file: {embeddings_file}")
                print(f"Resuming from previous run...")
                embeddings = np.memmap(
                    embeddings_file, 
                    dtype='float32', 
                    mode='r+',  # Read-write mode to continue
                    shape=(len(truncated_questions), embedding_dim)
                )
                resume_mode = True
            else:
                print(f"Found existing file but size mismatch (expected {expected_size}, got {actual_size})")
                print(f"Creating new embeddings file...")
                embeddings_file = os.path.join(temp_dir, f"embeddings_{os.getpid()}.npy")
        
        if embeddings is None:
            embeddings_file = os.path.join(temp_dir, f"embeddings_{os.getpid()}.npy")
            print(f"Creating memory-mapped file: {embeddings_file}")
            embeddings = np.memmap(
                embeddings_file, 
                dtype='float32', 
                mode='w+', 
                shape=(len(truncated_questions), embedding_dim)
            )
        
        # Helper function to embed with timeout
        import signal
        
        def embed_with_timeout(questions_batch, timeout_seconds=180):
            """Embed a batch with timeout. Returns outputs or raises TimeoutError."""
            def timeout_handler(signum, frame):
                raise TimeoutError(f"vLLM embed call timed out after {timeout_seconds}s")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            try:
                outputs = model.embed(questions_batch)
                signal.alarm(0)  # Cancel alarm
                return outputs
            except TimeoutError:
                signal.alarm(0)  # Cancel alarm
                raise
            except Exception as e:
                signal.alarm(0)  # Cancel alarm
                raise e
        
        def process_batch_with_fallback(batch, batch_start_idx, embeddings, timeout=180, min_batch=1):
            """
            Process a batch with fallback to smaller batches if timeout occurs.
            Returns number of questions that failed.
            """
            failed_count = 0
            
            # Skip if already completed (resuming)
            if resume_mode:
                batch_embeddings = embeddings[batch_start_idx:batch_start_idx+len(batch)]
                if np.any(batch_embeddings != 0):
                    return 0  # Already done
            
            try:
                # Try to process the full batch
                outputs = embed_with_timeout(batch, timeout_seconds=timeout)
                
                # Extract embeddings
                for j, output in enumerate(outputs):
                    try:
                        embedding = output.outputs.embedding
                        embeddings[batch_start_idx + j] = embedding
                    except Exception as e:
                        print(f"\nWarning: Failed to extract embedding at index {batch_start_idx + j}: {e}")
                        embeddings[batch_start_idx + j] = np.zeros(embedding_dim, dtype='float32')
                        failed_count += 1
                
                embeddings.flush()
                del outputs
                
                # Clear GPU cache
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass
                
                return failed_count
                
            except TimeoutError as e:
                print(f"\n{e} for batch starting at {batch_start_idx} (size: {len(batch)})")
                
                # If batch is small enough, skip these questions
                if len(batch) <= min_batch:
                    print(f"Skipping {len(batch)} question(s) that cause timeout")
                    for j in range(len(batch)):
                        embeddings[batch_start_idx + j] = np.zeros(embedding_dim, dtype='float32')
                    embeddings.flush()
                    return len(batch)
                
                # Otherwise, split batch and retry with smaller chunks
                print(f"Retrying with smaller sub-batches...")
                mid = len(batch) // 2
                
                # Process first half
                failed_count += process_batch_with_fallback(
                    batch[:mid], 
                    batch_start_idx, 
                    embeddings,
                    timeout=timeout // 2,
                    min_batch=min_batch
                )
                
                # Process second half
                failed_count += process_batch_with_fallback(
                    batch[mid:], 
                    batch_start_idx + mid, 
                    embeddings,
                    timeout=timeout // 2,
                    min_batch=min_batch
                )
                
                return failed_count
                
            except Exception as e:
                print(f"\nError embedding batch starting at {batch_start_idx}: {e}")
                print(f"Skipping {len(batch)} question(s)")
                for j in range(len(batch)):
                    embeddings[batch_start_idx + j] = np.zeros(embedding_dim, dtype='float32')
                embeddings.flush()
                return len(batch)
        
        # Process in batches to show progress and handle memory better
        num_failed = 0
        num_skipped = 0
        timeout_per_batch = timeout  # Timeout for full batch
        min_batch_size_param = min_batch_size  # Minimum batch size before giving up on individual questions
        
        for i in tqdm(range(0, len(truncated_questions), batch_size), desc="Batches"):
            batch = truncated_questions[i:i+batch_size]
            batch_start_idx = i
            
            failed = process_batch_with_fallback(
                batch, 
                batch_start_idx, 
                embeddings,
                timeout=timeout_per_batch,
                min_batch=min_batch_size_param
            )
            num_failed += failed
        
        if num_skipped > 0:
            print(f"\nResumed: Skipped {num_skipped} already-completed batches")
        if num_failed > 0:
            print(f"\nWarning: {num_failed}/{len(truncated_questions)} questions failed to embed and were zero-padded")
        
        # Convert memmap to regular array for further processing
        print("Loading embeddings into memory for normalization...")
        embeddings = np.array(embeddings)
        
        # Clean up temporary memmap file
        try:
            if os.path.exists(embeddings_file):
                os.remove(embeddings_file)
                print(f"Cleaned up temporary file: {embeddings_file}")
        except Exception as e:
            print(f"Warning: Could not remove temporary file {embeddings_file}: {e}")
        
        # vLLM doesn't automatically normalize in the same way, so we do it manually if requested
        if normalize:
            print("Normalizing embeddings...")
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            # Handle any zero embeddings (failed questions)
            norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
            embeddings = embeddings / (norms + 1e-12)
            
        return embeddings
    else:
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is not installed. Please install it with: pip install sentence-transformers")
            
        model = SentenceTransformer(model_name, trust_remote_code=True)
        
        print(f"Embedding {len(questions)} questions...")
        embeddings = model.encode(
            questions,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        
        return embeddings


def fit_kmeans(embeddings: np.ndarray, num_clusters: int, n_init: int = 10, max_iter: int = 300, random_state: int = 42) -> KMeans:
    """Fit K-Means clustering on embeddings."""
    print(f"Fitting K-Means with {num_clusters} clusters...")
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
    
    Args:
        output_dir: Directory to save files
        centroids: Cluster centroids array (num_clusters, embedding_dim)
        labels: Cluster assignments array (num_questions,)
        questions: List of metadata dicts (each with "question" and "source" fields, plus MathLake fields)
        embeddings: Question embeddings array (num_questions, embedding_dim)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_questions = len(questions)
    assert len(labels) == num_questions, f"Mismatch: {len(labels)} labels vs {num_questions} questions"
    assert len(embeddings) == num_questions, f"Mismatch: {len(embeddings)} embeddings vs {num_questions} questions"
    
    # Save centroids
    centroids_path = os.path.join(output_dir, "centroids.npy")
    np.save(centroids_path, centroids)
    print(f"Saved centroids to {centroids_path} (shape: {centroids.shape})")
    
    # Save labels (cluster assignments)
    labels_path = os.path.join(output_dir, "labels.npy")
    np.save(labels_path, labels)
    print(f"Saved labels to {labels_path} (shape: {labels.shape})")
    
    # Save embeddings
    embeddings_path = os.path.join(output_dir, "embeddings.npy")
    np.save(embeddings_path, embeddings)
    print(f"Saved embeddings to {embeddings_path} (shape: {embeddings.shape})")
    
    # Generate question IDs and create metadata (preserving all MathLake fields)
    question_metadata = []
    for index, q_meta in enumerate(questions):
        question_text = q_meta.get("question", "")
        # Use existing ID if present (from MathLake), otherwise generate one
        existing_id = q_meta.get("id", "")
        if existing_id:
            question_id = existing_id
        else:
            question_id = generate_question_id(question_text, index)
        
        # Build metadata preserving all original fields
        metadata = {
            "question_id": question_id,
            "index": index,  # Index in labels.npy and embeddings.npy
            "question": question_text,
            "cluster_id": int(labels[index]),
            "dataset_source": q_meta.get("source", q_meta.get("dataset_source", "unknown")),
        }
        
        # Preserve MathLake-specific fields if present
        if "id" in q_meta and q_meta["id"]:
            metadata["original_id"] = q_meta["id"]
        if "subject" in q_meta:
            metadata["subject"] = q_meta.get("subject", "")
        if "format" in q_meta:
            metadata["format"] = q_meta.get("format", "")
        if "difficulty" in q_meta:
            metadata["difficulty"] = q_meta.get("difficulty", "")
        
        question_metadata.append(metadata)
    
    # Save question metadata (mapping from question_id to all info)
    metadata_path = os.path.join(output_dir, "question_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(question_metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved question metadata to {metadata_path} ({len(question_metadata)} questions)")
    
    # Also save a reverse mapping: index -> question_id (for quick lookup)
    index_to_id = {item["index"]: item["question_id"] for item in question_metadata}
    index_mapping_path = os.path.join(output_dir, "index_to_question_id.json")
    with open(index_mapping_path, 'w') as f:
        json.dump(index_to_id, f, indent=2)
    print(f"Saved index mapping to {index_mapping_path}")
    
    # Save cluster stats
    unique, counts = np.unique(labels, return_counts=True)
    stats = {
        "num_clusters": len(unique),
        "num_questions": num_questions,
        "cluster_sizes": {int(k): int(v) for k, v in zip(unique, counts)},
        "embedding_dim": centroids.shape[1],
        "dataset_sources": list(set([q.get("source", q.get("dataset_source", "unknown")) for q in questions])),
    }
    
    # Add MathLake-specific stats if available
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
    print(f"Saved stats to {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Build cluster space for Rentropy")
    parser.add_argument("--corpus_file", type=str, help="Single JSON file with questions")
    parser.add_argument("--corpus_dir", type=str, help="Directory with JSON files")
    parser.add_argument("--dataset_source", type=str, default=None,
                       help="Dataset source name (if using --corpus_file)")
    parser.add_argument("--output_dir", type=str, default="./cluster_data")
    parser.add_argument("--num_clusters", type=int, default=2048)
    parser.add_argument("--embedding_model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM for encoding")
    parser.add_argument("--timeout", type=int, default=300,
                       help="Timeout in seconds for each batch (default: 300)")
    parser.add_argument("--min_batch_size", type=int, default=100,
                       help="Minimum batch size before giving up on questions (default: 100)")
    args = parser.parse_args()
    
    # Load questions (now returns list of metadata dicts)
    if args.corpus_file:
        dataset_source = args.dataset_source or Path(args.corpus_file).stem
        questions_metadata = load_questions_from_file(args.corpus_file, dataset_source)
    elif args.corpus_dir:
        questions_metadata = load_questions_from_dir(args.corpus_dir)
    else:
        raise ValueError("Must provide either --corpus_file or --corpus_dir")
    
    # Extract just question texts for embedding
    questions_text = [q["question"] for q in questions_metadata]
    sources = set(q.get("source", q.get("dataset_source", "unknown")) for q in questions_metadata)
    print(f"Loaded {len(questions_text)} questions from {len(sources)} dataset source(s)")
    
    # Print field statistics if MathLake format
    if questions_metadata and "subject" in questions_metadata[0]:
        from collections import Counter
        subjects = [q.get("subject", "") for q in questions_metadata if q.get("subject")]
        difficulties = [q.get("difficulty", "") for q in questions_metadata if q.get("difficulty")]
        if subjects:
            print(f"Subjects: {len(set(subjects))} unique")
            print(f"Difficulties: {Counter(difficulties)}")
    
    if len(questions_text) < args.num_clusters:
        print(f"Warning: fewer questions ({len(questions_text)}) than clusters ({args.num_clusters})")
        args.num_clusters = max(10, len(questions_text) // 5)
        print(f"Reducing to {args.num_clusters} clusters")
    
    # Embed
    embeddings = embed_questions(
        questions_text, 
        args.embedding_model, 
        args.batch_size, 
        args.normalize, 
        args.use_vllm,
        args.timeout,
        args.min_batch_size
    )
    
    # Cluster
    kmeans = fit_kmeans(embeddings, args.num_clusters)
    
    # Save everything (including embeddings and full metadata)
    save_cluster_data(
        args.output_dir,
        kmeans.cluster_centers_,
        kmeans.labels_,
        questions_metadata,  # Pass full metadata dicts
        embeddings,  # Pass embeddings to save
    )
    
    print("\n" + "="*60)
    print("Done! Files saved:")
    print(f"  - centroids.npy: Cluster centroids")
    print(f"  - labels.npy: Cluster assignments (indexed by question order)")
    print(f"  - embeddings.npy: Question embeddings (indexed by question order)")
    print(f"  - question_metadata.json: Full metadata with question_id mapping")
    print(f"  - index_to_question_id.json: Quick index -> question_id lookup")
    print(f"  - cluster_stats.json: Summary statistics")
    print("="*60)


if __name__ == "__main__":
    main()
