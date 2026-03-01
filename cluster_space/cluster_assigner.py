"""
Cluster assignment and count tracking for Rentropy diversity reward.
This module is imported by the reward function during training.
"""
import numpy as np
import os
import json
from typing import List, Tuple, Optional
from collections import defaultdict
import torch

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class ClusterAssigner:
    """Assigns questions to clusters and tracks visit counts."""
    
    def __init__(
        self,
        centroids_path: str,
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        ema_decay: float = 0.99,
        smoothing_alpha: float = 1.0,
        init_counts_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            centroids_path: Path to centroids.npy file
            embedding_model: Sentence transformer model for embedding
            ema_decay: Decay factor for exponential moving average of counts
            smoothing_alpha: Smoothing constant for log-inverse-frequency reward
            init_counts_path: Optional path to numpy file with previous cluster frequencies
                             (e.g., cluster_frequencies.npy)
        """
        if SentenceTransformer is None:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
        
        # Load centroids
        self.centroids = np.load(centroids_path)
        self.num_clusters = self.centroids.shape[0]
        print(f"[ClusterAssigner] Loaded {self.num_clusters} centroids from {centroids_path}")
        
        # Load embedding model
        # CUDA_VISIBLE_DEVICES should be set by caller before importing this module
        # to ensure we use the designated GPU (e.g., GPU 7)
        print(f"[ClusterAssigner] Loading embedding model: {embedding_model}")
        print(f"[ClusterAssigner] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        
        if device is None and torch.cuda.is_available():
            device = 'cuda:0'  # Use first visible GPU (should be the designated embedding GPU)
        if device is not None and torch.cuda.is_available():
            print(f"[ClusterAssigner] Using device: {device}")
        else:
            device = 'cpu'
            print(f"[ClusterAssigner] CUDA not available, falling back to CPU")
        
        self.embed_model = SentenceTransformer(embedding_model, trust_remote_code=True, device=device)
        
        # Count tracking (EMA style)
        self.ema_decay = ema_decay
        self.smoothing_alpha = smoothing_alpha
        
        # Initialize cluster counts - either from previous log or uniform
        if init_counts_path and os.path.exists(init_counts_path):
            self._load_counts_from_log(init_counts_path)
        else:
            self.cluster_counts = np.ones(self.num_clusters, dtype=np.float64) * smoothing_alpha
            self.total_count = float(self.num_clusters * smoothing_alpha)
            print(f"[ClusterAssigner] Initialized uniform cluster counts (1/{self.num_clusters})")
    
    def _load_counts_from_log(self, log_path: str):
        """Load cluster counts from a numpy file containing cluster frequencies."""
        try:
            # Load cluster counts from numpy file (ensure float dtype)
            self.cluster_counts = np.load(log_path).astype(np.float64)
            
            # Verify size matches
            if len(self.cluster_counts) != self.num_clusters:
                print(f"[ClusterAssigner] WARNING: Cluster count size mismatch! "
                      f"Expected {self.num_clusters}, got {len(self.cluster_counts)}")
                print(f"[ClusterAssigner] Falling back to uniform initialization")
                self.cluster_counts = np.ones(self.num_clusters, dtype=np.float64) * self.smoothing_alpha
            
            self.total_count = np.sum(self.cluster_counts)
            
            # Print some stats about the loaded distribution
            min_count = np.min(self.cluster_counts)
            max_count = np.max(self.cluster_counts)
            mean_count = np.mean(self.cluster_counts)
            nonuniform = np.sum(self.cluster_counts > mean_count * 1.1)
            
            print(f"[ClusterAssigner] Loaded cluster counts from {log_path}")
            print(f"[ClusterAssigner] Stats - min: {min_count:.4f}, max: {max_count:.4f}, "
                  f"mean: {mean_count:.4f}, clusters above mean: {nonuniform}")
            print(f"[ClusterAssigner] Total count: {self.total_count:.4f}")
            
        except Exception as e:
            print(f"[ClusterAssigner] ERROR loading counts from {log_path}: {e}")
            print(f"[ClusterAssigner] Falling back to uniform initialization")
            self.cluster_counts = np.ones(self.num_clusters, dtype=np.float64) * self.smoothing_alpha
            self.total_count = float(self.num_clusters * self.smoothing_alpha)
    
    def embed(self, questions: List[str]) -> np.ndarray:
        """Embed a list of questions."""
        embeddings = self.embed_model.encode(
            questions,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings
    
    def assign_clusters(self, questions: List[str]) -> np.ndarray:
        """Assign cluster IDs to a list of questions."""
        if not questions:
            return np.array([], dtype=np.int32)
        
        embeddings = self.embed(questions)
        # Compute distances to all centroids
        # Since embeddings are normalized, we can use dot product (higher = closer)
        similarities = embeddings @ self.centroids.T  # (n_questions, n_clusters)
        cluster_ids = np.argmax(similarities, axis=1)
        return cluster_ids
    
    def get_cluster_probabilities(self) -> np.ndarray:
        """Get current estimated cluster probabilities."""
        return self.cluster_counts / self.total_count
    
    def compute_rarity_reward(self, cluster_ids: np.ndarray) -> np.ndarray:
        """
        Compute rarity reward using exponential decay based on visit counts.
        
        Formula: reward = exp(-count / mean_count)
        
        This provides a MUCH stronger signal than the old -log(p)/18.4 approach:
        - Old method: 5x visit difference → 0.09 reward difference (too weak!)
        - New method: 5x visit difference → 0.54 reward difference (strong signal!)
        
        Properties:
        - Rare clusters (count < mean): reward > 0.37 (e^-1)
        - Average clusters (count = mean): reward = 0.37
        - Popular clusters (count > mean): reward < 0.37
        - Natural [0, 1] range without artificial normalization
        """
        counts = self.cluster_counts[cluster_ids]
        mean_count = np.mean(self.cluster_counts)
        
        # Prevent division by zero
        if mean_count < 1e-8:
            mean_count = 1e-8
        
        # Exponential decay: rare clusters get high reward, popular get low
        rewards = np.exp(-counts / mean_count)
        
        return rewards
    

    def update_counts(self, cluster_ids: np.ndarray):
        """Update cluster counts with EMA.
        
        Fixed: Only apply EMA decay when we actually have questions to update.
        This prevents stale cluster distributions from decaying when no questions are generated.
        """
        if len(cluster_ids) == 0:
            # No questions to update, don't decay (preserves current distribution)
            return
        
        # Apply decay ONCE per batch, not per question (fixes reward explosion bug)
        # But only when we have actual questions to process
        self.cluster_counts *= self.ema_decay
        
        for cid in cluster_ids:
            self.cluster_counts[cid] += (1 - self.ema_decay)
        
        self.total_count = np.sum(self.cluster_counts)
    
    def get_stats(self) -> dict:
        """Get current cluster statistics for logging."""
        return {
            "cluster_counts": self.cluster_counts.tolist(),
            "cluster_probabilities": self.get_cluster_probabilities().tolist(),
            "total_count": float(self.total_count),
            "num_clusters": self.num_clusters,
        }


# Global instance (lazy loaded)
_assigner: Optional[ClusterAssigner] = None


def get_assigner(centroids_path: str = None, **kwargs) -> ClusterAssigner:
    """Get or create the global ClusterAssigner instance."""
    global _assigner
    if _assigner is None:
        if centroids_path is None:
            # Default path - adjust as needed
            centroids_path = os.path.join(
                os.path.dirname(__file__),
                "cluster_data",
                "centroids.npy"
            )
        _assigner = ClusterAssigner(centroids_path, **kwargs)
    return _assigner


def reset_assigner():
    """Reset the global assigner (useful for testing)."""
    global _assigner
    _assigner = None
