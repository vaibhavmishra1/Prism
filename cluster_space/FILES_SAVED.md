# Files Saved by build_clusters.py

When you run `build_clusters.py`, it now saves the following files in the output directory:

## Core Files (Required for Runtime)

### 1. `centroids.npy`
- **Type**: NumPy array
- **Shape**: `(num_clusters, embedding_dim)`
- **Content**: K-Means cluster centroids
- **Usage**: Loaded by `ClusterAssigner` to assign new questions to clusters
- **Size**: ~200KB for 128 clusters × 384 dim

### 2. `labels.npy`
- **Type**: NumPy array
- **Shape**: `(num_questions,)`
- **Content**: Cluster assignment for each question (0 to num_clusters-1)
- **Usage**: Maps question index → cluster_id
- **Size**: ~50KB for 10K questions

## New Files (Question Data)

### 3. `embeddings.npy` ⭐ NEW
- **Type**: NumPy array
- **Shape**: `(num_questions, embedding_dim)`
- **Content**: Embedding vector for each question
- **Usage**: Pre-computed embeddings for analysis or reuse
- **Size**: ~15MB for 10K questions × 384 dim (float32)
- **Index mapping**: Row `i` corresponds to question at index `i` in `labels.npy`

### 4. `question_metadata.json` ⭐ NEW
- **Type**: JSON array
- **Content**: Full metadata for each question
- **Structure**:
  ```json
  [
    {
      "question_id": "q_a1b2c3d4e5f6_0",
      "index": 0,
      "question": "Find all integers n such that...",
      "cluster_id": 42,
      "dataset_source": "math12k"
    },
    ...
  ]
  ```
- **Usage**: Complete mapping from `question_id` to all question information
- **Size**: ~5-10MB for 10K questions (depends on question length)

### 5. `index_to_question_id.json` ⭐ NEW
- **Type**: JSON object
- **Content**: Quick lookup from index → question_id
- **Structure**:
  ```json
  {
    "0": "q_a1b2c3d4e5f6_0",
    "1": "q_f6e5d4c3b2a1_1",
    ...
  }
  ```
- **Usage**: Fast reverse lookup (index in labels.npy → question_id)
- **Size**: ~200KB for 10K questions

## Metadata Files

### 6. `cluster_stats.json`
- **Type**: JSON object
- **Content**: Summary statistics
- **Structure**:
  ```json
  {
    "num_clusters": 128,
    "num_questions": 12000,
    "cluster_sizes": {
      "0": 95,
      "1": 87,
      ...
    },
    "embedding_dim": 384,
    "dataset_sources": ["math12k", "gsm8k", "math"]
  }
  ```
- **Usage**: Quick overview of cluster distribution
- **Size**: ~5KB

## Index Mapping

All files use the **same index ordering**:

- `labels[i]` = cluster_id for question at index `i`
- `embeddings[i]` = embedding vector for question at index `i`
- `question_metadata[i]["index"]` = `i`
- `question_metadata[i]["question_id"]` = unique ID for question at index `i`

## Usage Examples

### Load embeddings for a specific question

```python
import numpy as np
import json

# Load data
embeddings = np.load("cluster_data/embeddings.npy")
labels = np.load("cluster_data/labels.npy")
with open("cluster_data/question_metadata.json") as f:
    metadata = json.load(f)

# Find question by ID
question_id = "q_a1b2c3d4e5f6_0"
question_info = next(m for m in metadata if m["question_id"] == question_id)
index = question_info["index"]

# Get embedding and cluster
embedding = embeddings[index]
cluster_id = labels[index]
print(f"Question: {question_info['question']}")
print(f"Cluster: {cluster_id}")
print(f"Embedding shape: {embedding.shape}")
```

### Find all questions in a cluster

```python
import numpy as np
import json

labels = np.load("cluster_data/labels.npy")
with open("cluster_data/question_metadata.json") as f:
    metadata = json.load(f)

target_cluster = 42
indices_in_cluster = np.where(labels == target_cluster)[0]

print(f"Questions in cluster {target_cluster}:")
for idx in indices_in_cluster:
    question_info = metadata[idx]
    print(f"  [{idx}] {question_info['question_id']}: {question_info['question'][:50]}...")
```

### Get dataset distribution

```python
import json
from collections import Counter

with open("cluster_data/question_metadata.json") as f:
    metadata = json.load(f)

sources = [m["dataset_source"] for m in metadata]
source_counts = Counter(sources)

print("Questions by dataset:")
for source, count in source_counts.most_common():
    print(f"  {source}: {count}")
```

## File Sizes (Approximate)

For 10,000 questions, 128 clusters, 384-dim embeddings:

| File | Size |
|------|------|
| `centroids.npy` | ~200 KB |
| `labels.npy` | ~50 KB |
| `embeddings.npy` | ~15 MB |
| `question_metadata.json` | ~5-10 MB |
| `index_to_question_id.json` | ~200 KB |
| `cluster_stats.json` | ~5 KB |
| **Total** | **~20-25 MB** |
