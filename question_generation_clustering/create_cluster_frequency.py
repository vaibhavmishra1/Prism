#!/usr/bin/env python3
"""
Create cluster frequency array from OpenAI validated matched questions.
"""

import json
import numpy as np
from collections import Counter

# Configuration
INPUT_FILE = "balanced_questions__darwin_iter2_openai_validated_matched.json"
OUTPUT_FILE = "cluster_frequency_darwin_iter2_dataset_verified_matched.npy"
NUM_CLUSTERS = 128

def main():
    # Load the JSON file
    print(f"[Load] Reading from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data)} questions")
    
    # Extract cluster IDs
    print(f"\n[Extract] Extracting cluster IDs...")
    cluster_ids = []
    missing_cluster = 0
    
    for item in data:
        if "cluster_id" in item:
            cluster_ids.append(item["cluster_id"])
        else:
            missing_cluster += 1
    
    print(f"✓ Extracted {len(cluster_ids)} cluster IDs")
    if missing_cluster > 0:
        print(f"  ⚠ Warning: {missing_cluster} items missing 'cluster_id' field")
    
    # Count frequencies
    print(f"\n[Count] Counting cluster frequencies...")
    cluster_counter = Counter(cluster_ids)
    
    # Create frequency array of size 128
    frequency_array = np.zeros(NUM_CLUSTERS, dtype=np.int32)
    for cluster_id, count in cluster_counter.items():
        if 0 <= cluster_id < NUM_CLUSTERS:
            frequency_array[cluster_id] = count
        else:
            print(f"  ⚠ Warning: Found cluster_id {cluster_id} outside range [0, {NUM_CLUSTERS-1}]")
    
    print(f"✓ Created frequency array of size {NUM_CLUSTERS}")
    
    # Print statistics
    print(f"\n[Stats] Cluster frequency statistics:")
    print(f"  Total questions: {len(cluster_ids)}")
    print(f"  Total clusters: {NUM_CLUSTERS}")
    print(f"  Clusters with questions: {np.count_nonzero(frequency_array)}")
    print(f"  Clusters without questions: {NUM_CLUSTERS - np.count_nonzero(frequency_array)}")
    print(f"  Average per cluster: {frequency_array.mean():.2f}")
    print(f"  Median per cluster: {np.median(frequency_array):.2f}")
    print(f"  Min per cluster: {frequency_array.min()}")
    print(f"  Max per cluster: {frequency_array.max()}")
    print(f"  Std deviation: {frequency_array.std():.2f}")
    
    # Show top 10 clusters
    print(f"\n[Top] Top 10 most frequent clusters:")
    top_indices = np.argsort(frequency_array)[::-1][:10]
    for i, cluster_id in enumerate(top_indices, 1):
        count = frequency_array[cluster_id]
        percent = count / len(cluster_ids) * 100
        print(f"  {i}. Cluster {cluster_id}: {count} questions ({percent:.1f}%)")
    
    # Show bottom 10 non-zero clusters
    print(f"\n[Bottom] Bottom 10 least frequent clusters (excluding empty):")
    non_zero_indices = np.where(frequency_array > 0)[0]
    if len(non_zero_indices) > 0:
        sorted_non_zero = sorted(non_zero_indices, key=lambda x: frequency_array[x])[:10]
        for i, cluster_id in enumerate(sorted_non_zero, 1):
            count = frequency_array[cluster_id]
            percent = count / len(cluster_ids) * 100
            print(f"  {i}. Cluster {cluster_id}: {count} questions ({percent:.1f}%)")
    
    # Distribution histogram
    print(f"\n[Distribution] Frequency distribution:")
    bins = [0, 5, 10, 15, 20, 30]
    labels = ["0", "1-5", "6-10", "11-15", "16-20", "20+"]
    for i in range(len(bins)):
        if i == 0:
            count = np.sum(frequency_array == 0)
            label = labels[0]
        elif i == len(bins) - 1:
            count = np.sum(frequency_array > bins[i])
            label = labels[i]
        else:
            count = np.sum((frequency_array > bins[i]) & (frequency_array <= bins[i+1]))
            label = labels[i]
        percent = count / NUM_CLUSTERS * 100
        print(f"  {label} questions: {count} clusters ({percent:.1f}%)")
    
    # Save as numpy array
    print(f"\n[Save] Saving frequency array to {OUTPUT_FILE}...")
    np.save(OUTPUT_FILE, frequency_array)
    print(f"✓ Saved successfully")
    
    # Verify saved file
    print(f"\n[Verify] Verifying saved file...")
    loaded_array = np.load(OUTPUT_FILE)
    assert loaded_array.shape == (NUM_CLUSTERS,), f"Shape mismatch: {loaded_array.shape}"
    assert np.array_equal(loaded_array, frequency_array), "Array contents mismatch"
    print(f"✓ Verification passed")
    print(f"  Shape: {loaded_array.shape}")
    print(f"  Dtype: {loaded_array.dtype}")
    print(f"  Total count: {loaded_array.sum()}")
    
    print("\n✅ All done!")

if __name__ == "__main__":
    main()
