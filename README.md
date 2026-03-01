<div align="center">

# 🔷 Prism

### **Preventing Curriculum Collapse in Self-Evolving Reasoning Systems**

<p align="center">
  <a href="https://huggingface.co/datasets/vibhuiitj/prism-math"><img src="https://img.shields.io/badge/🤗%20Dataset-Prism--Math-blue" alt="Dataset"></a>
  <a href="https://huggingface.co/vibhuiitj/Prism-Solver"><img src="https://img.shields.io/badge/🤗%20Models-Prism--Solver-orange" alt="Solver Model"></a>
  <a href="https://huggingface.co/vibhuiitj/Prism-Questioner"><img src="https://img.shields.io/badge/🤗%20Models-Prism--Solver-orange" alt="Questioner Model"></a>
  <a href="tree/euclid/darwin/paper/report.md"><img src="https://img.shields.io/badge/📄%20Paper-Preprint-green" alt="Paper"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-red" alt="License"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.7-ee4c2c" alt="PyTorch">
</p>

<p align="center">
  <strong>Vaibhav Mishra</strong><br>
  <a href="mailto:vaibhavm209625@gmail.com">vaibhavm209625@gmail.com</a>
</p>

---

*Self-evolving reasoning frameworks let LLMs improve by iteratively generating and solving problems — but they silently collapse into semantic repetition after a few iterations. **Prism** fixes this.*

</div>

<p align="center">
  <img src="figures/iterations.png" alt="Prism self-evolution loop" width="85%">
</p>

---

## 📌 Overview

Self-evolving reasoning systems train a **Questioner** to generate problems and a **Solver** to learn from them in a closed loop. Without explicit diversity pressure, the Questioner exploits familiar problem types and the curriculum gradually collapses — even when surface-level variation is preserved.

**Prism** introduces two targeted interventions:

| Component | What it does |
|---|---|
| 🎯 **Semantic Coverage Reward** | Maintains an embedding-based partition of mathematical problems and rewards questions from underexplored semantic regions using cross-iteration EMA memory |
| ⚡ **Solver-Initialised Questioner** | At each iteration, re-derives the Questioner from the latest Solver — eliminating capability lag and resetting distributional bias |

Together, these components resolve the two root causes of curriculum collapse: *semantic mode-seeking* and *capability-lag-induced narrowing*.

---

## 🏆 Results

Evaluated on **7 mathematical reasoning benchmarks** against 5 self-evolving baselines (all using Qwen3-4B-Base):

| Model | GSM8K | MATH-500 | AMC | Minerva | OlyBench | AIME 2024 | AIME 2025 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen3-4B-Base | 72.60 | 68.20 | 47.50 | 42.30 | 34.80 | 6.70 | 10.30 |
| R-Zero | 92.12 | 79.60 | 57.27 | 52.94 | 44.59 | 13.40 | 9.60 |
| R-Few (5%) | 92.60 | 78.00 | 52.40 | 53.20 | 42.80 | 14.50 | 9.90 |
| R-Few (1%) | 92.30 | 77.80 | 52.70 | 52.10 | 42.40 | 13.60 | 9.10 |
| Absolute Zero | 89.30 | 76.20 | 52.50 | 38.20 | 38.50 | 13.30 | 13.30 |
| SPICE | 92.70 | 78.00 | 57.50 | 51.90 | 42.70 | 12.20 | **19.10** |
| **Prism (ours)** | **93.45** | **81.02** | **61.25** | **56.62** | **45.58** | **16.77** | 12.92 |

> Prism achieves the highest accuracy on **6 out of 7** benchmarks, with gains of **+3.37** on AIME 2024, **+3.68** on Minerva Math, and **+3.75** on AMC over R-Zero.

---

## 🔬 Why Prism Works: Curriculum Diversity Analysis

We measured per-cluster question frequencies over a fixed K=128 semantic partition:

| Questioner | Active Clusters | Norm. Entropy | Gini | Top-10% Share |
|---|:---:|:---:|:---:|:---:|
| Base model | 89 / 128 | 0.71 | 0.81 | 61.2% |
| R-Zero | 65 / 128 | 0.53 | 0.90 | 79.5% |
| **Prism** | **107 / 128** | **0.83** | **0.66** | **42.4%** |

R-Zero *worsens* coverage vs. the base model — a single cluster accumulates ~50× the mean. Prism reverses this, covering 107/128 clusters with no dominant mode.

---

## 🏗️ How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Prism Self-Evolution Loop                        │
│                                                                      │
│  Iteration t:                                                        │
│                                                                      │
│  1. Q_t ← S_{t-1}          (Solver-initialised Questioner)          │
│                                                                      │
│  2. For each GRPO step:                                              │
│     ┌──────────────────────────────────────────────────────────┐    │
│     │  Generate batch {q_g} ~ Q_t                              │    │
│     │  Score via vLLM: p(q) = majority-vote solvability        │    │
│     │  For q with p(q) ∈ [0.3, 0.9]:                          │    │
│     │    c(q) = nearest centroid to embed(q)  ← cluster assign │    │
│     │    d(q) = exp(-n_c / n̄)                ← rarity bonus   │    │
│     │    r(q) = zpd(p(q)) · (1 + λ·d(q))    ← final reward   │    │
│     │  Update Q_t via GRPO                                     │    │
│     │  n_k ← γ·n_k + (1-γ)·𝟏[k visited]    ← EMA update     │    │
│     └──────────────────────────────────────────────────────────┘    │
│                                                                      │
│  3. Generate question pool 𝒫_t from Q_t                             │
│  4. Train S_t via GRPO on 𝒫_t with majority-vote rewards            │
└─────────────────────────────────────────────────────────────────────┘
```

**The reward formula:**

```
r(q) = zpd(p(q))  ×  (1 + λ · exp(-n_{c(q)} / n̄))
        └── quality ──┘  └────── diversity bonus ──────┘

zpd(p) = max(0, 1 - |p - 0.75| / 0.4),  p ∈ [0.3, 0.9]
λ = 5.0,  γ (EMA decay) = 0.99,  K = 128 clusters
```

---

## 📦 Project Structure

```
Prism/
├── rentropy_config.yaml              # Central config: centroids, embedding model, λ, γ
│
├── cluster_space/                    # Offline cluster construction (one-time)
│   ├── download_corpus.py            # Download MATH + GSM8K datasets
│   ├── build_clusters.py             # Embed → K-Means → save centroids
│   ├── cluster_from_embeddings.py    # Cluster from pre-computed embeddings
│   └── cluster_assigner.py           # Runtime cluster assignment + EMA tracking
│
├── question_generation_clustering/   # Question generation pipeline
│   ├── balanced_cluster_generation.py  # Generate questions balanced across clusters
│   ├── evaluator.py                    # vLLM-based majority-vote scoring
│   ├── create_cluster_frequency.py     # Compute cluster frequency arrays
│   ├── create_hf_dataset.py            # Upload evaluated questions to HuggingFace
│   └── create_hf_dataset_filtered.py   # Upload with score filtering (0.5–0.9)
│
└── train/
    ├── scripts/
    │   ├── main_rentropy.sh            # 🚀 Main entry point: full pipeline
    │   ├── questioner_train_rentropy.sh  # Questioner GRPO training
    │   └── solver_train.sh             # Solver GRPO training
    │
    ├── examples/
    │   ├── config.yaml                 # verl trainer config
    │   ├── reward_function/
    │   │   ├── caller_rentropy.py      # Prism diversity reward function
    │   │   └── math.py                 # Baseline math reward
    │   └── format_prompt/              # Jinja2 prompt templates
    │
    ├── question_generate/
    │   ├── question_generate.py        # Parallel question generation via vLLM
    │   ├── compute_diversity_scores.py # Add diversity scores to generated questions
    │   └── curate_balanced_dataset.py  # Pool + balance questions for solver training
    │
    ├── question_evaluate/
    │   ├── evaluate.py                 # Score questions via majority voting
    │   └── upload.py                   # Upload final dataset to HuggingFace
    │
    ├── vllm_service_init/
    │   └── start.sh                    # Launch vLLM solver servers
    │
    ├── evaluation/
    │   └── evaluate.bash               # Run all math benchmarks
    │
    └── benchmark_grpo/                 # GRPO fine-tuning on all benchmarks
        ├── prepare_data.py
        ├── config.yaml
        └── train.sh
```

---

## ⚙️ Installation

```bash
git clone https://github.com/PLACEHOLDER/prism.git
cd prism

pip install -r train/requirements.txt
```

**Key dependencies:** `torch==2.7`, `vllm==0.9.1`, `transformers==4.52.4`, `ray==2.46.0`, `verl` (included in `train/verl/`)

Set your credentials:
```bash
export HUGGINGFACENAME="your_hf_username"
export STORAGE_PATH="/path/to/storage"   # defaults to train/storage/
export HF_TOKEN="your_hf_token"
export WANDB_API_KEY="your_wandb_key"
```

---

## 🚀 Quick Start

### Step 1 — Build the Cluster Space (one-time, offline)

```bash
# Download the MATH corpus
python cluster_space/download_corpus.py

# Build K=128 semantic clusters using Qwen3-Embedding-0.6B
python cluster_space/build_clusters.py \
    --num_clusters 128 \
    --embedding_model Qwen/Qwen3-Embedding-0.6B \
    --output_dir cluster_space/cluster_data/
```

This produces `cluster_space/cluster_data/centroids.npy` — a fixed coordinate system over mathematical topics used throughout training.

### Step 2 — Run the Full Prism Training Pipeline

```bash
cd train

export HUGGINGFACENAME="your_hf_username"

bash scripts/main_rentropy.sh \
    Qwen/Qwen3-4B-Base \   # Questioner base model
    Qwen/Qwen3-4B-Base \   # Solver base model
    qwen3-4b               # Experiment name prefix
```

This orchestrates:
1. **Questioner training** — GRPO with Prism's coverage-aware reward (6 steps/iteration)
2. **Question generation** — parallel vLLM generation across GPUs
3. **Question evaluation** — majority-vote solvability scoring
4. **Dataset upload** — push filtered questions to HuggingFace
5. **Solver training** — GRPO on the curated question pool (20 steps/iteration)
6. **Evaluation** — benchmark suite after each solver update

### Step 3 — (Optional) Generate a Balanced Question Dataset

```bash
cd question_generation_clustering

python balanced_cluster_generation.py \
    --models your_hf_username/questioner-iter1 your_hf_username/questioner-iter2 \
    --centroids_dataset your_hf_username/math_clusters \
    --output_file balanced_questions.json \
    --max_per_cluster 100
```

---

## 🎛️ Configuration

All Prism-specific parameters are in `rentropy_config.yaml`:

```yaml
# Path to cluster centroids (built by cluster_space/build_clusters.py)
centroids_path: "cluster_space/cluster_data/centroids.npy"

# Embedding model for cluster assignment
embedding_model: "Qwen/Qwen3-Embedding-0.6B"

# Majority voting threshold — diversity reward applied only if p(q) >= this
majority_vote_threshold: 0.3

# Diversity weight λ
lambda_weight: 5.0

# EMA decay γ for cluster count tracking
ema_decay: 0.99

# Smoothing constant α (initial cluster count value)
smoothing_alpha: 1.0

# Optional: warm-start cluster counts from a previous iteration
# Set to null for uniform initialisation
init_cluster_counts_path: null
```

**Hyperparameter sensitivity** (from ablations):

| K (clusters) | MATH-500 | AIME 2024 |
|:---:|:---:|:---:|
| 64 | 63.60 | 14.78 |
| **128** | **64.40** | **16.67** |
| 256 | 63.80 | 15.42 |

| λ | MATH-500 | AIME 2024 | Norm. Entropy |
|:---:|:---:|:---:|:---:|
| 1.0 | 63.00 | 13.47 | 0.74 |
| 3.0 | 63.80 | 15.22 | 0.79 |
| **5.0** | **64.40** | **16.67** | **0.83** |
| 8.0 | 64.00 | 15.83 | 0.86 |

---

## 📊 Prism-Math Dataset

As a byproduct of training, we release **Prism-Math** — ~100K semantically diverse, difficulty-calibrated synthetic math questions generated by the Prism questioner.

| Property | Value |
|---|---|
| Total questions | ~100,000 |
| Source | Prism Questioner (iterations 1–4) |
| Base model | Qwen3-4B-Base |
| Semantic clusters covered | 125 / 128 (97.7%) |
| Median solvability p(q) | 0.72 |
| Solvability range | [0.30, 0.90] |
| Normalised entropy | 0.81 |
| Gini coefficient | 0.68 |
| Format | JSONL |
| Fields | `question`, `answer`, `cluster_id`, `solvability`, `iteration` |
| License | Apache 2.0 |

Unlike existing synthetic datasets, Prism-Math is explicitly optimised for **semantic breadth** and **edge-of-solvability difficulty** across all 128 semantic clusters.

📥 **[Download Prism-Math on HuggingFace](https://huggingface.co/datasets/PLACEHOLDER/prism-math)**

---

## 🧮 Computational Overhead

Prism's diversity mechanism is nearly free relative to GRPO training cost:

| Component | R-Zero | Prism |
|---|:---:|:---:|
| Questioner GRPO training | 38 min | 38 min |
| Embedding + cluster assignment | — | 2.4 min |
| Question generation & filtering | 25 min | 25 min |
| Solver GRPO training | 82 min | 82 min |
| **Total per iteration** | **145 min** | **147.4 min** |
| **Overhead** | — | **+1.7%** |

Hardware: 8× H100 80GB node.

---

## 📐 Method Details

### Cluster Space Construction (offline, once)

1. **Embed** — Every question in the MATH training set (~12.5K problems) is embedded with `Qwen3-Embedding-0.6B` → L₂-normalised vectors ∈ ℝ¹⁰²⁴
2. **Cluster** — K-Means with K=128 → centroids {μ₁, …, μ₁₂₈} saved as a static artifact
3. **Initialise** — Count vector **n** ∈ ℝᴷ set uniformly: nₖ = α ∀k

### Cluster Assignment (per-step, online)

```
c(q) = argmax_k  ⟨embed(q), μ_k⟩
```

### EMA Coverage Update (per-batch)

```
n_k ← γ · n_k + (1 - γ) · 𝟏[k visited in batch]
```

This provides **cross-iteration memory**: over-sampled clusters stay elevated across co-evolution rounds, unlike batch-local BLEU penalties that reset every step.

### Solver-Initialised Questioner

In R-Zero: `Q_t ← GRPO(Q_{t-1}, S_{t-1})`  
In Prism: `Q_t ← GRPO(S_{t-1}, vLLM(S_{t-1}))`

Initialising from the Solver at each iteration (1) eliminates capability lag, (2) removes generation bias, and (3) breaks the narrowing chain of `Q_{t-1} → Q_t`.

---

## 📋 Component Ablation

| Diversity Reward | Solver Init | MATH-500 | AIME 2024 |
|:---:|:---:|:---:|:---:|
| ✗ | ✗ | 62.40 ± 0.54 | 11.35 ± 1.43 |
| ✓ | ✗ | 63.20 ± 0.51 | 13.89 ± 1.52 |
| ✗ | ✓ | 63.60 ± 0.48 | 14.22 ± 1.47 |
| ✓ | ✓ | **64.40 ± 0.47** | **16.67 ± 1.38** |

Both components contribute independently; the combination is super-additive.

---

## 📖 Citation

If you use Prism or Prism-Math in your work, please cite:

```bibtex
@article{mishra2025prism,
  title   = {Preventing Curriculum Collapse in Self-Evolving Reasoning Systems},
  author  = {Mishra, Vaibhav},
  journal = {arXiv preprint},
  year    = {2025},
  url     = {https://github.com/PLACEHOLDER/prism}
}
```

---

## 🙏 Acknowledgements

Prism builds on and extends [R-Zero](https://github.com/Chengsong-Huang/R-Zero) and uses the [verl](https://github.com/volcengine/verl) framework for GRPO training. We thank the authors of R-Zero, R-Few, Absolute Zero, and SPICE for their work and open-source contributions.

---

<div align="center">
<sub>Apache 2.0 License · Questions? Open an issue or reach out at <a href="mailto:vaibhavm209625@gmail.com">vaibhavm209625@gmail.com</a></sub>
</div>
