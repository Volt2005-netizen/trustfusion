# TrustFusion
A Multi-Modal AI Middleware for Initial-Access Protection & Identity Verification on LinkedIn

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch--Geometric-2.4+-34a853.svg)](https://pyg.org/)
[![CUDA 12.3](https://img.shields.io/badge/CUDA-12.3-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Unit Tests](https://img.shields.io/badge/Unit%20Tests-28%2F28%20Passing-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Overview

**TrustFusion** is a real-time, multi-modal defensive framework engineered to protect enterprises and job seekers on professional platforms like LinkedIn from sophisticated social engineering campaigns. By dynamically fusing visual, linguistic, temporal, and relational interaction signals, TrustFusion detects and preempts two primary threat vectors:

1. **Operation Dream Job (Initial-Access Threats):** State-sponsored campaigns where malicious actors impersonate recruiters to deliver trojanized malware via fake assessment portals.
2. **Brand Impersonation & Recruitment Fraud:** Scammers cloning corporate logos, credentials, and executive profiles to execute advance-fee recruitment scams.

Unlike legacy single-modality filters, TrustFusion uses a **3-Gate Security Pipeline** backed by **Cross-Modal Attention Gating**, resolving the academic **"Cold-Start" feature sparsity problem** for newly registered profiles without photos, connections, or chat histories.

---

## 🏗️ System Architecture

```text
                             [ TRUSTFUSION PIPELINE ]

                              ┌─────────────────────────────┐
                              │ Raw LinkedIn & Chat Streams │
                              └──────────────┬──────────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
        [ Anti-Leakage Preprocessor ]                [ Section Tag Embeddings (STE) ]
         *MinMax Scaled on Train Only*                *[INTRO], [ABOUT], [EXPERIENCE]*
                       │                                           │
    ┌──────────────────┼───────────────────────────────────────────┼──────────────────┐
    │                  ▼                                           ▼                  ▼
    │          [ GATE 1: Visual ]                         [ GATE 2: Linguistic ]      │
    │    (64-bit DCT pHash + Forensics)               (RoBERTa STE + PCA-150 + IAT)   │
    │         *FFHQ vs. StyleGAN*                          *Shannon Entropy*          │
    │      *Cold-Start Modality Mask*                   *Off-Platforming Cues*        │
    │                                                                                 │
    │                                                     [ GATE 3: Relational ]      │
    │                                                    (PyG RGCN HIN + Louvain)     │
    │                                                     *16,970 Nodes / 45,536 Edges│
    └──────────────────┬───────────────────────────────────────────┬──────────────────┘
                       │                                           │
                       └─────────────────────┬─────────────────────┘
                                             ▼
                               [ Cross-Modal Attention Gating ]
                             *Softmax Weighting with Modality Masks*
                                             ▼
                               [ Decision Ensemble Classifiers ]
                                    (CatBoost / XGBoost)
                                             ▼
                               [ Unified Fraud Risk Score ]
                                     ([0.0000, 1.0000])
                                             ▼
                               [ SHAP & GNNExplainer Console ]
```

---

## 🛡️ The 3-Gate Security Pipeline

### 🖼️ Gate 1: Visual Identity & Cold-Start Verification ([`pipeline/gate1_visual.py`](pipeline/gate1_visual.py))
* **Technique:** 64-bit Hexadecimal DCT Perceptual Hashing (pHash) + Fourier High-Frequency Energy ($E_{\text{HF}}$) Artifact Analysis + Cold-Start Modality Masking.
* **Function:** 
  - Computes 64-bit perceptual image hashes to detect **Sybil photo reuse collisions** (Hamming distance $\le 5$).
  - Distinguishes genuine human headshots (**FFHQ**) from generative deepfakes (**StyleGAN**) and brand logo impersonation (**Logo-2K+**).
  - Implements **Cold-Start Modality Gating**: when photos are absent (`Photo == 'No'`), emits a neutral prior embedding ($\mathbf{0} \in \mathbb{R}^{64}$) and sets $M_{\text{visual}} = 0.0$ for downstream attention gating.
* **Performance on Unseen Test Photos:** **81.78% Accuracy**, **85.19% F1-Score**, **0.8283 ROC-AUC**, **1,427 Sybil photo reuse collisions detected**.

### 💬 Gate 2: Linguistic & Conversational Forensics ([`pipeline/gate2_linguistic.py`](pipeline/gate2_linguistic.py))
* **Technique:** Section Tag Embeddings (STE) + `roberta-base` Transformer + Anti-Leakage PCA-150 + Deception Heuristics + Shannon Temporal Entropy.
* **Function:** 
  - Wraps profile sections with structural markers (`[INTRO]`, `[ABOUT]`, `[EXPERIENCE]`, `[SKILLS]`) allowing RoBERTa's self-attention to evaluate cross-section semantic consistency.
  - Applies PCA-150 fitted strictly on training embeddings (**96.99% explained variance**).
  - Scans for spear-phishing syntax, advance-fee triggers, and evasion patterns (WhatsApp, Telegram, crypto).
  - Evaluates message Inter-Arrival Time (IAT) Shannon entropy ($H(T) = -\sum p_i \log_2 p_i$) to separate robotic outreach bots from human typing.
* **Performance on Unseen Test Profiles:** **98.34% Accuracy**, **98.36% Macro F1**, **0.9980 ROC-AUC**.

### 🕸️ Gate 3: Relational Workspace Graph ([`pipeline/gate3_relational.py`](pipeline/gate3_relational.py))
* **Technique:** Heterogeneous Information Network (HIN) in PyTorch Geometric (PyG) + 2-Layer Relational GCN (`RGCNConv`) + Louvain Community Clustering.
* **Function:** 
  - Maps profiles into a graph with **16,970 nodes** (3,619 Users, 8,687 Companies, 2,043 Institutes, 2,621 Locations) and **45,536 directed edges** across 6 canonical relation types.
  - Partitions user co-occurrences into **1,181 communities** with a high modularity of **0.5166**, isolating tightly coupled Sybil scam rings.
  - Employs **Anti-Leakage Inductive Masking**: loss is evaluated strictly on `train_mask`, ensuring zero supervision leaks into test nodes.
* **Performance on Unseen Test Profiles:** **94.66% Accuracy**, **94.79% Macro F1**, **0.9914 ROC-AUC**.

---

## 📊 Experimental Results & Benchmark Statistics

Evaluated on a benchmark of **3,619 profiles** (2,533 Train / 1,086 Strictly Unseen Test) across 4 threat classes:
- **Class 0:** Legitimate Profiles (1,800)
- **Class 1:** Manual Fakes / Parodies (599)
- **Class 2:** ChatGPT-Generated Fakes (1,200)
- **Class 3:** GPT-4 Adversarial Profiles (20)

### Comprehensive Gate Performance Matrix (Unseen Test Set)

| Security Gate | Pipeline Script | Primary Modality | Accuracy | Macro F1 | ROC-AUC | Key Forensic Capability |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Step 1: Ingestion & Anti-Leakage** | [`preprocess.py`](pipeline/preprocess.py) | Tabular / Text | *N/A* | *N/A* | *N/A* | Zero ID/Text Leakage, 100% Intro Recovery, MinMax on Train Only |
| **Gate 1: Visual Identity** | [`gate1_visual.py`](pipeline/gate1_visual.py) | Image / Vision | **81.78%** | **85.19%** | **0.8283** | 64-bit DCT pHash, StyleGAN/FFHQ, 1,427 Sybil Photo Collisions |
| **Gate 2: Linguistic Forensics** | [`gate2_linguistic.py`](pipeline/gate2_linguistic.py) | Natural Language | **98.34%** | **98.36%** | **0.9980** | RoBERTa STE, PCA-150 (96.99% var), Deception Scan, IAT Entropy |
| **Gate 3: Relational Graph** | [`gate3_relational.py`](pipeline/gate3_relational.py) | Graph Network | **94.66%** | **94.79%** | **0.9914** | 16,970 Nodes, 45,536 Edges, 1,181 Louvain Clusters (0.5166 Mod) |

---

## 🛠️ Project Structure

```text
trustfusion/
├── data/
│   ├── raw/                              # Source archives (FFHQ, StyleGAN, Logo-2K+, etc.)
│   └── processed/                        # Processed artifacts (gitignored)
│       ├── numerical/                    # Scaled numerical arrays & MinMax scalers
│       ├── text/                         # Parsed section-level JSON records
│       ├── visual/                       # Gate 1 visual embeddings, masks & pHash index
│       ├── linguistic/                   # Gate 2 RoBERTa PCA-150 representations & risk scores
│       └── relational/                   # Gate 3 RGCN models, embeddings & Louvain features
│
├── pipeline/
│   ├── preprocess.py                     # Step 1: Ingestion, schema validation & anti-leakage scaling
│   ├── curate_visual_dataset.py          # Visual asset curation & anti-leakage train/test isolation
│   ├── gate1_visual.py                   # Step 4: Visual Identity, 64-bit pHash & cold-start gating
│   ├── gate2_linguistic.py               # Step 2: RoBERTa STE, PCA-150 & conversational entropy
│   ├── gate3_relational.py               # Step 3: PyG RGCN heterogeneous graph & Louvain clustering
│   └── fusion_classifier.py              # Step 5: Cross-Modal Attention Gating & Decision Ensemble
│
├── xai/
│   ├── shap_explainer.py                 # Multi-modal SHAP feature attribution
│   └── gnn_explainer.py                  # GNN subgraph path isolation
│
├── backend/
│   └── main.py                           # FastAPI inference server & moderation API
│
└── tests/
    ├── test_preprocessing.py             # Schema & anti-leakage invariants (13 tests)
    ├── test_gate1.py                     # pHash, cold-start & visual dimensions (6 tests)
    ├── test_gate2.py                     # STE, PCA-150 & entropy calculations (4 tests)
    └── test_gate3.py                     # Graph builder, RGCN & inductive masking (5 tests)
```

---

## 🚀 Quickstart & Examiner Demonstration

### 1. Instant Demonstration Mode (`--demo`)
Demonstrate any pipeline stage live in front of examiners in **< 2 to 8 seconds** using precomputed artifacts:

```powershell
# Step 1: Preprocessing & Anti-Leakage Report (< 2s)
python pipeline/preprocess.py --demo

# Step 2: Gate 2 Linguistic & Temporal Forensics Report (< 8s)
python pipeline/gate2_linguistic.py --demo

# Step 3: Gate 3 Relational Workspace Mapping Report (< 7s)
python pipeline/gate3_relational.py --demo

# Step 4: Gate 1 Visual Identity & Cold-Start Report (< 2s)
python pipeline/gate1_visual.py --demo
```

### 2. Run Automated Unit Tests (28/28 Passing)
Verify the entire mathematical and scientific integrity of all four pipeline gates in **0.3s**:

```powershell
python -m unittest discover tests -v
```

---

## 📜 License
Distributed under the MIT License. See [LICENSE](LICENSE) for details.
