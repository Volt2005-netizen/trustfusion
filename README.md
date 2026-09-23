TrustFusion
A Multi-Modal AI Middleware for Initial-Access Protection& Identity Verification on LinkedIn

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch--Geometric-2.4+-34a853.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

📌 Overview

TrustFusion is a real-time, multi-modal defensive framework engineered to protect enterprises and job seekers on professional platforms like LinkedIn from sophisticated social engineering campaigns. By dynamically fusing visual, linguistic, temporal and relational interaction signals, TrustFusion detects and preempts two primary threat vectors:

1.  Operation Dream Job (Initial-Access Threats):  State-sponsored campaigns where malicious actors impersonate recruiters to deliver trojanized, memory-resident malware via fake assessment portals.
2.  Brand Impersonation & Recruitment Fraud:  Scammers cloning corporate logos, credentials and executive profiles to execute advance-fee recruitment scams.

Unlike legacy single-modality filters, TrustFusion uses a  3-Gate Security Pipeline  backed by  Cross-Modal Attention Gating , resolving the academic  "Cold-Start" feature sparsity problem  for newly registered profiles.

---

  🏗️ System Architecture

                              [ TRUSTFUSION PIPELINE ]

                               ┌─────────────────────────────┐
                               │ Raw LinkedIn  & Chat Streams │
                               └──────────────┬──────────────┘
                                              │
                        ┌─────────────────────┴─────────────────────┐
                        ▼                                           ▼
         [ Contrastive Preprocessor ]                [ Multilingual Tokenizer ]
          *Solves Gap 1 (AI-Polished)*               *Solves Gap 4 (Hinglish)*
                        │                                           │
     ┌──────────────────┼───────────────────────────────────────────┼──────────────────┐
     │                  ▼                                           ▼                  ▼
     │          [ GATE 1: Visual ]                         [ GATE 2: Linguistic ]      │
     │      (pHash + Milvus Cosine)                    (RoBERTa + Shannon Entropy)     │
     │                                                      *Solves Gap 2*             │
     │                                                 (Boundary Transitions)          │
     │                                                                                 │
     │                                                     [ GATE 3: Relational ]      │
     │                                                    (RGCN H-GNN + Louvain)       │
     └──────────────────┬───────────────────────────────────────────┬──────────────────┘
                        │                                           │
                        └─────────────────────┬─────────────────────┘
                                              ▼
                                [ Hierarchical Tripwire ] 
                               *Solves Gap 3 (Scalability)*
                                              ▼
                                [ Attention Gating Fusion ]
                             *Solves Cold-Start Feature Gaps*
                                              ▼
                                [ Ensemble CatBoost/XGBoost ]
                                              ▼
                                  [ Unified Risk Score ]
                                              ▼
                                [ SHAP  & GNNExplainer Console ]

---

  🛡️ The 3-Gate Security Pipeline

#  🖼️ Gate 1: Visual Identity Verification
*  Technique:  Perceptual Image Hashing (64-bit structural pHash) + Milvus Vector Database.
*  Function:  Normalizes images to \\(32 \times 32\\) grayscale grids and performs sub-150ms cosine similarity searches against verified corporate logos and deepfake templates (StyleGAN/FFHQ) to halt asset spoofing.

#  💬 Gate 2: Linguistic  & Temporal Forensics
*  Technique:  Fine-tuned RoBERTa-base / mBERT + Shannon Temporal Entropy.
*  Function:  Scans chat streams for spear-phishing syntax and off-platforming cues (e.g., *"chat on WhatsApp"*). Calculates Inter-Arrival Time (IAT) entropy across message timestamps to distinguish natural human typing from robotic outreach bots.

#  🕸️ Gate 3: Relational Workspace Graph
*  Technique:  Heterogeneous Information Network (HIN) in PyTorch Geometric (PyG) + RGCN + Louvain Clustering.
*  Function:  Maps relationships across User, Device ID, IP Subnet and Payment nodes. Uses Relational Graph Convolutions and community detection to expose coordinated scam compounds sharing local hardware footprints.

---

  🚀 Key Research Gap Solutions

| Research Gap | Technical Solution in TrustFusion |
| :--- | :--- |
|  Gap 1: AI-Polished Resume Overlap  |  Contrastive Preprocessor:  Uses Section Tag Embeddings (STE) and fact-triangulation against verified databases to avoid misclassifying legitimate candidates using ChatGPT. |
|  Gap 2: Off-Platforming Deception  |  Boundary Transition Indicators:  Scans private chat logs for platform evasion patterns and lookalike domain spoofing (e.g., `amazon-joining-portal.xyz`). |
|  Gap 3: Scalability  & Latency  |  Hierarchical Tripwire Layer:  Keeps heavy GNN/NLP models dormant on CPU, waking GPU gates only when low-power lexical/metadata anomalies are tripped. |
|  Gap 4: Code-Mixed Phishing  |  Multilingual Tokenizer:  Leverages mBERT and IndicBERT with Byte-Pair Encoding (BPE) to parse hybrid regional scripts (e.g., Hinglish). |

---

  📊 Dataset  & Benchmarks

Trained and benchmarked on a combined adversarial dataset of  4,200 profiles  (Ayoobi et al., ACM HT '23; Gulati et al., ASONAM '25):

*  Dataset Split:  1,800 Legitimate, 600 Manual Fakes, 1,200 GPT-3.5 Fakes and 600 GPT-4-Turbo Adversarial Profiles.
*  Classifier Ensemble:  XGBoost + CatBoost with Softmax Calibration.
*  Performance:  Achieves a  98.2% F1-Score  on multi-class profile fraud classification.

---
## 🛠️ Project Structure & Module Ownership

The project is organized into separate modules for data processing, multimodal analysis, explainable AI, backend services, and dashboard visualization.

```text
trustfusion/
│
├── data/
│   ├── raw/
│   │   └── Ayoobi & Gulati datasets
│   │       └── 4,200 profiles
│   │
│   └── processed/
│       └── Scaled arrays, PCA matrices & pHash indices
│
├── pipeline/
│   ├── preprocess.py
│   │   └── Min-Max Scaling & Section Tag Embeddings (STE)
│   │
│   ├── gate1_visual.py
│   │   └── pHash & Milvus Vector Search
│   │       └── Owner: Aadesh Lande
│   │
│   ├── gate2_linguistic.py
│   │   └── RoBERTa + IAT / Shannon Entropy
│   │       └── Owner: Prathmesh Mulje
│   │
│   ├── gate3_relational.py
│   │   └── PyG RGCN & Louvain Graph Clustering
│   │       └── Owner: Srushti Patil
│   │
│   └── fusion_classifier.py
│       └── Cross-Modal Attention Gating + Ensemble Models
│           └── Owner: Vedang Sharnarth
│
├── xai/
│   ├── shap_explainer.py
│   │   └── Text Highlighting
│   │
│   └── gnn_explainer.py
│       └── Graph Path Isolation
│
├── backend/
│   └── main.py
│       └── FastAPI Server & API Endpoints
│
└── dashboard/
    └── React.js + D3.js
        └── Moderation Console
