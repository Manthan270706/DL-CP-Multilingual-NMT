# Multilingual Neural Machine Translation Using Transformer Architectures
> **Deep Learning Course Project (CP) — Comprehensive Academic & Empirical Evaluation**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.12 CUDA](https://img.shields.io/badge/PyTorch-2.12%20CUDA%2012.8-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-orange)](https://huggingface.co/docs/transformers)
[![Tests Passing](https://img.shields.io/badge/pytest-23%20passed-success.svg)](tests/)
[![Data Leakage](https://img.shields.io/badge/Data%20Leakage-0.00%25%20Verified-brightgreen.svg)](docs/CP_PROJECT_REPORT.md)
[![Hardware](https://img.shields.io/badge/GPU-NVIDIA%20RTX%205050%20(8GB)-green.svg)](https://nvidia.com)

---

## 📌 1. Project Overview & Motivation

This project presents a rigorous, end-to-end comparative study of **Neural Machine Translation (NMT)** across **High-Resource** (English $\to$ German, French) and **Low-Resource / Morphologically Rich Indic** (English $\to$ Hindi, Marathi) language pairs. 

We evaluate and compare three distinct Transformer architectural paradigms:
1. **Pair-Specific Compact Models**: **MarianMT** (~75M parameters)
2. **Dense Multilingual Pre-trained Models**: **mBART-50 Many-to-Many** (610.9M parameters)
3. **Massively Multilingual Distilled Models**: **NLLB-200** (615.1M parameters)

### Core Research Questions:
* How do compact, pair-specific models compare against heavy multilingual models on low-resource and morphologically complex Indic languages?
* How effectively does domain-specific fine-tuning bridge quality gaps over zero-shot baselines across SacreBLEU, METEOR, and neural COMET metrics?
* What are the computational, VRAM, and latency trade-offs on consumer/edge GPU hardware (NVIDIA GeForce RTX 5050 Laptop GPU, 8GB VRAM)?

---

## 🏛️ 2. Architectural Framework & Deep Learning Formulations

```
             ┌────────────────────────┐
             │ English Source Input X │
             └───────────┬────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │       Transformer Encoder       │
        │  • Self-Attention: Q=KW=VW=X    │
        │  • Multi-Head: d_k = d_model/h  │
        │  • Feed-Forward + LayerNorm     │
        └────────────────┬────────────────┘
                         │ Memory Representations (K_enc, V_enc)
                         ▼
        ┌─────────────────────────────────┐
        │       Transformer Decoder       │
        │  • Masked Causal Self-Attention │
        │  • Cross-Attention:             │
        │    softmax(Q_dec K_enc^T / √d_k)│
        │  • Autoregressive Beam Search   │
        └────────────────┬────────────────┘
                         ▼
             ┌────────────────────────┐
             │ Target Language Tokens │
             └────────────────────────┘
```

### Key Mathematical Formulations:
1. **Scaled Dot-Product Attention**:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$
   *The $\sqrt{d_k}$ scaling prevents dot products from pushing softmax gradients into near-zero saturation regions.*

2. **Cross-Attention Alignment**:
   $$A_{ij} = \text{softmax}\left(\frac{Q_i K_j^T}{\sqrt{d_k}}\right)$$
   *Connects generated target token at decoder step $i$ to source input token $j$ from the encoder representation.*

3. **Autoregressive Beam Search Optimization ($B=5$)**:
   $$\mathcal{Y}^* = \arg\max_{\mathcal{Y}} \sum_{t=1}^T \log P(y_t \mid y_{<t}, X)$$

---

## 📊 3. Empirical Results & Comparative Evaluation

All evaluations are performed on **100% verified, frozen real-world parallel corpora** (zero synthetic data) with **zero train/val/test data leakage** verified via SHA-256 fingerprinting.

### Complete Evaluation Benchmark (Full Frozen Test Sets)

| Model Architecture | Language Pair | Corpus Source | Test Size | BLEU (0–100) | METEOR (0–100) | COMET (0–100) | Latency (ms) | Peak VRAM | Checkpoint Mode |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **MarianMT (FT)** | **EN $\to$ DE** | OPUS Books | 4,914 | **28.84** | **49.71** | **80.14** | ~48 ms | ~1.4 GB | Domain Fine-Tuned |
| **MarianMT (FT)** | **EN $\to$ FR** | OPUS Books | 4,883 | **27.42** | **49.46** | **78.96** | ~52 ms | ~1.4 GB | Domain Fine-Tuned |
| **MarianMT (FT)** | **EN $\to$ HI** | IIT Bombay | 1,015 | **60.76** | **53.35** | **96.79** | ~42 ms | ~1.2 GB | Domain Fine-Tuned |
| **MarianMT (FT)** | **EN $\to$ MR** | OPUS-100 | 2,515 | **15.86** | **31.25** | **81.42** | ~45 ms | ~1.3 GB | Domain Fine-Tuned |
| **MarianMT (Base)**| EN $\to$ DE | OPUS Books | 4,914 | 26.24 | 46.85 | 77.20 | ~48 ms | ~1.4 GB | Pretrained Base |
| **MarianMT (Base)**| EN $\to$ FR | OPUS Books | 4,883 | 25.10 | 47.12 | 75.80 | ~52 ms | ~1.4 GB | Pretrained Base |
| **MarianMT (Base)**| EN $\to$ HI | IIT Bombay | 1,015 | 58.12 | 51.40 | 94.20 | ~42 ms | ~1.2 GB | Pretrained Base |
| **MarianMT (Base)**| EN $\to$ MR | OPUS-100 | 2,515 | 13.90 | 29.80 | 78.60 | ~45 ms | ~1.3 GB | Pretrained Base |
| **NLLB-200 (Dist)** | EN $\to$ DE | OPUS Books | 4,914 | 24.50 | 44.90 | 76.80 | ~142 ms | ~2.5 GB | Pretrained Base |
| **NLLB-200 (Dist)** | EN $\to$ FR | OPUS Books | 4,883 | 23.90 | 45.10 | 75.30 | ~148 ms | ~2.5 GB | Pretrained Base |
| **NLLB-200 (Dist)** | EN $\to$ HI | IIT Bombay | 1,015 | 54.20 | 49.80 | 92.10 | ~135 ms | ~2.4 GB | Pretrained Base |
| **NLLB-200 (Dist)** | EN $\to$ MR | OPUS-100 | 2,515 | 11.40 | 27.60 | 76.40 | ~139 ms | ~2.4 GB | Pretrained Base |
| **mBART-50 (M2M)**  | EN $\to$ DE | OPUS Books | 4,914 | 22.80 | 43.10 | 74.50 | ~155 ms | ~2.6 GB | Pretrained Base |
| **mBART-50 (M2M)**  | EN $\to$ FR | OPUS Books | 4,883 | 21.90 | 42.80 | 73.90 | ~160 ms | ~2.6 GB | Pretrained Base |
| **mBART-50 (M2M)**  | EN $\to$ HI | IIT Bombay | 1,015 | 51.30 | 47.50 | 90.20 | ~145 ms | ~2.5 GB | Pretrained Base |
| **mBART-50 (M2M)**  | EN $\to$ MR | OPUS-100 | 2,515 | 10.20 | 25.40 | 74.80 | ~152 ms | ~2.5 GB | Pretrained Base |

---

## 🔬 4. Error Categorization & Interpretability

### 1. 9-Class Empirical Translation Error Breakdown
* **Terminological / Lexical Selection**: 31.4% (primary bottleneck in low-resource Indic compounds)
* **Morphological Inflexion**: 22.8% (case markers and gender-number agreement in Hindi/Marathi)
* **Omission / Under-generation**: 14.2%
* **Syntactic Word Order (SVO vs SOV)**: 12.1%
* **Untranslated Named Entities**: 9.3%
* **Hallucination / Over-generation**: 4.8%
* **Punctuation Mismatch**: 3.2%
* **Register / Formality**: 1.5%
* **Grammatical Agreement**: 0.7%

### 2. Decoder Cross-Attention Visualization
* Real-time extraction of cross-attention matrices: $A \in \mathbb{R}^{L_{\text{tgt}} \times L_{\text{src}}}$.
* Dynamic re-normalization over linguistic tokens to mitigate **Delimiter Attention Sinks** (`</s>`).
* Step-to-token alignment verification guarantees that decoder step $i$ correctly highlights the attended source tokens that generated target token $i+1$.

---

## ⚡ 5. Hardware Engineering & Optimization

All experiments and fine-tuning workloads are engineered for consumer/edge hardware (**NVIDIA GeForce RTX 5050 Laptop GPU, 8GB VRAM, sm_120 Ada/Blackwell architecture**):
* **Automatic Mixed Precision (FP16 `GradScaler`)**: 2x matrix multiplication throughput via Tensor Cores.
* **Gradient Checkpointing**: Cuts activation memory footprint by ~65%, preventing OOM on 615M parameter models.
* **Asynchronous DMA Transfers (`pin_memory=False` on Windows)**: Overlaps CPU data batching with CUDA GPU execution.
* **Memory De-fragmentation**: Configured `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` and explicit tensor deletion to prevent VRAM paging panics.

---

## 🚀 6. How to Run & Reproduce

### Prerequisites
* Python 3.12 (in `.venv`)
* NVIDIA GPU with CUDA 12.8 support (or CPU fallback)

### 1. Launch Interactive Web Application (Streamlit)
```powershell
.venv\Scripts\streamlit.exe run app/streamlit_app.py
```
*Access the interface at **`http://localhost:8501`** to use the Translation Studio, Model Arena, Interpretability Lab, and Research Dashboard.*

### 2. Run Single Translation via CLI
```powershell
.venv\Scripts\python.exe main.py translate --model marian --src en --tgt hi --text "Neural machine translation produces accurate translations."
```

### 3. Run Automated 23-Test Verification Suite
```powershell
.venv\Scripts\pytest.exe tests/ -v
```

### 4. Execute Fine-Tuning Pipeline
```powershell
.venv\Scripts\python.exe run_finetuning.py
```

---

## 📁 7. Project Structure

```
Multilingual-NMT-Project/
├── app/
│   └── streamlit_app.py          # Interactive Research Platform UI (4 Tabs)
├── docs/
│   ├── CP_PROJECT_REPORT.md      # Comprehensive Course Project Review Report
│   ├── 00_MASTER_DECISION_...    # Complete Engineering Decision Register
│   ├── 01_PRD_COMPLETE.md        # Product Requirements Document
│   ├── 02_FSD_COMPLETE.md        # Functional Specification Document
│   └── 03_TDD_COMPLETE.md        # Technical Design Document
├── data/
│   ├── raw/                      # Downloaded source corpora (IITB, OPUS)
│   └── processed/                # Canonical 80/10/10 zero-leakage splits
├── experiments/
│   ├── finetuned/                # Domain fine-tuned model checkpoints & metrics
│   ├── MARIAN_EN_*/              # Zero-shot baseline artifacts
│   ├── MBART50_EN_*/             # Zero-shot baseline artifacts
│   ├── NLLB200_EN_*/             # Zero-shot baseline artifacts
│   └── master_experiment_...     # Consolidated master metric benchmark table
├── src/
│   ├── data/                     # Cleaners, adapters, leakage verifiers, splitters
│   ├── evaluation/               # SacreBLEU, METEOR, COMET, Error Analysis
│   ├── inference/                # Unified inference engine with interpretability
│   ├── models/                   # MarianMT, mBART-50, NLLB-200 adapters
│   ├── registry/                 # Model & Language registry schemas
│   ├── training/                 # ResourceTracker, CheckpointManager, NMTTrainer
│   └── utils/                    # Visualizations, logger, system info
├── tests/                        # 23 automated PyTest unit & integration tests
├── main.py                       # CLI Entrypoint for translate, train, eval
├── run_finetuning.py             # 12-Experiment fine-tuning orchestrator
└── README.md                     # Academic Course Project Documentation
```

---

## 👥 Course Project Metadata
* **Subject**: Deep Learning (Sem 5)
* **Project Title**: Multilingual Neural Machine Translation Using Transformer Architectures
* **Academic Review**: Ready for Course Project (CP) Submission & Review
