# Comprehensive Course Project (CP) Report
## Multilingual Neural Machine Translation Using Transformer Architectures

---

### **Executive Summary**
This report documents the design, implementation, mathematical foundation, and empirical evaluation of an end-to-end **Multilingual Neural Machine Translation (NMT)** research platform. We benchmark three prominent Seq2Seq Transformer architectures across both high-resource European languages (**German, French**) and low-resource, morphologically complex Indic languages (**Hindi, Marathi**). The system is built with research integrity, featuring **0.00% train/val/test data leakage**, full-test evaluation, error categorization, cross-attention interpretability, and memory-efficient training on consumer GPU hardware.

---

## 1. Introduction, Objectives & Problem Statement

### 1.1 Objectives
1. **Cross-Architecture Benchmarking**: Compare pair-specific compact models (**MarianMT**, ~75M params) against dense multilingual models (**mBART-50**, 610.9M params; **NLLB-200**, 615.1M params).
2. **Linguistic Diversity**: Investigate translation behavior across diverse typological structures:
   - **SVO** (Subject-Verb-Object) in English, French, and German.
   - **SOV** (Subject-Object-Verb) with complex agglutination and postpositions in Hindi and Marathi (Devanagari script).
3. **Domain Fine-Tuning Impact**: Quantify gains from full domain adaptation over raw zero-shot pretrained checkpoints using SacreBLEU, METEOR, and neural COMET scores.
4. **Interpretability**: Demystify Transformer translation mechanics using decoder cross-attention alignment maps and token-level softmax entropy.
5. **Hardware Feasibility**: Demonstrate training and real-time inference on a laptop GPU (**NVIDIA GeForce RTX 5050, 8GB VRAM**).

---

## 2. Theoretical Framework & Mathematical Formulations

### 2.1 Transformer Encoder-Decoder Architecture

#### 1. Self-Attention Mechanism
Given input representations $X \in \mathbb{R}^{n \times d_{\text{model}}}$, queries ($Q$), keys ($K$), and values ($V$) are linearly projected:
$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V \quad (W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_k})$$

The scaled dot-product attention computes compatibility between queries and keys:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$
*The scaling factor $1/\sqrt{d_k}$ stabilizes variance when $d_k$ is large, preventing the softmax function from saturating and producing vanishing gradients.*

#### 2. Multi-Head Attention
Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions:
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$
$$\text{where} \quad \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

#### 3. Decoder Cross-Attention (Encoder-Decoder Attention)
In the decoder, queries originate from the previous decoder layer ($Q_{\text{dec}}$), while keys and values are the output memory of the encoder ($K_{\text{enc}}, V_{\text{enc}}$):
$$A_{ij} = \text{softmax}\left(\frac{Q_i K_j^T}{\sqrt{d_k}}\right)$$
*This matrix $A \in \mathbb{R}^{L_{\text{tgt}} \times L_{\text{src}}}$ forms the alignment matrix connecting target token $i$ to source token $j$.*

#### 4. Loss Function (Label-Smoothed Cross-Entropy)
During fine-tuning, the model minimizes categorical cross-entropy over target sequence tokens:
$$\mathcal{L}(\theta) = -\sum_{t=1}^T \log P(y_t \mid y_{<t}, X; \theta)$$

---

## 3. Data Pipeline & Research Integrity

All datasets were loaded from original verified sources, normalized, cleaned, and partitioned into deterministic **80% Train / 10% Validation / 10% Test** splits using SHA-256 sentence hashing (Seed 42).

### 3.1 Corpus Provenance & Statistics

| Direction | Language Pair | Corpus Provider | Usable Parallel Pairs | Train Rows | Val Rows | Test Rows | Pair Leakage | Source Leakage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EN $\to$ DE** | English $\to$ German | OPUS Books | 49,079 | 39,261 | 4,904 | **4,914** | **0.00%** | **0.00%** |
| **EN $\to$ FR** | English $\to$ French | OPUS Books | 48,835 | 39,078 | 4,874 | **4,883** | **0.00%** | **0.00%** |
| **EN $\to$ HI** | English $\to$ Hindi | IIT Bombay | 10,153 | 8,142 | 996 | **1,015** | **0.00%** | **0.00%** |
| **EN $\to$ MR** | English $\to$ Marathi | OPUS-100 | 25,101 | 20,045 | 2,541 | **2,515** | **0.00%** | **0.00%** |

*Leakage Verification: Verified via strict exact-match and hash-set disjointness assertions in `src/data/leakage.py`.*

---

## 4. Empirical Evaluation & Quantitative Results

### 4.1 Master Benchmark Comparison Table

All scores are evaluated on the **complete frozen test partitions**:

| Model Family | Direction | Model ID | Parameters | BLEU $\uparrow$ | METEOR $\uparrow$ | COMET $\uparrow$ | Mean Latency | Peak VRAM |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MarianMT (FT)** | **EN $\to$ DE** | `Helsinki-NLP/opus-mt-en-de` | 74.4M | **28.84** | **49.71** | **80.14** | **48.2 ms** | **1,420 MB** |
| **MarianMT (FT)** | **EN $\to$ FR** | `Helsinki-NLP/opus-mt-en-fr` | 74.4M | **27.42** | **49.46** | **78.96** | **51.8 ms** | **1,418 MB** |
| **MarianMT (FT)** | **EN $\to$ HI** | `Helsinki-NLP/opus-mt-en-hi` | 76.2M | **60.76** | **53.35** | **96.79** | **42.1 ms** | **1,210 MB** |
| **MarianMT (FT)** | **EN $\to$ MR** | `Helsinki-NLP/opus-mt-en-mr` | 76.2M | **15.86** | **31.25** | **81.42** | **45.3 ms** | **1,280 MB** |
| MarianMT (Base) | EN $\to$ DE | `Helsinki-NLP/opus-mt-en-de` | 74.4M | 26.24 | 46.85 | 77.20 | 48.2 ms | 1,420 MB |
| MarianMT (Base) | EN $\to$ FR | `Helsinki-NLP/opus-mt-en-fr` | 74.4M | 25.10 | 47.12 | 75.80 | 51.8 ms | 1,418 MB |
| MarianMT (Base) | EN $\to$ HI | `Helsinki-NLP/opus-mt-en-hi` | 76.2M | 58.12 | 51.40 | 94.20 | 42.1 ms | 1,210 MB |
| MarianMT (Base) | EN $\to$ MR | `Helsinki-NLP/opus-mt-en-mr` | 76.2M | 13.90 | 29.80 | 78.60 | 45.3 ms | 1,280 MB |
| NLLB-200 (Dist) | EN $\to$ DE | `facebook/nllb-200-distilled-600M` | 615.1M | 24.50 | 44.90 | 76.80 | 142.0 ms | 2,490 MB |
| NLLB-200 (Dist) | EN $\to$ FR | `facebook/nllb-200-distilled-600M` | 615.1M | 23.90 | 45.10 | 75.30 | 148.2 ms | 2,490 MB |
| NLLB-200 (Dist) | EN $\to$ HI | `facebook/nllb-200-distilled-600M` | 615.1M | 54.20 | 49.80 | 92.10 | 135.4 ms | 2,410 MB |
| NLLB-200 (Dist) | EN $\to$ MR | `facebook/nllb-200-distilled-600M` | 615.1M | 11.40 | 27.60 | 76.40 | 139.1 ms | 2,410 MB |
| mBART-50 (M2M)  | EN $\to$ DE | `facebook/mbart-large-50-many-to-many-mmt` | 610.9M | 22.80 | 43.10 | 74.50 | 155.0 ms | 2,580 MB |
| mBART-50 (M2M)  | EN $\to$ FR | `facebook/mbart-large-50-many-to-many-mmt` | 610.9M | 21.90 | 42.80 | 73.90 | 160.2 ms | 2,580 MB |
| mBART-50 (M2M)  | EN $\to$ HI | `facebook/mbart-large-50-many-to-many-mmt` | 610.9M | 51.30 | 47.50 | 90.20 | 145.0 ms | 2,520 MB |
| mBART-50 (M2M)  | EN $\to$ MR | `facebook/mbart-large-50-many-to-many-mmt` | 610.9M | 10.20 | 25.40 | 74.80 | 152.4 ms | 2,520 MB |

---

## 5. Key Findings & Discussion

1. **MarianMT Efficiency Advantage**:
   - MarianMT achieved the highest translation scores across all language pairs while operating with **~8.2x fewer parameters** and **~3.2x faster inference latency** than NLLB-200 / mBART-50.
   - Dedicated pair-specific subword vocabularies prevent capacity dilution across hundreds of unrelated languages.
2. **Indic Language Nuance (Hindi vs Marathi)**:
   - EN $\to$ HI achieved high scores (BLEU `60.76`, COMET `96.79`) on IIT Bombay test set due to strong corpus consistency.
   - EN $\to$ MR (BLEU `15.86`, COMET `81.42`) highlights the low-resource challenge of Marathi, where high agglutination and rich inflections require broader morpheme modeling.
3. **Cross-Attention Interpretability**:
   - In MarianMT, cross-attention directly traces grammatical word alignments.
   - In NLLB-200, attention sink behavior parks mass on `</s>`. Dynamic row re-normalization restores clean word-to-word visibility.

---

## 6. Error Analysis Taxonomy

We categorized failure cases across 9 distinct error classes using automated alignment heuristics (`src/evaluation/error_analyzer.py`):

```
┌─────────────────────────────────────────────────────────────┐
│              Translation Error Distribution                 │
├──────────────────────────────────────────┬──────────────────┤
│ Error Category                           │ Percentage       │
├──────────────────────────────────────────┼──────────────────┤
│ 1. Lexical / Terminology Selection       │ 31.4% ████████   │
│ 2. Morphological Inflexion               │ 22.8% ██████     │
│ 3. Omission / Under-generation           │ 14.2% ████       │
│ 4. Syntactic Reordering (SVO to SOV)     │ 12.1% ███        │
│ 5. Named Entity Transliteration          │  9.3% ██         │
│ 6. Hallucination / Over-generation       │  4.8% █          │
│ 7. Punctuation Inconsistency             │  3.2% █          │
│ 8. Formality / Register Mismatch         │  1.5%            │
│ 9. Grammatical Agreement                 │  0.7%            │
└──────────────────────────────────────────┴──────────────────┘
```

---

## 7. System Demonstration & Verification

### 7.1 Web Interface Features (Streamlit)
* **Translation Studio**: Real-time translation, model switching, fine-tuned checkpoint fallback, generation parameter tuning, and telemetry metrics (latency, tok/s, peak VRAM).
* **Model Arena**: Side-by-side translation generation comparing MarianMT, mBART-50, and NLLB-200 on identical inputs with interactive latency benchmarking charts.
* **Interpretability Lab**: Layer-wise cross-attention heatmap extraction, attention sink filtering with re-normalization, and softmax output distribution chips.
* **Research Dashboard**: Interactive empirical benchmark exploration, 9-category error plots, sentence-length breakdown, and SHA-256 data leakage audit tables.

### 7.2 Automated Testing
* **PyTest Suite**: 23 automated unit and integration tests passing (`100% test pass rate`).

---

## 8. Conclusion
The project establishes a complete, rigorous, and reproducible deep learning translation benchmark. Pair-specific Transformer architectures demonstrate exceptional resource-to-quality efficiency for targeted language pairs, while multilingual models provide zero-shot breadth. The system is documented, tested, and validated for academic review.
