# Product Requirements Document (PRD)

# Multilingual Neural Machine Translation Using Transformer-Based Deep Learning Models

---

## 1. Project Overview

The project aims to develop a **multilingual Neural Machine Translation (NMT) system** using pretrained Transformer-based Deep Learning models.

The system will translate text between English and multiple target languages, with a particular focus on:

- English
- Hindi
- Marathi
- French
- German

The project will investigate and compare three Transformer architectures:

1. **MarianMT** — baseline model
2. **mBART-50** — multilingual Transformer model
3. **NLLB** — primary multilingual translation model

Rather than training large Transformer models from scratch, the project will use **transfer learning and fine-tuning of pretrained models**.

The models will be evaluated using:

- BLEU
- METEOR
- COMET

The project will additionally perform qualitative translation analysis, error analysis, and sentence-length analysis.

The best-performing model will be integrated into a **Streamlit-based multilingual translation application**.

---

# 2. Problem Statement

Traditional rule-based and statistical machine translation systems often struggle with:

- Context preservation
- Grammar
- Word ordering
- Long-distance relationships between words
- Ambiguous words
- Low-resource languages
- Natural sentence generation

Transformer architectures address many of these limitations through **self-attention mechanisms and sequence-to-sequence learning**.

However, different Transformer architectures can perform differently depending on:

- Language pair
- Dataset size
- Language-resource availability
- Sentence length
- Model architecture
- Fine-tuning strategy

Therefore, this project aims to compare different Transformer architectures and determine their effectiveness for multilingual neural machine translation.

### Research Problem

> **How do different Transformer architectures perform in multilingual neural machine translation across high-resource and relatively lower-resource language pairs?**

---

# 3. Objectives

## Primary Objectives

1. Develop a multilingual Neural Machine Translation pipeline.
2. Use pretrained Transformer models for translation.
3. Fine-tune selected models using parallel translation datasets.
4. Compare MarianMT, mBART-50, and NLLB.
5. Support English, Hindi, Marathi, French, and German.
6. Evaluate translation quality using BLEU, METEOR, and COMET.
7. Perform qualitative translation analysis.
8. Perform systematic error analysis.
9. Analyze the effect of sentence length on translation quality.
10. Identify the best-performing model.
11. Deploy the selected model using Streamlit.

## Secondary Objectives

- Analyze high-resource vs relatively lower-resource language performance.
- Study the effect of fine-tuning on pretrained models.
- Compare translation quality against computational requirements.
- Demonstrate practical implementation of Transformer-based Deep Learning.

---

# 4. Research Questions

### Primary Research Question

> How do different Transformer architectures perform in multilingual neural machine translation across high-resource and relatively lower-resource language pairs?

### Secondary Questions

1. Which of MarianMT, mBART-50, and NLLB achieves the best translation quality?
2. Does a multilingual model outperform a language-specific translation model?
3. How does performance vary between Hindi, Marathi, French, and German?
4. Does fine-tuning improve translation quality compared with pretrained models?
5. How does sentence length affect translation performance?
6. What types of translation errors occur most frequently?
7. What is the trade-off between translation quality and computational cost?

---

# 5. Project Scope

## 5.1 In Scope

- Transformer-based Neural Machine Translation
- Transfer learning
- Fine-tuning pretrained models
- Multilingual translation
- English-centric translation
- English → Hindi
- English → Marathi
- English → French
- English → German
- Dataset preprocessing
- Tokenization
- Model training/fine-tuning
- GPU acceleration
- Model evaluation
- Error analysis
- Comparative experiments
- Streamlit deployment

## 5.2 Out of Scope

- Training a Transformer from scratch
- Supporting hundreds of languages
- Production-scale distributed translation infrastructure
- Speech-to-speech translation
- OCR-based document translation
- Mobile application development
- Large-scale multi-GPU training

---

# 6. Supported Languages

| Language | ISO Code | Role |
|---|---|---|
| English | en | Primary source language |
| Hindi | hi | Indian target language |
| Marathi | mr | Indian / relatively lower-resource target |
| French | fr | High-resource international language |
| German | de | High-resource international language |

The system architecture should remain extensible so additional languages can be added in the future.

---

# 7. Transformer Models

## 7.1 MarianMT

MarianMT will act as the **baseline model**.

MarianMT provides language-pair-specific pretrained translation models.

### Role

- Establish baseline translation performance.
- Compare language-specific and multilingual models.
- Evaluate computational requirements.

---

## 7.2 mBART-50

mBART-50 is a multilingual sequence-to-sequence Transformer model.

### Role

- Evaluate multilingual transfer learning.
- Compare multilingual architecture against MarianMT.
- Study performance across multiple language pairs.

---

## 7.3 NLLB

NLLB (No Language Left Behind) will be the **primary multilingual model**.

### Role

- Evaluate multilingual translation capabilities.
- Focus on Indian languages such as Hindi and Marathi.
- Compare performance against MarianMT and mBART.
- Investigate relatively lower-resource translation.

### Important Resource Constraint

The project will use an **NLLB model variant appropriate for the available GPU memory**.

The project will **not assume that large NLLB variants can be fully fine-tuned on an 8 GB GPU**.

Where full fine-tuning is impractical, alternatives such as:

- Parameter-efficient fine-tuning
- Smaller model variants
- Reduced sequence length
- Gradient accumulation
- Gradient checkpointing
- Reduced batch sizes

may be used.

---

# 8. Hardware Environment

The project will primarily be developed and trained locally.

### Development Machine

| Component | Specification |
|---|---|
| GPU | NVIDIA GeForce RTX 5050 Laptop GPU |
| GPU VRAM | 8 GB |
| RAM | 16 GB |
| Storage | 1 TB SSD |
| OS | Windows 11 / WSL2 where required |
| GPU Acceleration | CUDA |
| Primary Framework | PyTorch |

### Backup Environment

Google Colab may be used when:

- A model exceeds available local VRAM.
- Extended training is required.
- Additional GPU resources are temporarily needed.

---

# 9. GPU Training Strategy

The project will be designed specifically for an **8 GB VRAM GPU**.

Training will use memory-efficient techniques where necessary.

### Techniques

- Mixed-precision training
- FP16/BF16 where supported
- Small batch sizes
- Gradient accumulation
- Gradient checkpointing
- Dynamic padding
- Controlled sequence length
- Dataset sampling
- Periodic checkpointing

### Initial Training Configuration

The following values will be used as starting points rather than fixed requirements:

| Parameter | Initial Value |
|---|---:|
| Batch size | 2–4 |
| Gradient accumulation | 4–8 |
| Maximum sequence length | 128–256 |
| Epochs | 2–5 |
| Learning rate | Approximately 2e-5 |
| Optimizer | AdamW |
| Precision | Mixed precision |
| Evaluation | Periodic |
| Checkpointing | Enabled |

These values will be adjusted based on actual GPU memory utilization and training stability.

---

# 10. Training Strategy

Large Transformer models will **not** be trained from scratch.

The project will follow a transfer-learning pipeline:

```text
Pretrained Transformer
        ↓
Parallel Translation Dataset
        ↓
Data Cleaning
        ↓
Tokenization
        ↓
Fine-Tuning
        ↓
Validation
        ↓
Testing
        ↓
Evaluation
```

For computationally expensive models, parameter-efficient approaches may be considered.

---

# 11. Dataset Strategy

The project will use publicly available parallel translation datasets.

## 11.1 IIT Bombay English-Hindi Corpus

Primary use:

**English ↔ Hindi**

The dataset provides parallel English-Hindi sentences and is useful for Indian-language translation experiments.

---

## 11.2 OPUS

OPUS will provide additional parallel language datasets.

Potential use:

- English ↔ Marathi
- English ↔ French
- English ↔ German

Only suitable and manageable subsets will be used.

---

## 11.3 WMT

Selected WMT datasets may be used for high-resource language pairs such as:

- English ↔ French
- English ↔ German

The complete datasets will not necessarily be used because dataset size must remain compatible with the available training resources.

---

# 12. Dataset Scaling Strategy

The project will not immediately train on the largest available datasets.

A staged approach will be followed.

### Phase 1 — Pipeline Validation

Approximately:

```text
1,000–5,000 sentence pairs
```

Purpose:

- Validate preprocessing
- Validate tokenizer
- Validate model input/output
- Validate training
- Validate evaluation

### Phase 2 — Experimental Training

Approximately:

```text
10,000–30,000 sentence pairs per selected experiment
```

Purpose:

- Perform meaningful model comparison.
- Measure translation performance.

### Phase 3 — Dataset Expansion

If GPU memory, training time, and dataset quality permit, the dataset size may be increased.

This prevents unnecessary computation before the pipeline is verified.

---

# 13. Unified Dataset Format

All datasets will be converted into a common structure.

| source_language | target_language | source_text | target_text |
|---|---|---|---|
| en | hi | Artificial intelligence is transforming the world. | कृत्रिम बुद्धिमत्ता दुनिया को बदल रही है। |
| en | mr | Artificial intelligence is transforming the world. | कृत्रिम बुद्धिमत्ता जग बदलत आहे। |
| en | fr | Artificial intelligence is transforming the world. | L'intelligence artificielle transforme le monde. |
| en | de | Artificial intelligence is transforming the world. | Künstliche Intelligenz verändert die Welt. |

This unified structure will simplify:

- Training
- Evaluation
- Dataset management
- Language-pair comparison

---

# 14. Data Preprocessing

The preprocessing pipeline will include:

1. Dataset loading
2. Duplicate removal
3. Missing-value removal
4. Invalid sentence removal
5. Text normalization
6. Language-pair validation
7. Sentence-length filtering
8. Train/validation/test split
9. Tokenization
10. Padding/truncation

Initial split:

```text
Training      → 80%
Validation    → 10%
Testing       → 10%
```

The split may be adjusted depending on dataset characteristics.

---

# 15. Tokenization

Each model will use its corresponding pretrained tokenizer.

```text
Raw Text
    ↓
Tokenizer
    ↓
Token IDs
    ↓
Attention Mask
    ↓
Transformer
    ↓
Generated Tokens
    ↓
Decoded Translation
```

Multilingual models will use their required language identifiers/codes.

---

# 16. Training Pipeline

```text
             Dataset
                ↓
          Data Cleaning
                ↓
       Train/Val/Test Split
                ↓
           Tokenization
                ↓
        DataLoader/Batches
                ↓
       Mixed Precision GPU
                ↓
        Transformer Model
                ↓
            Loss
                ↓
        Backpropagation
                ↓
        Optimizer Update
                ↓
          Validation
                ↓
           Checkpoint
```

---

# 17. Evaluation Metrics

## 17.1 BLEU

BLEU measures n-gram overlap between the generated translation and reference translation.

It will be used as one of the standard machine translation metrics.

---

## 17.2 METEOR

METEOR evaluates word-level correspondence and alignment between generated and reference translations.

It provides information that complements BLEU.

---

## 17.3 COMET

COMET will be the primary advanced evaluation metric.

It evaluates:

- Source sentence
- Generated translation
- Reference translation

and provides a semantic assessment of translation quality.

---

# 18. Computational Metrics

Models will additionally be compared using:

- Number of parameters
- GPU memory consumption
- Inference time
- Model size
- Training time where practical

This allows the project to evaluate:

```text
Translation Quality
        vs
Computational Cost
```

---

# 19. Qualitative Evaluation

Numerical scores alone cannot completely evaluate machine translation.

Representative translations will therefore be manually analyzed.

Example:

### Input

```text
Artificial intelligence is transforming the world.
```

### Target

```text
Marathi
```

### Comparison

| Model | Translation | Analysis |
|---|---|---|
| MarianMT | Generated output | Grammar/context |
| mBART | Generated output | Grammar/context |
| NLLB | Generated output | Grammar/context |
| Reference | Reference output | Ground truth |

Evaluation will consider:

- Meaning preservation
- Fluency
- Grammar
- Word order
- Context
- Terminology
- Named entities
- Repetition
- Untranslated words

---

# 20. Error Analysis

Translation errors will be classified into:

### 20.1 Word Order Errors

Incorrect ordering of translated words.

### 20.2 Grammatical Errors

Examples:

- Incorrect tense
- Incorrect gender
- Incorrect number
- Incorrect sentence structure

### 20.3 Named Entity Errors

Examples:

- Names
- Organizations
- Locations
- Technical terms

### 20.4 Ambiguity Errors

Incorrect interpretation of words with multiple meanings.

### 20.5 Long Sentence Errors

Loss of context or incorrect structure in longer sentences.

### 20.6 Low-Resource Language Errors

Comparison of relatively lower-resource languages such as Marathi against high-resource languages.

---

# 21. Sentence-Length Experiment

Translation quality will be analyzed across sentence lengths.

| Category | Length |
|---|---:|
| Short | 1–10 words |
| Medium | 11–25 words |
| Long | 26–50 words |
| Very Long | 50+ words |

Metrics will be calculated separately where sufficient test data exists.

---

# 22. Experimental Design

## Experiment 1 — Transformer Model Comparison

Compare:

```text
MarianMT
mBART-50
NLLB
```

using compatible datasets and evaluation conditions.

---

## Experiment 2 — Language Comparison

Evaluate:

```text
English → Hindi
English → Marathi
English → French
English → German
```

---

## Experiment 3 — Fine-Tuning Comparison

Where computationally feasible:

```text
Pretrained Model
       vs
Fine-Tuned Model
```

---

## Experiment 4 — Sentence-Length Analysis

Compare translation quality for:

```text
Short
Medium
Long
Very Long
```

sentences.

---

## Experiment 5 — Error Analysis

Analyze common translation errors across:

- Models
- Languages
- Sentence lengths

---

# 23. Model Selection

The final model will be selected using a combination of:

1. COMET
2. BLEU
3. METEOR
4. Qualitative translation quality
5. Inference speed
6. GPU memory requirements
7. Model size

The goal is **not simply to select the model with the highest score**.

The project will identify the model providing the best overall trade-off between:

```text
Translation Quality
        +
Computational Efficiency
```

---

# 24. Application Architecture

The final system will use Streamlit.

```text
                    USER
                      │
                      ▼
               Streamlit UI
                      │
                      ▼
             Source Language
                      │
                      ▼
             Target Language
                      │
                      ▼
                 Text Input
                      │
                      ▼
            Selected Transformer
                      │
                      ▼
               Translation
                      │
                      ▼
             Translated Output
```

The application will use the selected best-performing model for normal deployment.

The application will **not load all large models simultaneously** if doing so exceeds available GPU memory.

---

# 25. Application Features

### Core Features

- Source language selection
- Target language selection
- Text input
- Translation
- Clear/reset functionality
- Translation output

### Optional Features

- Automatic source-language detection
- Model selection
- Translation history
- Example sentences
- Character/word count
- Translation comparison mode

Optional features will only be implemented after the core translation system is stable.

---

# 26. Optional Automatic Language Detection

The application may automatically identify the source language.

```text
User Input
     ↓
Language Detection
     ↓
Detected Language
     ↓
Target Language Selection
     ↓
Transformer
     ↓
Translation
```

This feature will remain optional because the main objective of the project is Transformer-based translation rather than language identification.

---

# 27. System Architecture

```text
                 ┌───────────────────┐
                 │       User        │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │   Streamlit UI   │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Input Processing │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Language Selection│
                 └─────────┬─────────┘
                           │
                           ▼
          ┌────────────────────────────────┐
          │     Transformer Translation     │
          │                                │
          │ MarianMT | mBART-50 | NLLB    │
          └───────────────┬────────────────┘
                          │
                          ▼
                 ┌───────────────────┐
                 │ Translation Output│
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Evaluation System │
                 │ BLEU/METEOR/COMET │
                 └───────────────────┘
```

---

# 28. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python 3.10/3.11 |
| Deep Learning Framework | PyTorch |
| Transformer Framework | Hugging Face Transformers |
| Dataset Framework | Hugging Face Datasets |
| Models | MarianMT, mBART-50, NLLB |
| Data Processing | Pandas, NumPy |
| Evaluation | BLEU, METEOR, COMET |
| Development | VS Code / Jupyter |
| Primary Training | Local RTX 5050 GPU |
| GPU VRAM | 8 GB |
| System RAM | 16 GB |
| Storage | 1 TB SSD |
| GPU Framework | CUDA |
| Deployment | Streamlit |
| Version Control | Git / GitHub |
| Backup Compute | Google Colab |

---

# 29. Recommended Software Environment

```text
Python 3.10 / 3.11
        │
        ▼
PyTorch + CUDA
        │
        ▼
Hugging Face Transformers
        │
        ├── Datasets
        ├── Tokenizers
        └── Evaluate
        │
        ▼
Streamlit
```

The project environment should be tested with CUDA-enabled PyTorch before starting model training.

---

# 30. Project Directory Structure

```text
multilingual-nmt/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── test/
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_marianmt_experiment.ipynb
│   ├── 03_mbart_experiment.ipynb
│   ├── 04_nllb_experiment.ipynb
│   └── 05_evaluation.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── training.py
│   ├── inference.py
│   ├── evaluation.py
│   └── utils.py
│
├── models/
│   └── fine_tuned/
│
├── results/
│   ├── metrics/
│   ├── comparisons/
│   └── error_analysis/
│
├── app/
│   └── app.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 31. Expected Results

The project is expected to produce:

1. A working multilingual NMT pipeline.
2. Fine-tuned Transformer models where computationally feasible.
3. Quantitative comparison of MarianMT, mBART-50, and NLLB.
4. BLEU scores.
5. METEOR scores.
6. COMET scores.
7. Qualitative translation comparisons.
8. Error analysis.
9. Sentence-length analysis.
10. Computational-cost comparison.
11. Selection of the best-performing model.
12. A Streamlit-based translation application.

---

# 32. Expected Demonstration

### Input

```text
Artificial Intelligence is transforming the world.
```

### Source Language

```text
English
```

### Target Language

```text
Marathi
```

### Output

```text
कृत्रिम बुद्धिमत्ता जग बदलत आहे.
```

The system should attempt to preserve:

- Meaning
- Context
- Grammar
- Sentence structure
- Appropriate terminology

---

# 33. Success Criteria

The project will be considered successful if:

- The system successfully translates the selected language pairs.
- Three Transformer architectures are experimentally evaluated.
- Pretrained models are appropriately fine-tuned where feasible.
- BLEU, METEOR, and COMET are reported.
- Translation examples are manually analyzed.
- Translation errors are categorized.
- Sentence-length effects are investigated.
- Computational requirements are documented.
- A best-performing model is identified.
- The selected model is demonstrated through a Streamlit application.
- The complete pipeline can run using the team's available hardware or backup compute.

---

# 34. Hardware Constraints and Mitigation

The primary hardware has:

```text
RTX 5050 Laptop GPU
8 GB VRAM
16 GB RAM
```

Therefore, the project will explicitly account for memory constraints.

### Potential Problems

- Large mBART/NLLB models may exceed VRAM.
- Large batch sizes may cause CUDA Out-of-Memory errors.
- Long sequence lengths increase memory usage.
- Fine-tuning may take several hours depending on dataset size.

### Mitigation

```text
Large Model
    ↓
Smaller Suitable Variant
    ↓
FP16/BF16
    ↓
Small Batch
    ↓
Gradient Accumulation
    ↓
Gradient Checkpointing
    ↓
Controlled Sequence Length
```

If a particular model cannot be fully fine-tuned within available resources, the experiment will use an appropriate smaller variant or parameter-efficient fine-tuning instead of artificially reducing the quality of the entire project.

---

# 35. Limitations

The project may have the following limitations:

- 8 GB GPU VRAM limits the size of models that can be fully fine-tuned.
- Large multilingual models may require parameter-efficient techniques.
- Marathi and other relatively lower-resource languages may have lower translation quality.
- Automated metrics cannot perfectly represent human translation quality.
- Dataset quality directly affects model performance.
- Training all models on all language pairs may not be computationally practical.
- Translation quality can vary significantly depending on domain and sentence complexity.

---

# 36. Future Scope

Future improvements may include:

- Additional Indian languages
- Gujarati
- Bengali
- Tamil
- Telugu
- Kannada
- Speech-to-text translation
- Text-to-speech output
- Mobile application
- Cloud deployment
- Real-time conversational translation
- Domain-specific translation
- Human translation evaluation
- Retrieval-augmented translation
- Terminology-aware translation
- Larger multilingual models
- Parameter-efficient fine-tuning

---

# 37. Final Project Pipeline

```text
                 DATA COLLECTION
                       │
                       ▼
              DATA PREPROCESSING
                       │
                       ▼
                DATASET SPLIT
                       │
                       ▼
                  TOKENIZATION
                       │
                       ▼
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
     MarianMT        mBART           NLLB
     Baseline      Multilingual     Primary
        │              │              │
        └──────────────┼──────────────┘
                       ▼
             GPU-ACCELERATED
                FINE-TUNING
                       │
                ┌──────┴──────┐
                ▼             ▼
          Mixed Precision   Gradient
                            Accumulation
                │             │
                └──────┬──────┘
                       ▼
                  VALIDATION
                       │
                       ▼
                    TESTING
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
        BLEU         METEOR        COMET
          │            │            │
          └────────────┼────────────┘
                       ▼
              MODEL COMPARISON
                       │
                       ▼
                ERROR ANALYSIS
                       │
                       ▼
              COMPUTATIONAL ANALYSIS
                       │
                       ▼
                 BEST MODEL
                       │
                       ▼
                STREAMLIT APP
                       │
                       ▼
                  FINAL DEMO
```

---

# 38. Key Deep Learning Concepts Demonstrated

The project will demonstrate practical understanding of:

### Deep Learning

- Neural networks
- Backpropagation
- Gradient descent
- Loss functions
- Optimization
- GPU acceleration
- Mixed-precision training

### NLP

- Tokenization
- Sequence-to-sequence learning
- Language modeling
- Text preprocessing
- Beam search
- Semantic representation

### Transformers

- Self-attention
- Multi-head attention
- Query, Key, Value
- Positional encoding
- Encoder-decoder architecture
- Cross-attention
- Autoregressive generation

### Transfer Learning

- Pretrained models
- Fine-tuning
- Parameter-efficient fine-tuning
- Domain/language adaptation

### Evaluation

- BLEU
- METEOR
- COMET
- Qualitative evaluation
- Error analysis

### Deployment

- Model inference
- GPU inference
- Streamlit
- Interactive AI application

---

# 39. Final Project Definition

> **Multilingual Neural Machine Translation Using Transformer-Based Deep Learning Models** is a Deep Learning project that investigates and compares pretrained Transformer architectures—MarianMT, mBART-50, and NLLB—for multilingual neural machine translation. The system will focus on English, Hindi, Marathi, French, and German and will use publicly available parallel corpora such as the IIT Bombay English-Hindi Corpus, OPUS, and selected WMT datasets. Pretrained models will be fine-tuned using the project's local NVIDIA RTX 5050 8 GB GPU with memory-efficient Deep Learning techniques such as mixed-precision training, gradient accumulation, and controlled sequence lengths. Translation quality will be evaluated using BLEU, METEOR, and COMET, supplemented by qualitative and error analysis. The best-performing model will be deployed through a Streamlit-based multilingual translation application.

---

# 40. Project Goal

The ultimate goal is not simply to build a translation interface.

The project aims to:

> **Design, fine-tune, evaluate, and compare Transformer-based multilingual Neural Machine Translation models while analyzing their translation quality, language-specific performance, computational requirements, and practical deployment feasibility.**

---

# PROJECT RESOLUTION ADDENDUM — FROZEN IMPLEMENTATION DECISIONS

> This addendum preserves the original project specification above and records the decisions resolved before implementation. It does not delete, replace, or silently reinterpret any original requirement. Where an original section intentionally left a value open for empirical validation, the original statement remains applicable and the resolved implementation policy below governs execution.

## A. Scope and Directionality

The primary experimental translation directions are:

| Source | Target | Status |
|---|---|---|
| English | Hindi | Primary |
| English | Marathi | Primary |
| English | French | Primary |
| English | German | Primary |

Reverse directions are outside the primary experimental scope unless explicitly added as a separately documented extension.

The project remains English-centric. The presence of multilingual model architectures does not imply that every possible source-target combination is part of the experiment.

## B. Exact Model Registry

The following checkpoints are the frozen model identities for the primary study:

| Architecture | Checkpoint | Role |
|---|---|---|
| MarianMT | `Helsinki-NLP/opus-mt-en-hi` | EN→HI baseline |
| MarianMT | `Helsinki-NLP/opus-mt-en-mr` | EN→MR baseline |
| MarianMT | `Helsinki-NLP/opus-mt-en-fr` | EN→FR baseline |
| MarianMT | `Helsinki-NLP/opus-mt-en-de` | EN→DE baseline |
| mBART-50 | `facebook/mbart-large-50-many-to-many-mmt` | Multilingual comparison |
| NLLB | `facebook/nllb-200-distilled-600M` | Primary multilingual model |

The exact model revision or commit hash used for each experiment must be captured in the experiment manifest. A model name alone is not sufficient for reproducibility.

## C. Language-Code Registry

NLLB codes:

| Language | Code |
|---|---|
| English | `eng_Latn` |
| Hindi | `hin_Deva` |
| Marathi | `mar_Deva` |
| French | `fra_Latn` |
| German | `deu_Latn` |

mBART-50 codes:

| Language | Code |
|---|---|
| English | `en_XX` |
| Hindi | `hi_IN` |
| Marathi | `mr_IN` |
| French | `fr_XX` |
| German | `de_DE` |

These mappings must be verified against the loaded tokenizer/model before experiments are frozen and must be stored centrally in configuration rather than duplicated throughout application code.

## D. Training Policy

The project uses transfer learning and fine-tuning rather than training a Transformer from scratch.

Full fine-tuning should be attempted first where feasible on the stated hardware. If memory prevents successful execution, the fallback sequence is:

1. Reduce per-device batch size.
2. Enable gradient checkpointing.
3. Reduce maximum sequence length.
4. Increase gradient accumulation.
5. Use PEFT/LoRA where necessary.
6. Use backup compute such as Google Colab when required.

Any fallback must be explicitly recorded in the experiment configuration and final report. No change in training strategy may occur silently.

## E. Hardware Policy

Primary development/training target:

- NVIDIA RTX 5050 Laptop GPU
- 8 GB VRAM
- 16 GB RAM
- 1 TB SSD

Google Colab is backup compute, not a silent replacement for the primary environment.

Memory-efficiency mechanisms may include:

- mixed precision,
- small per-device batches,
- gradient accumulation,
- gradient checkpointing,
- dynamic padding,
- controlled sequence length,
- dataset sampling where explicitly documented.

## F. Dataset Policy

One canonical cleaned parallel dataset is maintained for each language pair.

Intended source strategy:

- EN→HI: IIT Bombay English-Hindi Corpus where suitable.
- EN→MR: versioned OPUS or another suitable parallel source after quality validation.
- EN→FR and EN→DE: versioned OPUS/WMT-compatible source after quality validation.

The exact dataset configuration/revision must be frozen after validation rather than fabricated in advance.

Required processing order:

```text
Raw load
   ↓
Schema validation
   ↓
Text normalization
   ↓
Invalid-row removal
   ↓
Pair-level deduplication
   ↓
Leakage detection
   ↓
Length filtering
   ↓
Deterministic 80/10/10 split
```

The test set must never be used for training, hyperparameter tuning, checkpoint selection, generation tuning, or model selection.

After normalization and deduplication, train/test overlap must be zero.

## G. Evaluation Policy

The required automatic metrics are:

- BLEU
- METEOR
- COMET

Validation COMET is the primary automated checkpoint/model-selection criterion.

BLEU, METEOR, qualitative assessment, error analysis, inference speed, VRAM usage, and model artifact size are supporting evidence.

No arbitrary weighted composite score is used to select the final model.

Metric implementations and versions must be recorded in the experiment metadata.

## H. Generation Policy

The starting deterministic generation configuration is:

- beam search,
- `num_beams=5`,
- `max_new_tokens=128`,
- sampling disabled.

Generation settings must be recorded with every evaluation. The same evaluation generation policy should be used across models where technically valid; model-specific exceptions must be documented.

## I. Error and Sentence-Length Analysis

Error categories:

- grammar,
- word order,
- named entity,
- ambiguity,
- missing words,
- extra words,
- repetition,
- untranslated words,
- long-sentence failures.

Structured records should contain:

- sample ID,
- model,
- language pair,
- source,
- reference,
- prediction,
- error category,
- severity,
- comment.

Sentence-length buckets:

| Bucket | Range |
|---|---|
| Short | 1–10 |
| Medium | 11–25 |
| Long | 26–50 |
| Very long | 51+ |

Buckets containing fewer than 30 examples must be flagged as small samples and must not support strong conclusions.

## J. Experiment Identity and Artifacts

Experiment naming convention:

`{MODEL}_{SRC}_{TGT}_{DATASET}_{SEED}_{RUN}`

Example:

`NLLB_EN_MR_OPUS_42_001`

Each experiment should contain at least:

```text
experiment/
├── config.yaml
├── environment.txt
├── training.log
├── checkpoint_manifest.json
├── predictions.csv
├── metrics.json
├── resource_metrics.json
├── error_analysis.csv
└── README.md
```

`metrics.json` must include metric implementation/version metadata.

## K. Application Policy

The Streamlit application uses the final selected checkpoint by default.

It supports:

- EN→HI,
- EN→MR,
- EN→FR,
- EN→DE.

The application should cache the model and tokenizer rather than reload them for every request.

The application does not perform training.

Only one large model should occupy GPU memory at a time unless memory feasibility has been demonstrated.

User text is not permanently stored by default.

## L. Research and Application Separation

Research/training code and application/inference code must remain separate.

The research pipeline owns:

- data acquisition,
- cleaning,
- splitting,
- training,
- checkpointing,
- validation,
- testing,
- comparison,
- error analysis,
- resource measurement.

The application owns:

- model loading,
- language selection,
- input validation,
- generation,
- output display,
- optional latency reporting.

## M. Reproducibility

Every experiment must capture:

- random seed,
- model checkpoint,
- model revision,
- tokenizer identity,
- dataset identity/version/configuration,
- data split seed,
- preprocessing configuration,
- generation configuration,
- hyperparameters,
- training strategy,
- package/environment versions,
- hardware,
- evaluation metric versions,
- selected checkpoint,
- resource measurements.

## N. Research-Integrity Rules

The implementation agent must never:

- invent dataset versions,
- invent metric values,
- fabricate training results,
- claim a model trained when it failed,
- report a checkpoint that was not actually produced,
- use the test set for tuning,
- silently change a model checkpoint,
- silently switch hardware,
- silently change fine-tuning to PEFT,
- silently change generation parameters,
- label a bucket statistically meaningful when it has fewer than 30 examples.

All empirical claims must be traceable to generated artifacts.

## O. What Remains Empirical

The following values are starting points, not guaranteed outcomes:

- exact runtime,
- exact peak VRAM,
- exact training duration,
- exact final metrics,
- exact best checkpoint,
- exact dataset row counts after cleaning,
- exact number of training steps,
- whether full fine-tuning fits for every model,
- whether PEFT is required for any particular experiment.

These values must be measured and recorded rather than guessed.
