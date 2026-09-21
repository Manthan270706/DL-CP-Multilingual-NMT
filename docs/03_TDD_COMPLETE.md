# Technical Design Document (TDD)

# Multilingual Neural Machine Translation Using Transformer-Based Deep Learning Models

**Version:** 1.0\
**Status:** Proposed\
**Primary Framework:** PyTorch\
**NLP Framework:** Hugging Face Transformers\
**Deployment:** Streamlit\
**Hardware:** NVIDIA RTX 5050 Laptop GPU (8 GB VRAM), 16 GB RAM, 1 TB SSD

---

# 1. Technical Overview

The system is a Transformer-based Neural Machine Translation platform designed to translate text between English and selected target languages:

- Hindi
- Marathi
- French
- German

The system will experimentally evaluate three pretrained Transformer architectures:

1. MarianMT
2. mBART-50
3. NLLB

The project consists of two major components:

### Research/Training System

Responsible for:

- Dataset collection
- Data preprocessing
- Tokenization
- Model fine-tuning
- Validation
- Testing
- Metric calculation
- Model comparison
- Error analysis

### Inference/Application System

Responsible for:

- User input
- Language selection
- Model loading
- Translation
- Output generation
- Streamlit interface

---

# 2. High-Level Architecture

```text
                         ┌───────────────────────┐
                         │        DATASETS       │
                         │                       │
                         │ IIT Bombay │ OPUS     │
                         │ WMT       │           │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ DATA PREPROCESSING    │
                         │                       │
                         │ Cleaning              │
                         │ Deduplication         │
                         │ Normalization         │
                         │ Filtering             │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ DATASET MANAGEMENT    │
                         │                       │
                         │ Train                 │
                         │ Validation            │
                         │ Test                  │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ TOKENIZATION          │
                         │                       │
                         │ MarianTokenizer       │
                         │ MBartTokenizer        │
                         │ NLLBTokenizer         │
                         └───────────┬───────────┘
                                     │
                                     ▼
               ┌───────────────────────────────────────────┐
               │              MODEL LAYER                  │
               │                                           │
               │   ┌──────────┐ ┌─────────┐ ┌─────────┐   │
               │   │ MarianMT │ │ mBART   │ │  NLLB   │   │
               │   └──────────┘ └─────────┘ └─────────┘   │
               └─────────────────────┬─────────────────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ TRAINING ENGINE       │
                         │                       │
                         │ PyTorch               │
                         │ AdamW                 │
                         │ Mixed Precision       │
                         │ Gradient Accumulation │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ MODEL CHECKPOINTS     │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ EVALUATION ENGINE     │
                         │                       │
                         │ BLEU                  │
                         │ METEOR                │
                         │ COMET                 │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ MODEL COMPARISON      │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ BEST MODEL            │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ STREAMLIT APPLICATION  │
                         └───────────────────────┘
```

---

# 3. Architectural Principles

The system will follow these principles:

### 3.1 Modularity

Dataset processing, training, evaluation, inference, and UI should be independent modules.

### 3.2 Reproducibility

Experiments should record:

- Random seed
- Model name
- Dataset version
- Hyperparameters
- Training configuration
- Evaluation metrics

### 3.3 Model Agnosticism

The training and evaluation pipeline should support multiple Transformer architectures without rewriting the entire system.

### 3.4 Resource Awareness

The architecture must operate within:

```text
GPU VRAM: 8 GB
RAM:      16 GB
Storage:  1 TB
```

### 3.5 Separation of Training and Inference

Training code must not be tightly coupled with the Streamlit application.

---

# 4. System Components

The system will contain the following major components:

```text
1. Dataset Manager
2. Data Preprocessor
3. Dataset Splitter
4. Tokenization Manager
5. Model Manager
6. Training Engine
7. Checkpoint Manager
8. Evaluation Engine
9. Error Analysis Module
10. Model Comparison Module
11. Inference Engine
12. Streamlit UI
13. Configuration Manager
14. Logging System
```

---

# 5. Component Architecture

## 5.1 Dataset Manager

### Responsibility

Load and organize translation datasets.

### Inputs

- IIT Bombay English-Hindi Corpus
- OPUS
- WMT

### Output

Unified parallel dataset.

```text
source_language
target_language
source_text
target_text
```

### Example

```text
{
    "source_language": "en",
    "target_language": "mr",
    "source_text": "Artificial intelligence is transforming the world.",
    "target_text": "कृत्रिम बुद्धिमत्ता जग बदलत आहे."
}
```

---

# 6. Data Preprocessor

### Responsibilities

- Remove null values
- Remove duplicates
- Normalize text
- Remove invalid samples
- Filter extreme sentence lengths
- Validate language pairs
- Remove malformed records

### Pipeline

```text
Raw Dataset
     │
     ▼
Missing Value Removal
     │
     ▼
Duplicate Removal
     │
     ▼
Text Normalization
     │
     ▼
Length Filtering
     │
     ▼
Language Validation
     │
     ▼
Clean Dataset
```

---

# 7. Dataset Splitter

The cleaned dataset will be divided into:

```text
80% → Training
10% → Validation
10% → Testing
```

The test set must remain isolated from training.

### Important Requirement

The test dataset must not be used during:

- Training
- Hyperparameter tuning
- Model selection

This prevents test-set leakage.

---

# 8. Tokenization Architecture

Each model uses its own tokenizer.

```text
                  Input Text
                      │
                      ▼
             Model Selection
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
     MarianTokenizer MBartTokenizer NLLBTokenizer
          │           │           │
          └───────────┼───────────┘
                      ▼
                 Token IDs
                      │
                      ▼
               Attention Mask
                      │
                      ▼
                Model Input
```

---

# 9. Model Manager

The Model Manager provides a unified interface for loading Transformer models.

### Responsibilities

- Load pretrained model
- Load tokenizer
- Configure language codes
- Move model to GPU
- Enable evaluation mode
- Load fine-tuned checkpoints

### Supported Models

```text
MarianMT
mBART-50
NLLB
```

---

# 10. Model Abstraction

The application should use a common conceptual interface:

```python
class TranslationModel:

    def load_model():
        pass

    def tokenize():
        pass

    def translate():
        pass

    def save_checkpoint():
        pass

    def load_checkpoint():
        pass
```

Each architecture can implement model-specific behavior internally.

This prevents the Streamlit application from depending directly on a specific Transformer implementation.

---

# 11. Training Engine

The Training Engine manages model fine-tuning.

### Responsibilities

- Load training dataset
- Create batches
- Move tensors to GPU
- Forward pass
- Calculate loss
- Backpropagation
- Optimizer update
- Validation
- Checkpointing
- Logging

### Training Flow

```text
Training Dataset
       │
       ▼
    DataLoader
       │
       ▼
    Batch Data
       │
       ▼
      GPU
       │
       ▼
 Transformer Model
       │
       ▼
      Loss
       │
       ▼
 Backpropagation
       │
       ▼
   AdamW Update
       │
       ▼
Next Batch
```

---

# 12. Mixed Precision Training

Because the GPU has 8 GB VRAM, mixed-precision training will be used where supported.

```text
FP32 Training
     │
     ▼
Higher Memory Usage
     │
     ▼
Potential OOM
```

Instead:

```text
Mixed Precision
     │
     ├── FP16/BF16 operations
     │
     ├── FP32 where necessary
     │
     ▼
Reduced GPU Memory Usage
```

This allows larger models or batch-equivalent workloads to fit within the available VRAM.

---

# 13. Gradient Accumulation

Small GPU batch sizes may not provide an effective batch size large enough for stable training.

Therefore:

```text
Batch 1
  ↓
Gradient

Batch 2
  ↓
Gradient

Batch 3
  ↓
Gradient

Batch 4
  ↓
Optimizer Update
```

For example:

```text
Batch Size = 2
Gradient Accumulation = 4

Effective Batch Size ≈ 8
```

---

# 14. Gradient Checkpointing

For models that exceed the available memory during training, gradient checkpointing may be enabled.

Instead of storing all intermediate activations:

```text
Normal Training
→ Store many activations
→ Higher VRAM usage
```

Gradient checkpointing:

```text
Store selected activations
        ↓
Recompute others when required
        ↓
Lower VRAM usage
```

This trades additional computation time for reduced memory consumption.

---

# 15. Sequence Length Management

Sequence length strongly affects Transformer memory consumption.

The system will initially use:

```text
Maximum Sequence Length:
128–256 tokens
```

Longer sequences may be tested separately.

The system should avoid unnecessarily padding every sentence to the global maximum.

Dynamic padding will be preferred.

---

# 16. Checkpoint Manager

The Checkpoint Manager will save:

- Model weights
- Tokenizer
- Optimizer state where required
- Training configuration
- Epoch/step
- Validation loss
- Evaluation metrics

Example:

```text
models/
└── fine_tuned/
    ├── marianmt/
    │   └── en_hi/
    ├── mbart/
    │   └── multilingual/
    └── nllb/
        └── multilingual/
```

---

# 17. Evaluation Engine

The Evaluation Engine will calculate:

```text
BLEU
METEOR
COMET
```

### Pipeline

```text
Test Dataset
     │
     ▼
Model Inference
     │
     ▼
Generated Translation
     │
     ├───────────────┐
     ▼               ▼
Reference         Source
Translation       Sentence
     │               │
     └───────┬───────┘
             ▼
       Evaluation
             │
     ┌───────┼────────┐
     ▼       ▼        ▼
   BLEU    METEOR    COMET
```

---

# 18. BLEU Module

### Input

- Generated translations
- Reference translations

### Output

```text
BLEU Score
```

BLEU will primarily measure n-gram overlap.

---

# 19. METEOR Module

### Input

- Generated translations
- Reference translations

### Output

```text
METEOR Score
```

It provides additional word-level alignment information.

---

# 20. COMET Module

### Input

- Source sentence
- Generated translation
- Reference translation

### Output

```text
COMET Score
```

COMET will provide semantic translation evaluation.

---

# 21. Error Analysis Module

The Error Analysis Module will identify and categorize translation errors.

### Categories

```text
Grammar
Word Order
Named Entity
Ambiguity
Missing Words
Extra Words
Repetition
Untranslated Terms
Long Sentence Failure
```

The module can initially use rule-based analysis and manual annotation.

Automated error classification is optional.

---

# 22. Model Comparison Module

The Model Comparison Module combines results from all experiments.

Example output:

| Model    | Language Pair | BLEU | METEOR | COMET | Inference Time |
| -------- | ------------- | ---: | -----: | ----: | -------------: |
| MarianMT | EN → HI       |    X |      X |     X |              X |
| mBART    | EN → HI       |    X |      X |     X |              X |
| NLLB     | EN → HI       |    X |      X |     X |              X |
| MarianMT | EN → MR       |    X |      X |     X |              X |
| mBART    | EN → MR       |    X |      X |     X |              X |
| NLLB     | EN → MR       |    X |      X |     X |              X |

The actual values will be generated after experiments.

---

# 23. Model Selection Logic

The model selection process should not rely on one metric.

Conceptually:

```text
                 Evaluation Results
                        │
       ┌────────────────┼────────────────┐
       ▼                ▼                ▼
      BLEU           METEOR           COMET
       │                │                │
       └────────────────┼────────────────┘
                        ▼
               Qualitative Analysis
                        │
                        ▼
             Computational Analysis
                        │
                        ▼
                 Final Selection
```

COMET should generally receive strong consideration because it is designed for machine translation evaluation, while BLEU and METEOR provide complementary information.

---

# 24. Inference Engine

The Inference Engine is responsible for generating translations from user input.

### Input

```text
Source Text
Source Language
Target Language
Model
```

### Process

```text
Input Text
    ↓
Tokenizer
    ↓
Language Configuration
    ↓
GPU
    ↓
Transformer
    ↓
Beam Search / Generation
    ↓
Tokenizer Decode
    ↓
Translation
```

---

# 25. Translation Generation

The system may use beam search for improved generation quality.

Conceptually:

```text
Input
  ↓
Encoder
  ↓
Decoder
  ↓
Candidate 1 ──┐
Candidate 2 ──┤
Candidate 3 ──┤
Candidate 4 ──┘
       ↓
Best Sequence
       ↓
Translation
```

Generation parameters should be configurable.

Potential parameters:

```text
num_beams
max_new_tokens
length_penalty
no_repeat_ngram_size
```

---

# 26. Streamlit Application Architecture

```text
┌─────────────────────────────────────────┐
│             STREAMLIT UI                │
│                                         │
│  Source Language: [English ▼]           │
│  Target Language: [Marathi ▼]           │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │ Enter text...                     │  │
│  └───────────────────────────────────┘  │
│                                         │
│          [ Translate ]                  │
│                                         │
│  Translation:                           │
│  ┌───────────────────────────────────┐  │
│  │ Translated text                   │  │
│  └───────────────────────────────────┘  │
└────────────────────┬────────────────────┘
                     │
                     ▼
              Inference Engine
                     │
                     ▼
              Selected Model
                     │
                     ▼
                  GPU/CPU
```

---

# 27. Streamlit Application Components

### Input Component

Accepts source text.

### Language Selector

Provides supported languages.

### Translation Button

Triggers inference.

### Output Component

Displays generated translation.

### Optional Components

- Model selector
- Example input
- Translation history
- Copy button
- Word count
- Execution time

---

# 28. Application Model Loading Strategy

Models should not be repeatedly loaded for every translation.

Bad:

```text
User clicks Translate
       ↓
Load model
       ↓
Translate
       ↓
Delete model
```

Better:

```text
Application Start
       ↓
Load Selected Model
       ↓
Keep in Memory
       ↓
Multiple Translations
```

Streamlit caching mechanisms should be used to prevent unnecessary model reloading.

Because the GPU has 8 GB VRAM, the application should avoid keeping multiple large models on the GPU simultaneously.

---

# 29. Configuration Management

Experiment settings should not be hardcoded throughout the project.

Example:

```text
config/
├── marianmt.yaml
├── mbart.yaml
└── nllb.yaml
```

Configuration may include:

```text
model_name
source_language
target_language
learning_rate
batch_size
gradient_accumulation
max_length
epochs
fp16
gradient_checkpointing
```

This makes experiments easier to reproduce.

---

# 30. Logging

Each experiment should record:

```text
Experiment ID
Model
Dataset
Language Pair
Epoch
Training Loss
Validation Loss
Learning Rate
Batch Size
Sequence Length
GPU
Evaluation Metrics
```

Example:

```text
Experiment: NLLB_EN_MR_001
Model: NLLB
Language Pair: EN → MR
Batch Size: 2
Gradient Accumulation: 4
Epochs: 3
BLEU: X
METEOR: X
COMET: X
```

---

# 31. Experiment Reproducibility

A fixed random seed should be used where possible.

Example:

```text
Seed = 42
```

The project should record:

- Dataset version
- Model version
- Python version
- PyTorch version
- Transformers version
- CUDA version
- Hyperparameters

This makes the experiments reproducible.

---

# 32. Data Flow

## Training Data Flow

```text
Dataset
   ↓
Cleaning
   ↓
Filtering
   ↓
Splitting
   ↓
Tokenization
   ↓
DataLoader
   ↓
Transformer
   ↓
Loss
   ↓
Backpropagation
   ↓
Optimizer
   ↓
Checkpoint
```

## Evaluation Data Flow

```text
Test Dataset
   ↓
Tokenizer
   ↓
Transformer
   ↓
Generated Translation
   ↓
BLEU
METEOR
COMET
   ↓
Results
```

## Application Data Flow

```text
User
 ↓
Streamlit
 ↓
Input Validation
 ↓
Tokenizer
 ↓
Transformer
 ↓
Generation
 ↓
Decode
 ↓
Output
```

---

# 33. Storage Architecture

The 1 TB SSD will be organized approximately as:

```text
Project/
│
├── datasets/
├── pretrained_models/
├── fine_tuned_models/
├── checkpoints/
├── results/
├── logs/
└── application/
```

Large raw datasets should be stored separately from processed datasets where possible.

---

# 34. Model Storage Strategy

Pretrained and fine-tuned models should be separated.

```text
models/
│
├── pretrained/
│   ├── marianmt/
│   ├── mbart/
│   └── nllb/
│
└── fine_tuned/
    ├── marianmt/
    ├── mbart/
    └── nllb/
```

This prevents accidental overwriting of the original pretrained models.

---

# 35. Project Directory Structure

```text
multilingual-nmt/
│
├── app/
│   ├── app.py
│   ├── ui.py
│   └── components.py
│
├── configs/
│   ├── marianmt.yaml
│   ├── mbart.yaml
│   └── nllb.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── test/
│
├── models/
│   ├── pretrained/
│   └── fine_tuned/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_marianmt.ipynb
│   ├── 04_mbart.ipynb
│   ├── 05_nllb.ipynb
│   └── 06_evaluation.ipynb
│
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   ├── cleaner.py
│   │   └── splitter.py
│   │
│   ├── models/
│   │   ├── base.py
│   │   ├── marianmt.py
│   │   ├── mbart.py
│   │   └── nllb.py
│   │
│   ├── training/
│   │   ├── trainer.py
│   │   ├── checkpoint.py
│   │   └── callbacks.py
│   │
│   ├── evaluation/
│   │   ├── bleu.py
│   │   ├── meteor.py
│   │   ├── comet.py
│   │   ├── error_analysis.py
│   │   └── comparison.py
│   │
│   ├── inference/
│   │   └── translator.py
│   │
│   └── utils/
│       ├── config.py
│       ├── logging.py
│       └── seed.py
│
├── results/
│   ├── metrics/
│   ├── plots/
│   └── error_analysis/
│
├── logs/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 36. Dependency Stack

The primary dependencies will include:

```text
Python
PyTorch
Transformers
Datasets
Tokenizers
Evaluate
SentencePiece
SacreBLEU
NLTK
COMET
Pandas
NumPy
Matplotlib
Streamlit
PyYAML
```

Exact package versions will be pinned after the working environment is established.

---

# 37. GPU Resource Management

The system should monitor GPU memory during training.

Conceptual resource flow:

```text
RTX 5050
8 GB VRAM
    │
    ├── Model Weights
    ├── Activations
    ├── Gradients
    ├── Optimizer States
    └── CUDA Overhead
```

The total must remain below available GPU memory.

If CUDA Out-of-Memory occurs:

```text
Reduce Batch Size
       ↓
Increase Gradient Accumulation
       ↓
Reduce Sequence Length
       ↓
Enable Gradient Checkpointing
       ↓
Use Smaller Model
       ↓
Consider PEFT
```

---

# 38. Training Resource Strategy

The project will use a staged training approach.

### Stage 1

```text
1,000–5,000 samples
```

Purpose:

- Validate pipeline
- Detect implementation errors
- Measure VRAM usage

### Stage 2

```text
10,000–30,000 samples
```

Purpose:

- Main experiments
- Model comparison

### Stage 3

Increase dataset size only if:

- Training is stable
- GPU memory is sufficient
- Performance improves
- Training time remains practical

---

# 39. Failure Handling

## CUDA Out-of-Memory

Actions:

1. Reduce batch size.
2. Increase gradient accumulation.
3. Reduce sequence length.
4. Enable gradient checkpointing.
5. Use a smaller model.
6. Use parameter-efficient fine-tuning.

---

## Dataset Corruption

Actions:

1. Validate dataset schema.
2. Remove malformed rows.
3. Check language-pair consistency.
4. Re-run preprocessing.

---

## Invalid Language Code

Actions:

1. Validate language code.
2. Check model-supported language codes.
3. Stop inference with a meaningful error message.

---

## Model Loading Failure

Actions:

1. Verify model path.
2. Verify tokenizer compatibility.
3. Verify Hugging Face model availability.
4. Fall back to CPU only for debugging if necessary.

---

# 40. Security and Privacy

The initial project does not require personal user data.

The application should:

- Process user input locally where possible.
- Avoid storing translation input by default.
- Avoid logging sensitive user text.
- Avoid uploading user input to external APIs.

The final deployed system should clearly define whether translation requests remain local or are sent to external services.

---

# 41. Performance Requirements

### Training

The training pipeline should:

- Utilize GPU acceleration.
- Use mixed precision where appropriate.
- Avoid unnecessary CPU-GPU transfers.
- Use efficient batching.

### Inference

The application should:

- Load the model once.
- Reuse the model between requests.
- Avoid unnecessary tokenizer/model initialization.
- Use GPU inference when available.

---

# 42. Testing Strategy

## Unit Testing

Test individual components:

- Dataset loader
- Cleaner
- Tokenizer
- Model loader
- Translation function
- Metric calculation

## Integration Testing

Test:

```text
Dataset
 ↓
Preprocessing
 ↓
Tokenizer
 ↓
Model
 ↓
Translation
 ↓
Evaluation
```

## Application Testing

Test:

- Valid input
- Empty input
- Unsupported language pair
- Long input
- Special characters
- Multilingual Unicode text
- Model loading failures

---

# 43. Experimental Validation

Each experiment should produce:

```text
experiment/
├── config
├── logs
├── checkpoint
├── metrics
├── predictions
└── analysis
```

Example:

```text
results/
└── nllb_en_mr/
    ├── predictions.csv
    ├── metrics.json
    ├── training_log.csv
    └── error_analysis.csv
```

---

# 44. Results Visualization

The project should generate visualizations such as:

### Model Comparison

```text
Model
 │
 ├── BLEU
 ├── METEOR
 └── COMET
```

### Language Comparison

```text
English → Hindi
English → Marathi
English → French
English → German
```

### Sentence-Length Analysis

```text
Short
Medium
Long
Very Long
```

### Computational Comparison

```text
Model
 │
 ├── VRAM
 ├── Inference Time
 └── Model Size
```

---

# 45. Deployment Architecture

The final deployment will be:

```text
                  User
                   │
                   ▼
            Streamlit Web UI
                   │
                   ▼
           Inference Controller
                   │
                   ▼
          Cached Transformer
                   │
                   ▼
              GPU / CUDA
                   │
                   ▼
             Translation
                   │
                   ▼
             Streamlit UI
```

For the first version, deployment will be local.

Future versions may use:

```text
Streamlit
     ↓
FastAPI
     ↓
Model Server
     ↓
GPU
```

if a production-style architecture is required.

---

# 46. Deployment Model

The first implementation will use:

```text
Local Machine
     │
     ├── PyTorch
     ├── CUDA
     ├── Transformer
     └── Streamlit
```

The application can later be deployed to cloud infrastructure if required.

---

# 47. API-Like Interface

Although Streamlit will be the primary frontend, the inference engine should expose a clean internal interface.

Conceptual request:

```json
{
    "source_language": "en",
    "target_language": "mr",
    "text": "Artificial intelligence is transforming the world."
}
```

Conceptual response:

```json
{
    "source_language": "en",
    "target_language": "mr",
    "translation": "कृत्रिम बुद्धिमत्ता जग बदलत आहे.",
    "model": "NLLB"
}
```

This abstraction makes future FastAPI deployment easier.

---

# 48. End-to-End System Flow

```text
                       START
                         │
                         ▼
                 Load Configuration
                         │
                         ▼
                   Load Dataset
                         │
                         ▼
                  Clean Dataset
                         │
                         ▼
                 Split Dataset
                         │
                         ▼
                    Tokenize
                         │
                         ▼
              Select Transformer
                         │
                         ▼
                Load Pretrained
                     Model
                         │
                         ▼
              Configure GPU/AMP
                         │
                         ▼
                   Fine-Tune
                         │
                         ▼
                  Validate Model
                         │
                         ▼
                Save Checkpoint
                         │
                         ▼
                  Test Dataset
                         │
                         ▼
                Generate Output
                         │
                         ▼
              BLEU/METEOR/COMET
                         │
                         ▼
              Compare All Models
                         │
                         ▼
                Error Analysis
                         │
                         ▼
               Select Best Model
                         │
                         ▼
                Streamlit Deploy
                         │
                         ▼
                       END
```

---

# 49. Non-Functional Requirements

## Performance

The system should use GPU acceleration efficiently.

## Reliability

The application should handle invalid inputs without crashing.

## Maintainability

Training and inference components should remain independent.

## Reproducibility

Experiments should be configurable and logged.

## Scalability

The architecture should allow additional models and languages to be added.

## Usability

The Streamlit application should provide a simple translation workflow.

---

# 50. Technical Risks

| Risk                     | Impact | Mitigation                      |
| ------------------------ | ------ | ------------------------------- |
| CUDA OOM                 | High   | Small batch, AMP, checkpointing |
| Large model              | High   | Smaller variant / PEFT          |
| Dataset too large        | Medium | Sampling                        |
| Poor Marathi performance | Medium | Better dataset / fine-tuning    |
| Long training time       | Medium | Staged experiments              |
| Metric disagreement      | Medium | Multiple metrics                |
| Model loading time       | Medium | Streamlit caching               |
| Dataset quality          | High   | Cleaning and validation         |
| Overfitting              | Medium | Validation + early stopping     |

---

# 51. Recommended Development Phases

## Phase 1 — Environment Setup

```text
Python
PyTorch
CUDA
Transformers
Datasets
```

Verify:

```text
torch.cuda.is_available()
```

---

## Phase 2 — Dataset Pipeline

Implement:

- Loading
- Cleaning
- Splitting
- Tokenization

---

## Phase 3 — MarianMT Baseline

Implement:

- Model loading
- Fine-tuning
- Evaluation
- Inference

---

## Phase 4 — mBART

Implement:

- Multilingual tokenizer
- Language configuration
- Fine-tuning
- Evaluation

---

## Phase 5 — NLLB

Implement:

- Suitable NLLB variant
- Language-code handling
- Fine-tuning / PEFT if required
- Evaluation

---

## Phase 6 — Comparative Evaluation

Generate:

- BLEU
- METEOR
- COMET
- Inference time
- VRAM usage
- Model size

---

## Phase 7 — Error Analysis

Analyze:

- Grammar
- Word order
- Named entities
- Ambiguity
- Long sentences
- Low-resource translation

---

## Phase 8 — Application

Develop Streamlit UI.

---

## Phase 9 — Final Validation

Perform:

- Unit testing
- Integration testing
- Application testing
- Final experiments

---

# 52. Definition of Done

The project will be considered technically complete when:

- [ ] Dataset pipeline works.
- [ ] Train/validation/test separation is implemented.
- [ ] MarianMT baseline is functional.
- [ ] mBART experiment is functional.
- [ ] NLLB experiment is functional.
- [ ] GPU training works on RTX 5050.
- [ ] Memory-efficient training is implemented.
- [ ] Checkpoints can be saved and restored.
- [ ] BLEU evaluation works.
- [ ] METEOR evaluation works.
- [ ] COMET evaluation works.
- [ ] Model comparison is generated.
- [ ] Error analysis is completed.
- [ ] Sentence-length experiment is completed.
- [ ] Best model is selected.
- [ ] Streamlit application works.
- [ ] Application performs GPU inference.
- [ ] README contains setup instructions.
- [ ] Experiments are reproducible.

---

# 53. Final Technical Architecture

```text
                           ┌───────────────────┐
                           │      DATASETS     │
                           │                   │
                           │ IITB │ OPUS │ WMT│
                           └─────────┬─────────┘
                                     │
                                     ▼
                           ┌───────────────────┐
                           │ DATA PREPROCESSOR │
                           └─────────┬─────────┘
                                     │
                                     ▼
                           ┌───────────────────┐
                           │ DATASET SPLITTER  │
                           └─────────┬─────────┘
                                     │
                                     ▼
                           ┌───────────────────┐
                           │    TOKENIZER      │
                           └─────────┬─────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
       ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
       │  MarianMT   │       │   mBART-50  │       │    NLLB     │
       │  Baseline   │       │ Multilingual │       │   Primary   │
       └──────┬──────┘       └──────┬──────┘       └──────┬──────┘
              │                     │                     │
              └─────────────────────┼─────────────────────┘
                                    │
                                    ▼
                           ┌───────────────────┐
                           │ TRAINING ENGINE   │
                           │                   │
                           │ PyTorch           │
                           │ CUDA              │
                           │ AMP               │
                           │ AdamW             │
                           │ Gradient Accum.   │
                           └─────────┬─────────┘
                                     │
                                     ▼
                           ┌───────────────────┐
                           │   CHECKPOINTS     │
                           └─────────┬─────────┘
                                     │
                                     ▼
                           ┌───────────────────┐
                           │ EVALUATION ENGINE │
                           │                   │
                           │ BLEU              │
                           │ METEOR            │
                           │ COMET             │
                           └─────────┬─────────┘
                                     │
                                     ▼
                           ┌───────────────────┐
                           │ MODEL COMPARISON  │
                           └─────────┬─────────┘
                                     │
                                     ▼
                           ┌───────────────────┐
                           │  ERROR ANALYSIS   │
                           └─────────┬─────────┘
                                     │
                                     ▼
                           ┌───────────────────┐
                           │    BEST MODEL     │
                           └─────────┬─────────┘
                                     │
                                     ▼
                           ┌───────────────────┐
                           │ INFERENCE ENGINE  │
                           └─────────┬─────────┘
                                     │
                                     ▼
                           ┌───────────────────┐
                           │    STREAMLIT      │
                           │    APPLICATION    │
                           └───────────────────┘
```

---

# 54. Final Technical Design Summary

The proposed system uses a **modular Transformer-based architecture** where data processing, model training, evaluation, and inference are separated into independent components.

The system will use **PyTorch and Hugging Face Transformers** and will be optimized for the project's **NVIDIA RTX 5050 8 GB GPU** using mixed precision, gradient accumulation, controlled sequence lengths, and gradient checkpointing where necessary.

Three Transformer architectures—**MarianMT, mBART-50, and NLLB**—will be compared using **BLEU, METEOR, and COMET**, supplemented with qualitative and error analysis.

The final selected model will be exposed through a **Streamlit application**, while the underlying inference engine will remain independent from the UI so that the system can later be migrated to a REST API or production model-serving architecture.

The architecture prioritizes:

```text
Modularity
+
Reproducibility
+
GPU Efficiency
+
Model Comparability
+
Evaluation Rigor
+
Deployment Simplicity
```

The result is a complete Deep Learning pipeline rather than simply a pretrained translation demo.


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
