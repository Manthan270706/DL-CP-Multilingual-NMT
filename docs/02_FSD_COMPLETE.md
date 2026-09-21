# Functional Specification Document (FSD)

# Multilingual Neural Machine Translation Using Transformer-Based Deep Learning Models

**Version:** 1.0  
**Status:** Proposed  
**Project Type:** Deep Learning Course Project  
**Primary Framework:** PyTorch  
**NLP Framework:** Hugging Face Transformers  
**Deployment:** Streamlit  
**Hardware:** NVIDIA RTX 5050 Laptop GPU — 8 GB VRAM, 16 GB RAM, 1 TB SSD

---

# 1. Document Purpose

This Functional Specification Document defines the functional behavior of the **Multilingual Neural Machine Translation System**.

The document specifies:

- System functionality
- User interactions
- Data processing behavior
- Model training workflow
- Model evaluation workflow
- Translation workflow
- Error handling
- Functional requirements
- Non-functional requirements
- System acceptance criteria

The FSD is derived from the project's PRD and TDD and serves as the implementation-level functional specification.

---

# 2. System Overview

The system is a Transformer-based Neural Machine Translation platform capable of translating text between English and selected target languages.

### Supported Languages

- English
- Hindi
- Marathi
- French
- German

### Transformer Models

- MarianMT
- mBART-50
- NLLB

The system has two major functional areas:

```text
┌──────────────────────────────────────┐
│      RESEARCH / TRAINING SYSTEM      │
│                                      │
│ Dataset → Preprocessing → Training  │
│ → Evaluation → Comparison            │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│        TRANSLATION APPLICATION       │
│                                      │
│ Input → Model → Translation → Output │
└──────────────────────────────────────┘
```

---

# 3. User Types

The system primarily has two user roles.

## 3.1 Researcher / Developer

Responsible for:

- Dataset preparation
- Model configuration
- Training
- Evaluation
- Experiment comparison
- Error analysis
- Model selection

## 3.2 End User

Responsible for:

- Selecting source language
- Selecting target language
- Entering text
- Requesting translation
- Viewing translated output

---

# 4. Functional Scope

## 4.1 Research Functions

The system shall provide functionality for:

1. Dataset loading
2. Dataset cleaning
3. Dataset validation
4. Dataset splitting
5. Tokenization
6. Model loading
7. Model fine-tuning
8. Validation
9. Checkpointing
10. Model evaluation
11. Model comparison
12. Error analysis
13. Sentence-length analysis
14. Experiment logging

## 4.2 Application Functions

The application shall provide:

1. Language selection
2. Text input
3. Input validation
4. Model selection where enabled
5. Translation generation
6. Output display
7. Translation reset
8. Optional execution-time display

---

# 5. Supported Translation Pairs

The initial application shall support:

| Source | Target |
|---|---|
| English | Hindi |
| English | Marathi |
| English | French |
| English | German |

The architecture should allow additional language pairs to be added later.

The application should not claim support for a language pair unless the selected model has been configured and tested for that pair.

---

# 6. Functional Architecture

```text
                    ┌────────────────────┐
                    │       USER         │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │    STREAMLIT UI    │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ INPUT VALIDATION   │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ INFERENCE ENGINE   │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ MODEL MANAGER      │
                    └─────────┬──────────┘
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
        ┌─────────┐      ┌─────────┐      ┌─────────┐
        │MarianMT │      │  mBART  │      │  NLLB   │
        └─────────┘      └─────────┘      └─────────┘
             │                │                │
             └────────────────┼────────────────┘
                              ▼
                    ┌────────────────────┐
                    │ TRANSLATION OUTPUT │
                    └────────────────────┘
```

---

# 7. Functional Module Specifications

# 7.1 Dataset Management Module

## Purpose

Load and organize parallel translation datasets.

## Supported Sources

- IIT Bombay English-Hindi Corpus
- OPUS
- Selected WMT datasets

## Input

Raw dataset files or Hugging Face datasets.

## Processing

The module shall:

1. Load dataset.
2. Identify source and target columns.
3. Validate required fields.
4. Convert records into a common format.
5. Store the processed dataset.

## Output

```text
source_language
target_language
source_text
target_text
```

---

# 7.2 Data Cleaning Module

## Purpose

Remove invalid or unsuitable training samples.

## Functional Requirements

The module shall:

- Remove null records.
- Remove duplicate sentence pairs.
- Remove empty strings.
- Normalize text.
- Remove malformed records.
- Filter extremely long sequences.
- Validate language-pair consistency.

## Processing Flow

```text
Raw Dataset
     ↓
Check Missing Values
     ↓
Remove Duplicates
     ↓
Normalize Text
     ↓
Validate Language Pair
     ↓
Filter Length
     ↓
Clean Dataset
```

---

# 7.3 Dataset Splitting Module

The system shall divide the cleaned dataset into:

```text
Training      → 80%
Validation    → 10%
Testing       → 10%
```

## Requirements

- Random seed shall be configurable.
- Test data shall remain isolated.
- No test samples shall be used during training.
- Validation data shall be used for model monitoring.
- Dataset split information shall be logged.

---

# 7.4 Tokenization Module

## Purpose

Convert natural-language sentences into model-compatible tokens.

## Supported Tokenizers

```text
MarianTokenizer
MBartTokenizer / MBart50Tokenizer
NLLB tokenizer
```

## Input

```text
Source sentence
Target sentence
```

## Output

```text
input_ids
attention_mask
labels
```

## Functional Behavior

The tokenizer shall:

1. Load the appropriate tokenizer.
2. Apply required language codes.
3. Tokenize source text.
4. Tokenize target text.
5. Apply truncation.
6. Apply dynamic padding where appropriate.
7. Return tensors suitable for model training.

---

# 7.5 Model Management Module

## Purpose

Manage pretrained and fine-tuned Transformer models.

## Supported Models

```text
MarianMT
mBART-50
NLLB
```

## Functions

### Load Model

Input:

```text
Model name
Model path
Language configuration
```

Output:

```text
Loaded Transformer model
```

### Load Checkpoint

Input:

```text
Checkpoint path
```

Output:

```text
Fine-tuned model
```

### Move Model to Device

The module shall determine whether CUDA is available.

```text
CUDA Available
      │
     Yes
      ↓
    GPU

CUDA Not Available
      │
      ↓
    CPU
```

GPU execution shall be preferred for training and inference.

---

# 7.6 Training Module

## Purpose

Fine-tune pretrained Transformer models.

## Input

- Training dataset
- Validation dataset
- Pretrained model
- Tokenizer
- Training configuration

## Configuration

The module shall support:

- Learning rate
- Batch size
- Number of epochs
- Maximum sequence length
- Gradient accumulation
- Mixed precision
- Gradient checkpointing
- Random seed

## Training Process

```text
Load Model
    ↓
Load Dataset
    ↓
Tokenize
    ↓
Create DataLoader
    ↓
Move Batch to GPU
    ↓
Forward Pass
    ↓
Calculate Loss
    ↓
Backpropagation
    ↓
Gradient Accumulation
    ↓
Optimizer Update
    ↓
Validation
    ↓
Checkpoint
```

---

# 7.7 Mixed Precision Module

The system shall support mixed-precision training to reduce GPU memory consumption.

The default configuration should attempt to use mixed precision where supported.

## Purpose

Reduce:

- GPU memory usage
- Training overhead

while maintaining acceptable numerical stability.

---

# 7.8 Gradient Accumulation Module

The system shall support gradient accumulation.

Example:

```text
Batch Size = 2
Accumulation Steps = 4

Effective Batch Size ≈ 8
```

The optimizer shall update model weights only after the configured number of accumulation steps.

---

# 7.9 Gradient Checkpointing Module

Gradient checkpointing shall be optionally enabled for memory-intensive models.

It shall be enabled when:

- GPU memory is insufficient.
- The selected model requires additional memory optimization.

---

# 7.10 Checkpoint Management Module

## Purpose

Save and restore training state.

The module shall save:

- Model weights
- Tokenizer
- Training configuration
- Epoch
- Training step
- Validation loss
- Relevant optimizer state

## Checkpoint Naming

Example:

```text
nllb_en_mr_epoch3_step1200
```

## Functional Requirements

The system shall:

- Save checkpoints periodically.
- Identify the best validation checkpoint.
- Allow checkpoint restoration.
- Prevent accidental overwriting of unrelated experiments.

---

# 7.11 Evaluation Module

The system shall evaluate models using:

- BLEU
- METEOR
- COMET

## Input

```text
Source sentences
Reference translations
Generated translations
```

## Output

```text
Metric name
Metric score
Language pair
Model
Experiment ID
```

---

# 7.12 BLEU Evaluation

The system shall calculate BLEU using generated and reference translations.

Example result:

```text
Model: NLLB
Language Pair: EN → MR
BLEU: XX.XX
```

The exact score shall be generated from the experiment.

---

# 7.13 METEOR Evaluation

The system shall calculate METEOR using generated and reference translations.

The resulting score shall be stored with the experiment results.

---

# 7.14 COMET Evaluation

COMET evaluation shall use:

```text
Source
+
Generated Translation
+
Reference Translation
```

The resulting COMET score shall be stored with the experiment.

Because COMET models can consume significant computational resources, evaluation may be performed after model inference rather than during every training step.

---

# 7.15 Model Comparison Module

## Purpose

Compare trained models under consistent evaluation conditions.

The module shall compare:

- BLEU
- METEOR
- COMET
- Inference time
- Model size
- GPU memory usage where measurable

## Example

| Model | Pair | BLEU | METEOR | COMET | Inference Time |
|---|---|---:|---:|---:|---:|
| MarianMT | EN→HI | X | X | X | X |
| mBART | EN→HI | X | X | X | X |
| NLLB | EN→HI | X | X | X | X |

---

# 7.16 Error Analysis Module

## Purpose

Analyze translation failures.

The module shall classify observed errors into:

1. Grammar errors
2. Word-order errors
3. Named-entity errors
4. Missing words
5. Extra words
6. Repetition
7. Untranslated words
8. Ambiguous-word errors
9. Long-sentence failures

## Output

An error-analysis dataset such as:

| Input | Reference | Prediction | Model | Error Type |
|---|---|---|---|---|
| Example | Reference | Output | NLLB | Word Order |

Manual annotation shall be acceptable for the initial implementation.

---

# 7.17 Sentence-Length Analysis Module

The system shall divide test samples into:

```text
Short       → 1–10 words
Medium      → 11–25 words
Long        → 26–50 words
Very Long   → 50+ words
```

Metrics shall be calculated separately when sufficient samples are available.

## Purpose

Determine whether translation quality decreases as sentence complexity increases.

---

# 7.18 Inference Module

## Purpose

Generate translations for user-provided text.

## Input

```text
Source Language
Target Language
Source Text
Model
```

## Output

```text
Translated Text
```

## Processing

```text
User Text
   ↓
Input Validation
   ↓
Language Configuration
   ↓
Tokenizer
   ↓
Transformer
   ↓
Text Generation
   ↓
Decode Tokens
   ↓
Translation
```

---

# 8. Translation Generation Requirements

The inference module shall support configurable generation parameters.

### Parameters

```text
num_beams
max_new_tokens
length_penalty
no_repeat_ngram_size
```

Default values shall be selected during experimentation.

The final application shall use the configuration validated during model evaluation.

---

# 9. Streamlit User Interface

# 9.1 Main Screen

The application shall contain:

```text
Multilingual Translation System

Source Language: [ English ▼ ]

Target Language: [ Marathi ▼ ]

┌─────────────────────────────────────────┐
│ Enter text here...                      │
└─────────────────────────────────────────┘

              [ Translate ]

Translation:

┌─────────────────────────────────────────┐
│ Generated translation                   │
└─────────────────────────────────────────┘
```

---

# 9.2 Source Language Selection

The user shall select a source language from the supported language list.

Initial options:

```text
English
Hindi
Marathi
French
German
```

However, the application shall only enable language combinations supported by the currently loaded model.

---

# 9.3 Target Language Selection

The user shall select the desired target language.

The target language shall not be allowed to be identical to the source language unless explicitly supported.

---

# 9.4 Text Input

The system shall accept multilingual Unicode text.

Examples:

```text
English
Hindi
मराठी
Français
Deutsch
```

The input area shall support multi-word and multi-sentence text.

---

# 9.5 Input Validation

Before translation, the system shall verify:

### Empty Input

If no text is entered:

```text
Please enter text to translate.
```

### Unsupported Language Pair

```text
Selected language pair is not supported by this model.
```

### Excessively Long Input

The system shall either:

- Reject the input with a meaningful message, or
- Process it using controlled truncation/chunking.

The behavior shall be documented in the final application.

---

# 9.6 Translation Button

When the user clicks **Translate**:

1. Validate input.
2. Validate language pair.
3. Load/retrieve cached model.
4. Tokenize input.
5. Perform inference.
6. Decode output.
7. Display translation.

---

# 9.7 Output Display

The translated sentence shall be displayed clearly.

Optional information:

```text
Model: NLLB
Source: English
Target: Marathi
Inference Time: X seconds
```

---

# 9.8 Clear Button

A Clear/Reset function may be provided.

It shall:

- Remove source text.
- Remove translation output.
- Reset optional status information.

---

# 10. Model Loading Behavior

The application shall avoid loading the model for every translation.

## Required Behavior

```text
Application Start
       ↓
Load Model
       ↓
Cache Model
       ↓
User Translation Requests
       ↓
Reuse Model
```

Only one large model should normally occupy GPU memory at a time.

---

# 11. Model Selection Behavior

Two modes may be supported.

## Mode A — Best Model

The application automatically uses the final selected best-performing model.

This is the recommended default for the final demonstration.

## Mode B — Model Comparison

Optional research mode allowing the user to select:

```text
MarianMT
mBART-50
NLLB
```

This mode is useful for demonstrating the project's comparative research component.

---

# 12. Application State

The application may maintain:

```text
Current Model
Source Language
Target Language
Input Text
Translation Output
Inference Time
```

The application shall not permanently store user translation text by default.

---

# 13. Training Configuration

Training configuration shall be externally configurable.

Example:

```yaml
model_name: nllb
source_language: eng_Latn
target_language: mar_Deva

batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 2e-5
epochs: 3

max_length: 256

fp16: true
gradient_checkpointing: true

seed: 42
```

Actual values will be adjusted based on hardware and experiments.

---

# 14. Experiment Management

Every training run shall receive a unique experiment identifier.

Example:

```text
NLLB_EN_MR_001
```

The system shall store:

```text
Experiment ID
Model
Dataset
Language Pair
Hyperparameters
Training Metrics
Validation Metrics
Test Metrics
Checkpoint
```

---

# 15. Experiment Result Format

Example:

```json
{
    "experiment_id": "NLLB_EN_MR_001",
    "model": "NLLB",
    "language_pair": "EN-MR",
    "epochs": 3,
    "batch_size": 2,
    "learning_rate": 0.00002,
    "bleu": 0.0,
    "meteor": 0.0,
    "comet": 0.0
}
```

Actual metric values will be populated after evaluation.

---

# 16. Dataset Requirements

Each training record must contain:

```text
source_text
target_text
source_language
target_language
```

Invalid records shall not enter the training pipeline.

---

# 17. Data Quality Requirements

The dataset should satisfy:

- No empty source sentences.
- No empty target sentences.
- No duplicate pairs.
- Correct language pairing.
- Reasonable sentence lengths.
- Valid Unicode encoding.
- No corrupted records.

---

# 18. Model Training Acceptance Criteria

A model training experiment is considered successful when:

- Dataset loads successfully.
- Tokenization completes.
- Model loads onto the GPU.
- At least one training epoch completes.
- Validation completes.
- Loss values are recorded.
- Checkpoint is created.
- Model can be reloaded.
- Model can generate translations.

---

# 19. Evaluation Acceptance Criteria

An evaluation experiment is successful when:

- Test dataset loads.
- Model generates predictions.
- BLEU is calculated.
- METEOR is calculated.
- COMET is calculated.
- Results are saved.
- Results can be compared against other models.

---

# 20. Translation Acceptance Criteria

A translation request is successful when:

1. Valid source text is provided.
2. Valid source language is selected.
3. Valid target language is selected.
4. Model supports the selected pair.
5. Model generates output.
6. Output is displayed without application failure.

---

# 21. Error Handling Requirements

## 21.1 CUDA Out-of-Memory

The system should recommend or automatically attempt:

```text
Reduce Batch Size
        ↓
Reduce Sequence Length
        ↓
Enable Gradient Checkpointing
        ↓
Use Smaller Model
        ↓
Use PEFT
```

---

## 21.2 Model Loading Error

Display:

```text
Unable to load the selected translation model.
Please verify the model configuration or checkpoint.
```

Detailed technical information should be recorded in logs rather than shown to the end user.

---

## 21.3 Dataset Error

Display/log:

```text
Dataset validation failed.
Please check the dataset format and required fields.
```

---

## 21.4 Unsupported Language

Display:

```text
The selected language pair is not supported by this model.
```

---

## 21.5 Empty Input

Display:

```text
Please enter text to translate.
```

---

# 22. Functional Requirements

## FR-001 — Dataset Loading

The system shall load supported parallel translation datasets.

## FR-002 — Dataset Cleaning

The system shall remove invalid and duplicate records.

## FR-003 — Dataset Splitting

The system shall split data into training, validation, and testing sets.

## FR-004 — Tokenization

The system shall tokenize source and target sentences using model-specific tokenizers.

## FR-005 — Model Loading

The system shall load MarianMT, mBART-50, and NLLB models.

## FR-006 — GPU Acceleration

The system shall use CUDA when available.

## FR-007 — Model Fine-Tuning

The system shall fine-tune supported pretrained models.

## FR-008 — Mixed Precision

The system shall support memory-efficient mixed-precision training.

## FR-009 — Gradient Accumulation

The system shall support gradient accumulation.

## FR-010 — Gradient Checkpointing

The system shall optionally support gradient checkpointing.

## FR-011 — Checkpointing

The system shall save and restore model checkpoints.

## FR-012 — BLEU Evaluation

The system shall calculate BLEU.

## FR-013 — METEOR Evaluation

The system shall calculate METEOR.

## FR-014 — COMET Evaluation

The system shall calculate COMET.

## FR-015 — Model Comparison

The system shall compare translation models.

## FR-016 — Error Analysis

The system shall support categorization of translation errors.

## FR-017 — Sentence-Length Analysis

The system shall evaluate performance across sentence-length categories.

## FR-018 — Text Translation

The application shall translate user-provided text.

## FR-019 — Language Selection

The application shall allow source and target language selection.

## FR-020 — Output Display

The application shall display the generated translation.

## FR-021 — Input Validation

The application shall validate user input before inference.

## FR-022 — Model Caching

The application shall reuse loaded models instead of loading them for every request.

## FR-023 — Experiment Logging

The system shall record experiment configurations and results.

## FR-024 — Reproducibility

The system shall record the configuration required to reproduce experiments.

---

# 23. Non-Functional Requirements

## NFR-001 — Performance

The system should use GPU acceleration whenever available.

## NFR-002 — Memory Efficiency

Training should be optimized for an 8 GB VRAM GPU.

## NFR-003 — Reliability

Invalid inputs should not crash the application.

## NFR-004 — Maintainability

Training, evaluation, and inference should remain modular.

## NFR-005 — Reproducibility

Experiments should be reproducible using recorded configurations.

## NFR-006 — Scalability

Additional language pairs and models should be addable without redesigning the entire system.

## NFR-007 — Usability

The translation interface should require minimal user interaction.

## NFR-008 — Privacy

User translation input should not be permanently stored by default.

---

# 24. User Workflow

## End-User Translation Workflow

```text
START
  │
  ▼
Open Application
  │
  ▼
Select Source Language
  │
  ▼
Select Target Language
  │
  ▼
Enter Text
  │
  ▼
Click Translate
  │
  ▼
Validate Input
  │
  ├── Invalid → Display Error
  │
  ▼
Load Cached Model
  │
  ▼
Tokenize
  │
  ▼
Generate Translation
  │
  ▼
Decode Output
  │
  ▼
Display Translation
  │
  ▼
END
```

---

# 25. Researcher Workflow

```text
START
  │
  ▼
Collect Dataset
  │
  ▼
Clean Dataset
  │
  ▼
Split Dataset
  │
  ▼
Select Model
  │
  ▼
Configure Experiment
  │
  ▼
Fine-Tune
  │
  ▼
Validate
  │
  ▼
Save Checkpoint
  │
  ▼
Evaluate
  │
  ├── BLEU
  ├── METEOR
  └── COMET
  │
  ▼
Error Analysis
  │
  ▼
Model Comparison
  │
  ▼
Select Best Model
  │
  ▼
Deploy
  │
  ▼
END
```

---

# 26. Functional Use Cases

## UC-01: Translate Text

### Actor

End User

### Preconditions

- Application is running.
- Valid model is loaded.
- Selected language pair is supported.

### Main Flow

1. User selects source language.
2. User selects target language.
3. User enters text.
4. User clicks Translate.
5. System validates input.
6. System tokenizes text.
7. Transformer generates translation.
8. System decodes tokens.
9. Translation is displayed.

### Postcondition

Translated text is available to the user.

---

# 27. UC-02: Train Model

### Actor

Researcher

### Preconditions

- Dataset is available.
- Model is available.
- GPU/software environment is configured.

### Main Flow

1. Researcher selects model.
2. Researcher selects dataset.
3. Configuration is loaded.
4. Dataset is tokenized.
5. Training starts.
6. Loss is calculated.
7. Model parameters are updated.
8. Validation is performed.
9. Checkpoint is saved.
10. Training completes.

### Postcondition

Fine-tuned model checkpoint is available.

---

# 28. UC-03: Evaluate Model

### Actor

Researcher

### Main Flow

1. Load trained checkpoint.
2. Load test dataset.
3. Generate translations.
4. Calculate BLEU.
5. Calculate METEOR.
6. Calculate COMET.
7. Save metrics.
8. Generate comparison data.

### Postcondition

Model evaluation results are stored.

---

# 29. UC-04: Compare Models

### Actor

Researcher

### Main Flow

1. Load experiment results.
2. Group results by model.
3. Compare evaluation metrics.
4. Compare computational metrics.
5. Generate comparison table.
6. Identify best-performing model.

---

# 30. UC-05: Analyze Translation Errors

### Actor

Researcher

### Main Flow

1. Load test predictions.
2. Compare predictions with references.
3. Identify representative failures.
4. Categorize errors.
5. Store error classifications.
6. Generate analysis report.

---

# 31. UC-06: Analyze Sentence Length

### Actor

Researcher

### Main Flow

1. Load test dataset.
2. Calculate sentence lengths.
3. Categorize sentences.
4. Generate translations.
5. Calculate metrics by category.
6. Compare results.

---

# 32. Language Validation Rules

The application shall validate:

```text
Source Language ≠ Target Language
```

unless explicitly supported.

The selected model must support the requested language pair.

For multilingual models such as NLLB and mBART, the system must use the appropriate language codes.

Example:

```text
English → Marathi

English:
eng_Latn

Marathi:
mar_Deva
```

Exact language-code mappings shall be maintained in configuration rather than hardcoded throughout the application.

---

# 33. Model-Specific Functional Behavior

## MarianMT

MarianMT models are generally language-pair specific.

Therefore, the application shall load the appropriate MarianMT checkpoint for the requested pair.

Example:

```text
EN → HI
```

requires a compatible English-Hindi MarianMT model.

---

## mBART-50

mBART-50 uses multilingual language identifiers.

The system shall configure the source and target language codes before generation.

---

## NLLB

NLLB shall use the appropriate language codes for source and target languages.

The system shall verify that the selected NLLB checkpoint supports the requested languages.

---

# 34. Resource Management Requirements

The system shall account for:

```text
GPU:
8 GB VRAM

RAM:
16 GB

Storage:
1 TB SSD
```

The system should avoid loading multiple large models into GPU memory simultaneously.

When necessary, models shall be moved between CPU and GPU or loaded only when required.

---

# 35. Training Dataset Scaling

The system shall support staged dataset experimentation.

### Stage 1

```text
1,000–5,000 samples
```

Purpose:

- Pipeline testing
- Debugging
- Memory testing

### Stage 2

```text
10,000–30,000 samples
```

Purpose:

- Main experiments
- Model comparison

### Stage 3

Larger datasets may be used if:

- Training remains stable.
- GPU memory remains sufficient.
- Training time is acceptable.
- Performance improves.

---

# 36. Output Artifacts

The system shall generate:

```text
1. Cleaned datasets
2. Train/validation/test splits
3. Fine-tuned models
4. Model checkpoints
5. Predictions
6. BLEU scores
7. METEOR scores
8. COMET scores
9. Comparison tables
10. Error-analysis results
11. Sentence-length analysis
12. Visualizations
13. Streamlit application
14. Experiment logs
```

---

# 37. Results Storage

Results shall follow a consistent structure:

```text
results/
│
├── metrics/
│   ├── marianmt/
│   ├── mbart/
│   └── nllb/
│
├── predictions/
│
├── error_analysis/
│
└── plots/
```

---

# 38. Acceptance Criteria

The complete project shall satisfy the following:

### Dataset

- [ ] Supported datasets can be loaded.
- [ ] Invalid data is removed.
- [ ] Duplicate records are removed.
- [ ] Dataset is split correctly.

### Models

- [ ] MarianMT can be loaded.
- [ ] mBART-50 can be loaded.
- [ ] NLLB can be loaded.
- [ ] Appropriate language codes are configured.

### Training

- [ ] Models can be fine-tuned.
- [ ] GPU acceleration works.
- [ ] Mixed precision works.
- [ ] Gradient accumulation works.
- [ ] Checkpoints are created.

### Evaluation

- [ ] BLEU works.
- [ ] METEOR works.
- [ ] COMET works.
- [ ] Results are stored.
- [ ] Models can be compared.

### Analysis

- [ ] Error analysis is completed.
- [ ] Sentence-length analysis is completed.
- [ ] Language-pair comparison is completed.

### Application

- [ ] Streamlit launches successfully.
- [ ] User can select languages.
- [ ] User can enter text.
- [ ] User can request translation.
- [ ] Translation is displayed.
- [ ] Invalid inputs are handled.
- [ ] Model is cached.

---

# 39. Definition of Done

The project will be considered functionally complete when:

```text
Dataset
   ↓
Preprocessing
   ↓
Training
   ↓
Validation
   ↓
Checkpoint
   ↓
Evaluation
   ↓
Model Comparison
   ↓
Error Analysis
   ↓
Best Model
   ↓
Streamlit Application
   ↓
Successful Translation
```

can be executed successfully from beginning to end.

---

# 40. Final Functional Specification

The **Multilingual Neural Machine Translation System** shall provide an end-to-end Deep Learning workflow for multilingual translation.

The research pipeline shall allow the project team to collect and preprocess parallel datasets, fine-tune pretrained Transformer architectures, evaluate their performance using BLEU, METEOR, and COMET, analyze translation errors, compare models, and select the best-performing architecture.

The application pipeline shall provide an intuitive Streamlit interface where users can select languages, enter text, and receive Transformer-generated translations.

The system shall be designed around the available **RTX 5050 8 GB VRAM, 16 GB RAM, and 1 TB SSD**, using memory-efficient training techniques where necessary.

The final system should demonstrate not merely that a pretrained Transformer can translate text, but that the project team can **design, train, evaluate, compare, analyze, and deploy a complete Neural Machine Translation system**.

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
