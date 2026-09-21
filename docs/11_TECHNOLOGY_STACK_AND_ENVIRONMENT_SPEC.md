# Technology Stack & Environment Specification
## Multilingual Neural Machine Translation Using Transformer-Based Deep Learning Models

**Document status:** Pre-development / implementation baseline  
**Purpose:** Freeze the technology, environment, tooling, configuration, testing, and application-stack decisions that the AI coding agent must follow before implementation begins.

---

## 1. Purpose

This document defines the technology stack and development-environment contract for the project:

**“Multilingual Neural Machine Translation Using Transformer-Based Deep Learning Models.”**

The project is a deep-learning research and application project focused on English-centric neural machine translation for:

- English → Hindi
- English → Marathi
- English → French
- English → German

The project compares and evaluates:

- MarianMT
- mBART-50
- NLLB-200 distilled 600M

The technology stack must support:

1. deterministic dataset preparation,
2. tokenizer/model loading,
3. transfer learning and fine-tuning,
4. GPU-aware training on an NVIDIA RTX 5050 Laptop GPU with 8 GB VRAM,
5. resource-constrained training strategies,
6. reproducible evaluation,
7. structured experiment artifacts,
8. qualitative/error analysis,
9. final model inference,
10. Streamlit deployment.

This document is an implementation constraint, not a request to introduce additional frameworks merely because they are available.

---

# 2. Technology Stack Summary

| Layer | Selected Technology | Role |
|---|---|---|
| Programming language | Python | Primary implementation language |
| Deep learning framework | PyTorch | Tensor computation, training, inference |
| NLP/model framework | Hugging Face Transformers | MarianMT, mBART, NLLB model/tokenizer integration |
| Dataset framework | Hugging Face Datasets | Dataset loading and dataset abstractions |
| Data processing | Pandas, NumPy | Cleaning, analysis, tabular processing |
| Training orchestration | Transformers Trainer and/or Accelerate | Fine-tuning and device/training management |
| PEFT fallback | Hugging Face PEFT / LoRA | Fallback when full fine-tuning is infeasible |
| BLEU | SacreBLEU or pinned equivalent | Standardized BLEU evaluation |
| METEOR | Pinned reproducible implementation | METEOR evaluation |
| COMET | Pinned COMET implementation/model | Semantic translation evaluation |
| Visualization | Matplotlib; Seaborn optional | Research plots and analysis |
| Configuration | YAML | Centralized configuration |
| Testing | pytest | Unit, integration, and end-to-end tests |
| Linting | Ruff | Static linting |
| Formatting | Black | Code formatting |
| Version control | Git | Source/version history |
| Remote repository | GitHub | Repository hosting and collaboration |
| Application | Streamlit | Final translation interface |
| GPU acceleration | NVIDIA CUDA through PyTorch | GPU training/inference |
| Backup compute | Google Colab | Fallback compute |
| Secrets | Environment variables / `.env` | Credentials and tokens |
| Experiment storage | Structured local experiment directories | Reproducibility and artifact tracking |

---

# 3. Technology Selection Principles

The AI coding agent must follow these principles.

## 3.1 Prefer the existing project ecosystem

The project is designed around:

**Python → PyTorch → Hugging Face Transformers → Streamlit**

The agent must not replace this architecture with another deep-learning framework or application architecture without explicit approval.

## 3.2 Avoid unnecessary technology

The following are not part of the baseline stack:

- TensorFlow
- JAX
- Flax
- Fairseq
- OpenNMT
- React
- Node.js
- FastAPI
- Django
- Flask
- PostgreSQL
- MongoDB
- Redis
- Kafka
- Kubernetes
- Spark
- Airflow
- Ray
- MLflow

They must not be introduced merely for convenience or personal preference.

If a future requirement genuinely requires one of these technologies, the agent must stop and report the requirement before introducing it.

## 3.3 Research reproducibility takes priority

The selected technology must make it possible to reproduce:

- dataset preparation,
- train/validation/test splitting,
- model loading,
- training configuration,
- generation,
- metric computation,
- final model selection,
- resource measurements.

## 3.4 Version information must be recorded

Exact dependency versions used for an experiment must be captured.

The project must never rely on an undocumented “latest” package state for final experimental results.

---

# 4. Python Environment

Python is the primary implementation language.

The exact Python version must be selected during environment setup and then pinned for the project.

The agent must record:

```text
Python version
Operating system
PyTorch version
Transformers version
Datasets version
Accelerate version
PEFT version
evaluation-library versions
Streamlit version
CUDA runtime/version
NVIDIA driver version
```

The project must not silently switch Python versions between experiments.

A final environment manifest must be generated and retained with the experiment artifacts.

---

# 5. Deep Learning Stack

## 5.1 PyTorch

PyTorch is the required deep-learning framework.

It is responsible for:

- tensors,
- model computation,
- GPU execution,
- automatic differentiation,
- optimization,
- mixed precision,
- training,
- inference.

The agent must verify GPU availability before expensive training.

A basic environment check must establish:

```python
import torch

assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0).total_memory)
```

The exact validation implementation may be improved by the agent, but the check must exist.

## 5.2 CUDA

CUDA is required for GPU acceleration through PyTorch.

The agent must verify compatibility among:

```text
NVIDIA driver
CUDA runtime/toolkit
PyTorch build
GPU
```

The project must not assume that a system CUDA installation automatically means that PyTorch has usable CUDA support.

---

# 6. Transformer/NLP Stack

## 6.1 Hugging Face Transformers

Hugging Face Transformers is the primary model framework.

The following models are frozen by the project specification.

### MarianMT

```text
Helsinki-NLP/opus-mt-en-hi
Helsinki-NLP/opus-mt-en-mr
Helsinki-NLP/opus-mt-en-fr
Helsinki-NLP/opus-mt-en-de
```

### mBART

```text
facebook/mbart-large-50-many-to-many-mmt
```

### NLLB

```text
facebook/nllb-200-distilled-600M
```

The exact model revision/commit used in experiments must be resolved and recorded when the model is frozen for the experiment.

The agent must not substitute another checkpoint because it appears easier to run.

If a checkpoint cannot be executed under the intended hardware constraints, the agent must follow the documented fallback policy and record the change.

---

# 7. Dataset Stack

## 7.1 Hugging Face Datasets

The Hugging Face Datasets library may be used for:

- loading datasets,
- representing parallel data,
- mapping preprocessing operations,
- caching processed datasets,
- dataset inspection.

## 7.2 Pandas

Pandas should be used where tabular operations are clearer or more appropriate, including:

- error analysis,
- prediction tables,
- metric summaries,
- experiment summaries,
- CSV artifacts.

## 7.3 NumPy

NumPy may be used for:

- numerical operations,
- aggregation,
- statistical calculations,
- array manipulation.

---

# 8. Dataset Architecture

The agent must implement dataset processing as a pipeline rather than as ad-hoc preprocessing inside training scripts.

Required conceptual flow:

```text
Raw Dataset
    ↓
Dataset Adapter
    ↓
Schema Validation
    ↓
Normalization
    ↓
Invalid-row Removal
    ↓
Pair-level Deduplication
    ↓
Leakage Detection
    ↓
Length Filtering
    ↓
Deterministic 80/10/10 Split
    ↓
Processed Dataset
```

The canonical parallel-data representation should contain at least:

```text
sample_id
source_language
target_language
source_text
target_text
dataset_source
split
```

The exact schema may include additional metadata.

Raw data must remain separate from processed data.

---

# 9. Model Registry

Model identifiers must be centralized.

They must not be duplicated throughout source files.

The implementation should provide one model registry/configuration layer containing at least:

```text
model family
checkpoint identifier
revision
supported source languages
supported target languages
tokenizer requirements
language-code requirements
training policy
inference policy
```

The registry must support:

```text
MarianMT
mBART
NLLB
```

This registry becomes the single source for model loading.

---

# 10. Language Registry

Language mappings must be centralized.

## 10.1 NLLB

```text
English   = eng_Latn
Hindi     = hin_Deva
Marathi   = mar_Deva
French    = fra_Latn
German    = deu_Latn
```

## 10.2 mBART

```text
English   = en_XX
Hindi     = hi_IN
Marathi   = mr_IN
French    = fr_XX
German    = de_DE
```

The agent must verify tokenizer/model support for these codes during model smoke testing.

MarianMT is pair-specific and must use its corresponding checkpoint rather than a generic language-code mechanism.

No language code should be hardcoded independently in multiple modules.

---

# 11. Training Stack

Training should use Hugging Face Transformers training infrastructure, with Accelerate available for device and mixed-precision management.

The exact implementation may use:

- `Trainer`,
- lower-level PyTorch training loops,
- or a combination,

provided that the implementation satisfies the project’s reproducibility and artifact requirements.

The agent must not introduce a new training framework merely for convenience.

---

# 12. Hardware-Aware Training

Primary hardware:

```text
GPU: NVIDIA RTX 5050 Laptop GPU
VRAM: 8 GB
RAM: 16 GB
Storage: 1 TB SSD
```

Training must be designed for this constraint.

Required strategies include:

- small batch size,
- gradient accumulation,
- mixed precision where supported,
- gradient checkpointing where beneficial,
- controlled sequence length,
- careful model loading,
- memory monitoring.

The project starts from the following documented training values:

```text
Batch size: 2–4
Gradient accumulation: 4–8
Maximum sequence length: 128–256
Epochs: 2–5
Learning rate: approximately 2e-5
Optimizer: AdamW
Mixed precision: enabled where supported
```

These are starting configuration values and not claimed final empirical optima.

The agent must record any changes.

---

# 13. OOM Recovery Policy

If training produces an out-of-memory error, the agent must follow the project’s ordered fallback strategy:

```text
1. Reduce batch size
2. Enable gradient checkpointing
3. Reduce sequence length
4. Increase gradient accumulation
5. Use PEFT/LoRA
6. Move the experiment to backup compute
```

The agent must not silently change the training strategy.

Every fallback must be recorded in the experiment configuration and README.

---

# 14. PEFT / LoRA

PEFT/LoRA is a fallback mechanism.

The primary objective is to attempt full fine-tuning where empirically feasible.

PEFT must not automatically replace full fine-tuning.

If PEFT is used, the experiment must clearly state:

```text
why full fine-tuning was not used
PEFT method
target modules
rank/alpha/dropout where applicable
trainable parameter count
base model revision
```

---

# 15. Tokenization and Generation

Tokenization must use the tokenizer associated with each selected model.

The agent must not implement a custom tokenizer unless explicitly required.

The conceptual pipeline is:

```text
Raw English Text
        ↓
Tokenizer
        ↓
Input IDs / Attention Mask
        ↓
Transformer Encoder/Decoder
        ↓
Generated Token IDs
        ↓
Tokenizer Decode
        ↓
Translated Text
```

Generation must initially use deterministic beam search.

Starting configuration:

```text
num_beams = 5
max_new_tokens = 128
do_sample = false
```

Any model-specific generation exception must be documented.

The same evaluation generation policy should be used across models wherever technically valid.

---

# 16. Evaluation Stack

The project requires:

```text
BLEU
METEOR
COMET
```

## 16.1 BLEU

Use a standardized implementation such as SacreBLEU or another explicitly pinned implementation.

The implementation and version must be recorded.

## 16.2 METEOR

Use a reproducible pinned implementation.

The exact implementation, package version, and relevant configuration must be recorded.

## 16.3 COMET

Use a pinned COMET implementation and explicitly record the COMET model/checkpoint used.

COMET is the primary automated criterion for validation-based model/checkpoint selection.

BLEU and METEOR are supporting metrics.

---

# 17. Model Selection

The stack must support validation-based checkpoint selection.

The frozen policy is:

```text
Primary automated selection criterion:
Validation COMET
```

Supporting evidence:

```text
BLEU
METEOR
qualitative analysis
resource usage
inference speed
model size
```

The agent must not invent an arbitrary weighted composite score.

The final test set is used only after the final model/checkpoint and generation configuration are frozen.

---

# 18. Visualization Stack

Use Matplotlib for research visualizations.

Expected visualizations may include:

- BLEU comparison,
- METEOR comparison,
- COMET comparison,
- model comparison,
- sentence-length analysis,
- error-category distribution,
- inference latency,
- training/resource comparison.

Seaborn may be used if useful, but it is optional.

The agent must not introduce a browser-based visualization framework unless the requirements change.

---

# 19. Configuration Management

YAML is the baseline configuration format.

Recommended structure:

```text
configs/
├── base.yaml
├── languages.yaml
├── models.yaml
├── datasets.yaml
├── training.yaml
├── evaluation.yaml
└── experiments/
```

Configuration should contain:

```text
seed
dataset
language pair
model
model revision
training parameters
generation parameters
evaluation parameters
resource settings
```

The source code should consume configuration rather than embedding experimental constants throughout modules.

---

# 20. Experiment Reproducibility

Every experiment must record:

```text
Experiment ID
Git commit
Python version
OS
GPU
GPU memory
NVIDIA driver
CUDA version
PyTorch version
Transformers version
Datasets version
Accelerate version
PEFT version if used
Evaluation package versions
Model checkpoint
Model revision
Dataset source
Dataset version
Dataset hash where available
Seed
Training configuration
Generation configuration
Metrics
Training time
Inference latency
Peak VRAM
Final checkpoint
```

This information must be automatically generated where possible.

The agent should not depend on the user manually copying values into experiment files.

---

# 21. Experiment Directory Contract

Each experiment must use:

```text
{MODEL}_{SRC}_{TGT}_{DATASET}_{SEED}_{RUN}
```

Example:

```text
NLLB_EN_MR_OPUS_42_001
```

Each experiment directory should contain:

```text
config.yaml
environment.txt
training.log
checkpoint_manifest.json
predictions.csv
metrics.json
resource_metrics.json
error_analysis.csv
README.md
```

`metrics.json` must include implementation/version metadata for the evaluation metrics.

---

# 22. Testing Stack

pytest is the required testing framework.

Tests must cover:

### Environment

```text
GPU availability
CUDA availability
dependency imports
```

### Language registry

```text
language names
language codes
model-specific mappings
```

### Dataset

```text
schema validation
normalization
invalid-row handling
deduplication
leakage detection
deterministic split
```

### Tokenization

```text
tokenizer loading
encoding
decoding
special-token handling
```

### Models

```text
checkpoint loading
language-code configuration
forward/generation smoke tests
```

### Evaluation

```text
BLEU
METEOR
COMET
```

### Application

```text
translation service
model caching
language selection
inference
error handling
```

No full training run should be required for ordinary unit tests.

---

# 23. Code Quality

The baseline tools are:

```text
Ruff
Black
pytest
```

The agent should run:

```text
ruff check .
black --check .
pytest
```

before a major implementation milestone is considered complete.

Formatting and linting failures must not be ignored without explanation.

---

# 24. Version Control

Git is required.

GitHub is the intended remote repository.

The repository must contain:

```text
source code
tests
configuration
documentation
small metadata
scripts
```

The repository must not contain:

```text
raw datasets
large checkpoints
model caches
large generated artifacts
secrets
.env
```

Large generated artifacts should remain in the structured experiment storage location or appropriate external storage.

---

# 25. Secrets and Credentials

Credentials must be provided through environment variables or an ignored `.env` file.

Example:

```text
HF_TOKEN=
```

The actual token must never be committed.

The repository should contain:

```text
.env.example
```

but not the populated `.env`.

The AI agent must not print or expose secret values in logs.

---

# 26. Hugging Face Access

The agent must verify access to the required model checkpoints before beginning full experiments.

The smoke test must attempt to load:

```text
Helsinki-NLP/opus-mt-en-hi
Helsinki-NLP/opus-mt-en-mr
Helsinki-NLP/opus-mt-en-fr
Helsinki-NLP/opus-mt-en-de
facebook/mbart-large-50-many-to-many-mmt
facebook/nllb-200-distilled-600M
```

The test should verify:

1. checkpoint can be resolved,
2. tokenizer can load,
3. model can load,
4. expected language configuration can be applied,
5. a small generation test can execute.

---

# 27. Streamlit Application Stack

The final application uses Streamlit.

The application must:

- load the final selected checkpoint,
- load the corresponding tokenizer,
- accept English input,
- provide target-language selection,
- generate translation,
- display translation,
- handle inference errors gracefully,
- cache model/tokenizer objects,
- avoid reloading the model on every request.

Supported target languages:

```text
Hindi
Marathi
French
German
```

The application must not train models.

The application must not modify experiment artifacts.

The application must not use the test set.

---

# 28. Streamlit Model Loading

Model loading should be separated from UI code.

Recommended conceptual architecture:

```text
Streamlit UI
     ↓
Translation Service
     ↓
Model Registry
     ↓
Tokenizer + Model Loader
     ↓
Final Checkpoint
```

Caching should be applied to expensive model/tokenizer initialization.

Only the required large model should occupy GPU memory at a time unless the hardware constraints explicitly permit otherwise.

---

# 29. Logging

The project must use structured logging rather than scattered `print()` statements for important events.

Logs should capture:

- experiment start,
- model loading,
- dataset loading,
- training start/end,
- checkpoint saves,
- evaluation start/end,
- OOM/fallback events,
- resource measurements,
- errors.

Training logs must be saved inside the experiment artifact directory.

---

# 30. Resource Measurement

The technology stack must support recording:

```text
training duration
inference latency
peak GPU VRAM
model artifact size
```

Where possible, GPU measurements should use PyTorch/NVIDIA-compatible measurement methods.

The measurement method must be consistent across comparable experiments.

---

# 31. Backup Compute

Google Colab is the designated backup compute environment.

The codebase must therefore remain portable.

The agent should not hardcode:

- local absolute paths,
- machine-specific GPU names,
- local usernames,
- Windows-only paths,
- notebook-only state.

Paths should be configuration-driven or relative to the project root.

---

# 32. Path Handling

Use Python's `pathlib` rather than hardcoded operating-system-specific path strings.

Bad:

```python
"/home/manthan/project/data/train.csv"
```

Good:

```python
Path(project_root) / "data" / "processed" / "train.csv"
```

This is particularly important because the same code may run on:

- the local RTX 5050 machine,
- Google Colab,
- another evaluation environment.

---

# 33. Environment Manifest

At the beginning of an experiment, the system must generate an environment manifest containing at least:

```text
Python
OS
CPU
RAM
GPU
GPU VRAM
NVIDIA driver
CUDA
PyTorch
Transformers
Datasets
Accelerate
PEFT
Pandas
NumPy
evaluation packages
Streamlit
Git commit
```

This manifest becomes part of the experiment artifact.

---

# 34. Dependency Management Policy

The agent may install additional dependencies only when:

1. the existing stack cannot satisfy a documented requirement,
2. the dependency has a clear purpose,
3. its version is pinned/recorded,
4. it does not unnecessarily duplicate an existing capability.

If the new dependency materially changes architecture, the agent must stop and report the proposed change before implementation.

---

# 35. Technology Change-Control Rule

The following are considered **frozen unless explicitly changed**:

- Python ecosystem
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Streamlit
- pytest
- Git/GitHub
- YAML configuration
- primary evaluation stack
- model registry
- primary GPU strategy

The AI agent must not replace these technologies based on personal preference.

---

# 36. Agent Preflight Requirements

Before implementation begins, the agent must verify:

```text
[ ] Repository is accessible
[ ] Documentation is accessible
[ ] Python environment works
[ ] Git works
[ ] GPU is visible
[ ] CUDA is usable
[ ] PyTorch sees GPU
[ ] Required packages import
[ ] Hugging Face access works
[ ] All required model checkpoints can be resolved
[ ] Tokenizers can load
[ ] Small model generation works
[ ] Dataset source access is available
[ ] Storage is sufficient
[ ] .gitignore protects large files/secrets
[ ] configuration system works
[ ] pytest works
[ ] linting works
```

The agent must produce a preflight report before moving into full implementation.

---

# 37. Agent Stop Conditions

The agent must stop and ask for clarification if:

1. a required model cannot be accessed,
2. a language code is unsupported,
3. a dataset source is unavailable,
4. two specifications conflict,
5. a required metric cannot be reproduced,
6. GPU memory makes the documented experiment impossible even after the defined fallback sequence,
7. a new framework appears necessary,
8. a major dependency must be replaced,
9. the test set would need to be used for tuning,
10. the agent would need to invent an undocumented research decision.

The agent must never silently resolve a research-critical ambiguity.

---

# 38. Technologies the Agent Must Not Introduce Without Approval

The agent must not introduce a replacement or additional major framework for:

```text
deep learning
NLP
web application
database
distributed training
experiment tracking
workflow orchestration
cloud infrastructure
```

without documenting why the existing stack cannot satisfy the requirement.

Examples include:

```text
TensorFlow instead of PyTorch
JAX instead of PyTorch
FastAPI alongside Streamlit
React instead of Streamlit
MLflow as a mandatory experiment system
Docker as a mandatory deployment layer
```

These are architectural changes, not ordinary implementation details.

---

# 39. Recommended Repository Layout

The final implementation should follow a structure similar to:

```text
multilingual-nmt/
│
├── app/
│   ├── streamlit_app.py
│   └── ...
│
├── configs/
│   ├── base.yaml
│   ├── languages.yaml
│   ├── models.yaml
│   ├── datasets.yaml
│   ├── training.yaml
│   ├── evaluation.yaml
│   └── experiments/
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── metadata/
│
├── docs/
│
├── experiments/
│
├── scripts/
│
├── src/
│   └── ...
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── data/
│   ├── models/
│   ├── evaluation/
│   └── app/
│
├── reports/
│
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
└── requirements.txt
```

The exact module names may be refined by the agent, but the separation of concerns must remain.

---

# 40. Definition of Technology-Stack Completion

The technology stack setup is complete only when:

```text
Python environment is reproducible
        AND
PyTorch sees the RTX 5050
        AND
CUDA works
        AND
required packages import
        AND
all frozen models can be loaded
        AND
language mappings are verified
        AND
dataset adapters can be initialized
        AND
evaluation libraries work
        AND
pytest passes
        AND
linting/formatting pass
        AND
Streamlit can start
        AND
environment metadata can be recorded
```

Only then should the project proceed to the main development tasks.

---

# 41. Relationship to the Other Project Documents

This document complements rather than replaces:

- Product Requirements Document
- Functional Specification Document
- Technical Design Document
- Master Decision Register
- AI Agent Development Master Specification
- Experiment Protocol
- Test and Validation Plan
- Dataset/Model Manifest
- Deployment Runbook
- Decisions and Assumptions
- Final Agent Checklist

The technology stack specifies **how the system is to be built**.

The PRD specifies **what the product/research project must accomplish**.

The FSD specifies **what the system must functionally do**.

The TDD specifies **how the technical architecture is organized**.

The experiment protocol specifies **how research experiments are conducted and compared**.

The test plan specifies **how implementation correctness is verified**.

The deployment runbook specifies **how the final application is run**.

---

# 42. Final Agent Instruction

Before writing substantial implementation code, the AI coding agent must:

1. Read all documents under `docs/`.
2. Treat the Master Decision Register as authoritative for frozen decisions.
3. Read this Technology Stack & Environment Specification.
4. Validate the local environment.
5. Validate the GPU/CUDA/PyTorch stack.
6. Validate required model access.
7. Validate tokenizer/language mappings.
8. Validate dataset access.
9. Validate evaluation dependencies.
10. Validate testing and linting.
11. Produce a preflight report.
12. Stop on unresolved research-critical conflicts.
13. Proceed sequentially through the AI Agent Development Master Specification only after preflight passes.

The agent must not begin full model training during preflight.

---

# 43. Final Technology Baseline

The project's baseline stack is therefore:

```text
Python
│
├── PyTorch
│
├── Hugging Face Transformers
│   ├── MarianMT
│   ├── mBART-50
│   └── NLLB-200 distilled 600M
│
├── Hugging Face Datasets
│
├── Accelerate
│
├── PEFT / LoRA (fallback)
│
├── Pandas / NumPy
│
├── SacreBLEU
│
├── METEOR implementation
│
├── COMET
│
├── Matplotlib
│
├── YAML configuration
│
├── pytest
│
├── Ruff / Black
│
└── Streamlit
       │
       ▼
   Final translation application
```

**This is the technology baseline the AI coding agent should implement against unless a documented requirement forces a change.**
