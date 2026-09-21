# 04 AI AGENT DEVELOPMENT MASTER SPEC — COMPLETE

## 1. Mission
Implement the full NMT project according to the original PRD/FSD/TDD and the resolved decision register. Preserve every documented requirement and do not silently substitute assumptions.

## 2. Ordered Development Plan

### TASK-001 — Environment
Create the Python environment and verify PyTorch, CUDA, Transformers, tokenizers, evaluation dependencies, and Streamlit. Record versions.

### TASK-002 — Repository
Create modular directories for configuration, data, models, training, evaluation, inference, application, tests, and results.

### TASK-003 — Language Registry
Create one authoritative registry for names, ISO identifiers, model-specific language codes, and supported directions.

### TASK-004 — Dataset Adapters
Implement adapters for the approved dataset sources. Keep retrieval separate from cleaning.

### TASK-005 — Cleaning
Implement normalization, invalid-row filtering, pair-level deduplication, and length filtering.

### TASK-006 — Leakage Detection
Check normalized source-target pairs across splits and detect train/test overlap.

### TASK-007 — Deterministic Split
Create the 80/10/10 split with a fixed seed and save split metadata.

### TASK-008 — Model Registry
Implement the exact MarianMT, mBART-50, and NLLB checkpoints.

### TASK-009 — Model Smoke Tests
Load each tokenizer/model and execute minimal translation for every supported direction where technically valid.

### TASK-010 — Training Configuration
Implement configuration-driven training and persist the effective configuration.

### TASK-011 — Resource Controls
Implement mixed precision, accumulation, checkpointing, controlled sequence length, and memory monitoring.

### TASK-012 — Fine-Tuning
Run the controlled experiments. Attempt full fine-tuning first.

### TASK-013 — Validation
Evaluate validation data and select checkpoints using validation COMET.

### TASK-014 — Test Freeze
Freeze checkpoint, preprocessing, generation configuration, and model identity before test evaluation.

### TASK-015 — Final Test
Generate predictions and calculate BLEU, METEOR, and COMET on the untouched test set.

### TASK-016 — Resource Measurement
Measure training time, inference latency, peak VRAM, and artifact size.

### TASK-017 — Error Analysis
Create structured error records using the predefined categories.

### TASK-018 — Sentence-Length Analysis
Calculate results for the four buckets and flag small samples.

### TASK-019 — Comparative Analysis
Produce descriptive comparisons without inventing values or rankings unsupported by evidence.

### TASK-020 — Inference Engine
Create a reusable translation interface independent of Streamlit.

### TASK-021 — Streamlit
Create the application around the inference engine and cache the model/tokenizer.

### TASK-022 — Testing
Run unit, integration, end-to-end, leakage, reproducibility, and application tests.

### TASK-023 — Artifact Validation
Verify required experiment files and metadata.

### TASK-024 — Documentation
Generate final methodology and results documentation from actual artifacts.

### TASK-025 — Final Gate
Complete only when all quality, research-integrity, reproducibility, and deployment checks pass.

## 3. Mandatory Stop Conditions
Stop before proceeding when:
- a required model cannot load;
- language-code mapping is invalid;
- dataset schema fails;
- leakage is detected;
- test data enters tuning;
- required metrics cannot execute;
- a final checkpoint cannot be restored;
- application configuration does not match the final artifact.

## 4. Agent Rules
Configuration must be centralized. Fallbacks must be logged. Raw data must be preserved. Destructive transformations are prohibited. Empirical claims require artifacts.
