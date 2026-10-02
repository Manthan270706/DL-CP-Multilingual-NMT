# 08 DEPLOYMENT RUNBOOK — COMPLETE

## 1. Objective
Deploy the final selected NMT checkpoint through Streamlit while keeping inference separate from research/training.

## 2. Pre-Deployment Requirements
Verify:
- final checkpoint exists;
- tokenizer exists;
- model revision is recorded;
- language registry is correct;
- generation settings are recorded;
- inference smoke test passes;
- environment versions are recorded.

## 3. Application Flow
```text
User
 ↓
Streamlit UI
 ↓
Input Validation
 ↓
Language Registry
 ↓
Cached Inference Engine
 ↓
Tokenizer
 ↓
Transformer
 ↓
Generation
 ↓
Decoded Translation
 ↓
Output
```

## 4. Supported Directions
EN→HI, EN→MR, EN→FR, EN→DE.

## 5. Caching
Load the model/tokenizer once and reuse them. Avoid repeated initialization.

## 6. Errors
Handle empty input, unsupported pair, model loading failure, generation failure, excessive input length, and Unicode text.

User-facing messages should be concise; technical details belong in logs.

## 7. OOM
Training fallback:
batch reduction → sequence reduction → checkpointing → accumulation → PEFT → backup compute.

Inference should avoid loading unnecessary models simultaneously.

## 8. Demo Checklist & Launch Commands

### 8.1 Launch Web Platform
```powershell
.venv\Scripts\streamlit.exe run app/streamlit_app.py
```
*Access URL: `http://localhost:8501`*

### 8.2 Run Command-Line Inference
```powershell
.venv\Scripts\python.exe main.py translate --model marian --src en --tgt hi --text "The model translates technical documents."
```

### 8.3 Run Verification Test Suite
```powershell
.venv\Scripts\pytest.exe tests/ -v
```

### 8.4 Verification Checklist
- [x] Streamlit web platform launches cleanly without port conflicts;
- [x] All 4 directions (EN $\to$ DE, FR, HI, MR) translate accurately;
- [x] Unicode rendering for Devanagari (Hindi/Marathi) glyphs is verified;
- [x] Cross-attention interpretability heatmap is extracted with dynamic re-normalization;
- [x] Zero-shot and fine-tuned checkpoints switch dynamically;
- [x] 23/23 automated PyTest test cases pass in ~11s.

## 9. Future Architecture
The inference layer supports modular expansion:
Streamlit $\to$ FastAPI $\to$ Triton/vLLM Model Server $\to$ Distributed Multi-GPU Cluster.
