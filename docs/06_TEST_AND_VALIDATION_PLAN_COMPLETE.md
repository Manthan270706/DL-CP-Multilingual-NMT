# 06 TEST AND VALIDATION PLAN — COMPLETE

## 1. Purpose
Verify the complete system before full training, final test evaluation, and submission.

## 2. Unit Tests

### Dataset
- required fields;
- invalid rows;
- deterministic normalization;
- duplicate removal;
- split proportions;
- no cross-split overlap.

### Language Registry
- valid language metadata;
- valid NLLB codes;
- valid mBART codes;
- unsupported directions rejected.

### Models
- tokenizer loads;
- checkpoint loads;
- language code applies;
- generation produces output.

### Metrics
- BLEU runs;
- METEOR runs;
- COMET runs;
- metric versions are stored.

### Inference
- valid text;
- empty text;
- unsupported pair;
- long input;
- Unicode;
- cached model.

## 3. Integration Tests
Validate:
dataset → preprocessing → tokenizer → model → generation → predictions → evaluation.

Validate:
configuration → checkpoint → inference engine → Streamlit.

## 4. End-to-End Tests
Run a small controlled experiment before full-scale experiments and verify all expected artifacts.

## 5. Leakage Tests
Fail if test examples appear in training or if test results influence checkpoint selection or generation tuning.

## 6. Resource Tests
Record GPU visibility, peak VRAM, inference latency, and relevant system-resource measurements.

## 7. Application Acceptance
The app must launch, expose supported directions, accept input, translate, display output, handle invalid inputs, cache the model, and use the documented final checkpoint.

## 8. Final Gates
A — data valid and leak-free.
B — models load and smoke tests pass.
C — experiments complete or documented as infeasible.
D — metrics and resource artifacts exist.
E — error and sentence-length analysis exist.
F — application uses final selected artifact.
G — reproducibility metadata is complete.
H — no research-integrity violations exist.
