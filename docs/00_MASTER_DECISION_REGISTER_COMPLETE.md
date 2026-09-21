# 00 MASTER DECISION REGISTER — COMPLETE

## 1. Purpose
This document is the authoritative implementation decision register. It consolidates resolved decisions while preserving the original PRD, FSD, and TDD.

## 2. Frozen Scope
Primary experimental directions:
- English → Hindi
- English → Marathi
- English → French
- English → German

Reverse directions are outside the primary study.

## 3. Frozen Model Registry
### MarianMT
- `Helsinki-NLP/opus-mt-en-hi`
- `Helsinki-NLP/opus-mt-en-mr`
- `Helsinki-NLP/opus-mt-en-fr`
- `Helsinki-NLP/opus-mt-en-de`

### mBART-50
- `facebook/mbart-large-50-many-to-many-mmt`

### NLLB
- `facebook/nllb-200-distilled-600M`

The exact revision/commit hash used must be recorded per experiment.

## 4. Training Policy
Transfer learning and fine-tuning are required; training from scratch is out of scope.

Full fine-tuning is attempted first. If infeasible, use:
1. smaller batch;
2. gradient checkpointing;
3. shorter sequence length;
4. larger gradient accumulation;
5. PEFT/LoRA;
6. backup compute.

Every fallback is recorded.

## 5. Data Policy
For each language pair, maintain one canonical cleaned parallel dataset.

Processing:
raw load → schema validation → normalization → invalid-row removal → pair deduplication → leakage detection → length filtering → deterministic 80/10/10 split.

The test set is never used for training, tuning, checkpoint selection, generation tuning, or model selection.

## 6. Evaluation Policy
Mandatory metrics:
- BLEU
- METEOR
- COMET

Validation COMET is the primary automated checkpoint/model-selection criterion. BLEU, METEOR, qualitative review, error analysis, inference speed, VRAM, and model size are supporting evidence.

No arbitrary weighted composite score is used.

## 7. Generation
Initial deterministic configuration:
- beam search;
- `num_beams=5`;
- `max_new_tokens=128`;
- sampling disabled.

Record actual generation settings for every evaluation.

## 8. Analysis
Error categories:
grammar, word order, named entity, ambiguity, missing words, extra words, repetition, untranslated words, long-sentence failures.

Sentence-length buckets:
1–10, 11–25, 26–50, 51+.

Buckets below 30 examples are flagged as small samples.

## 9. Artifacts
Every experiment:
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

## 10. Application
Streamlit supports EN→HI, EN→MR, EN→FR, and EN→DE. The final selected checkpoint is the default. Model and tokenizer are cached. Training does not run inside the application.

## 11. Reproducibility
Record seed, model/revision, tokenizer, dataset/version, preprocessing, hyperparameters, generation settings, environment, hardware, metric versions, selected checkpoint, and resource measurements.

## 12. Research Integrity
Never fabricate results, dataset revisions, model runs, metric values, checkpoints, resource measurements, or qualitative examples. Never silently change models, datasets, compute, training strategy, or generation configuration.
