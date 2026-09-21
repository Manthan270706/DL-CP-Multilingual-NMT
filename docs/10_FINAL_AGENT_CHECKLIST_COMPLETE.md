# 10 FINAL AGENT CHECKLIST — COMPLETE

## Documentation
- [ ] Original PRD preserved.
- [ ] Original FSD preserved.
- [ ] Original TDD preserved.
- [ ] Resolved decisions appended.
- [ ] Terminology is consistent.

## Data
- [ ] Source and revision recorded.
- [ ] Schema validated.
- [ ] Normalization applied.
- [ ] Invalid rows removed.
- [ ] Duplicates removed.
- [ ] Leakage check passed.
- [ ] 80/10/10 split created.
- [ ] Split seed recorded.

## Models
- [ ] Four MarianMT checkpoints load.
- [ ] mBART-50 loads.
- [ ] NLLB-200 distilled 600M loads.
- [ ] Language codes verified.
- [ ] Model revisions recorded.

## Training
- [ ] Transfer learning used.
- [ ] Effective configuration recorded.
- [ ] Precision recorded.
- [ ] Accumulation recorded.
- [ ] Checkpointing recorded.
- [ ] OOM fallback recorded if used.

## Evaluation
- [ ] Validation/test isolation maintained.
- [ ] BLEU computed.
- [ ] METEOR computed.
- [ ] COMET computed.
- [ ] Metric versions recorded.
- [ ] Generation settings recorded.
- [ ] Final checkpoint frozen before test.

## Analysis
- [ ] Qualitative examples retained.
- [ ] Error analysis completed.
- [ ] Sentence-length analysis completed.
- [ ] Small buckets flagged.
- [ ] Resource measurements recorded.

## Application
- [ ] Final checkpoint integrated.
- [ ] Model cached.
- [ ] Tokenizer cached.
- [ ] Four directions supported.
- [ ] Invalid input handled.
- [ ] Unicode handled.
- [ ] No training in app.

## Reproducibility
- [ ] config.yaml
- [ ] environment.txt
- [ ] training.log
- [ ] checkpoint_manifest.json
- [ ] predictions.csv
- [ ] metrics.json
- [ ] resource_metrics.json
- [ ] error_analysis.csv
- [ ] README.md

## Research Integrity
- [ ] No fabricated results.
- [ ] No test leakage.
- [ ] No undocumented model changes.
- [ ] No undocumented dataset changes.
- [ ] No undocumented compute changes.
- [ ] No undocumented generation changes.
