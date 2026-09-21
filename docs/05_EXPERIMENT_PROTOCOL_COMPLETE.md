# 05 EXPERIMENT PROTOCOL — COMPLETE

## 1. Objective
Define the controlled protocol for comparing MarianMT, mBART-50, and NLLB on the four primary English-centric directions.

## 2. Primary Experiment Matrix

| Model | EN→HI | EN→MR | EN→FR | EN→DE |
|---|---|---|---|---|
| MarianMT | Yes | Yes | Yes | Yes |
| mBART-50 | Yes | Yes | Yes | Yes |
| NLLB | Yes | Yes | Yes | Yes |

This is a target matrix of 12 primary experiments, subject to actual model loading and documented hardware feasibility.

## 3. Controlled Conditions
Keep constant where technically valid:
- cleaned data;
- split;
- seed;
- evaluation references;
- generation policy;
- metric implementations;
- checkpoint-selection rule.

Document every exception.

## 4. Starting Configuration
- batch size: 2–4;
- gradient accumulation: 4–8;
- max sequence length: 128–256;
- epochs: 2–5;
- learning rate: approximately 2e-5;
- AdamW;
- mixed precision where supported.

These are starting values and not guaranteed final values.

## 5. Checkpoint Selection
Use validation COMET as the primary automated criterion. Do not use test results to select checkpoints.

## 6. Final Test
Once the configuration is frozen:
1. generate test predictions;
2. calculate BLEU;
3. calculate METEOR;
4. calculate COMET;
5. save predictions;
6. save metric versions;
7. save resource metrics.

## 7. Generation
Use deterministic beam search with initial settings `num_beams=5`, `max_new_tokens=128`, and sampling disabled.

## 8. Fine-Tuning Effect
Where feasible, evaluate the original pretrained checkpoint under the same evaluation conditions to support analysis of fine-tuning effects.

## 9. Resource Reporting
Measure rather than estimate training time, peak VRAM, inference latency, and model artifact size.

## 10. Naming
`{MODEL}_{SRC}_{TGT}_{DATASET}_{SEED}_{RUN}`

Example:
`NLLB_EN_MR_OPUS_42_001`

## 11. Experiment Manifest
Record model revision, tokenizer, dataset revision/configuration, language codes, seeds, preprocessing, training settings, generation settings, metric versions, hardware, fallback strategy, selected checkpoint, and output locations.

## 12. Reproducibility
Repeated execution should use the same model identity, data split, configuration, and evaluation policy. Environment-dependent nondeterminism must be documented.

## 13. Interpretation
Metrics must be interpreted together with qualitative examples, error analysis, sentence-length analysis, and resource measurements.
