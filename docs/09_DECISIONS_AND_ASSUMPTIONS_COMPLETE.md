# 09 DECISIONS AND ASSUMPTIONS — COMPLETE

## 1. Frozen Decisions
1. English is the primary source language.
2. Hindi, Marathi, French, and German are the primary targets.
3. MarianMT provides pair-specific baselines.
4. mBART-50 provides multilingual comparison.
5. NLLB-200 distilled 600M is the primary multilingual model.
6. Fine-tuning is used; training from scratch is excluded.
7. RTX 5050 8 GB is the primary GPU.
8. Google Colab is backup compute.
9. Data follows the specified cleaning/leakage/splitting sequence.
10. Split is 80/10/10.
11. Test data is isolated.
12. BLEU, METEOR, and COMET are mandatory.
13. Validation COMET drives automated checkpoint selection.
14. Generation starts with deterministic beam search.
15. Error categories are predefined.
16. Sentence-length buckets are predefined.
17. Streamlit is the deployment UI.
18. Model/tokenizer caching is required.

## 2. Assumptions
- suitable parallel corpora can be obtained;
- required model checkpoints remain accessible;
- the environment can run the required libraries;
- COMET can be executed locally or on backup compute.

## 3. Empirical Unknowns
Must be measured:
- final row counts;
- training duration;
- peak VRAM;
- inference latency;
- final metrics;
- best checkpoint;
- whether full fine-tuning fits;
- whether PEFT is needed.

## 4. Change Control
Any change to a frozen decision requires documentation of the previous decision, proposed change, reason, affected experiments, comparability impact, and updated documentation.
