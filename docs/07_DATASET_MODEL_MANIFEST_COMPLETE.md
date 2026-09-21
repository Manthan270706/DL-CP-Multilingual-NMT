# 07 DATASET AND MODEL MANIFEST — COMPLETE

## 1. Model Registry

| ID | Checkpoint | Role |
|---|---|---|
| MARIAN_EN_HI | Helsinki-NLP/opus-mt-en-hi | Baseline |
| MARIAN_EN_MR | Helsinki-NLP/opus-mt-en-mr | Baseline |
| MARIAN_EN_FR | Helsinki-NLP/opus-mt-en-fr | Baseline |
| MARIAN_EN_DE | Helsinki-NLP/opus-mt-en-de | Baseline |
| MBART50 | facebook/mbart-large-50-many-to-many-mmt | Multilingual |
| NLLB600M | facebook/nllb-200-distilled-600M | Primary multilingual |

## 2. Language Registry

| Language | ISO | NLLB | mBART |
|---|---|---|---|
| English | en | eng_Latn | en_XX |
| Hindi | hi | hin_Deva | hi_IN |
| Marathi | mr | mar_Deva | mr_IN |
| French | fr | fra_Latn | fr_XX |
| German | de | deu_Latn | de_DE |

## 3. Dataset Strategy

EN→HI: IIT Bombay English-Hindi Corpus where suitable.

EN→MR: versioned OPUS or another suitable validated parallel source.

EN→FR: versioned OPUS/WMT-compatible source.

EN→DE: versioned OPUS/WMT-compatible source.

The exact dataset configuration and revision are frozen only after validation and recorded in the manifest.

## 4. Required Data Contract
Canonical records should retain source text, target text, language pair, source identifier when available, dataset source, and dataset revision/configuration.

## 5. Processing
raw → schema validation → normalization → invalid-row removal → pair-level deduplication → leakage detection → length filtering → 80/10/10 deterministic split.

## 6. Manifest Metadata
Record dataset identity, revision, configuration, retrieval date, preprocessing version, split seed, row counts, and filtering counts once empirically available.
