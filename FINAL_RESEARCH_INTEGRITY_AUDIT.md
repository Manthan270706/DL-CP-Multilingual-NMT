# FINAL RESEARCH INTEGRITY AUDIT REPORT

**Date**: 2026-09-22  
**Repository**: `Manthan270706/DL-CP-Multilingual-NMT`  
**Platform**: Windows 11 | NVIDIA GeForce RTX 5050 Laptop GPU (8,151 MiB VRAM, Compute Capability `sm_120` Blackwell)  
**Environment**: Python 3.12.3 | PyTorch `2.12.0.dev20260408+cu128` (CUDA 12.8 nightly)

---

## 1. Executive Summary Table

| Audit Area | Status | Primary File / Artifact Evidence | Summary Finding |
|---|:---:|---|---|
| **1. Test Sample Size (100 vs Full)** | **WARNING** | `run_pipeline.py` (L39, L111, L130), `docs/05_EXPERIMENT_PROTOCOL_COMPLETE.md` | 100 samples was a developmental smoke-test default in `run_pipeline.py`, not the project-specified final test set size. Complete frozen test sets range from 1,015 to 4,914 samples. |
| **2. Marian EN→HI BLEU = 100.00** | **FAIL (Metric Bug)**<br>**PASS (Data Leakage)** | `src/evaluation/bleu.py` (L35), `experiments/MARIAN_EN_HI_IITB_42_001/metrics.json` (L15-18), `data/processed/en_hi/freeze_manifest.json` | BLEU 100 was caused by a reference formatting bug in `bleu.py` (`ref_list = [[r] for r in references]`), causing SacreBLEU to evaluate only sample #1 (`sys_len = 12`). True 100-sample BLEU is **80.89**. Zero data leakage confirmed. |
| **3. Model Fine-Tuning Verification** | **WARNING** | `experiments/*/resource_metrics.json`, `run_pipeline.py` (L144-154), `src/training/trainer.py` | The 12 reported experiments evaluated pretrained base model checkpoints directly in zero-shot/baseline mode. No multi-epoch fine-tuning was executed. |
| **4. Checkpoint Selection & Test Isolation** | **PASS** | `run_pipeline.py` (L129-135), `data/processed/*/freeze_manifest.json` | Test sets were strictly isolated and never used for training or checkpoint tuning. Base pretrained checkpoints were evaluated directly. |
| **5. Data Integrity & Dual Leakage** | **PASS** | `data/processed/{en_de,en_fr,en_hi,en_mr}/freeze_manifest.json`, split CSV SHA-256 hashes | 100% verified zero pair leakage and zero source-sentence leakage across all splits. On-disk CSV SHA-256 checksums match freeze manifests. |
| **6. Metric & Generation Settings** | **PASS** | `src/evaluation/bleu.py`, `src/evaluation/meteor.py`, `src/evaluation/comet_eval.py`, `tests/test_evaluation.py` | SacreBLEU 2.6.0 (13a), NLTK 3.10.3 METEOR, Unbabel/wmt22-comet-da (COMET 2.2.7), deterministic beam search (`num_beams=5`, `max_new_tokens=128`, `do_sample=False`). |
| **7. Hardware & Architecture Claims** | **PASS** | `environment.txt` (L5-11), `torch.cuda.get_arch_list()` verification | RTX 5050 (`sm_120`) native support is fully verified with PyTorch nightly cu128 (`['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']`). Peak VRAM: 3.53 GB max. |

---

## 2. Detailed Audit Inquiries

### 1. Why were only 100 test samples previously evaluated?
- **Finding**: In `run_pipeline.py` (line 39), the function was defined with `def run_full_pipeline(num_samples: int = 100, fast_eval: bool = True):`. Line 111 resolved `eval_test_size = num_samples or 200`, and Line 130 sliced `full_test_df.iloc[: min(eval_test_size, len(full_test_df))]`.
- **Specification vs Implementation**: Project specifications (`docs/05_EXPERIMENT_PROTOCOL_COMPLETE.md` Section 6, `docs/00_MASTER_DECISION_REGISTER_COMPLETE.md` Section 5) mandate evaluation on the frozen test set without restricting size. The 100-sample limit was an iteration default in the script.
- **Evidence**:
  - `run_pipeline.py`: Lines 39, 111, 130
  - Frozen test set counts on disk:
    - `data/processed/en_de/test.csv`: **4,914 samples**
    - `data/processed/en_fr/test.csv`: **4,883 samples**
    - `data/processed/en_hi/test.csv`: **1,015 samples**
    - `data/processed/en_mr/test.csv`: **2,515 samples**
- **Conclusion**: The existing 100-sample numbers represent a **smoke-test subset benchmark**. Complete test set evaluation requires running inference on the full splits.

---

### 2. Was Marian EN→HI BLEU = 100.00 caused by leakage or an evaluation error?
- **Finding**: **Evaluation Bug in `bleu.py`**.
  - In `src/evaluation/bleu.py` (Line 35), the code executed:
    ```python
    ref_list = [[r] for r in references]
    bleu = sacrebleu.corpus_bleu(predictions, ref_list, tokenize=tokenize, lowercase=lowercase)
    ```
  - In `sacrebleu`, passing a list of $N$ single-element lists causes the engine to interpret the input as $N$ independent reference streams for a 1-sentence corpus. Consequently, SacreBLEU evaluated **only sample #1** against reference #1.
  - Sample #1 in `experiments/MARIAN_EN_HI_IITB_42_001/predictions.csv`:
    - *Source*: `"A list of plugins that are disabled by default"`
    - *Target*: `"उन प्लग-इनों की सूची जिन्हें डिफोल्ट रूप से निष्क्रिय किया गया है"`
    - *Prediction*: `"उन प्लग-इनों की सूची जिन्हें डिफोल्ट रूप से निष्क्रिय किया गया है"`
  - Because sample #1 matched exactly, SacreBLEU output `score = 100.0`, `sys_len = 12`, `ref_len = 12` (documented in `experiments/MARIAN_EN_HI_IITB_42_001/metrics.json` lines 15–18).
- **Recalculation on the 100 Samples**:
  - Fixing `ref_list = [references]` evaluates all 100 sentences (`hyp_len = 605, ref_len = 592`), yielding:
    - **SacreBLEU**: **80.89**
    - **METEOR**: **77.19**
    - **COMET**: **88.01**
- **Data Leakage Check**:
  - `data/processed/en_hi/freeze_manifest.json` confirms **0.00% pair leakage** (`pair_overlaps.train_test = 0`) and **0.00% source leakage** (`source_overlaps.train_test = 0`).
  - High performance of MarianMT on IIT Bombay stems from Helsinki-NLP's OPUS pretraining corpus (which shares UI/software localization strings with GNOME/Ubuntu).
- **Resolution**: Bug in `src/evaluation/bleu.py` has been fixed to `ref_list = [references]`, and verified with updated unit tests in `tests/test_evaluation.py`.

---

### 3. Were all 12 experiments actually fine-tuned?
- **Finding**: **No**. The 12 experiments in `experiments/` reflect **zero-shot / baseline inference benchmarks of pretrained foundation checkpoints**.
  - Evaluated Checkpoints:
    - MarianMT: `Helsinki-NLP/opus-mt-en-de`, `opus-mt-en-fr`, `opus-mt-en-hi`, `opus-mt-en-mr`
    - mBART-50: `facebook/mbart-large-50-many-to-many-mmt`
    - NLLB-200: `facebook/nllb-200-distilled-600M`
  - In `run_pipeline.py` (lines 144–154), `model_wrapper = ModelManager.load_model(...)` was loaded and evaluated directly.
  - `src/training/trainer.py` contains the complete PyTorch training engine (`NMTTrainer`), but multi-epoch training was not invoked.
  - Resource logs in `experiments/*/resource_metrics.json` record total elapsed execution times of 15–30 seconds per direction, confirming inference-only passes.
- **Evidence**:
  - `experiments/MARIAN_EN_DE_OPUS_42_001/resource_metrics.json`: `elapsed_seconds: 17.905`
  - `experiments/MBART50_EN_HI_IITB_42_001/resource_metrics.json`: `elapsed_seconds: 14.12`
  - Absence of `checkpoint_manifest.json` and `training.log` in all experiment folders.

---

### 4. Was the test set isolated?
- **Finding**: **Yes**.
  - `DatasetSplitter.split()` partitions datasets by normalized SHA-256 source sentence hashes.
  - Zero test examples or validation examples were exposed to model weights or used for threshold selection.
- **Evidence**:
  - `src/data/splitter.py`: Lines 35–65
  - `data/processed/*/freeze_manifest.json`: `leakage_audit` sections confirm 0 overlap across all sets.

---

### 5. Data Integrity & Split Verification
- **Finding**: **Pass**. All 4 canonical datasets are intact and validated against their frozen SHA-256 manifests.

| Split | EN-DE Checksum (SHA-256) | EN-FR Checksum (SHA-256) | EN-HI Checksum (SHA-256) | EN-MR Checksum (SHA-256) |
|---|---|---|---|---|
| **Train** | `87b57002c2d3d877353f2bfa...` | `19facc51d153539cb1c824e3...` | `996807f9c2344d22af160d01...` | `5956ff8548a3bd8c6440b19e...` |
| **Val** | `479e783613aa16dd02d7c3eb...` | `513f45e6c893ec64dcb6001a...` | `5a037ac871d65e87b7e160a3...` | `a6741d09148a8d41ce7945d3...` |
| **Test** | `db5624e081c35090bfbd45c5...` | `47308d821e87bbaec0edc601...` | `9b0a400a514b4db1ff66a58f...` | `18e23787adcaddfc89466641...` |
| **Disk Verification** | **MATCH** | **MATCH** | **MATCH** | **MATCH** |

---

### 6. Metric & Generation Settings Audit
- **SacreBLEU**:
  - Version: `2.6.0`
  - Tokenizer: `13a`
  - Reference stream format: `[references]` (fixed and verified).
- **METEOR**:
  - Implementation: `nltk.translate.meteor_score.meteor_score`
  - Version: `nltk-3.10.3`
- **COMET**:
  - Model: `Unbabel/wmt22-comet-da`
  - Version: `comet-2.2.7`
  - Execution: GPU accelerated on RTX 5050 (`cuda`).
- **Generation Parameters**:
  - `num_beams = 5`
  - `max_new_tokens = 128`
  - `length_penalty = 1.0`
  - `no_repeat_ngram_size = 3`
  - `do_sample = False`
  - `early_stopping = True`
- **Language Code Configuration**:
  - mBART-50: `de_DE`, `fr_XX`, `hi_IN`, `mr_IN`, `en_XX`
  - NLLB-200: `deu_Latn`, `fra_Latn`, `hin_Deva`, `mar_Deva`, `eng_Latn`

---

### 7. Hardware & GPU Acceleration Audit
- **GPU**: NVIDIA GeForce RTX 5050 Laptop GPU (8,151 MiB VRAM)
- **Compute Capability**: `sm_120` (Blackwell architecture)
- **PyTorch Build**: `2.12.0.dev20260408+cu128`
- **Why this build was required**: Stable PyTorch releases (e.g., PyTorch 2.6 CUDA 12.4/12.6) do not include binary kernels for Blackwell (`sm_120`), raising `CUDA error: no kernel image is available for execution on the device`. The nightly `cu128` build compiles PTX/CUBIN targets up to `sm_120` (`CUDA Arch List: ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']`).
- **VRAM Utilization**: Peak memory during chunked inference (`batch_size=8`) was measured at **3,534.6 MB**, remaining well within the 8,151 MiB hardware limit.

---

## 3. Corrected 100-Sample Subset Benchmark Results

With the SacreBLEU reference list bug resolved, here are the true metrics for the 100-sample subset benchmark:

| Experiment ID | Model | Direction | Dataset | Eval N | SacreBLEU (13a) | METEOR | COMET (`wmt22-da`) | Peak VRAM |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `MARIAN_EN_DE_OPUS_42_001` | MarianMT | EN → DE | OPUS | 100 | **15.12** | 31.23 | 72.64 | 925.5 MB |
| `MARIAN_EN_FR_OPUS_42_001` | MarianMT | EN → FR | OPUS | 100 | **19.66** | 36.73 | 74.10 | 796.1 MB |
| `MARIAN_EN_HI_IITB_42_001` | MarianMT | EN → HI | IITB | 100 | **80.89** | 77.19 | 88.01 | 507.1 MB |
| `MARIAN_EN_MR_OPUS_42_001` | MarianMT | EN → MR | OPUS | 100 | **23.39** | 25.25 | 63.82 | 498.5 MB |
| `MBART50_EN_DE_OPUS_42_001` | mBART-50 | EN → DE | OPUS | 100 | **13.73** | 29.96 | 72.20 | 3,534.6 MB |
| `MBART50_EN_FR_OPUS_42_001` | mBART-50 | EN → FR | OPUS | 100 | **14.83** | 31.09 | 70.77 | 3,303.6 MB |
| `MBART50_EN_HI_IITB_42_001` | mBART-50 | EN → HI | IITB | 100 | **21.71** | 38.49 | 69.52 | 2,694.2 MB |
| `MBART50_EN_MR_OPUS_42_001` | mBART-50 | EN → MR | OPUS | 100 | **4.89** | 13.45 | 60.69 | 2,858.1 MB |
| `NLLB200_EN_DE_OPUS_42_001` | NLLB-200 | EN → DE | OPUS | 100 | **15.42** | 30.45 | 70.88 | 3,527.6 MB |
| `NLLB200_EN_FR_OPUS_42_001` | NLLB-200 | EN → FR | OPUS | 100 | **19.07** | 35.40 | 74.93 | 3,210.3 MB |
| `NLLB200_EN_HI_IITB_42_001` | NLLB-200 | EN → HI | IITB | 100 | **4.94** | 19.84 | 62.51 | 2,737.6 MB |
| `NLLB200_EN_MR_OPUS_42_001` | NLLB-200 | EN → MR | OPUS | 100 | **24.46** | 32.74 | 76.83 | 2,754.5 MB |

---

## 4. Audit Conclusions & Recommendations

1. **Test Size**: The 100-sample evaluations are valid as **smoke-test subset benchmarks**, but should be designated as such. Full-scale evaluations on the complete test sets (`1,015` to `4,914` sentences) should be run when final full-corpus metrics are required.
2. **Evaluation Metric Integrity**: The SacreBLEU formatting bug has been fixed in `src/evaluation/bleu.py` and validated through unit tests.
3. **Model Characterization**: The 12 experiments represent foundation model zero-shot/baseline capabilities. The `NMTTrainer` infrastructure is present and tested, and can be used for domain fine-tuning when training compute is scheduled.
4. **Hardware & Data Verification**: Both data integrity (0.00% leakage, SHA-256 matched) and hardware support (NVIDIA RTX 5050 `sm_120` CUDA 12.8 acceleration) pass all audit criteria.
