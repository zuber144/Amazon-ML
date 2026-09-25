# Business Entity Resolution

ML pipeline for the Amazon ML Challenge 2026 — Business Entity Resolution.

> **Quickstart Setup**:
> - **Windows (PowerShell)**: `.\setup_venv.ps1`
> - **Windows (CMD)**: `setup_venv.bat`
> - **Linux / macOS**: `chmod +x setup_venv.sh && ./setup_venv.sh`
> - Full root guide & troubleshooting: [`../README.md`](../README.md) or jump to [Section 11: How to Run](#11-how-to-run).

---

## 1. Problem Statement

Given business records from three independent data sources (Source 1, Source 2, Source 3) with noisy, inconsistent fields, determine which records across sources refer to the same real-world business entity.

- **Source 1** is the deduplicated reference/master source.
- **Source 2 and Source 3** are noisy duplicates.
- A Source 1 entity may match **zero, one, or many** records from S2/S3.

The primary metric is **macro-averaged F0.5** (precision-heavy).

---

## 2. Dataset Structure

```
dataset/
├── train/
│   ├── train_source1.tsv      (2.2M records: US, India)
│   ├── train_source2.tsv      (5.0M records)
│   ├── train_source3.tsv      (5.3M records)
│   └── train_ground_truth.tsv (2.2M rows; ~123K singletons)
└── test/
    ├── test_source1.tsv       (1.73M records: US, India, France)
    ├── test_source2.tsv       (4.9M records)
    └── test_source3.tsv       (5.1M records)
```

All files are **tab-separated** (`.tsv`). Always read with `sep="\t"`.

Each source file contains:
| Column | Description |
|--------|-------------|
| `entity_id` | Unique ID. Prefix `S1-` / `S2-` / `S3-` indicates source. |
| `business_name` | Business name (noisy, abbreviated, possibly non-ASCII) |
| `business_address` | Address (abbreviations, transliterations, landmarks) |
| `country` | Open-set string. Training: `US`, `India`. Test adds `France`. |

Ground truth:
| Column | Description |
|--------|-------------|
| `source1_entity_id` | S1 entity ID |
| `matched_entity_ids` | Comma-separated S2/S3 IDs (empty for singletons) |

---

## 3. Pipeline Architecture

```
RAW TSV DATA
     │
     ▼
DATA LOADER (src/data_loader.py)
     │
     ▼
PREPROCESSING (src/preprocessing.py + src/normalization.py)
  - Name normalization
  - Address normalization
  - Country normalization
  - Tokenization
     │
     ▼
BLOCKING ENGINE (src/blocking.py + src/candidate_generation.py)
  - Exact name blocking
  - Token overlap blocking
  - TF-IDF n-gram blocking
  - Address token blocking
  - Country-aware blocking
     │
     ▼
CANDIDATE PAIRS (S1 ↔ S2/S3)
     │
     ▼
FEATURE ENGINE (src/features.py)
  - Name similarities (Jaccard, edit, n-gram, TF-IDF)
  - Address similarities
  - Country features
  - Missingness indicators
     │
     ▼
MATCHING MODEL (src/model.py)
  - Logistic Regression (baseline)
  - Random Forest / LightGBM (experimental)
     │
     ▼
DECISION ENGINE (src/threshold.py)
  - Threshold tuned for F0.5 on validation set
  - Singleton handling (empty = no match)
     │
     ▼
POST-PROCESSING (src/postprocessing.py)
  - Deduplication
  - Valid ID enforcement
  - Candidate subset enforcement
     │
     ▼
OUTPUT (src/output.py)
  ├── output/matching_results.tsv
  └── output/candidate_pairs.tsv
```

---

## 4. Normalization

**Name normalization** (`normalize_name`):
- Unicode NFC
- Lowercase
- Punctuation normalization
- Legal suffix canonicalization: `Corporation → Corp`, `Private Limited → Pvt Ltd`, `LLC → llc`, etc.
- `and` / `&` normalization
- Whitespace normalization

**Address normalization** (`normalize_address`):
- Lowercase
- Punctuation normalization
- Road/street abbreviations: `Road → Rd`, `Street → St`, `Avenue → Ave`, etc.
- Directional abbreviations: `North → N`, etc.
- Preserves meaningful numbers (house numbers, postal codes)

**Country**: lowercase + strip. Country is an **open set** — France and any future country pass through unchanged.

---

## 5. Blocking

Blocking is critical at this scale (>2M S1 × >10M S2/S3 = impossible all-pairs comparison).

Multiple strategies are unioned:

| Strategy | Description |
|---|---|
| `exact_name` | Exact normalized name match |
| `token_name` | Inverted index on name tokens (≥1 shared token) |
| `tfidf_name` | Top-K TF-IDF character n-gram similarity |
| `token_address` | Inverted index on address tokens (≥2 shared) |
| `country_aware` | Apply inner strategy per-country group |

Configure in `src/config.py`:
```python
BLOCKING_STRATEGIES = ["exact_name", "token_name", "tfidf_name", "token_address"]
MAX_CANDIDATES_PER_S1 = 500
```

---

## 6. Feature Engineering

26 pairwise features per candidate pair:

- **Name**: exact match, Jaccard, token overlap (both directions), edit similarity, partial ratio, token sort ratio, bigram/trigram overlap, length differences
- **Address**: Jaccard, token overlap, edit similarity, trigram overlap, length difference
- **Country**: exact match (open-set, no hard-coding)
- **Missingness**: 6 indicators (missing name/address/country for S1 and candidate)
- **Source**: is_S2 indicator

---

## 7. Model

**Baseline**: Logistic Regression with StandardScaler (sklearn Pipeline)

**Experimental**: Random Forest, LightGBM (MIT/Apache 2.0, <8B params)

Configure via `src/config.py`:
```python
MODEL_TYPE = "logistic_regression"  # or "random_forest" or "gradient_boosting"
```

---

## 8. Threshold Tuning

Threshold is tuned on the validation set to maximize **F0.5**:
```python
# Grid search between 0.1 and 0.95
THRESHOLD_SEARCH_MIN = 0.1
THRESHOLD_SEARCH_MAX = 0.95
THRESHOLD_SEARCH_STEPS = 85
THRESHOLD_METRIC = "f0_5"
```

Saved to `models/threshold.json`. Override with `--threshold` flag.

---

## 9. Validation

Reproducible split (seed=42, 15% held out):
```python
VALIDATION_FRACTION = 0.15
RANDOM_SEED = 42
STRATIFY_VALIDATION = True  # preserves singleton ratio
```

Reports:
- Macro Precision / Recall / F0.5
- Candidate recall (upper bound on matching recall)
- Singleton accuracy

---

## 10. Output Format

### matching_results.tsv
```
source1_entity_id\tmatched_entity_ids
S1-00001\tS2-00047,S3-00812
S1-00002\tS3-00004
S1-00003\t
```

### candidate_pairs.tsv
```
source1_entity_id\tcandidate_entity_ids
S1-00001\tS2-00047,S3-00812,S3-00999
S1-00002\tS3-00004
S1-00003\t
```

Rules:
- Every S1 entity has exactly one row
- Empty `matched_entity_ids` = singleton (no match predicted)
- Only S2-/S3- IDs, no self-matches
- `matching_results ⊆ candidate_pairs`

---

## 11. How to Run

### Setup & Virtual Environment

To ensure seamless execution across any device or OS without version conflicts, use a dedicated virtual environment with the verified pinned dependencies (`requirements-lock.txt`):

#### Automatic Setup (Recommended)
- **Windows (Command Prompt)**:
  ```cmd
  setup_venv.bat
  ```
- **Windows (PowerShell)**:
  ```powershell
  .\setup_venv.ps1
  ```
- **Linux / macOS / WSL**:
  ```bash
  chmod +x setup_venv.sh
  ./setup_venv.sh
  ```

#### Manual Setup
```bash
# 1. Create virtual environment
python -m venv .venv

# 2. Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Windows (CMD):
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# 3. Install verified pinned dependencies
pip install --upgrade pip
pip install -r requirements-lock.txt

# Or install minimum requirement bounds:
pip install -r requirements.txt
```

### Symlink or copy dataset files
Place the provided TSV files in:
```
dataset/train/train_source1.tsv
dataset/train/train_source2.tsv
dataset/train/train_source3.tsv
dataset/train/train_ground_truth.tsv
dataset/test/test_source1.tsv
dataset/test/test_source2.tsv
dataset/test/test_source3.tsv
```

### Run pipeline
```bash
# Full pipeline: train + validate + predict
python run_pipeline.py --mode all

# Train only (saves model and threshold)
python run_pipeline.py --mode train

# Validate on held-out split
python run_pipeline.py --mode validate

# Generate test predictions (requires saved model)
python run_pipeline.py --mode predict

# Override model and blocking
python run_pipeline.py --mode train --model gradient_boosting --blocking exact_name token_name tfidf_name
```

### Run tests
```bash
python -m pytest tests/ -v
```

### Validate submission
```bash
cd E:\Amazon ML\student_resource
python utils/validate_submission.py --matching ../business-entity-resolution/output/matching_results.tsv --candidate ../business-entity-resolution/output/candidate_pairs.tsv --test-dir dataset/test
```

---

## 12. Team Responsibilities

| Member | Owns |
|--------|------|
| **Member 1** (Data/Normalization) | `src/data_loader.py`, `src/normalization.py`, `src/preprocessing.py` |
| **Member 2** (Blocking) | `src/blocking.py`, `src/candidate_generation.py` |
| **Member 3** (ML Matching) | `src/features.py`, `src/model.py`, `src/threshold.py` |
| **Member 4** (Integration/Eval) | `validation/`, `src/prediction.py`, `src/postprocessing.py`, `src/output.py`, `src/pipeline.py` |

**Avoid circular imports**: each module only imports from modules to its left in the dependency chain.

---

## 13. Fair-Play Restrictions

**STRICTLY PROHIBITED**:
- External business databases or APIs
- Google Maps / geocoding APIs
- Government business registries
- Commercial entity-resolution APIs
- Any internet-based data augmentation

The pipeline uses only the provided training data and locally computed features.

---

## 14. Reproducibility

- All random operations seeded with `RANDOM_SEED = 42` (configurable in `src/config.py`)
- Validation split is deterministic given the same seed
- Model is saved to `models/matcher_model.joblib`
- Threshold is saved to `models/threshold.json`
- All paths centralized in `src/config.py`

---

## 15. Development Phases

- [x] Phase 1: Project structure
- [x] Phase 2: Data loading
- [x] Phase 3: Normalization
- [x] Phase 4: Baseline blocking
- [x] Phase 5: Basic similarity features
- [x] Phase 6: Baseline matching model
- [x] Phase 7: Validation and F0.5
- [x] Phase 8: Threshold tuning
- [ ] Phase 9: Improved blocking strategies
- [ ] Phase 10: Additional features / models
- [ ] Phase 11: Test predictions
- [ ] Phase 12: Validate submission
- [ ] Phase 13: Final reproducible package
