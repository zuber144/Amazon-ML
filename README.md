# Business Entity Resolution — Amazon ML Challenge 2026

An end-to-end Machine Learning pipeline for the **Amazon ML Challenge 2026: Business Entity Resolution Challenge**.

This repository contains:
- `business-entity-resolution/`: Complete modular ML pipeline (preprocessing, blocking, feature extraction, ML classification, threshold tuning, and submission formatting).
- `student_resource/`: Official challenge dataset (`dataset/`), validation script (`utils/validate_submission.py`), and competition documentation.

---

## Quick Navigation

- [1. System Requirements](#1-system-requirements)
- [2. Quickstart (1-Click Setup)](#2-quickstart-1-click-setup)
- [3. Manual Setup (Step-by-Step)](#3-manual-setup-step-by-step)
- [4. Verifying the Installation](#4-verifying-the-installation)
- [5. Dataset Configuration](#5-dataset-configuration)
- [6. Running the Pipeline](#6-running-the-pipeline)
- [7. Validating Submission Files](#7-validating-submission-files)
- [8. Project Structure](#8-project-structure)
- [9. Troubleshooting & FAQ](#9-troubleshooting--faq)

---

## 1. System Requirements

- **Operating System**: Windows 10/11, Ubuntu/Debian/Linux, or macOS
- **Python**: **Python 3.12** is strongly recommended and verified (Python 3.10+ supported)
- **RAM**: Minimum 8 GB recommended (16 GB+ recommended for full training and candidate indexing)
- **Storage**: ~5 GB free space for dataset, virtual environment, and generated candidate pairs

---

## 2. Quickstart (1-Click Setup)

Automated setup scripts are provided in the `business-entity-resolution/` directory. They will automatically:
1. Create a local `.venv` virtual environment.
2. Upgrade `pip`.
3. Install exact verified dependencies from `requirements-lock.txt`.
4. Run the full unit test suite (47 tests) to confirm everything works.

### Windows (PowerShell)
```powershell
cd business-entity-resolution
.\setup_venv.ps1
```
> *Note for PowerShell*: If script execution is restricted on your machine, enable it for your current session:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

### Windows (Command Prompt)
```cmd
cd business-entity-resolution
setup_venv.bat
```

### Linux / macOS / WSL
```bash
cd business-entity-resolution
chmod +x setup_venv.sh
./setup_venv.sh
```

---

## 3. Manual Setup (Step-by-Step)

If you prefer to configure the environment manually:

### Step 1: Navigate to the pipeline directory
```bash
cd business-entity-resolution
```

### Step 2: Create a virtual environment
```bash
# Windows
python -m venv .venv

# Linux / macOS
python3 -m venv .venv
```

### Step 3: Activate the virtual environment
- **Windows (PowerShell)**:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- **Windows (Command Prompt)**:
  ```cmd
  .venv\Scripts\activate.bat
  ```
- **Linux / macOS**:
  ```bash
  source .venv/bin/activate
  ```

### Step 4: Upgrade pip & install dependencies
To install the **exact, battle-tested versions** verified with Python 3.12:
```bash
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
```

Alternatively, to install using flexible minimum versions:
```bash
pip install -r requirements.txt
```

---

## 4. Verifying the Installation

With your virtual environment activated, run the full test suite:

```bash
pytest tests/ -v
```

Or run directly using the virtual environment's Python executable without activating:
```powershell
# Windows
& ".\.venv\Scripts\python.exe" -m pytest tests/ -v

# Linux / macOS
./.venv/bin/python -m pytest tests/ -v
```

Expected result:
```
============================= 47 passed in ~2.5s ==============================
```

All 47 unit tests verify:
- Unicode & text normalization (NFC, ampersands, legal suffixes, addresses, country open-set handling).
- Multi-pass blocking (no `S1-S1` leaks, token overlap, country awareness).
- Pairwise feature extraction (26 features, bounds `[0, 1]`, NaN-free guarantees).
- Submission output format integrity (singleton handling, candidate subset enforcement).

---

## 5. Dataset Configuration

The pipeline's configuration ([`src/config.py`](file:///E:/Amazon%20ML/business-entity-resolution/src/config.py)) has **automatic path resolution**:

1. **Automatic Resolution**: The pipeline automatically detects the dataset inside `student_resource/dataset/` as a fallback if `business-entity-resolution/dataset/` is empty. You do not need to move or duplicate the ~1 GB dataset files!
2. **Local Directory (Optional)**: If you prefer dataset files inside the project folder:
   - Create a directory junction on Windows:
     ```cmd
     mklink /J "business-entity-resolution\dataset" "student_resource\dataset"
     ```
   - Or create a symlink on Linux/macOS:
     ```bash
     ln -s ../student_resource/dataset business-entity-resolution/dataset
     ```

### Expected Dataset Layout
```
student_resource/dataset/  (or business-entity-resolution/dataset/)
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

---

## 6. Running the Pipeline

All pipeline operations are executed via [`run_pipeline.py`](file:///E:/Amazon%20ML/business-entity-resolution/run_pipeline.py) inside `business-entity-resolution/`.

### 1. End-to-End Execution (Train + Validate + Predict)
Runs the complete workflow: loads data, generates candidate pairs via blocking, extracts pairwise similarity features, trains the classifier, tunes the decision threshold on validation data for macro $F_{0.5}$, and generates test submission files:
```bash
python run_pipeline.py --mode all
```

### 2. Train Only
Trains the matching model and saves the trained artifact to `models/matcher_model.joblib`:
```bash
python run_pipeline.py --mode train
```

### 3. Validate Only
Evaluates performance against a held-out, stratified 15% validation split and prints precision, recall, candidate recall upper bound, and macro $F_{0.5}$:
```bash
python run_pipeline.py --mode validate
```

### 4. Predict / Generate Test Submissions
Generates submission files from `test_source1.tsv`, `test_source2.tsv`, and `test_source3.tsv`:
```bash
python run_pipeline.py --mode predict
```
This produces:
- `output/matching_results.tsv` (Leaderboard submission)
- `output/candidate_pairs.tsv` (Blocking candidate pairs)

### 5. CLI Customization & Experiments
You can override models, blocking strategies, and thresholds directly from the CLI:
```bash
# Example: Use Random Forest with specific blocking strategies
python run_pipeline.py --mode train --model random_forest --blocking exact_name token_name tfidf_name

# Example: Run prediction with a fixed decision threshold
python run_pipeline.py --mode predict --threshold 0.65
```

---

## 7. Validating Submission Files

Before uploading to the competition portal, run the official validation script provided by Amazon:

```bash
# From workspace root:
python student_resource/utils/validate_submission.py `
  --matching business-entity-resolution/output/matching_results.tsv `
  --candidate business-entity-resolution/output/candidate_pairs.tsv `
  --test-dir student_resource/dataset/test
```

This verifies:
- Exactly matches all S1 test entities (including singletons with blank matches).
- Tab-separated format without quoting anomalies.
- All matched IDs belong to the candidate set (`matching_results ⊆ candidate_pairs`).
- No self-matches (`S1-*` matching `S1-*`).

---

## 8. Project Structure

```
Amazon ML/
├── README.md                          <-- You are here (Repository Overview & Setup)
├── student_resource/                  <-- Official challenge resources
│   ├── dataset/                       <-- TSV datasets (train & test)
│   ├── utils/
│   │   └── validate_submission.py     <-- Official submission validator
│   ├── Documentation_template.md      <-- Competition report template
│   └── README.md                      <-- Competition problem statement & rules
│
└── business-entity-resolution/        <-- Core ML solution
    ├── .venv/                         <-- Virtual environment (ignored by git)
    ├── setup_venv.bat                 <-- 1-click setup for Windows CMD
    ├── setup_venv.ps1                 <-- 1-click setup for Windows PowerShell
    ├── setup_venv.sh                  <-- 1-click setup for Linux / macOS / WSL
    ├── requirements.txt               <-- General dependencies with minimum bounds
    ├── requirements-lock.txt          <-- Exact pinned dependencies verified for Python 3.12
    ├── run_pipeline.py                <-- Main CLI entry point
    ├── README.md                      <-- Pipeline architecture & technical specification
    │
    ├── src/                           <-- Modular pipeline source code
    │   ├── config.py                  <-- Centralized paths, thresholds, and hyperparameters
    │   ├── data_loader.py             <-- Memory-efficient TSV loading & chunking
    │   ├── normalization.py           <-- Text, address, and legal suffix normalization
    │   ├── preprocessing.py          <-- Dataset-wide cleaning and tokenization
    │   ├── blocking.py                <-- Multi-strategy candidate blocking
    │   ├── candidate_generation.py    <-- Candidate pair generation & filtering
    │   ├── features.py                <-- 26 pairwise similarity features
    │   ├── model.py                   <-- Logistic Regression / Random Forest classifiers
    │   ├── threshold.py               <-- F0.5-optimized threshold tuning
    │   ├── prediction.py              <-- Test inference engine
    │   ├── postprocessing.py          <-- Deduplication & candidate-subset enforcement
    │   ├── output.py                  <-- TSV submission file writer
    │   └── pipeline.py                <-- Orchestrator tying all phases together
    │
    ├── tests/                         <-- Pytest test suite (47 unit tests)
    │   ├── test_normalization.py
    │   ├── test_blocking.py
    │   ├── test_features.py
    │   └── test_output.py
    │
    ├── validation/                    <-- Evaluation metrics & validation split
    │   ├── metrics.py                 <-- Macro F0.5, Precision, Recall, Candidate Recall
    │   └── split.py                   <-- Stratified singleton-aware validation split
    │
    ├── models/                        <-- Saved model artifacts (.joblib, threshold.json)
    ├── output/                        <-- Submission output files (.tsv)
    └── logs/                          <-- Execution logs
```

---

## 9. Troubleshooting & FAQ

### Q1: `Activate.ps1 cannot be loaded because running scripts is disabled on this system`
**Solution**: Open PowerShell and run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
Then run `.\setup_venv.ps1` or `.\.venv\Scripts\Activate.ps1` again.

### Q2: Unicode / Character display issues in PowerShell
**Cause**: The Windows console code page may default to legacy OEM encoding.
**Solution**: The pipeline internally reads, processes, and writes UTF-8 data correctly. To fix PowerShell terminal display:
```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

### Q3: `No module named 'src'` when running custom scripts
**Solution**: Run python with `-m` or set `PYTHONPATH`:
```bash
# Windows PowerShell:
$env:PYTHONPATH = "E:\Amazon ML\business-entity-resolution"

# Linux / macOS:
export PYTHONPATH="."
```
Alternatively, always execute through `run_pipeline.py` or run `pytest` from `business-entity-resolution/`.

### Q4: Open-set country test data (France)
**Solution**: The pipeline is strictly designed for open-set countries. Test records from France or any unknown country are handled natively through string pass-through normalization and dynamic blocking. No country code is hard-coded or filtered out.
