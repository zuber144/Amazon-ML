"""
src/config.py
=============
Centralized configuration for the Business Entity Resolution pipeline.

All paths, hyperparameters, and tunable knobs live here.
No other module should hard-code dataset paths or magic numbers.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# ROOT PATHS
# ---------------------------------------------------------------------------
# Resolve the project root as the parent of this file's directory (src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# DATASET PATHS
# ---------------------------------------------------------------------------
# Primary: look for dataset in the project directory.
# Fallback: look in the student_resource directory (where the actual data is).
DATASET_DIR = PROJECT_ROOT / "dataset"

_STUDENT_RESOURCE = PROJECT_ROOT.parent / "student_resource" / "dataset"

def _find_file(local_path: Path, fallback_path: Path) -> Path:
    """Return local path if it exists, otherwise the fallback."""
    if local_path.is_file():
        return local_path
    if fallback_path.is_file():
        return fallback_path
    # Return local path (will raise FileNotFoundError at load time with a clear message)
    return local_path

TRAIN_DIR = DATASET_DIR / "train"
TRAIN_SOURCE1 = _find_file(TRAIN_DIR / "train_source1.tsv", _STUDENT_RESOURCE / "train" / "train_source1.tsv")
TRAIN_SOURCE2 = _find_file(TRAIN_DIR / "train_source2.tsv", _STUDENT_RESOURCE / "train" / "train_source2.tsv")
TRAIN_SOURCE3 = _find_file(TRAIN_DIR / "train_source3.tsv", _STUDENT_RESOURCE / "train" / "train_source3.tsv")
TRAIN_GROUND_TRUTH = _find_file(TRAIN_DIR / "train_ground_truth.tsv", _STUDENT_RESOURCE / "train" / "train_ground_truth.tsv")

TEST_DIR = DATASET_DIR / "test"
TEST_SOURCE1 = _find_file(TEST_DIR / "test_source1.tsv", _STUDENT_RESOURCE / "test" / "test_source1.tsv")
TEST_SOURCE2 = _find_file(TEST_DIR / "test_source2.tsv", _STUDENT_RESOURCE / "test" / "test_source2.tsv")
TEST_SOURCE3 = _find_file(TEST_DIR / "test_source3.tsv", _STUDENT_RESOURCE / "test" / "test_source3.tsv")

# ---------------------------------------------------------------------------
# OUTPUT PATHS
# ---------------------------------------------------------------------------
OUTPUT_DIR = PROJECT_ROOT / "output"
MATCHING_RESULTS_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_PATH = OUTPUT_DIR / "candidate_pairs.tsv"

# ---------------------------------------------------------------------------
# MODEL PERSISTENCE
# ---------------------------------------------------------------------------
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_SAVE_PATH = MODELS_DIR / "matcher_model.joblib"
THRESHOLD_SAVE_PATH = MODELS_DIR / "threshold.json"

# ---------------------------------------------------------------------------
# LOGS
# ---------------------------------------------------------------------------
LOGS_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOGS_DIR / "pipeline.log"

# ---------------------------------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# DATA SCHEMA
# ---------------------------------------------------------------------------
SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]

SOURCE1_PREFIX = "S1-"
SOURCE2_PREFIX = "S2-"
SOURCE3_PREFIX = "S3-"

# ---------------------------------------------------------------------------
# VALIDATION SPLIT
# ---------------------------------------------------------------------------
VALIDATION_FRACTION = 0.15
STRATIFY_VALIDATION = True

# ---------------------------------------------------------------------------
# BLOCKING PARAMETERS
# ---------------------------------------------------------------------------
BLOCKING_MIN_TOKEN_LENGTH = 3
MAX_CANDIDATES_PER_S1 = 500
TOKEN_OVERLAP_MIN_SHARED = 1
NGRAM_RANGE = (2, 3)
TFIDF_TOP_K_CANDIDATES = 10
TFIDF_BATCH_SIZE = 5000

BLOCKING_STRATEGIES = [
    "exact_name",
    "token_name",
    "tfidf_name",
    "token_address",
]

# ---------------------------------------------------------------------------
# FEATURE ENGINEERING PARAMETERS
# ---------------------------------------------------------------------------
FEATURE_TFIDF_MAX_FEATURES = 50_000
MAX_STRING_LENGTH_FOR_EDIT = 200

# ---------------------------------------------------------------------------
# MODEL PARAMETERS
# ---------------------------------------------------------------------------
MODEL_TYPE = "logistic_regression"

LOGISTIC_REGRESSION_PARAMS = {
    "C": 1.0,
    "max_iter": 1000,
    "solver": "lbfgs",
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_leaf": 5,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}

GRADIENT_BOOSTING_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 8,
    "num_leaves": 63,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# ---------------------------------------------------------------------------
# THRESHOLD / DECISION LAYER
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLD = 0.5
THRESHOLD_SEARCH_MIN = 0.1
THRESHOLD_SEARCH_MAX = 0.95
THRESHOLD_SEARCH_STEPS = 85
THRESHOLD_METRIC = "f0_5"

# ---------------------------------------------------------------------------
# POST-PROCESSING
# ---------------------------------------------------------------------------
ENFORCE_MATCH_SUBSET_OF_CANDIDATES = True

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
LOG_LEVEL = "INFO"
LOG_TO_FILE = True
LOG_TO_CONSOLE = True


def _ensure_dirs():
    for d in [OUTPUT_DIR, MODELS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


_ensure_dirs()
