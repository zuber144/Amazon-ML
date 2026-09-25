"""
src/threshold.py
================
Decision layer: converts match probabilities into binary match decisions.
Keeps threshold selection separate from the model so it can be tuned
independently on the validation set.

The primary optimization target is F0.5 (precision-heavy).

Owner: Team Member 3 (ML Matching)
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# METRIC: F0.5
# ---------------------------------------------------------------------------

def f05_score(precision: float, recall: float) -> float:
    """Compute F0.5 given precision and recall.

    F0.5 = (1.25 * P * R) / (0.25 * P + R)
    """
    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0
    return (1.25 * precision * recall) / denom


def compute_pair_level_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Compute precision, recall, F1, and F0.5 from binary arrays.

    Parameters
    ----------
    y_true: np.ndarray of 0/1
    y_pred: np.ndarray of 0/1

    Returns
    -------
    dict with keys: precision, recall, f1, f0_5, tp, fp, fn, tn
    """
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    f05 = f05_score(precision, recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f0_5": f05,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# ---------------------------------------------------------------------------
# THRESHOLD TUNING (grid search over the validation set)
# ---------------------------------------------------------------------------

def tune_threshold(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    metric: str = None,
    search_min: float = None,
    search_max: float = None,
    search_steps: int = None,
) -> tuple[float, dict]:
    """Grid search for the optimal decision threshold on the validation set.

    Parameters
    ----------
    probabilities: np.ndarray
        Match probabilities from model.predict_proba().
    y_true: np.ndarray
        Ground-truth binary labels (1 = match, 0 = no match).
    metric: str
        Metric to maximize. Default from config ("f0_5").
    search_min, search_max: float
        Search range for threshold.
    search_steps: int
        Number of threshold candidates.

    Returns
    -------
    tuple of:
        best_threshold (float)
        best_metrics (dict)
    """
    metric = metric or config.THRESHOLD_METRIC
    search_min = search_min if search_min is not None else config.THRESHOLD_SEARCH_MIN
    search_max = search_max if search_max is not None else config.THRESHOLD_SEARCH_MAX
    search_steps = search_steps or config.THRESHOLD_SEARCH_STEPS

    thresholds = np.linspace(search_min, search_max, search_steps)
    best_threshold = config.DEFAULT_THRESHOLD
    best_score = -1.0
    best_metrics = {}

    for thr in thresholds:
        y_pred = (probabilities >= thr).astype(int)
        metrics = compute_pair_level_metrics(y_true, y_pred)
        score = metrics[metric]
        if score > best_score:
            best_score = score
            best_threshold = float(thr)
            best_metrics = metrics

    logger.info(
        "Threshold tuning complete: best_threshold=%.4f, %s=%.4f",
        best_threshold, metric, best_score,
    )
    return best_threshold, best_metrics


# ---------------------------------------------------------------------------
# DECISION APPLICATION
# ---------------------------------------------------------------------------

def apply_threshold(
    pairs: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Apply a threshold to produce binary match predictions.

    Parameters
    ----------
    pairs: pd.DataFrame
        Must have columns: source1_entity_id, candidate_entity_id.
    probabilities: np.ndarray
        Match probability per pair.
    threshold: float
        Pairs with probability >= threshold are predicted as matches.

    Returns
    -------
    pd.DataFrame with columns: source1_entity_id, candidate_entity_id,
        probability, is_match.
    """
    result = pairs[["source1_entity_id", "candidate_entity_id"]].copy()
    result["probability"] = probabilities
    result["is_match"] = (probabilities >= threshold).astype(int)
    return result


def get_positive_pairs(
    pairs_with_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Return only the rows predicted as matches (is_match == 1).

    Parameters
    ----------
    pairs_with_predictions: pd.DataFrame
        Output of apply_threshold().

    Returns
    -------
    pd.DataFrame of matched pairs.
    """
    return pairs_with_predictions[pairs_with_predictions["is_match"] == 1].copy()


# ---------------------------------------------------------------------------
# SERIALIZATION
# ---------------------------------------------------------------------------

def save_threshold(threshold: float, path: Optional[Path] = None) -> None:
    """Save the tuned threshold to a JSON file."""
    path = Path(path) if path else config.THRESHOLD_SAVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"threshold": threshold}, f)
    logger.info("Threshold %.4f saved to: %s", threshold, path)


def load_threshold(path: Optional[Path] = None) -> float:
    """Load the threshold from a JSON file.

    Returns config.DEFAULT_THRESHOLD if the file does not exist.
    """
    path = Path(path) if path else config.THRESHOLD_SAVE_PATH
    if not path.is_file():
        logger.warning(
            "Threshold file not found: %s. Using default: %.4f",
            path, config.DEFAULT_THRESHOLD,
        )
        return config.DEFAULT_THRESHOLD
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    threshold = float(data["threshold"])
    logger.info("Threshold %.4f loaded from: %s", threshold, path)
    return threshold
