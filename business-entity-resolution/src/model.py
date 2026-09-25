"""
src/model.py
============
Modular matching model that takes pairwise features and outputs match probabilities.

Owner: Team Member 3 (ML Matching)

The model is deliberately kept separate from threshold selection.
See src/threshold.py for the decision layer.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src import config

logger = logging.getLogger(__name__)

# Optional: try to import LightGBM for gradient_boosting model type
try:
    from lightgbm import LGBMClassifier as _LGBM
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False


# ---------------------------------------------------------------------------
# MODEL FACTORY
# ---------------------------------------------------------------------------

def build_model(model_type: Optional[str] = None) -> Pipeline:
    """Build an sklearn Pipeline containing a scaler and a classifier.

    Parameters
    ----------
    model_type: str, optional
        "logistic_regression" | "random_forest" | "gradient_boosting"
        Defaults to config.MODEL_TYPE.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    model_type = model_type or config.MODEL_TYPE

    if model_type == "logistic_regression":
        clf = LogisticRegression(**config.LOGISTIC_REGRESSION_PARAMS)
        steps = [("scaler", StandardScaler()), ("clf", clf)]
    elif model_type == "random_forest":
        clf = RandomForestClassifier(**config.RANDOM_FOREST_PARAMS)
        steps = [("clf", clf)]  # RF doesn't need scaling
    elif model_type == "gradient_boosting":
        if _HAS_LGBM:
            params = {k: v for k, v in config.GRADIENT_BOOSTING_PARAMS.items()
                      if k != "num_leaves"}
            params["num_leaves"] = config.GRADIENT_BOOSTING_PARAMS.get("num_leaves", 31)
            clf = _LGBM(**params)
        else:
            logger.warning("LightGBM not available; falling back to sklearn GradientBoostingClassifier.")
            params = {
                k: v for k, v in config.GRADIENT_BOOSTING_PARAMS.items()
                if k not in ("num_leaves", "n_jobs", "class_weight")
            }
            clf = GradientBoostingClassifier(**params)
        steps = [("clf", clf)]
    else:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. "
            f"Choose from: logistic_regression, random_forest, gradient_boosting"
        )

    logger.info("Building model: %s", model_type)
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: Optional[str] = None,
) -> Pipeline:
    """Train a matching model on labeled pair features.

    Parameters
    ----------
    X_train: pd.DataFrame
        Feature matrix (output of features.extract_features_for_pairs).
    y_train: pd.Series
        Binary labels: 1 = match, 0 = no match.
    model_type: str, optional

    Returns
    -------
    Trained sklearn Pipeline.
    """
    logger.info(
        "Training model on %d pairs (%d positive, %d negative)...",
        len(X_train), int(y_train.sum()), int((1 - y_train).sum()),
    )
    model = build_model(model_type)
    model.fit(X_train.values, y_train.values)
    logger.info("Training complete.")
    return model


# ---------------------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------------------

def predict_proba(
    model: Pipeline,
    X: pd.DataFrame,
) -> np.ndarray:
    """Predict match probabilities for candidate pairs.

    Parameters
    ----------
    model: Pipeline
        Trained model from train_model().
    X: pd.DataFrame
        Feature matrix.

    Returns
    -------
    np.ndarray of shape (n_pairs,): match probability for each pair.
    """
    proba = model.predict_proba(X.values)
    # Column index 1 = positive class (match)
    return proba[:, 1]


# ---------------------------------------------------------------------------
# SERIALIZATION
# ---------------------------------------------------------------------------

def save_model(model: Pipeline, path: Optional[Path] = None) -> None:
    """Save the trained model to disk using joblib.

    Parameters
    ----------
    model: Pipeline
    path: Path, optional
        Defaults to config.MODEL_SAVE_PATH.
    """
    path = Path(path) if path else config.MODEL_SAVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Model saved to: %s", path)


def load_model(path: Optional[Path] = None) -> Pipeline:
    """Load a trained model from disk.

    Parameters
    ----------
    path: Path, optional
        Defaults to config.MODEL_SAVE_PATH.

    Returns
    -------
    Trained Pipeline.

    Raises
    ------
    FileNotFoundError if the model file does not exist.
    """
    path = Path(path) if path else config.MODEL_SAVE_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {path}. "
            "Run the pipeline in 'train' mode first."
        )
    model = joblib.load(path)
    logger.info("Model loaded from: %s", path)
    return model
