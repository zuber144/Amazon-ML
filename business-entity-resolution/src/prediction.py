"""
src/prediction.py
=================
Runs inference on a set of candidate pairs and returns match predictions.
Ties together: features -> model -> threshold -> decisions.

Owner: Team Member 4 (Evaluation / Integration)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src import config, features, model, threshold

logger = logging.getLogger(__name__)


def predict(
    pairs: pd.DataFrame,
    s1: pd.DataFrame,
    candidates: pd.DataFrame,
    trained_model,
    decision_threshold: float,
    batch_size: int = 50_000,
) -> pd.DataFrame:
    """Run end-to-end prediction on candidate pairs.

    Parameters
    ----------
    pairs: pd.DataFrame
        Candidate pairs with columns: source1_entity_id, candidate_entity_id.
    s1: pd.DataFrame
        Preprocessed Source 1 records.
    candidates: pd.DataFrame
        Combined S2 + S3 preprocessed records.
    trained_model:
        Model loaded/trained via src.model.
    decision_threshold: float
        Decision threshold from src.threshold.
    batch_size: int
        Feature extraction and scoring batch size to manage memory.

    Returns
    -------
    pd.DataFrame with columns:
        source1_entity_id, candidate_entity_id, probability, is_match
    """
    logger.info("Running prediction on %d candidate pairs...", len(pairs))

    if len(pairs) == 0:
        logger.warning("No candidate pairs to predict on.")
        return pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id", "probability", "is_match"]
        )

    # Process in batches to manage memory
    all_results = []
    n = len(pairs)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_pairs = pairs.iloc[start:end]
        logger.debug("Scoring batch %d-%d / %d...", start, end, n)

        X_batch = features.extract_features_for_pairs(batch_pairs, s1, candidates)
        proba_batch = model.predict_proba(trained_model, X_batch)
        result_batch = threshold.apply_threshold(batch_pairs, proba_batch, decision_threshold)
        all_results.append(result_batch)

    result = pd.concat(all_results, ignore_index=True)
    n_matches = result["is_match"].sum()
    logger.info(
        "Prediction complete: %d matches out of %d pairs (threshold=%.4f).",
        n_matches, len(result), decision_threshold,
    )
    return result
