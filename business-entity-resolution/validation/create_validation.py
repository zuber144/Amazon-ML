"""
validation/create_validation.py
================================
Creates a reproducible validation split from the training data.

Owner: Team Member 4 (Evaluation / Integration)

IMPORTANT: Ground truth labels are NEVER used during feature generation
or blocking -- only for label assignment after candidate pairs are generated.
"""

import logging

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def create_split(
    s1: pd.DataFrame,
    ground_truth: pd.DataFrame,
    val_fraction: float = None,
    seed: int = None,
    stratify: bool = None,
) -> dict:
    """Create a reproducible train/validation split of S1 entities.

    Splits at the S1-entity level (not at the pair level) to prevent
    data leakage between train and validation candidate sets.

    Parameters
    ----------
    s1: pd.DataFrame
        Full Source 1 DataFrame (must have entity_id column).
    ground_truth: pd.DataFrame
        Full ground truth (must have source1_entity_id, matched_entity_ids).
    val_fraction: float, optional
        Fraction of S1 entities to hold out for validation.
        Defaults to config.VALIDATION_FRACTION.
    seed: int, optional
        Random seed for reproducibility. Defaults to config.RANDOM_SEED.
    stratify: bool, optional
        If True, preserve singleton / non-singleton ratio in the split.
        Defaults to config.STRATIFY_VALIDATION.

    Returns
    -------
    dict with keys:
        s1_train, s1_val         - S1 DataFrames
        gt_train, gt_val         - Ground truth DataFrames
    """
    val_fraction = val_fraction if val_fraction is not None else config.VALIDATION_FRACTION
    seed = seed if seed is not None else config.RANDOM_SEED
    stratify = stratify if stratify is not None else config.STRATIFY_VALIDATION

    rng = np.random.default_rng(seed)

    s1_ids = s1["entity_id"].tolist()
    gt_indexed = ground_truth.set_index("source1_entity_id")

    if stratify:
        # Identify singletons vs entities with matches
        singletons = []
        non_singletons = []
        for eid in s1_ids:
            if eid in gt_indexed.index:
                matched = gt_indexed.loc[eid, "matched_entity_ids"]
                has_match = isinstance(matched, str) and matched.strip()
            else:
                has_match = False

            if has_match:
                non_singletons.append(eid)
            else:
                singletons.append(eid)

        # Sample val from each stratum
        n_val_singleton = max(1, int(len(singletons) * val_fraction))
        n_val_non_singleton = max(1, int(len(non_singletons) * val_fraction))

        val_singletons = rng.choice(singletons, size=n_val_singleton, replace=False).tolist()
        val_non_singletons = rng.choice(
            non_singletons, size=n_val_non_singleton, replace=False
        ).tolist()

        val_ids = set(val_singletons) | set(val_non_singletons)
        logger.info(
            "Stratified split: val singletons=%d, val non-singletons=%d",
            len(val_singletons), len(val_non_singletons),
        )
    else:
        n_val = max(1, int(len(s1_ids) * val_fraction))
        val_ids = set(rng.choice(s1_ids, size=n_val, replace=False).tolist())

    train_ids = set(s1_ids) - val_ids

    # Filter DataFrames
    s1_train = s1[s1["entity_id"].isin(train_ids)].reset_index(drop=True)
    s1_val = s1[s1["entity_id"].isin(val_ids)].reset_index(drop=True)
    gt_train = ground_truth[ground_truth["source1_entity_id"].isin(train_ids)].reset_index(drop=True)
    gt_val = ground_truth[ground_truth["source1_entity_id"].isin(val_ids)].reset_index(drop=True)

    logger.info(
        "Validation split (seed=%d, fraction=%.2f): train=%d, val=%d S1 entities.",
        seed, val_fraction, len(s1_train), len(s1_val),
    )
    return {
        "s1_train": s1_train,
        "s1_val": s1_val,
        "gt_train": gt_train,
        "gt_val": gt_val,
        "train_ids": train_ids,
        "val_ids": val_ids,
    }
