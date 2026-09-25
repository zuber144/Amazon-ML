"""
validation/metrics.py
=====================
Core evaluation metrics for the Business Entity Resolution challenge.

Primary metric: F0.5 (precision-heavy, macro-averaged over S1 entities).
"""

import numpy as np
from typing import Union


def f05_score(precision: float, recall: float) -> float:
    """Compute F0.5 from precision and recall.

    F0.5 = (1.25 * P * R) / (0.25 * P + R)
    """
    denom = 0.25 * precision + recall
    if denom == 0.0:
        return 0.0
    return (1.25 * precision * recall) / denom


def entity_precision(predicted_ids: set, true_ids: set) -> float:
    """Compute precision for a single S1 entity.

    If the entity is a true singleton (true_ids empty) and we predict
    no match, precision = 1.0. If we predict matches for a true singleton,
    precision = 0.0 (all are false positives).
    """
    if not predicted_ids and not true_ids:
        return 1.0
    if not predicted_ids:
        return 0.0  # no prediction, but there are true matches -> P=0 (handled in F)
    if not true_ids:
        return 0.0  # predicted something for a singleton -> all FP
    tp = len(predicted_ids & true_ids)
    return tp / len(predicted_ids)


def entity_recall(predicted_ids: set, true_ids: set) -> float:
    """Compute recall for a single S1 entity."""
    if not predicted_ids and not true_ids:
        return 1.0
    if not true_ids:
        return 1.0  # singleton, and any prediction is wrong (P=0 drives F=0)
    if not predicted_ids:
        return 0.0
    tp = len(predicted_ids & true_ids)
    return tp / len(true_ids)


def entity_f05(predicted_ids: set, true_ids: set) -> float:
    """Compute F0.5 for a single S1 entity.

    Handles the singleton case correctly:
      - True singleton + correct empty prediction -> F0.5 = 1.0
      - True singleton + any prediction          -> F0.5 = 0.0
    """
    if not predicted_ids and not true_ids:
        return 1.0  # correctly predicted singleton
    p = entity_precision(predicted_ids, true_ids)
    r = entity_recall(predicted_ids, true_ids)
    return f05_score(p, r)


def macro_f05(
    predictions: dict[str, set],
    ground_truth: dict[str, set],
) -> dict:
    """Compute macro-averaged F0.5 over all S1 entities.

    Parameters
    ----------
    predictions: dict mapping source1_entity_id -> set of matched ids
    ground_truth: dict mapping source1_entity_id -> set of matched ids
        (empty set for singletons)

    Returns
    -------
    dict with keys: f0_5, precision, recall, n_entities, n_singletons_correct,
        n_singletons_total, n_with_matches_total.
    """
    all_s1_ids = set(predictions.keys()) | set(ground_truth.keys())
    f05_scores = []
    prec_scores = []
    rec_scores = []
    n_singleton_correct = 0
    n_singleton_total = 0
    n_with_matches = 0

    for s1_id in all_s1_ids:
        pred_ids = predictions.get(s1_id, set())
        true_ids = ground_truth.get(s1_id, set())

        if not true_ids:
            n_singleton_total += 1
            if not pred_ids:
                n_singleton_correct += 1
        else:
            n_with_matches += 1

        p = entity_precision(pred_ids, true_ids)
        r = entity_recall(pred_ids, true_ids)
        f = entity_f05(pred_ids, true_ids)

        f05_scores.append(f)
        prec_scores.append(p)
        rec_scores.append(r)

    n = len(all_s1_ids)
    return {
        "f0_5": float(np.mean(f05_scores)) if f05_scores else 0.0,
        "precision": float(np.mean(prec_scores)) if prec_scores else 0.0,
        "recall": float(np.mean(rec_scores)) if rec_scores else 0.0,
        "n_entities": n,
        "n_singletons_total": n_singleton_total,
        "n_singletons_correct": n_singleton_correct,
        "n_with_matches_total": n_with_matches,
    }
