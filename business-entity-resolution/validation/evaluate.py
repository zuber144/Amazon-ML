"""
validation/evaluate.py
======================
Evaluation tools: entity-level F0.5, candidate recall, and detailed reporting.

Owner: Team Member 4 (Evaluation / Integration)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from validation import metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _ground_truth_to_dict(gt: pd.DataFrame) -> dict[str, set]:
    """Convert ground truth DataFrame to {source1_entity_id: set(matched_ids)}."""
    gt_dict: dict[str, set] = {}
    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        matched = row.get("matched_entity_ids", "")
        if isinstance(matched, str) and matched.strip():
            gt_dict[s1_id] = set(matched.strip().split(","))
        else:
            gt_dict[s1_id] = set()
    return gt_dict


def _predictions_to_dict(results: pd.DataFrame) -> dict[str, set]:
    """Convert matching_results DataFrame to {source1_entity_id: set(matched_ids)}."""
    pred_dict: dict[str, set] = {}
    for _, row in results.iterrows():
        s1_id = row["source1_entity_id"]
        matched = row.get("matched_entity_ids", "")
        if isinstance(matched, str) and matched.strip():
            pred_dict[s1_id] = set(matched.strip().split(","))
        else:
            pred_dict[s1_id] = set()
    return pred_dict


# ---------------------------------------------------------------------------
# ENTITY-LEVEL F0.5
# ---------------------------------------------------------------------------

def entity_level_f05(
    matching_results: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> dict:
    """Compute macro-averaged entity-level F0.5.

    Parameters
    ----------
    matching_results: pd.DataFrame
        Must have columns: source1_entity_id, matched_entity_ids.
    ground_truth: pd.DataFrame
        Must have columns: source1_entity_id, matched_entity_ids.

    Returns
    -------
    dict with evaluation metrics.
    """
    pred_dict = _predictions_to_dict(matching_results)
    gt_dict = _ground_truth_to_dict(ground_truth)

    result = metrics.macro_f05(pred_dict, gt_dict)
    logger.info(
        "Entity-level F0.5=%.4f | Precision=%.4f | Recall=%.4f | "
        "N=%d | Singletons=%d/%d correct",
        result["f0_5"], result["precision"], result["recall"],
        result["n_entities"],
        result["n_singletons_correct"], result["n_singletons_total"],
    )
    return result


# ---------------------------------------------------------------------------
# CANDIDATE RECALL
# ---------------------------------------------------------------------------

def candidate_recall(
    candidate_pairs: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> float:
    """Compute candidate recall (fraction of true matches covered by candidates).

    This is the upper bound on recall for the matching stage.

    Parameters
    ----------
    candidate_pairs: pd.DataFrame
        Columns: source1_entity_id, candidate_entity_id.
    ground_truth: pd.DataFrame
        Columns: source1_entity_id, matched_entity_ids.

    Returns
    -------
    float: candidate recall in [0, 1].
    """
    # Build set of candidate pairs
    cand_set = set(
        zip(candidate_pairs["source1_entity_id"], candidate_pairs["candidate_entity_id"])
    )

    true_positives = 0
    false_negatives = 0

    for _, row in ground_truth.iterrows():
        s1_id = row["source1_entity_id"]
        matched = row.get("matched_entity_ids", "")
        if not isinstance(matched, str) or not matched.strip():
            continue
        for cand_id in matched.strip().split(","):
            cand_id = cand_id.strip()
            if not cand_id:
                continue
            if (s1_id, cand_id) in cand_set:
                true_positives += 1
            else:
                false_negatives += 1

    total_true = true_positives + false_negatives
    recall = true_positives / total_true if total_true > 0 else 0.0
    logger.info(
        "Candidate recall: %.4f (%d / %d true matches covered)",
        recall, true_positives, total_true,
    )
    return recall


# ---------------------------------------------------------------------------
# DETAILED EVALUATION REPORT
# ---------------------------------------------------------------------------

def full_evaluation_report(
    matching_results: pd.DataFrame,
    ground_truth: pd.DataFrame,
    candidate_pairs: Optional[pd.DataFrame] = None,
) -> dict:
    """Produce a full evaluation report with all key metrics.

    Parameters
    ----------
    matching_results: pd.DataFrame
    ground_truth: pd.DataFrame
    candidate_pairs: pd.DataFrame, optional

    Returns
    -------
    dict with all metrics.
    """
    report = {}

    # Entity-level metrics
    entity_metrics = entity_level_f05(matching_results, ground_truth)
    report.update(entity_metrics)

    # Candidate recall (if provided)
    if candidate_pairs is not None:
        report["candidate_recall"] = candidate_recall(candidate_pairs, ground_truth)

    # Prediction statistics
    pred_dict = _predictions_to_dict(matching_results)
    gt_dict = _ground_truth_to_dict(ground_truth)

    total_predicted = sum(len(v) for v in pred_dict.values())
    total_true = sum(len(v) for v in gt_dict.values())
    report["total_predicted_matches"] = total_predicted
    report["total_true_matches"] = total_true

    # Per-entity analysis
    f05_scores = []
    per_entity = []
    for s1_id in gt_dict:
        pred = pred_dict.get(s1_id, set())
        true = gt_dict[s1_id]
        f = metrics.entity_f05(pred, true)
        f05_scores.append(f)
        per_entity.append({"s1_id": s1_id, "f0_5": f, "n_pred": len(pred), "n_true": len(true)})

    report["f0_5_std"] = float(np.std(f05_scores))
    report["f0_5_min"] = float(np.min(f05_scores)) if f05_scores else 0.0
    report["f0_5_max"] = float(np.max(f05_scores)) if f05_scores else 0.0

    # Print formatted report
    print("=" * 50)
    print("EVALUATION REPORT")
    print("=" * 50)
    print(f"Entities evaluated:    {report['n_entities']}")
    print(f"True singletons:       {report['n_singletons_total']}")
    print(f"Singleton accuracy:    {report['n_singletons_correct']} / {report['n_singletons_total']}")
    print(f"Entities with matches: {report['n_with_matches_total']}")
    print(f"Total predicted:       {report['total_predicted_matches']}")
    print(f"Total true matches:    {report['total_true_matches']}")
    if "candidate_recall" in report:
        print(f"Candidate recall:      {report['candidate_recall']:.4f}")
    print("-" * 50)
    print(f"Macro Precision:       {report['precision']:.4f}")
    print(f"Macro Recall:          {report['recall']:.4f}")
    print(f"Macro F0.5:            {report['f0_5']:.4f}")
    print(f"F0.5 std:              {report['f0_5_std']:.4f}")
    print("=" * 50)

    return report
