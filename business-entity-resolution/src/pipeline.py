"""
src/pipeline.py
===============
Orchestrates the full end-to-end pipeline for training, validation,
and test prediction modes.

Owner: Team Member 4 (Evaluation / Integration)
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src import (
    config,
    data_loader,
    preprocessing,
    candidate_generation,
    features,
    model as model_module,
    threshold as threshold_module,
    prediction,
    postprocessing,
    output,
)
from validation import create_validation, evaluate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LABEL GENERATION
# ---------------------------------------------------------------------------

def build_pair_labels(
    pairs: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> np.ndarray:
    """Create binary labels (1=match, 0=no-match) for candidate pairs.

    Parameters
    ----------
    pairs: pd.DataFrame
        Columns: source1_entity_id, candidate_entity_id
    ground_truth: pd.DataFrame
        Columns: source1_entity_id, matched_entity_ids (comma-sep string or NaN)

    Returns
    -------
    np.ndarray of shape (n_pairs,) with values 0 or 1.
    """
    # Build a set of (s1_id, cand_id) true match pairs for O(1) lookup
    true_pairs: set[tuple[str, str]] = set()
    for _, row in ground_truth.iterrows():
        s1_id = row["source1_entity_id"]
        matched = row["matched_entity_ids"]
        if isinstance(matched, str) and matched.strip():
            for cand_id in matched.split(","):
                cand_id = cand_id.strip()
                if cand_id:
                    true_pairs.add((s1_id, cand_id))

    labels = np.array(
        [
            1 if (s1_id, cand_id) in true_pairs else 0
            for s1_id, cand_id in zip(
                pairs["source1_entity_id"], pairs["candidate_entity_id"]
            )
        ]
    )
    n_pos = labels.sum()
    logger.info(
        "Labels built: %d pairs | %d positive (%.1f%%) | %d negative.",
        len(labels), n_pos, 100 * n_pos / max(len(labels), 1), len(labels) - n_pos,
    )
    return labels


# ---------------------------------------------------------------------------
# TRAINING PIPELINE
# ---------------------------------------------------------------------------

def run_train(
    blocking_strategies: Optional[list[str]] = None,
    model_type: Optional[str] = None,
):
    """Load data, generate candidates, extract features, train and save model.

    Returns
    -------
    dict with pipeline artifacts (model, threshold, metrics, etc.)
    """
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("PIPELINE: TRAIN MODE")
    logger.info("=" * 60)

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    logger.info("[1/6] Loading training data...")
    s1, s2, s3, gt = data_loader.load_train_data()

    # -----------------------------------------------------------------------
    # 2. Validation split
    # -----------------------------------------------------------------------
    logger.info("[2/6] Creating validation split (fraction=%.2f)...", config.VALIDATION_FRACTION)
    split = create_validation.create_split(s1, gt, seed=config.RANDOM_SEED)
    s1_train = split["s1_train"]
    s1_val = split["s1_val"]
    gt_train = split["gt_train"]
    gt_val = split["gt_val"]
    logger.info(
        "Train: %d S1 entities | Val: %d S1 entities",
        len(s1_train), len(s1_val),
    )

    # -----------------------------------------------------------------------
    # 3. Preprocessing
    # -----------------------------------------------------------------------
    logger.info("[3/6] Preprocessing...")
    s1_train_proc, s2_proc, s3_proc = preprocessing.preprocess_all(s1_train, s2, s3)
    s1_val_proc = preprocessing.preprocess_source(s1_val, label="Source1_val")

    # -----------------------------------------------------------------------
    # 4. Candidate generation on training portion
    # -----------------------------------------------------------------------
    logger.info("[4/6] Generating training candidates...")
    cand_pairs_train = candidate_generation.generate_candidates(
        s1_train_proc, s2_proc, s3_proc, strategies=blocking_strategies
    )
    logger.info("Training candidate pairs: %d", len(cand_pairs_train))

    # Candidate recall on train
    cand_recall = evaluate.candidate_recall(cand_pairs_train, gt_train)
    logger.info("Training candidate recall: %.4f", cand_recall)

    # -----------------------------------------------------------------------
    # 5. Feature extraction and model training
    # -----------------------------------------------------------------------
    logger.info("[5/6] Extracting features and training model...")
    candidates_proc = pd.concat([s2_proc, s3_proc], ignore_index=True)
    X_train = features.extract_features_for_pairs(cand_pairs_train, s1_train_proc, candidates_proc)
    y_train = build_pair_labels(cand_pairs_train, gt_train)

    trained_model = model_module.train_model(X_train, pd.Series(y_train), model_type=model_type)
    model_module.save_model(trained_model)

    # -----------------------------------------------------------------------
    # 6. Validate and tune threshold
    # -----------------------------------------------------------------------
    logger.info("[6/6] Validating and tuning threshold...")
    cand_pairs_val = candidate_generation.generate_candidates(
        s1_val_proc, s2_proc, s3_proc, strategies=blocking_strategies
    )
    X_val = features.extract_features_for_pairs(cand_pairs_val, s1_val_proc, candidates_proc)
    y_val = build_pair_labels(cand_pairs_val, gt_val)
    proba_val = model_module.predict_proba(trained_model, X_val)

    best_threshold, best_metrics = threshold_module.tune_threshold(proba_val, y_val)
    threshold_module.save_threshold(best_threshold)

    # Entity-level F0.5 evaluation on validation
    pred_val_df = threshold_module.apply_threshold(cand_pairs_val, proba_val, best_threshold)
    val_results = postprocessing.build_match_output(
        pred_val_df, s1_val_proc["entity_id"].tolist(), candidate_pairs=cand_pairs_val
    )
    entity_metrics = evaluate.entity_level_f05(val_results, gt_val)
    logger.info(
        "VALIDATION RESULTS: Precision=%.4f | Recall=%.4f | F0.5=%.4f",
        entity_metrics["precision"], entity_metrics["recall"], entity_metrics["f0_5"],
    )

    elapsed = time.time() - t0
    logger.info("Training pipeline complete in %.1f seconds.", elapsed)

    return {
        "model": trained_model,
        "threshold": best_threshold,
        "val_metrics": entity_metrics,
        "candidate_recall_train": cand_recall,
        "elapsed_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# VALIDATION-ONLY PIPELINE
# ---------------------------------------------------------------------------

def run_validate(blocking_strategies: Optional[list[str]] = None):
    """Run blocking + feature extraction + prediction on the validation split.

    Loads the saved model and threshold.
    """
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("PIPELINE: VALIDATE MODE")
    logger.info("=" * 60)

    s1, s2, s3, gt = data_loader.load_train_data()
    split = create_validation.create_split(s1, gt, seed=config.RANDOM_SEED)
    s1_val = split["s1_val"]
    gt_val = split["gt_val"]

    s2_proc = preprocessing.preprocess_source(s2, "Source2")
    s3_proc = preprocessing.preprocess_source(s3, "Source3")
    s1_val_proc = preprocessing.preprocess_source(s1_val, "Source1_val")

    cand_pairs = candidate_generation.generate_candidates(
        s1_val_proc, s2_proc, s3_proc, strategies=blocking_strategies
    )

    candidates_proc = pd.concat([s2_proc, s3_proc], ignore_index=True)
    X_val = features.extract_features_for_pairs(cand_pairs, s1_val_proc, candidates_proc)
    trained_model = model_module.load_model()
    proba = model_module.predict_proba(trained_model, X_val)
    decision_threshold = threshold_module.load_threshold()

    pred_df = threshold_module.apply_threshold(cand_pairs, proba, decision_threshold)
    val_results = postprocessing.build_match_output(
        pred_df, s1_val_proc["entity_id"].tolist(), candidate_pairs=cand_pairs
    )
    entity_metrics = evaluate.entity_level_f05(val_results, gt_val)
    cand_recall = evaluate.candidate_recall(cand_pairs, gt_val)

    logger.info(
        "VALIDATION: Precision=%.4f | Recall=%.4f | F0.5=%.4f | CandRecall=%.4f",
        entity_metrics["precision"], entity_metrics["recall"],
        entity_metrics["f0_5"], cand_recall,
    )
    logger.info("Validation complete in %.1f seconds.", time.time() - t0)
    return entity_metrics


# ---------------------------------------------------------------------------
# PREDICTION (TEST) PIPELINE
# ---------------------------------------------------------------------------

def run_predict(blocking_strategies: Optional[list[str]] = None):
    """Generate test predictions and write output files."""
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("PIPELINE: PREDICT MODE")
    logger.info("=" * 60)

    # Load test data
    ts1, ts2, ts3 = data_loader.load_test_data()
    ts1_proc, ts2_proc, ts3_proc = preprocessing.preprocess_all(ts1, ts2, ts3)

    # Generate candidates
    cand_pairs = candidate_generation.generate_candidates(
        ts1_proc, ts2_proc, ts3_proc, strategies=blocking_strategies
    )

    # Feature extraction and scoring
    candidates_proc = pd.concat([ts2_proc, ts3_proc], ignore_index=True)
    trained_model = model_module.load_model()
    decision_threshold = threshold_module.load_threshold()
    pred_df = prediction.predict(cand_pairs, ts1_proc, candidates_proc, trained_model, decision_threshold)

    # Valid candidate IDs (all S2 and S3 test IDs)
    valid_ids = set(ts2["entity_id"]) | set(ts3["entity_id"])

    # Post-process
    matching_results = postprocessing.build_match_output(
        pred_df, ts1["entity_id"].tolist(), valid_candidate_ids=valid_ids, candidate_pairs=cand_pairs
    )

    # Group candidates for output
    cand_grouped = candidate_generation.ensure_all_s1_in_candidates(
        candidate_generation.candidates_to_grouped(cand_pairs), ts1_proc
    )

    # Validate
    errors = postprocessing.validate_output(
        matching_results, cand_grouped, ts1["entity_id"].tolist()
    )
    if errors:
        for err in errors:
            logger.error("OUTPUT VALIDATION ERROR: %s", err)
        raise RuntimeError(f"Output validation failed with {len(errors)} error(s).")

    # Write output files
    output.write_matching_results(matching_results)
    output.write_candidate_pairs(cand_grouped)

    logger.info("Test prediction complete in %.1f seconds.", time.time() - t0)
    return matching_results


# ---------------------------------------------------------------------------
# ALL MODE
# ---------------------------------------------------------------------------

def run_all(blocking_strategies: Optional[list[str]] = None, model_type: Optional[str] = None):
    """Run train -> validate -> predict in sequence."""
    logger.info("=" * 60)
    logger.info("PIPELINE: ALL MODE (train + validate + predict)")
    logger.info("=" * 60)
    train_result = run_train(blocking_strategies=blocking_strategies, model_type=model_type)
    run_predict(blocking_strategies=blocking_strategies)
    return train_result
