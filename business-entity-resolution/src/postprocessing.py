"""
src/postprocessing.py
=====================
Post-processing of raw match predictions before output generation.

Responsibilities:
- Remove duplicate matched IDs per S1 entity
- Ensure only S2/S3 IDs are present
- Ensure matched IDs exist in the test pool
- Ensure every S1 entity has a row (empty for singletons)
- Enforce that final matches are a subset of candidates

Owner: Team Member 4 (Evaluation / Integration)
"""

import logging
from typing import Optional

import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def build_match_output(
    predictions: pd.DataFrame,
    all_s1_ids: list[str],
    valid_candidate_ids: Optional[set[str]] = None,
    candidate_pairs: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Convert raw pair-level predictions into the final grouped output format.

    Parameters
    ----------
    predictions: pd.DataFrame
        Output of prediction.predict() or threshold.apply_threshold().
        Must have: source1_entity_id, candidate_entity_id, is_match.
    all_s1_ids: list[str]
        All Source 1 entity IDs that must appear in the output.
    valid_candidate_ids: set[str], optional
        If provided, any matched ID not in this set is removed (invalid ID guard).
    candidate_pairs: pd.DataFrame, optional
        The full candidate set (columns: source1_entity_id, candidate_entity_id).
        If provided, enforces that every match is a subset of candidates.

    Returns
    -------
    pd.DataFrame with columns:
        source1_entity_id  (str)
        matched_entity_ids (str) - comma-separated, empty string for singletons
    """
    logger.info("Post-processing predictions for %d S1 entities...", len(all_s1_ids))

    # Filter to positive predictions
    matched = predictions[predictions["is_match"] == 1][
        ["source1_entity_id", "candidate_entity_id"]
    ].copy()

    # Guard: remove S1-S1 self-matches
    s1_s1 = matched[matched["candidate_entity_id"].str.startswith(config.SOURCE1_PREFIX, na=False)]
    if not s1_s1.empty:
        logger.error(
            "Post-processing: removing %d S1-S1 self-match pairs.", len(s1_s1)
        )
        matched = matched[~matched["candidate_entity_id"].str.startswith(config.SOURCE1_PREFIX, na=False)]

    # Guard: remove IDs with wrong prefix
    bad_prefix = matched[
        ~matched["candidate_entity_id"].str.startswith(
            (config.SOURCE2_PREFIX, config.SOURCE3_PREFIX), na=False
        )
    ]
    if not bad_prefix.empty:
        logger.warning("Removing %d matched IDs with invalid prefix.", len(bad_prefix))
        matched = matched[
            matched["candidate_entity_id"].str.startswith(
                (config.SOURCE2_PREFIX, config.SOURCE3_PREFIX), na=False
            )
        ]

    # Guard: remove IDs not in the valid pool
    if valid_candidate_ids is not None:
        invalid = matched[~matched["candidate_entity_id"].isin(valid_candidate_ids)]
        if not invalid.empty:
            logger.warning(
                "Removing %d matched IDs not present in test set.", len(invalid)
            )
            matched = matched[matched["candidate_entity_id"].isin(valid_candidate_ids)]

    # Guard: enforce match ⊆ candidates
    if config.ENFORCE_MATCH_SUBSET_OF_CANDIDATES and candidate_pairs is not None:
        cand_set = set(zip(
            candidate_pairs["source1_entity_id"],
            candidate_pairs["candidate_entity_id"],
        ))
        n_before = len(matched)
        pair_tuples = list(zip(matched["source1_entity_id"], matched["candidate_entity_id"]))
        mask = [pair in cand_set for pair in pair_tuples]
        matched = matched[mask].copy()
        n_dropped = n_before - len(matched)
        if n_dropped > 0:
            logger.warning(
                "Removed %d matched IDs that were not in candidate_pairs "
                "(pipeline bug - match not in candidate set).", n_dropped
            )

    # Deduplicate within each S1 entity
    matched = matched.drop_duplicates(subset=["source1_entity_id", "candidate_entity_id"])

    # Group into comma-separated lists
    grouped = (
        matched.groupby("source1_entity_id")["candidate_entity_id"]
        .apply(lambda x: ",".join(sorted(set(x))))
        .reset_index()
        .rename(columns={"candidate_entity_id": "matched_entity_ids"})
    )

    # Merge with all S1 IDs to ensure every entity has a row
    all_s1_df = pd.DataFrame({"source1_entity_id": all_s1_ids})
    result = all_s1_df.merge(grouped, on="source1_entity_id", how="left")
    result["matched_entity_ids"] = result["matched_entity_ids"].fillna("")

    n_singletons = (result["matched_entity_ids"] == "").sum()
    n_with_matches = len(result) - n_singletons
    logger.info(
        "Post-processing complete: %d S1 entities | %d with matches | %d singletons.",
        len(result), n_with_matches, n_singletons,
    )
    return result


def validate_output(
    matching_results: pd.DataFrame,
    candidate_pairs_grouped: pd.DataFrame,
    all_s1_ids: list[str],
) -> list[str]:
    """Validate the output DataFrames against submission rules.

    Parameters
    ----------
    matching_results: pd.DataFrame
        Must have columns: source1_entity_id, matched_entity_ids.
    candidate_pairs_grouped: pd.DataFrame
        Must have columns: source1_entity_id, candidate_entity_ids.
    all_s1_ids: list[str]
        Complete list of expected S1 entity IDs.

    Returns
    -------
    list[str]: list of validation error messages. Empty if all checks pass.
    """
    errors = []

    # Check 1: Exactly one row per S1 entity
    dupes = matching_results[matching_results["source1_entity_id"].duplicated()]
    if not dupes.empty:
        errors.append(f"Duplicate source1_entity_ids: {dupes['source1_entity_id'].head(5).tolist()}")

    # Check 2: All S1 entities present
    missing = set(all_s1_ids) - set(matching_results["source1_entity_id"])
    if missing:
        errors.append(f"Missing S1 entities ({len(missing)}): {sorted(missing)[:5]}")

    # Check 3: No S1 IDs in matched_entity_ids
    for _, row in matching_results.iterrows():
        if not row["matched_entity_ids"]:
            continue
        ids = row["matched_entity_ids"].split(",")
        bad = [i for i in ids if i.startswith(config.SOURCE1_PREFIX)]
        if bad:
            errors.append(f"S1 self-match in output: {bad[:3]}")
            break

    # Check 4: No duplicate IDs within a list
    for _, row in matching_results.iterrows():
        if not row["matched_entity_ids"]:
            continue
        ids = row["matched_entity_ids"].split(",")
        if len(ids) != len(set(ids)):
            errors.append(
                f"Duplicate IDs within matched_entity_ids for {row['source1_entity_id']}"
            )
            break

    # Check 5: Matches ⊆ Candidates
    if candidate_pairs_grouped is not None:
        cand_map = dict(
            zip(
                candidate_pairs_grouped["source1_entity_id"],
                candidate_pairs_grouped["candidate_entity_ids"].fillna("").str.split(","),
            )
        )
        for _, row in matching_results.iterrows():
            if not row["matched_entity_ids"]:
                continue
            s1_id = row["source1_entity_id"]
            match_ids = set(row["matched_entity_ids"].split(","))
            cand_ids = set(cand_map.get(s1_id, []))
            non_candidates = match_ids - cand_ids
            if non_candidates:
                errors.append(
                    f"Matched IDs not in candidates for {s1_id}: {list(non_candidates)[:3]}"
                )
                break

    if errors:
        logger.warning("Output validation found %d issues.", len(errors))
    else:
        logger.info("Output validation passed.")
    return errors
