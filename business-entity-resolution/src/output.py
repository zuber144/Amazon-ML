"""
src/output.py
=============
Generates the required TSV output files:
  output/matching_results.tsv
  output/candidate_pairs.tsv

Owner: Team Member 4 (Evaluation / Integration)
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def write_matching_results(
    results: pd.DataFrame,
    path: Optional[Path] = None,
) -> None:
    """Write matching_results.tsv in the required format.

    Parameters
    ----------
    results: pd.DataFrame
        Must have columns: source1_entity_id, matched_entity_ids.
        matched_entity_ids should be a comma-separated string or "".
    path: Path, optional
        Defaults to config.MATCHING_RESULTS_PATH.
    """
    path = Path(path) if path else config.MATCHING_RESULTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    # Validate required columns
    required = ["source1_entity_id", "matched_entity_ids"]
    missing = [c for c in required if c not in results.columns]
    if missing:
        raise ValueError(f"Missing required columns for matching_results: {missing}")

    # Ensure no None/NaN in matched_entity_ids (empty string = singleton)
    out = results[required].copy()
    out["matched_entity_ids"] = out["matched_entity_ids"].fillna("")

    out.to_csv(path, sep="\t", index=False, encoding="utf-8")
    n_matches = (out["matched_entity_ids"] != "").sum()
    logger.info(
        "Wrote matching_results.tsv: %d rows (%d with matches, %d singletons) -> %s",
        len(out), n_matches, len(out) - n_matches, path,
    )


def write_candidate_pairs(
    candidates: pd.DataFrame,
    path: Optional[Path] = None,
) -> None:
    """Write candidate_pairs.tsv in the required format.

    Parameters
    ----------
    candidates: pd.DataFrame
        Must have columns: source1_entity_id, candidate_entity_ids.
        candidate_entity_ids should be a comma-separated string or "".
    path: Path, optional
        Defaults to config.CANDIDATE_PAIRS_PATH.
    """
    path = Path(path) if path else config.CANDIDATE_PAIRS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    required = ["source1_entity_id", "candidate_entity_ids"]
    missing = [c for c in required if c not in candidates.columns]
    if missing:
        raise ValueError(f"Missing required columns for candidate_pairs: {missing}")

    out = candidates[required].copy()
    out["candidate_entity_ids"] = out["candidate_entity_ids"].fillna("")

    out.to_csv(path, sep="\t", index=False, encoding="utf-8")
    n_with = (out["candidate_entity_ids"] != "").sum()
    logger.info(
        "Wrote candidate_pairs.tsv: %d rows (%d with candidates) -> %s",
        len(out), n_with, path,
    )


def pair_df_to_grouped(
    pair_df: pd.DataFrame,
    s1_id_col: str = "source1_entity_id",
    cand_id_col: str = "candidate_entity_id",
    all_s1_ids: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Convert a long-format pair DataFrame to the grouped output format.

    Parameters
    ----------
    pair_df: pd.DataFrame
        Long format with one pair per row.
    s1_id_col: str
    cand_id_col: str
    all_s1_ids: list[str], optional
        If provided, ensures every S1 entity appears in output.

    Returns
    -------
    pd.DataFrame with source1_entity_id and candidate_entity_ids (comma-sep).
    """
    grouped = (
        pair_df.groupby(s1_id_col)[cand_id_col]
        .apply(lambda x: ",".join(sorted(set(x))))
        .reset_index()
        .rename(columns={cand_id_col: "candidate_entity_ids"})
    )
    if all_s1_ids is not None:
        base = pd.DataFrame({"source1_entity_id": all_s1_ids})
        grouped = base.merge(grouped, on="source1_entity_id", how="left")
        grouped["candidate_entity_ids"] = grouped["candidate_entity_ids"].fillna("")
    return grouped
