"""
src/candidate_generation.py
============================
Combines multiple blocking strategies to produce the final candidate set.
Returns a DataFrame of (source1_entity_id, candidate_entity_id, candidate_source) triples.

Owner: Team Member 2 (Blocking)
"""

import logging
from typing import Optional

import pandas as pd

from src import config, blocking

logger = logging.getLogger(__name__)


def generate_candidates(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    strategies: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Generate candidate (S1, S2/S3) pairs using the configured blocking strategies.

    Each strategy is run independently and results are unioned.
    The final set is deduplicated.

    Parameters
    ----------
    s1: pd.DataFrame
        Preprocessed Source 1 records.
    s2: pd.DataFrame
        Preprocessed Source 2 records.
    s3: pd.DataFrame
        Preprocessed Source 3 records.
    strategies: list[str], optional
        Blocking strategies to apply. Defaults to config.BLOCKING_STRATEGIES.
        Options: "exact_name", "token_name", "tfidf_name", "token_address",
                 "country_aware"

    Returns
    -------
    pd.DataFrame with columns:
        source1_entity_id    (str)
        candidate_entity_id  (str)
        candidate_source     (str) - "S2" or "S3"
    """
    strategies = strategies or config.BLOCKING_STRATEGIES
    logger.info("Starting candidate generation with strategies: %s", strategies)

    # Combine S2 and S3 into a single candidates DataFrame
    candidates = pd.concat([s2, s3], ignore_index=True)
    logger.info("Total candidate pool: %d records (S2=%d, S3=%d)", len(candidates), len(s2), len(s3))

    all_pairs: set[tuple[str, str]] = set()

    for strategy in strategies:
        logger.info("Running blocking strategy: %s", strategy)
        try:
            if strategy == "exact_name":
                pairs = blocking.exact_name_blocking(s1, candidates)
            elif strategy == "token_name":
                pairs = blocking.token_name_blocking(
                    s1, candidates, min_shared=config.TOKEN_OVERLAP_MIN_SHARED
                )
            elif strategy == "tfidf_name":
                pairs = blocking.tfidf_name_blocking(
                    s1, candidates,
                    top_k=config.TFIDF_TOP_K_CANDIDATES,
                    ngram_range=config.NGRAM_RANGE,
                    batch_size=config.TFIDF_BATCH_SIZE,
                )
            elif strategy == "token_address":
                pairs = blocking.token_address_blocking(s1, candidates)
            elif strategy == "country_aware":
                pairs = blocking.country_aware_blocking(s1, candidates)
            else:
                logger.warning("Unknown blocking strategy: '%s'. Skipping.", strategy)
                continue

            prev_size = len(all_pairs)
            all_pairs |= pairs
            logger.info(
                "Strategy '%s' contributed %d new pairs (total: %d).",
                strategy, len(all_pairs) - prev_size, len(all_pairs),
            )
        except Exception as e:
            logger.error("Blocking strategy '%s' failed: %s", strategy, e, exc_info=True)
            raise

    logger.info("Total candidate pairs before post-processing: %d", len(all_pairs))

    if not all_pairs:
        logger.warning("No candidate pairs generated! Check blocking strategies.")
        return pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id", "candidate_source"]
        )

    # Build DataFrame
    pair_list = list(all_pairs)
    df_pairs = pd.DataFrame(pair_list, columns=["source1_entity_id", "candidate_entity_id"])

    # Safety check: ensure no S1-S1 pairs
    s1_s1 = df_pairs[df_pairs["candidate_entity_id"].str.startswith(config.SOURCE1_PREFIX, na=False)]
    if not s1_s1.empty:
        logger.error(
            "PIPELINE BUG: %d S1-S1 pairs found. Removing them.", len(s1_s1)
        )
        df_pairs = df_pairs[~df_pairs["candidate_entity_id"].str.startswith(config.SOURCE1_PREFIX, na=False)]

    # Add candidate_source column
    df_pairs["candidate_source"] = df_pairs["candidate_entity_id"].apply(
        lambda x: "S2" if x.startswith(config.SOURCE2_PREFIX) else "S3"
    )

    # Cap candidates per S1 entity if configured
    if config.MAX_CANDIDATES_PER_S1 is not None:
        before_cap = len(df_pairs)
        # Keep top MAX_CANDIDATES_PER_S1 per S1 entity (random order since no score yet)
        df_pairs = (
            df_pairs.groupby("source1_entity_id", group_keys=False)
            .apply(lambda g: g.head(config.MAX_CANDIDATES_PER_S1))
            .reset_index(drop=True)
        )
        after_cap = len(df_pairs)
        if before_cap != after_cap:
            logger.info(
                "Capped candidates: %d -> %d pairs (cap=%d per S1).",
                before_cap, after_cap, config.MAX_CANDIDATES_PER_S1,
            )

    logger.info(
        "Candidate generation complete: %d pairs, %d unique S1 entities covered.",
        len(df_pairs),
        df_pairs["source1_entity_id"].nunique(),
    )
    return df_pairs


def candidates_to_grouped(df_pairs: pd.DataFrame) -> pd.DataFrame:
    """Convert candidate pair rows into the grouped output format.

    Parameters
    ----------
    df_pairs: pd.DataFrame
        Must have columns: source1_entity_id, candidate_entity_id.

    Returns
    -------
    pd.DataFrame with columns:
        source1_entity_id  (str)
        candidate_entity_ids  (str) - comma-separated list, empty string for no candidates
    """
    grouped = (
        df_pairs.groupby("source1_entity_id")["candidate_entity_id"]
        .apply(lambda x: ",".join(sorted(set(x))))
        .reset_index()
        .rename(columns={"candidate_entity_id": "candidate_entity_ids"})
    )
    return grouped


def ensure_all_s1_in_candidates(
    candidates_grouped: pd.DataFrame,
    s1: pd.DataFrame,
) -> pd.DataFrame:
    """Ensure every S1 entity has a row in the grouped candidates DataFrame.

    Entities with no candidates get an empty candidate_entity_ids string.

    Parameters
    ----------
    candidates_grouped: pd.DataFrame
        Output of candidates_to_grouped().
    s1: pd.DataFrame
        Full S1 DataFrame (must have entity_id column).

    Returns
    -------
    pd.DataFrame with one row per S1 entity.
    """
    all_s1_ids = pd.DataFrame({"source1_entity_id": s1["entity_id"].tolist()})
    merged = all_s1_ids.merge(candidates_grouped, on="source1_entity_id", how="left")
    merged["candidate_entity_ids"] = merged["candidate_entity_ids"].fillna("")
    return merged
