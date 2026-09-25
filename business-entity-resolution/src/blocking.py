"""
src/blocking.py
===============
Individual blocking strategy implementations.

Each strategy produces a set of (source1_entity_id, candidate_entity_id) pairs.
Strategies are combined in candidate_generation.py using set union.

Owner: Team Member 2 (Blocking)

IMPORTANT: This module must NEVER produce S1-S1 pairs.
All candidates must come from S2 or S3 sources.
"""

import logging
from collections import defaultdict
from typing import Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
import numpy as np

from src import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HELPER: build an inverted index  token -> list of entity_ids
# ---------------------------------------------------------------------------

def _build_inverted_index(
    df: pd.DataFrame,
    token_col: str,
    id_col: str = "entity_id",
    min_token_length: int = 3,
) -> dict[str, list[str]]:
    """Build an inverted index mapping token -> [entity_id, ...].

    Parameters
    ----------
    df: pd.DataFrame
        Must contain id_col and token_col (a list of strings per row).
    token_col: str
        Column containing list of tokens (produced by preprocessing).
    id_col: str
    min_token_length: int
        Skip tokens shorter than this.

    Returns
    -------
    dict mapping token -> list of entity_ids containing that token.
    """
    index: dict[str, list[str]] = defaultdict(list)
    for eid, tokens in zip(df[id_col], df[token_col]):
        for tok in tokens:
            if len(tok) >= min_token_length:
                index[tok].append(eid)
    return dict(index)


# ---------------------------------------------------------------------------
# STRATEGY 1: Exact normalized name match
# ---------------------------------------------------------------------------

def exact_name_blocking(
    s1: pd.DataFrame,
    candidates: pd.DataFrame,
) -> set[tuple[str, str]]:
    """Block on exact normalized business name equality.

    Parameters
    ----------
    s1: pd.DataFrame
        Preprocessed Source 1 records. Must have entity_id, business_name_normalized.
    candidates: pd.DataFrame
        Combined S2 + S3 preprocessed records.

    Returns
    -------
    set of (s1_entity_id, candidate_entity_id) pairs.
    """
    logger.info("[Blocking:exact_name] Building lookup...")
    # Map normalized_name -> list of S2/S3 entity_ids
    cand_name_map: dict[str, list[str]] = defaultdict(list)
    for eid, name in zip(candidates["entity_id"], candidates["business_name_normalized"]):
        if name:
            cand_name_map[name].append(eid)

    pairs: set[tuple[str, str]] = set()
    for s1_id, name in zip(s1["entity_id"], s1["business_name_normalized"]):
        if name and name in cand_name_map:
            for cand_id in cand_name_map[name]:
                pairs.add((s1_id, cand_id))

    logger.info("[Blocking:exact_name] Generated %d pairs.", len(pairs))
    return pairs


# ---------------------------------------------------------------------------
# STRATEGY 2: Token overlap on name
# ---------------------------------------------------------------------------

def token_name_blocking(
    s1: pd.DataFrame,
    candidates: pd.DataFrame,
    min_shared: int = 1,
    min_token_length: int = 3,
) -> set[tuple[str, str]]:
    """Block on shared name tokens (inverted index lookup).

    A (s1_id, cand_id) pair is generated for every shared token.
    min_shared controls how many tokens must be shared to form a pair
    (higher = fewer but more precise candidates).

    Parameters
    ----------
    s1: pd.DataFrame
    candidates: pd.DataFrame
    min_shared: int
        Minimum number of shared tokens to include a pair. Default 1.
    min_token_length: int
        Only consider tokens at least this long.

    Returns
    -------
    set of (s1_entity_id, candidate_entity_id) pairs.
    """
    logger.info("[Blocking:token_name] Building inverted index on candidates...")
    cand_index = _build_inverted_index(candidates, "name_tokens", min_token_length=min_token_length)

    pairs: set[tuple[str, str]] = set()
    shared_counter: dict[tuple[str, str], int] = defaultdict(int)

    for s1_id, tokens in zip(s1["entity_id"], s1["name_tokens"]):
        for tok in tokens:
            if len(tok) < min_token_length:
                continue
            if tok in cand_index:
                for cand_id in cand_index[tok]:
                    shared_counter[(s1_id, cand_id)] += 1

    if min_shared <= 1:
        pairs = set(shared_counter.keys())
    else:
        pairs = {pair for pair, cnt in shared_counter.items() if cnt >= min_shared}

    logger.info("[Blocking:token_name] Generated %d pairs (min_shared=%d).", len(pairs), min_shared)
    return pairs


# ---------------------------------------------------------------------------
# STRATEGY 3: TF-IDF character n-gram blocking on name
# ---------------------------------------------------------------------------

def tfidf_name_blocking(
    s1: pd.DataFrame,
    candidates: pd.DataFrame,
    top_k: int = None,
    ngram_range: tuple = None,
    batch_size: int = None,
) -> set[tuple[str, str]]:
    """Block using TF-IDF character n-gram similarity on business names.

    Retrieves the top-K most similar S2/S3 candidates for each S1 entity
    using sparse matrix cosine similarity.

    Parameters
    ----------
    s1, candidates: pd.DataFrame
    top_k: int
        Number of top candidates to retrieve per S1 entity.
    ngram_range: tuple (min_n, max_n)
    batch_size: int
        Process S1 records in batches to control memory usage.

    Returns
    -------
    set of (s1_entity_id, candidate_entity_id) pairs.
    """
    top_k = top_k or config.TFIDF_TOP_K_CANDIDATES
    ngram_range = ngram_range or config.NGRAM_RANGE
    batch_size = batch_size or config.TFIDF_BATCH_SIZE

    logger.info("[Blocking:tfidf_name] Fitting TF-IDF (ngram=%s)...", ngram_range)

    s1_names = s1["business_name_normalized"].fillna("").tolist()
    cand_names = candidates["business_name_normalized"].fillna("").tolist()
    s1_ids = s1["entity_id"].tolist()
    cand_ids = candidates["entity_id"].tolist()

    # Filter empties for the vectorizer (keep indices aligned)
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        min_df=2,
        sublinear_tf=True,
    )

    # Fit on all names combined for consistent vocabulary
    all_names = s1_names + cand_names
    vectorizer.fit(all_names)

    cand_matrix = vectorizer.transform(cand_names)   # shape: (n_cand, n_features)
    s1_matrix = vectorizer.transform(s1_names)        # shape: (n_s1, n_features)

    logger.info(
        "[Blocking:tfidf_name] S1 matrix: %s, Cand matrix: %s",
        s1_matrix.shape, cand_matrix.shape,
    )

    pairs: set[tuple[str, str]] = set()
    n_s1 = len(s1_ids)

    for start in range(0, n_s1, batch_size):
        end = min(start + batch_size, n_s1)
        batch = s1_matrix[start:end]
        # Compute cosine similarity: (batch_size, n_cand)
        sims = cosine_similarity(batch, cand_matrix)
        # For each S1 row, pick top_k candidates
        top_indices = np.argsort(sims, axis=1)[:, -top_k:][:, ::-1]
        for i, s1_idx in enumerate(range(start, end)):
            s1_id = s1_ids[s1_idx]
            for cand_idx in top_indices[i]:
                if sims[i, cand_idx] > 0.0:
                    pairs.add((s1_id, cand_ids[cand_idx]))

        if (start // batch_size) % 10 == 0:
            logger.debug(
                "[Blocking:tfidf_name] Processed %d / %d S1 records...", end, n_s1
            )

    logger.info("[Blocking:tfidf_name] Generated %d pairs.", len(pairs))
    return pairs


# ---------------------------------------------------------------------------
# STRATEGY 4: Token overlap on address
# ---------------------------------------------------------------------------

def token_address_blocking(
    s1: pd.DataFrame,
    candidates: pd.DataFrame,
    min_shared: int = 2,
    min_token_length: int = 3,
) -> set[tuple[str, str]]:
    """Block on shared address tokens.

    Higher min_shared reduces noise from common tokens like 'rd', 'st'.
    Default min_shared=2 means at least 2 address tokens must overlap.

    Parameters
    ----------
    s1, candidates: pd.DataFrame
    min_shared: int
    min_token_length: int

    Returns
    -------
    set of (s1_entity_id, candidate_entity_id) pairs.
    """
    logger.info("[Blocking:token_address] Building inverted index on candidates...")
    cand_index = _build_inverted_index(
        candidates, "address_tokens", min_token_length=min_token_length
    )

    shared_counter: dict[tuple[str, str], int] = defaultdict(int)
    for s1_id, tokens in zip(s1["entity_id"], s1["address_tokens"]):
        for tok in tokens:
            if len(tok) < min_token_length:
                continue
            if tok in cand_index:
                for cand_id in cand_index[tok]:
                    shared_counter[(s1_id, cand_id)] += 1

    pairs = {pair for pair, cnt in shared_counter.items() if cnt >= min_shared}
    logger.info(
        "[Blocking:token_address] Generated %d pairs (min_shared=%d).", len(pairs), min_shared
    )
    return pairs


# ---------------------------------------------------------------------------
# STRATEGY 5: Country-aware combined blocking
# ---------------------------------------------------------------------------

def country_aware_blocking(
    s1: pd.DataFrame,
    candidates: pd.DataFrame,
    inner_strategy: str = "token_name",
    min_shared: int = 1,
) -> set[tuple[str, str]]:
    """Apply a blocking strategy independently per country group.

    This avoids cross-country false positives (e.g., a French business
    being blocked against an Indian one). Treats country as an open-set
    field -- unknown countries are handled gracefully.

    Parameters
    ----------
    s1, candidates: pd.DataFrame
    inner_strategy: str
        "token_name" or "exact_name"
    min_shared: int
        Passed to token_name_blocking if used.

    Returns
    -------
    set of (s1_entity_id, candidate_entity_id) pairs.
    """
    logger.info("[Blocking:country_aware] Grouping by country...")
    pairs: set[tuple[str, str]] = set()

    countries = set(s1["country_normalized"].unique())
    for country in countries:
        s1_c = s1[s1["country_normalized"] == country]
        cand_c = candidates[candidates["country_normalized"] == country]
        if s1_c.empty or cand_c.empty:
            continue

        logger.debug(
            "[Blocking:country_aware] Country='%s': S1=%d, Cand=%d",
            country, len(s1_c), len(cand_c),
        )

        if inner_strategy == "exact_name":
            country_pairs = exact_name_blocking(s1_c, cand_c)
        else:
            country_pairs = token_name_blocking(s1_c, cand_c, min_shared=min_shared)

        pairs |= country_pairs

    # Also handle S1 entities with missing/unknown country (don't drop them!)
    s1_unknown = s1[s1["country_normalized"] == ""]
    if not s1_unknown.empty:
        logger.debug(
            "[Blocking:country_aware] %d S1 entities have no country; "
            "running global blocking for them.", len(s1_unknown)
        )
        if inner_strategy == "exact_name":
            pairs |= exact_name_blocking(s1_unknown, candidates)
        else:
            pairs |= token_name_blocking(s1_unknown, candidates, min_shared=min_shared)

    logger.info("[Blocking:country_aware] Generated %d pairs total.", len(pairs))
    return pairs
