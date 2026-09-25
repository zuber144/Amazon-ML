"""
src/features.py
===============
Feature engineering for candidate pairs.

For every (s1_entity_id, candidate_entity_id) pair, computes pairwise
similarity features covering name, address, country, and missingness.

Owner: Team Member 3 (ML Matching)
"""

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src import config

logger = logging.getLogger(__name__)

# Try to import rapidfuzz for fast edit distance; fall back to simple ratio
try:
    from rapidfuzz.distance import Levenshtein as _Lev
    from rapidfuzz import fuzz as _fuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False
    logger.info("rapidfuzz not found; falling back to simple edit-similarity approximation.")


# ---------------------------------------------------------------------------
# LOW-LEVEL SIMILARITY PRIMITIVES
# ---------------------------------------------------------------------------

def _safe_str(x) -> str:
    """Return str(x) if x is not None/NaN, else ''."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x)


def jaccard_similarity(a: str, b: str, tokenize: bool = True) -> float:
    """Compute Jaccard similarity between two strings.

    If tokenize=True, compares token sets. Otherwise compares character sets.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if tokenize:
        set_a = set(a.split())
        set_b = set(b.split())
    else:
        set_a = set(a)
        set_b = set(b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def token_overlap(a: str, b: str) -> float:
    """Proportion of tokens in a that also appear in b (recall-style overlap)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if not tokens_a:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a)


def edit_similarity(a: str, b: str, max_len: int = None) -> float:
    """Normalized edit similarity in [0, 1].

    Uses rapidfuzz if available; otherwise a simple approximation.
    """
    max_len = max_len or config.MAX_STRING_LENGTH_FOR_EDIT
    a = a[:max_len]
    b = b[:max_len]
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if _HAS_RAPIDFUZZ:
        return _fuzz.ratio(a, b) / 100.0
    # Simple fallback: character overlap ratio
    len_a, len_b = len(a), len(b)
    if len_a == 0 and len_b == 0:
        return 1.0
    common = sum(ca == cb for ca, cb in zip(a, b))
    return 2 * common / (len_a + len_b)


def partial_ratio(a: str, b: str) -> float:
    """Best substring match ratio (rapidfuzz partial_ratio if available)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if _HAS_RAPIDFUZZ:
        return _fuzz.partial_ratio(a, b) / 100.0
    return edit_similarity(a, b)


def token_sort_ratio(a: str, b: str) -> float:
    """Token-sort ratio (rapidfuzz if available)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if _HAS_RAPIDFUZZ:
        return _fuzz.token_sort_ratio(a, b) / 100.0
    # Fallback: sort tokens then edit similarity
    a_sorted = " ".join(sorted(a.split()))
    b_sorted = " ".join(sorted(b.split()))
    return edit_similarity(a_sorted, b_sorted)


def ngram_overlap(a: str, b: str, n: int = 3) -> float:
    """Character n-gram Jaccard overlap."""
    def ngrams(s, n):
        return set(s[i:i+n] for i in range(len(s) - n + 1))
    ng_a = ngrams(a, n)
    ng_b = ngrams(b, n)
    if not ng_a and not ng_b:
        return 1.0
    if not ng_a or not ng_b:
        return 0.0
    return len(ng_a & ng_b) / len(ng_a | ng_b)


# ---------------------------------------------------------------------------
# TFIDF COSINE SIMILARITY (batch, for a set of string pairs)
# ---------------------------------------------------------------------------

def _compute_tfidf_cosine_batch(
    texts_a: list[str],
    texts_b: list[str],
    ngram_range: tuple = (1, 2),
    max_features: int = 10_000,
) -> np.ndarray:
    """Compute TF-IDF cosine similarity for paired lists.

    Returns a 1D numpy array of cosine similarity scores (one per pair).

    NOTE: This fits a fresh vectorizer on each call -- only suitable for
    the training feature extraction stage. For production batch scoring,
    pre-fit and pass the vectorizer.
    """
    if not texts_a:
        return np.array([])

    all_texts = texts_a + texts_b
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range,
                          max_features=max_features, sublinear_tf=True)
    try:
        vec.fit(all_texts)
    except ValueError:
        return np.zeros(len(texts_a))

    mat_a = vec.transform(texts_a)
    mat_b = vec.transform(texts_b)
    # Diagonal of the similarity matrix gives pair-wise cosine
    scores = np.array(mat_a.multiply(mat_b).sum(axis=1)).flatten()
    # Normalize by norms
    norms_a = np.sqrt(np.array(mat_a.power(2).sum(axis=1)).flatten())
    norms_b = np.sqrt(np.array(mat_b.power(2).sum(axis=1)).flatten())
    denom = norms_a * norms_b
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.where(denom > 0, scores / denom, 0.0)
    return scores


# ---------------------------------------------------------------------------
# FEATURE EXTRACTION FOR A BATCH OF PAIRS
# ---------------------------------------------------------------------------

def extract_features_for_pairs(
    pairs: pd.DataFrame,
    s1: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Extract pairwise features for a DataFrame of candidate pairs.

    Parameters
    ----------
    pairs: pd.DataFrame
        Must have columns: source1_entity_id, candidate_entity_id.
    s1: pd.DataFrame
        Preprocessed Source 1 records (indexed by entity_id).
    candidates: pd.DataFrame
        Combined S2 + S3 preprocessed records (indexed by entity_id).

    Returns
    -------
    pd.DataFrame of features, one row per pair.
        All features are floats in [0, 1] or indicator integers (0/1).
    """
    logger.info("Extracting features for %d candidate pairs...", len(pairs))

    # Build entity_id -> row lookups
    s1_lookup = s1.set_index("entity_id")
    cand_lookup = candidates.set_index("entity_id")

    # Pull aligned columns
    s1_ids = pairs["source1_entity_id"].tolist()
    cand_ids = pairs["candidate_entity_id"].tolist()

    def get_col(lookup, ids, col):
        return [_safe_str(lookup[col].get(eid, "")) for eid in ids]

    s1_name_norm  = get_col(s1_lookup,   s1_ids,   "business_name_normalized")
    cand_name_norm= get_col(cand_lookup, cand_ids,  "business_name_normalized")
    s1_addr_norm  = get_col(s1_lookup,   s1_ids,   "business_address_normalized")
    cand_addr_norm= get_col(cand_lookup, cand_ids,  "business_address_normalized")
    s1_country    = get_col(s1_lookup,   s1_ids,   "country_normalized")
    cand_country  = get_col(cand_lookup, cand_ids,  "country_normalized")
    s1_name_raw   = get_col(s1_lookup,   s1_ids,   "business_name")
    cand_name_raw = get_col(cand_lookup, cand_ids,  "business_name")

    features: dict[str, list] = {}

    # -----------------------------------------------------------------------
    # NAME FEATURES
    # -----------------------------------------------------------------------
    logger.debug("Computing name similarity features...")

    features["name_exact_match"] = [
        1.0 if a == b and a != "" else 0.0
        for a, b in zip(s1_name_norm, cand_name_norm)
    ]
    features["name_jaccard"] = [
        jaccard_similarity(a, b) for a, b in zip(s1_name_norm, cand_name_norm)
    ]
    features["name_token_overlap_s1"] = [
        token_overlap(a, b) for a, b in zip(s1_name_norm, cand_name_norm)
    ]
    features["name_token_overlap_cand"] = [
        token_overlap(b, a) for a, b in zip(s1_name_norm, cand_name_norm)
    ]
    features["name_edit_similarity"] = [
        edit_similarity(a, b) for a, b in zip(s1_name_norm, cand_name_norm)
    ]
    features["name_partial_ratio"] = [
        partial_ratio(a, b) for a, b in zip(s1_name_norm, cand_name_norm)
    ]
    features["name_token_sort_ratio"] = [
        token_sort_ratio(a, b) for a, b in zip(s1_name_norm, cand_name_norm)
    ]
    features["name_trigram_overlap"] = [
        ngram_overlap(a, b, n=3) for a, b in zip(s1_name_norm, cand_name_norm)
    ]
    features["name_bigram_overlap"] = [
        ngram_overlap(a, b, n=2) for a, b in zip(s1_name_norm, cand_name_norm)
    ]

    # Length difference features
    features["name_len_diff"] = [
        abs(len(a) - len(b)) / (max(len(a), len(b)) + 1)
        for a, b in zip(s1_name_norm, cand_name_norm)
    ]
    features["name_token_count_diff"] = [
        abs(len(a.split()) - len(b.split())) / (max(len(a.split()), len(b.split())) + 1)
        for a, b in zip(s1_name_norm, cand_name_norm)
    ]

    # -----------------------------------------------------------------------
    # ADDRESS FEATURES
    # -----------------------------------------------------------------------
    logger.debug("Computing address similarity features...")

    features["addr_jaccard"] = [
        jaccard_similarity(a, b) for a, b in zip(s1_addr_norm, cand_addr_norm)
    ]
    features["addr_token_overlap_s1"] = [
        token_overlap(a, b) for a, b in zip(s1_addr_norm, cand_addr_norm)
    ]
    features["addr_token_overlap_cand"] = [
        token_overlap(b, a) for a, b in zip(s1_addr_norm, cand_addr_norm)
    ]
    features["addr_edit_similarity"] = [
        edit_similarity(a, b) for a, b in zip(s1_addr_norm, cand_addr_norm)
    ]
    features["addr_trigram_overlap"] = [
        ngram_overlap(a, b, n=3) for a, b in zip(s1_addr_norm, cand_addr_norm)
    ]
    features["addr_len_diff"] = [
        abs(len(a) - len(b)) / (max(len(a), len(b)) + 1)
        for a, b in zip(s1_addr_norm, cand_addr_norm)
    ]

    # -----------------------------------------------------------------------
    # COUNTRY FEATURES
    # -----------------------------------------------------------------------
    features["country_exact_match"] = [
        1.0 if a == b and a != "" else 0.0
        for a, b in zip(s1_country, cand_country)
    ]

    # -----------------------------------------------------------------------
    # MISSINGNESS INDICATORS
    # -----------------------------------------------------------------------
    features["s1_name_missing"] = [1.0 if not a else 0.0 for a in s1_name_norm]
    features["cand_name_missing"] = [1.0 if not b else 0.0 for b in cand_name_norm]
    features["s1_addr_missing"] = [1.0 if not a else 0.0 for a in s1_addr_norm]
    features["cand_addr_missing"] = [1.0 if not b else 0.0 for b in cand_addr_norm]
    features["s1_country_missing"] = [1.0 if not a else 0.0 for a in s1_country]
    features["cand_country_missing"] = [1.0 if not b else 0.0 for b in cand_country]

    # -----------------------------------------------------------------------
    # CANDIDATE SOURCE (S2 vs S3)
    # -----------------------------------------------------------------------
    features["is_s2"] = [
        1.0 if eid.startswith("S2-") else 0.0 for eid in cand_ids
    ]

    result = pd.DataFrame(features, index=pairs.index)

    # Fill any remaining NaN with 0 (safety net)
    result = result.fillna(0.0)

    logger.info("Feature extraction complete: %d features for %d pairs.", result.shape[1], len(result))
    return result


def get_feature_names() -> list[str]:
    """Return the list of feature column names in the order they are produced."""
    return [
        # Name features
        "name_exact_match",
        "name_jaccard",
        "name_token_overlap_s1",
        "name_token_overlap_cand",
        "name_edit_similarity",
        "name_partial_ratio",
        "name_token_sort_ratio",
        "name_trigram_overlap",
        "name_bigram_overlap",
        "name_len_diff",
        "name_token_count_diff",
        # Address features
        "addr_jaccard",
        "addr_token_overlap_s1",
        "addr_token_overlap_cand",
        "addr_edit_similarity",
        "addr_trigram_overlap",
        "addr_len_diff",
        # Country features
        "country_exact_match",
        # Missingness
        "s1_name_missing",
        "cand_name_missing",
        "s1_addr_missing",
        "cand_addr_missing",
        "s1_country_missing",
        "cand_country_missing",
        # Source
        "is_s2",
    ]
