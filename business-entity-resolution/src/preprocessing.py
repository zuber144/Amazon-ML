"""
src/preprocessing.py
====================
Applies normalization to raw DataFrames, adding normalized columns while
preserving the originals. Handles missing values consistently.

Owner: Team Member 1 (Data / Normalization)
"""

import logging
import pandas as pd

from src import normalization

logger = logging.getLogger(__name__)


def preprocess_source(df: pd.DataFrame, label: str = "source") -> pd.DataFrame:
    """Apply normalization to a source DataFrame (S1, S2, or S3).

    Adds the following columns alongside the originals:
        business_name_normalized      - normalized business name
        business_address_normalized   - normalized business address
        country_normalized            - normalized country
        name_tokens                   - list of name tokens (for blocking)
        address_tokens                - list of address tokens (for blocking)

    Missing values in business_name / business_address are filled with ""
    before normalization so downstream code does not need to guard for NaN.

    Parameters
    ----------
    df: pd.DataFrame
        Must contain columns: entity_id, business_name, business_address, country
    label: str
        Human-readable label for log messages.

    Returns
    -------
    pd.DataFrame: copy of df with additional normalized columns.
    """
    logger.info("[%s] Preprocessing %d records...", label, len(df))
    df = df.copy()

    # Fill missing raw values with empty string for safe string operations
    df["business_name"] = df["business_name"].fillna("")
    df["business_address"] = df["business_address"].fillna("")
    df["country"] = df["country"].fillna("")

    # Normalize
    logger.debug("[%s] Normalizing names...", label)
    df["business_name_normalized"] = df["business_name"].apply(normalization.normalize_name)

    logger.debug("[%s] Normalizing addresses...", label)
    df["business_address_normalized"] = df["business_address"].apply(
        normalization.normalize_address
    )

    logger.debug("[%s] Normalizing countries...", label)
    df["country_normalized"] = df["country"].apply(normalization.normalize_country)

    # Token lists (used by blocking)
    df["name_tokens"] = df["business_name_normalized"].apply(
        lambda x: normalization.tokenize_name(x, min_length=2)
    )
    df["address_tokens"] = df["business_address_normalized"].apply(
        lambda x: normalization.tokenize_address(x, min_length=2)
    )

    logger.info("[%s] Preprocessing complete.", label)
    return df


def preprocess_all(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Preprocess all three source DataFrames.

    Parameters
    ----------
    s1, s2, s3: pd.DataFrame
        Raw source DataFrames from data_loader.

    Returns
    -------
    tuple of (s1_proc, s2_proc, s3_proc) preprocessed DataFrames.
    """
    logger.info("Preprocessing all sources...")
    s1_proc = preprocess_source(s1, label="Source1")
    s2_proc = preprocess_source(s2, label="Source2")
    s3_proc = preprocess_source(s3, label="Source3")
    logger.info("All sources preprocessed.")
    return s1_proc, s2_proc, s3_proc
