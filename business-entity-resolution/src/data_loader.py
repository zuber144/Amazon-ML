"""
src/data_loader.py
==================
Responsible for loading TSV files, validating schemas, detecting missing
files, and returning clean pandas DataFrames.

Owner: Team Member 1 (Data / Normalization)
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# INTERNAL HELPERS
# ---------------------------------------------------------------------------


def _load_tsv(path: Path, expected_columns: list[str], label: str) -> pd.DataFrame:
    """Load a single TSV file with schema validation.

    Parameters
    ----------
    path: Path
        Absolute path to the TSV file.
    expected_columns: list[str]
        Columns the file must contain (order-insensitive).
    label: str
        Human-readable name used in error messages.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame with only the expected columns.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If expected columns are missing or duplicate entity_id rows exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"[{label}] Required file not found: {path}. "
            "Please place the dataset files in the correct directory."
        )

    logger.info("[%s] Loading %s ...", label, path)
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)

    # Strip accidental leading/trailing whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    # Validate expected columns
    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"[{label}] Missing expected columns: {sorted(missing_cols)}. "
            f"Found columns: {df.columns.tolist()}"
        )

    df = df[expected_columns].copy()

    logger.info("[%s] Loaded %d records.", label, len(df))
    return df


def _validate_entity_ids(df: pd.DataFrame, prefix: str, label: str) -> None:
    """Check that all entity_ids start with the expected prefix and are unique.

    Raises
    ------
    ValueError
        On prefix violations or duplicate IDs.
    """
    bad_prefix = df["entity_id"][~df["entity_id"].str.startswith(prefix, na=False)]
    if not bad_prefix.empty:
        sample = bad_prefix.head(5).tolist()
        raise ValueError(
            f"[{label}] Found {len(bad_prefix)} entity_ids not starting with "
            f"'{prefix}'. Examples: {sample}"
        )

    dupes = df["entity_id"][df["entity_id"].duplicated(keep=False)]
    if not dupes.empty:
        sample = dupes.head(5).tolist()
        raise ValueError(
            f"[{label}] Found {dupes.nunique()} duplicate entity_ids. "
            f"Examples: {sample}"
        )


def _report_missing_values(df: pd.DataFrame, label: str) -> None:
    """Log missing-value counts per column (informational only)."""
    nulls = df.isnull().sum()
    if nulls.any():
        details = ", ".join(f"{c}={n}" for c, n in nulls.items() if n > 0)
        logger.info("[%s] Missing values: %s", label, details)
    else:
        logger.debug("[%s] No missing values detected.", label)


# ---------------------------------------------------------------------------
# PUBLIC LOADER FUNCTIONS
# ---------------------------------------------------------------------------


def load_source1(path: Optional[Path] = None) -> pd.DataFrame:
    """Load Source 1 (deduplicated reference) records.

    Parameters
    ----------
    path: Path, optional
        Override the default path from config.

    Returns
    -------
    pd.DataFrame with columns: entity_id, business_name, business_address, country
    """
    path = Path(path) if path else config.TRAIN_SOURCE1
    df = _load_tsv(path, config.SOURCE_COLUMNS, "Source1")
    _validate_entity_ids(df, config.SOURCE1_PREFIX, "Source1")
    _report_missing_values(df, "Source1")
    return df


def load_source2(path: Optional[Path] = None) -> pd.DataFrame:
    """Load Source 2 (noisy) records.

    Parameters
    ----------
    path: Path, optional
        Override the default path from config.

    Returns
    -------
    pd.DataFrame with columns: entity_id, business_name, business_address, country
    """
    path = Path(path) if path else config.TRAIN_SOURCE2
    df = _load_tsv(path, config.SOURCE_COLUMNS, "Source2")
    _validate_entity_ids(df, config.SOURCE2_PREFIX, "Source2")
    _report_missing_values(df, "Source2")
    return df


def load_source3(path: Optional[Path] = None) -> pd.DataFrame:
    """Load Source 3 (noisy) records.

    Parameters
    ----------
    path: Path, optional
        Override the default path from config.

    Returns
    -------
    pd.DataFrame with columns: entity_id, business_name, business_address, country
    """
    path = Path(path) if path else config.TRAIN_SOURCE3
    df = _load_tsv(path, config.SOURCE_COLUMNS, "Source3")
    _validate_entity_ids(df, config.SOURCE3_PREFIX, "Source3")
    _report_missing_values(df, "Source3")
    return df


def load_ground_truth(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the training ground truth file.

    Parameters
    ----------
    path: Path, optional
        Override the default path from config.

    Returns
    -------
    pd.DataFrame with columns: source1_entity_id, matched_entity_ids
        matched_entity_ids is a comma-separated string or NaN for singletons.
    """
    path = Path(path) if path else config.TRAIN_GROUND_TRUTH
    df = _load_tsv(path, config.GROUND_TRUTH_COLUMNS, "GroundTruth")

    # Validate source1_entity_id prefix and uniqueness
    bad = df["source1_entity_id"][
        ~df["source1_entity_id"].str.startswith(config.SOURCE1_PREFIX, na=False)
    ]
    if not bad.empty:
        raise ValueError(
            f"[GroundTruth] Found {len(bad)} source1_entity_ids not starting with "
            f"'{config.SOURCE1_PREFIX}'. Examples: {bad.head(5).tolist()}"
        )

    dupes = df["source1_entity_id"][df["source1_entity_id"].duplicated(keep=False)]
    if not dupes.empty:
        raise ValueError(
            f"[GroundTruth] Duplicate source1_entity_ids: {dupes.head(5).tolist()}"
        )

    n_singletons = df["matched_entity_ids"].isna().sum() + (
        df["matched_entity_ids"].str.strip().eq("").sum()
    )
    logger.info(
        "[GroundTruth] %d rows | %d singletons | %d with matches.",
        len(df),
        n_singletons,
        len(df) - n_singletons,
    )
    return df


def load_test_data(
    source1_path: Optional[Path] = None,
    source2_path: Optional[Path] = None,
    source3_path: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all three test source files.

    Parameters
    ----------
    source1_path, source2_path, source3_path: Path, optional
        Override the default paths from config.

    Returns
    -------
    tuple of (test_s1, test_s2, test_s3) DataFrames
    """
    logger.info("Loading test data...")
    ts1 = load_source1(source1_path or config.TEST_SOURCE1)
    ts2 = load_source2(source2_path or config.TEST_SOURCE2)
    ts3 = load_source3(source3_path or config.TEST_SOURCE3)
    logger.info(
        "Test data loaded: S1=%d, S2=%d, S3=%d",
        len(ts1), len(ts2), len(ts3),
    )
    return ts1, ts2, ts3


def load_train_data(
    source1_path: Optional[Path] = None,
    source2_path: Optional[Path] = None,
    source3_path: Optional[Path] = None,
    ground_truth_path: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all training files: three source files and ground truth.

    Parameters
    ----------
    source1_path, source2_path, source3_path, ground_truth_path: Path, optional
        Override the default paths from config.

    Returns
    -------
    tuple of (s1, s2, s3, ground_truth) DataFrames
    """
    logger.info("Loading training data...")
    s1 = load_source1(source1_path or config.TRAIN_SOURCE1)
    s2 = load_source2(source2_path or config.TRAIN_SOURCE2)
    s3 = load_source3(source3_path or config.TRAIN_SOURCE3)
    gt = load_ground_truth(ground_truth_path or config.TRAIN_GROUND_TRUTH)
    logger.info(
        "Training data loaded: S1=%d, S2=%d, S3=%d, GT=%d",
        len(s1), len(s2), len(s3), len(gt),
    )
    return s1, s2, s3, gt
