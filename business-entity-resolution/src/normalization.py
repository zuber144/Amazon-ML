"""
src/normalization.py
====================
Text normalization functions for business names, addresses, and country fields.

All functions are CONSERVATIVE: they remove obvious noise while preserving
meaningful tokens. Raw values are never modified in-place - callers receive
new normalized columns.

Owner: Team Member 1 (Data / Normalization)
"""

import re
import unicodedata
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LEGAL SUFFIX NORMALIZATION MAP
# Normalize common legal entity suffix abbreviations to a canonical form.
# Keep this conservative - only normalize clear-cut cases.
# ---------------------------------------------------------------------------
_LEGAL_SUFFIX_MAP = {
    r"\bprivate limited\b": "pvt ltd",
    r"\bpvt\. limited\b": "pvt ltd",
    r"\bpvt ltd\b": "pvt ltd",
    r"\bpvt\. ltd\b": "pvt ltd",
    r"\bprivate ltd\b": "pvt ltd",
    r"\bpte ltd\b": "pvt ltd",
    r"\bp\.?v\.?t\.? l\.?t\.?d\.?\b": "pvt ltd",
    r"\bllimited\b": "ltd",
    r"\blimited\b": "ltd",
    r"\bltd\b": "ltd",
    r"\bllp\b": "llp",
    r"\bllc\b": "llc",
    r"\bl\.l\.c\.?\b": "llc",
    r"\bl\.l\.p\.?\b": "llp",
    r"\bincorporated\b": "inc",
    r"\binc\.\b": "inc",
    r"\binc\b": "inc",
    r"\bcorporation\b": "corp",
    r"\bcorp\.\b": "corp",
    r"\bcorp\b": "corp",
    r"\bco\.\b": "co",
    r"\bcompany\b": "co",
    r"\band\b": "&",
    r"\benterprises\b": "enterprises",
    r"\benterprise\b": "enterprise",
    r"\bservices\b": "services",
    r"\bsolutions\b": "solutions",
    r"\btechnologies\b": "tech",
    r"\btechnology\b": "tech",
}

# Compile once for performance
_LEGAL_SUFFIX_PATTERNS = [
    (re.compile(pat, re.IGNORECASE), repl)
    for pat, repl in _LEGAL_SUFFIX_MAP.items()
]

# ---------------------------------------------------------------------------
# ADDRESS ABBREVIATION MAP
# Road/street abbreviations, directional tokens, etc.
# Only clearly unambiguous expansions/compressions.
# ---------------------------------------------------------------------------
_ADDRESS_ABBR_MAP = {
    r"\broad\b": "rd",
    r"\brd\b": "rd",
    r"\bstreet\b": "st",
    r"\bst\b": "st",
    r"\bavenue\b": "ave",
    r"\bave\b": "ave",
    r"\bboulevard\b": "blvd",
    r"\bblvd\b": "blvd",
    r"\bdrive\b": "dr",
    r"\bdr\b": "dr",
    r"\blane\b": "ln",
    r"\bln\b": "ln",
    r"\bcourt\b": "ct",
    r"\bct\b": "ct",
    r"\bplace\b": "pl",
    r"\bpl\b": "pl",
    r"\bnorth\b": "n",
    r"\bsouth\b": "s",
    r"\beast\b": "e",
    r"\bwest\b": "w",
    r"\bnortheast\b": "ne",
    r"\bnorthwest\b": "nw",
    r"\bsoutheast\b": "se",
    r"\bsouthwest\b": "sw",
    r"\bsuite\b": "ste",
    r"\bste\b": "ste",
    r"\bapartment\b": "apt",
    r"\bapt\b": "apt",
    r"\bfloor\b": "fl",
    r"\bfl\b": "fl",
    r"\bbuilding\b": "bldg",
    r"\bbldg\b": "bldg",
    r"\bblock\b": "blk",
    r"\bblk\b": "blk",
    r"\bnagar\b": "ngr",
    r"\bngr\b": "ngr",
    r"\bmarg\b": "marg",
    r"\bcolony\b": "col",
}

_ADDRESS_ABBR_PATTERNS = [
    (re.compile(pat, re.IGNORECASE), repl)
    for pat, repl in _ADDRESS_ABBR_MAP.items()
]

# Punctuation to strip (but preserve meaningful characters like digits and
# alphanumeric characters)
_PUNCT_RE = re.compile(r"[^\w\s&/-]", re.UNICODE)
_MULTI_SPACE_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# CORE UTILITIES
# ---------------------------------------------------------------------------


def unicode_normalize(text: str) -> str:
    """Apply Unicode NFC normalization and strip leading/trailing whitespace."""
    return unicodedata.normalize("NFC", text).strip()


def to_ascii_safe(text: str) -> str:
    """Transliterate Latin-extended characters to ASCII where safe.

    Non-Latin scripts (Devanagari, Kannada, etc.) are left intact because
    transliteration would destroy meaningful tokens.
    """
    try:
        return unicodedata.normalize("NFKD", text).encode("ascii", errors="ignore").decode("ascii")
    except Exception:
        return text


def _clean_punctuation(text: str) -> str:
    """Remove most punctuation while preserving & / - and alphanumeric content."""
    text = _PUNCT_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# PUBLIC NORMALIZATION FUNCTIONS
# ---------------------------------------------------------------------------


def normalize_name(text: Optional[str]) -> str:
    """Normalize a business name string.

    Steps (in order):
    1. Handle None / NaN / empty → return ""
    2. Unicode NFC normalization
    3. Lowercase
    4. Punctuation normalization (preserve & / -)
    5. Whitespace normalization
    6. Legal suffix normalization

    The result is suitable for use as a blocking key or feature input.
    Raw business_name is preserved alongside this in the DataFrame.

    Parameters
    ----------
    text: str or None

    Returns
    -------
    str: normalized name, possibly empty string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    t = unicode_normalize(text)
    t = t.lower()
    t = _clean_punctuation(t)

    # Apply legal suffix normalization
    for pattern, replacement in _LEGAL_SUFFIX_PATTERNS:
        t = pattern.sub(replacement, t)

    t = _MULTI_SPACE_RE.sub(" ", t).strip()
    return t


def normalize_address(text: Optional[str]) -> str:
    """Normalize a business address string.

    Steps:
    1. Handle None / NaN / empty → return ""
    2. Unicode NFC normalization
    3. Lowercase
    4. Punctuation normalization
    5. Whitespace normalization
    6. Road/street abbreviation normalization
    7. Preserve meaningful numbers (house numbers, PIN codes, etc.)

    Parameters
    ----------
    text: str or None

    Returns
    -------
    str: normalized address, possibly empty string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    t = unicode_normalize(text)
    t = t.lower()
    t = _clean_punctuation(t)

    # Apply address abbreviation normalization
    for pattern, replacement in _ADDRESS_ABBR_PATTERNS:
        t = pattern.sub(replacement, t)

    t = _MULTI_SPACE_RE.sub(" ", t).strip()
    return t


def normalize_country(text: Optional[str]) -> str:
    """Normalize a country field.

    Conservative: lowercase + strip. Never rejects or transforms unknown countries.
    Country is an open-set field (US, India in train; France also in test).

    Parameters
    ----------
    text: str or None

    Returns
    -------
    str: lowercase stripped country string, or "" for missing values.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    return text.strip().lower()


# ---------------------------------------------------------------------------
# TOKENIZATION HELPERS
# ---------------------------------------------------------------------------


def tokenize_name(normalized_name: str, min_length: int = 2) -> list[str]:
    """Tokenize a normalized business name into meaningful tokens.

    Filters very short tokens that are likely noise (e.g., single characters),
    but preserves tokens like 'co', 'llc', 'llp' which have entity-resolution value.

    Parameters
    ----------
    normalized_name: str
    min_length: int
        Minimum character length to keep a token. Default 2.

    Returns
    -------
    list[str]: list of tokens, may be empty.
    """
    if not normalized_name:
        return []
    tokens = normalized_name.split()
    return [t for t in tokens if len(t) >= min_length]


def tokenize_address(normalized_address: str, min_length: int = 2) -> list[str]:
    """Tokenize a normalized address string.

    Parameters
    ----------
    normalized_address: str
    min_length: int

    Returns
    -------
    list[str]: list of tokens.
    """
    if not normalized_address:
        return []
    tokens = normalized_address.split()
    return [t for t in tokens if len(t) >= min_length]


def get_name_tokens_set(normalized_name: str, min_length: int = 3) -> frozenset:
    """Return a frozenset of tokens (for blocking key construction).

    min_length=3 to skip noise and 2-letter abbreviations at blocking stage.
    """
    return frozenset(tokenize_name(normalized_name, min_length))


def get_address_tokens_set(normalized_address: str, min_length: int = 3) -> frozenset:
    """Return a frozenset of address tokens (for blocking key construction)."""
    return frozenset(tokenize_address(normalized_address, min_length))
