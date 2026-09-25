"""
tests/test_blocking.py
======================
Unit tests for the blocking module.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
import pandas as pd
from src import blocking, preprocessing, config


def _make_source_df(records: list[dict]) -> pd.DataFrame:
    """Create a minimal preprocessed source DataFrame for testing."""
    df = pd.DataFrame(records)
    return preprocessing.preprocess_source(df, label="test")


class TestBlockingNoS1S1(unittest.TestCase):
    """Ensure blocking never produces S1 -> S1 candidate pairs."""

    def setUp(self):
        self.s1 = _make_source_df([
            {"entity_id": "S1-001", "business_name": "ABC Corp", "business_address": "123 Main St", "country": "US"},
            {"entity_id": "S1-002", "business_name": "XYZ Ltd", "business_address": "456 Elm St", "country": "US"},
        ])
        self.s2 = _make_source_df([
            {"entity_id": "S2-001", "business_name": "ABC Corporation", "business_address": "123 Main Street", "country": "US"},
        ])
        self.s3 = _make_source_df([
            {"entity_id": "S3-001", "business_name": "XYZ Limited", "business_address": "456 Elm Street", "country": "US"},
        ])
        self.candidates = pd.concat([self.s2, self.s3], ignore_index=True)

    def _assert_no_s1_candidates(self, pairs):
        for s1_id, cand_id in pairs:
            self.assertFalse(
                cand_id.startswith(config.SOURCE1_PREFIX),
                f"Found S1-S1 pair: {s1_id} -> {cand_id}"
            )

    def test_exact_name_no_s1_s1(self):
        pairs = blocking.exact_name_blocking(self.s1, self.candidates)
        self._assert_no_s1_candidates(pairs)

    def test_token_name_no_s1_s1(self):
        pairs = blocking.token_name_blocking(self.s1, self.candidates)
        self._assert_no_s1_candidates(pairs)

    def test_token_address_no_s1_s1(self):
        pairs = blocking.token_address_blocking(self.s1, self.candidates, min_shared=1)
        self._assert_no_s1_candidates(pairs)


class TestBlockingRetrieval(unittest.TestCase):
    """Ensure obvious matches are retrieved by at least one blocking strategy."""

    def setUp(self):
        self.s1 = _make_source_df([
            {"entity_id": "S1-001", "business_name": "ABC Technologies", "business_address": "12 MG Road", "country": "India"},
        ])
        self.candidates = _make_source_df([
            {"entity_id": "S2-001", "business_name": "ABC Tech", "business_address": "12 MG Rd", "country": "India"},
            {"entity_id": "S3-001", "business_name": "Completely Different Business", "business_address": "99 Other Street", "country": "US"},
        ])

    def test_exact_or_token_retrieves_abc_tech(self):
        pairs_exact = blocking.exact_name_blocking(self.s1, self.candidates)
        pairs_token = blocking.token_name_blocking(self.s1, self.candidates, min_shared=1)
        all_candidates = {cand_id for _, cand_id in (pairs_exact | pairs_token)}
        # ABC Tech should be retrieved (shares 'abc' token)
        self.assertIn("S2-001", all_candidates, "ABC Tech should be a candidate")


class TestBlockingCountryAware(unittest.TestCase):
    """Country-aware blocking must handle France and unknown countries."""

    def setUp(self):
        self.s1 = _make_source_df([
            {"entity_id": "S1-001", "business_name": "Boulangerie Martin", "business_address": "5 Rue de la Paix", "country": "France"},
            {"entity_id": "S1-002", "business_name": "Unknown Co", "business_address": "Some Address", "country": ""},
        ])
        self.candidates = _make_source_df([
            {"entity_id": "S2-001", "business_name": "Boulangerie Martin", "business_address": "5 Rue de la Paix", "country": "France"},
            {"entity_id": "S2-002", "business_name": "Some Other Business", "business_address": "Other Address", "country": "US"},
        ])

    def test_french_entity_gets_candidates(self):
        pairs = blocking.country_aware_blocking(self.s1, self.candidates)
        candidates_for_s1_001 = {cand_id for s1_id, cand_id in pairs if s1_id == "S1-001"}
        self.assertIn("S2-001", candidates_for_s1_001)

    def test_unknown_country_handled(self):
        # Should not raise even if country is empty
        try:
            pairs = blocking.country_aware_blocking(self.s1, self.candidates)
        except Exception as e:
            self.fail(f"country_aware_blocking raised with unknown country: {e}")


if __name__ == "__main__":
    unittest.main()
