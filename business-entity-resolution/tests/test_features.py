"""
tests/test_features.py
======================
Unit tests for the features module.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
import pandas as pd
import numpy as np
from src import features, preprocessing


def _make_proc_df(records):
    df = pd.DataFrame(records)
    return preprocessing.preprocess_source(df, label="test")


class TestSimilarityPrimitives(unittest.TestCase):

    def test_jaccard_identical(self):
        self.assertAlmostEqual(features.jaccard_similarity("abc def", "abc def"), 1.0)

    def test_jaccard_disjoint(self):
        self.assertAlmostEqual(features.jaccard_similarity("abc", "xyz"), 0.0)

    def test_jaccard_empty_both(self):
        self.assertAlmostEqual(features.jaccard_similarity("", ""), 1.0)

    def test_edit_similarity_identical(self):
        self.assertAlmostEqual(features.edit_similarity("abc", "abc"), 1.0)

    def test_edit_similarity_empty_a(self):
        self.assertAlmostEqual(features.edit_similarity("", "abc"), 0.0)

    def test_ngram_overlap_identical(self):
        self.assertAlmostEqual(features.ngram_overlap("abcdef", "abcdef", n=3), 1.0)

    def test_ngram_overlap_disjoint(self):
        self.assertAlmostEqual(features.ngram_overlap("aaaa", "zzzz", n=3), 0.0)


class TestExtractFeatures(unittest.TestCase):

    def setUp(self):
        self.s1 = _make_proc_df([
            {"entity_id": "S1-001", "business_name": "ABC Corp", "business_address": "123 Main St", "country": "US"},
        ])
        self.cands = _make_proc_df([
            {"entity_id": "S2-001", "business_name": "ABC Corporation", "business_address": "123 Main Street", "country": "US"},
            {"entity_id": "S3-001", "business_name": "XYZ Ltd", "business_address": "456 Elm Ave", "country": "India"},
        ])
        self.pairs = pd.DataFrame({
            "source1_entity_id": ["S1-001", "S1-001"],
            "candidate_entity_id": ["S2-001", "S3-001"],
        })

    def test_returns_dataframe(self):
        feats = features.extract_features_for_pairs(self.pairs, self.s1, self.cands)
        self.assertIsInstance(feats, pd.DataFrame)

    def test_correct_row_count(self):
        feats = features.extract_features_for_pairs(self.pairs, self.s1, self.cands)
        self.assertEqual(len(feats), len(self.pairs))

    def test_no_nan_in_output(self):
        feats = features.extract_features_for_pairs(self.pairs, self.s1, self.cands)
        self.assertFalse(feats.isnull().any().any(), "Feature matrix contains NaN values")

    def test_exact_match_for_same_name(self):
        s1 = _make_proc_df([{"entity_id": "S1-001", "business_name": "abc corp", "business_address": "", "country": "US"}])
        cands = _make_proc_df([{"entity_id": "S2-001", "business_name": "abc corp", "business_address": "", "country": "US"}])
        pairs = pd.DataFrame({"source1_entity_id": ["S1-001"], "candidate_entity_id": ["S2-001"]})
        feats = features.extract_features_for_pairs(pairs, s1, cands)
        # After normalization, both will have the same name
        self.assertEqual(feats["name_jaccard"].iloc[0], 1.0)

    def test_country_match_feature(self):
        feats = features.extract_features_for_pairs(self.pairs, self.s1, self.cands)
        # S2-001 same country (US), S3-001 different country (India)
        self.assertEqual(feats["country_exact_match"].iloc[0], 1.0)
        self.assertEqual(feats["country_exact_match"].iloc[1], 0.0)

    def test_is_s2_feature(self):
        feats = features.extract_features_for_pairs(self.pairs, self.s1, self.cands)
        self.assertEqual(feats["is_s2"].iloc[0], 1.0)   # S2-001
        self.assertEqual(feats["is_s2"].iloc[1], 0.0)   # S3-001

    def test_all_features_in_expected_range(self):
        feats = features.extract_features_for_pairs(self.pairs, self.s1, self.cands)
        for col in feats.columns:
            vals = feats[col].values
            self.assertTrue(
                np.all(vals >= 0.0) and np.all(vals <= 1.0),
                f"Feature '{col}' has values outside [0, 1]: {vals}"
            )


if __name__ == "__main__":
    unittest.main()
