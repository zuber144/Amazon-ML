"""
tests/test_output.py
====================
Unit tests for output format validation and postprocessing.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
import pandas as pd
from src import postprocessing, config


def _make_predictions(rows):
    return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id", "probability", "is_match"])


class TestBuildMatchOutput(unittest.TestCase):

    def test_every_s1_appears(self):
        preds = _make_predictions([
            ("S1-001", "S2-001", 0.9, 1),
        ])
        all_s1 = ["S1-001", "S1-002", "S1-003"]
        result = postprocessing.build_match_output(preds, all_s1)
        self.assertEqual(len(result), 3)
        self.assertSetEqual(set(result["source1_entity_id"]), set(all_s1))

    def test_singleton_has_empty_matched(self):
        preds = _make_predictions([
            ("S1-001", "S2-001", 0.9, 1),
        ])
        all_s1 = ["S1-001", "S1-002"]
        result = postprocessing.build_match_output(preds, all_s1)
        s1_002_row = result[result["source1_entity_id"] == "S1-002"]
        self.assertEqual(s1_002_row["matched_entity_ids"].iloc[0], "")

    def test_no_nan_in_matched_ids(self):
        preds = _make_predictions([])
        all_s1 = ["S1-001", "S1-002"]
        result = postprocessing.build_match_output(preds, all_s1)
        self.assertFalse(result["matched_entity_ids"].isnull().any())

    def test_no_s1_ids_in_output(self):
        # Even if we pass an S1-S1 pair, it should be removed
        preds = pd.DataFrame({
            "source1_entity_id": ["S1-001"],
            "candidate_entity_id": ["S1-002"],  # S1 ID - should be removed
            "probability": [0.9],
            "is_match": [1],
        })
        all_s1 = ["S1-001", "S1-002"]
        result = postprocessing.build_match_output(preds, all_s1)
        for _, row in result.iterrows():
            if row["matched_entity_ids"]:
                ids = row["matched_entity_ids"].split(",")
                for mid in ids:
                    self.assertFalse(mid.startswith("S1-"), f"Found S1 ID in output: {mid}")

    def test_no_duplicate_ids_in_list(self):
        preds = _make_predictions([
            ("S1-001", "S2-001", 0.9, 1),
            ("S1-001", "S2-001", 0.9, 1),  # duplicate
        ])
        all_s1 = ["S1-001"]
        result = postprocessing.build_match_output(preds, all_s1)
        row = result[result["source1_entity_id"] == "S1-001"]
        ids = row["matched_entity_ids"].iloc[0].split(",")
        self.assertEqual(len(ids), len(set(ids)))

    def test_only_s2_s3_ids(self):
        preds = _make_predictions([
            ("S1-001", "S2-100", 0.9, 1),
            ("S1-001", "S3-200", 0.8, 1),
        ])
        all_s1 = ["S1-001"]
        result = postprocessing.build_match_output(preds, all_s1)
        row = result[result["source1_entity_id"] == "S1-001"]
        ids = row["matched_entity_ids"].iloc[0].split(",")
        for mid in ids:
            self.assertTrue(
                mid.startswith("S2-") or mid.startswith("S3-"),
                f"Unexpected ID prefix: {mid}"
            )


class TestValidateOutput(unittest.TestCase):

    def _make_results(self, rows):
        return pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"])

    def _make_candidates(self, rows):
        return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_ids"])

    def test_valid_output_has_no_errors(self):
        results = self._make_results([
            ("S1-001", "S2-001,S3-001"),
            ("S1-002", ""),
        ])
        candidates = self._make_candidates([
            ("S1-001", "S2-001,S3-001,S2-002"),
            ("S1-002", ""),
        ])
        all_s1 = ["S1-001", "S1-002"]
        errors = postprocessing.validate_output(results, candidates, all_s1)
        self.assertEqual(errors, [])

    def test_detects_missing_s1(self):
        results = self._make_results([
            ("S1-001", ""),
        ])
        candidates = self._make_candidates([("S1-001", "")])
        all_s1 = ["S1-001", "S1-002"]  # S1-002 missing from results
        errors = postprocessing.validate_output(results, candidates, all_s1)
        self.assertTrue(any("missing" in e.lower() for e in errors))

    def test_detects_match_not_in_candidates(self):
        results = self._make_results([
            ("S1-001", "S2-999"),  # S2-999 not in candidates
        ])
        candidates = self._make_candidates([
            ("S1-001", "S2-001"),  # different candidate
        ])
        all_s1 = ["S1-001"]
        errors = postprocessing.validate_output(results, candidates, all_s1)
        self.assertTrue(len(errors) > 0)


if __name__ == "__main__":
    unittest.main()
