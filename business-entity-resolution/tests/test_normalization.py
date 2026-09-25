"""
tests/test_normalization.py
============================
Unit tests for the normalization module.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from src.normalization import normalize_name, normalize_address, normalize_country


class TestNormalizeName(unittest.TestCase):

    def test_empty_returns_empty(self):
        self.assertEqual(normalize_name(""), "")
        self.assertEqual(normalize_name(None), "")

    def test_lowercase(self):
        self.assertEqual(normalize_name("ABC Corp").lower(), normalize_name("ABC Corp"))

    def test_legal_suffix_corporation_to_corp(self):
        n1 = normalize_name("ABC Corporation")
        n2 = normalize_name("ABC Corp.")
        # Both should normalize to same base
        self.assertIn("abc", n1)
        self.assertIn("corp", n1)
        self.assertIn("corp", n2)

    def test_private_limited_variants(self):
        variants = [
            "ABC Technologies Pvt Ltd",
            "ABC Technologies Private Limited",
            "ABC Technologies Pvt. Ltd.",
        ]
        normalized = [normalize_name(v) for v in variants]
        # All should contain "pvt ltd" or similar normalized form
        for n in normalized:
            self.assertIn("abc", n)
            self.assertIn("tech", n)  # "technologies" -> "tech"

    def test_and_ampersand(self):
        n1 = normalize_name("Smith and Jones")
        n2 = normalize_name("Smith & Jones")
        # Both should produce the same (or similar) result
        self.assertIn("smith", n1)
        self.assertIn("jones", n1)

    def test_whitespace_normalization(self):
        n = normalize_name("  ABC   Corp  ")
        self.assertFalse(n.startswith(" "))
        self.assertFalse(n.endswith(" "))
        self.assertNotIn("  ", n)

    def test_unicode_normalization(self):
        # Should not crash on non-ASCII
        n = normalize_name("राम मार्केटिंग प्राइवेट लिमिटेड")
        self.assertIsInstance(n, str)

    def test_none_is_safe(self):
        self.assertEqual(normalize_name(None), "")

    def test_nan_is_safe(self):
        import math
        self.assertEqual(normalize_name(float("nan")), "")


class TestNormalizeAddress(unittest.TestCase):

    def test_road_abbreviation(self):
        a1 = normalize_address("12 MG Road, Bengaluru")
        a2 = normalize_address("12 MG Rd, Bengaluru")
        # Both should normalize "road/rd" to same token
        self.assertIn("rd", a1)
        self.assertIn("rd", a2)

    def test_empty_returns_empty(self):
        self.assertEqual(normalize_address(""), "")
        self.assertEqual(normalize_address(None), "")

    def test_numbers_preserved(self):
        a = normalize_address("797 Lake Town Block A, Kolkata")
        self.assertIn("797", a)

    def test_lowercase(self):
        a = normalize_address("12 MG ROAD, BENGALURU")
        self.assertEqual(a, normalize_address("12 mg road, bengaluru"))


class TestNormalizeCountry(unittest.TestCase):

    def test_us(self):
        self.assertEqual(normalize_country("US"), "us")

    def test_india(self):
        self.assertEqual(normalize_country("India"), "india")

    def test_france(self):
        # France must not be rejected
        self.assertEqual(normalize_country("France"), "france")

    def test_empty(self):
        self.assertEqual(normalize_country(""), "")
        self.assertEqual(normalize_country(None), "")

    def test_unknown_country_passes_through(self):
        result = normalize_country("Germany")
        self.assertEqual(result, "germany")


if __name__ == "__main__":
    unittest.main()
