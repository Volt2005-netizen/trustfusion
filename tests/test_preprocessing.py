"""
TrustFusion - Unit Tests for Phase 1 & Phase 2 Preprocessing.

Tests schema validation, structured field parsing, label normalization,
deterministic ID generation, and strict train-test anti-leakage constraints.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.preprocess import (
    EXPECTED_COLUMNS,
    NUMERICAL_FEATURES,
    LABEL_MAP,
    clean_numerical_features,
    clean_scalar,
    create_profile_id,
    extract_intro_text,
    fit_minmax_and_save,
    normalize_labels,
    parse_count,
    parse_structured_value,
    stratified_split,
    validate_schema,
)


class TestPreprocessing(unittest.TestCase):

    def test_schema_validation_success(self):
        df = pd.DataFrame(columns=EXPECTED_COLUMNS)
        try:
            validate_schema(df, "test_df")
        except ValueError as e:
            self.fail(f"validate_schema raised unexpected ValueError: {e}")

    def test_schema_validation_missing_columns(self):
        df = pd.DataFrame(columns=["Intro", "Full Name", "Label"])
        with self.assertRaises(ValueError):
            validate_schema(df, "incomplete_df")

    def test_parse_count_various_formats(self):
        self.assertEqual(parse_count(500), 500.0)
        self.assertEqual(parse_count("500+"), 500.0)
        self.assertEqual(parse_count("1,234"), 1234.0)
        self.assertEqual(parse_count("1.5K"), 1500.0)
        self.assertEqual(parse_count("2M"), 2000000.0)
        self.assertTrue(np.isnan(parse_count(None)))
        self.assertTrue(np.isnan(parse_count("invalid")))

    def test_clean_numerical_features_negative_values(self):
        data = {col: [0] for col in EXPECTED_COLUMNS}
        data["Connections"] = [-10]
        data["Followers"] = [50]
        df = pd.DataFrame(data)
        cleaned = clean_numerical_features(df)
        self.assertTrue(np.isnan(cleaned.loc[0, "Connections"]))
        self.assertEqual(cleaned.loc[0, "Followers"], 50.0)

    def test_extract_intro_text_standard_dict(self):
        standard_intro = {
            "Full Name": "Alice Smith",
            "Workplace": "CyberCorp",
            "Location": "San Francisco, CA",
            "Connections": "500+",
            "Photo": "Yes",
            "Followers": "1000",
        }
        text = extract_intro_text(standard_intro)
        self.assertIn("Full Name: Alice Smith", text)
        self.assertIn("Workplace: CyberCorp", text)
        self.assertIn("Location: San Francisco, CA", text)

    def test_extract_intro_text_chatgpt_dict(self):
        chatgpt_intro = {
            "chatGPT": "1. John Doe#@Recruiter#@Tech Solutions#@New York\n2. Jane Roe#@Analyst#@Global#@Chicago",
            "random": 0,
        }
        text = extract_intro_text(chatgpt_intro)
        self.assertTrue(len(text) > 0)
        self.assertIn("John Doe", text)
        self.assertIn("Tech Solutions", text)

    def test_label_normalization(self):
        df = pd.DataFrame({
            "Label": [0, 1, 10, 11, 12],
        })
        normalized = normalize_labels(df)
        self.assertListEqual(
            normalized["class_4"].tolist(),
            [0, 1, 2, 2, 3]
        )
        self.assertListEqual(
            normalized["is_fake"].tolist(),
            [0, 1, 1, 1, 1]
        )
        self.assertListEqual(
            normalized["class_name"].tolist(),
            ["legitimate", "manual_fake", "chatgpt_fake", "chatgpt_fake", "gpt4_adversarial"]
        )

    def test_deterministic_profile_id_label_invariance(self):
        data1 = {col: f"val_{col}" for col in EXPECTED_COLUMNS}
        data1["Label"] = 0
        row1 = pd.Series(data1, dtype=object)

        data2 = {col: f"val_{col}" for col in EXPECTED_COLUMNS}
        data2["Label"] = 1  # Different label, identical profile content
        row2 = pd.Series(data2, dtype=object)

        id1 = create_profile_id(row1)
        id2 = create_profile_id(row2)

        self.assertEqual(id1, id2)
        self.assertTrue(id1.startswith("TF_"))
        self.assertEqual(len(id1), 19)  # "TF_" + 16 chars hex

    def test_train_test_split_and_no_leakage(self):
        # Create a synthetic dataset of 100 profiles across all 4 classes
        rows = []
        for i in range(100):
            label = 0 if i < 50 else (1 if i < 70 else (10 if i < 90 else 12))
            row = {col: f"val_{i}_{col}" for col in EXPECTED_COLUMNS}
            row["Label"] = label
            row["Connections"] = float(i * 10)
            row["Followers"] = float(i * 20)
            for num_col in NUMERICAL_FEATURES:
                row[num_col] = float(i % 5)
            rows.append(row)

        df = pd.DataFrame(rows)
        df = clean_numerical_features(df)
        df = normalize_labels(df)
        df["profile_id"] = [f"TF_{i:04d}" for i in range(len(df))]

        train, test = stratified_split(df, test_size=0.30)

        # Assert no overlap in profile IDs
        train_ids = set(train["profile_id"])
        test_ids = set(test["profile_id"])
        self.assertEqual(len(train_ids & test_ids), 0)
        self.assertEqual(len(train) + len(test), len(df))

        # Check stratification preserves class proportions approximately
        train_classes = set(train["class_4"])
        test_classes = set(test["class_4"])
        self.assertEqual(train_classes, test_classes)


if __name__ == "__main__":
    unittest.main()
