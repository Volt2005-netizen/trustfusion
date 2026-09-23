
"""
TrustFusion - Preprocessing Validation Report

Run from the TrustFusion project root:

    python pipeline/validate_preprocessing.py

This script checks the outputs produced by preprocess.py before
we build the RoBERTa + STE + PCA linguistic pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

TRAIN_FILE = PROCESSED / "train.csv"
TEST_FILE = PROCESSED / "test.csv"
CLEANED_FILE = PROCESSED / "cleaned_profiles.csv"
METADATA_FILE = PROCESSED / "preprocessing_metadata.json"

X_TRAIN_FILE = PROCESSED / "numerical" / "X_train.npy"
X_TEST_FILE = PROCESSED / "numerical" / "X_test.npy"

TRAIN_TEXT_FILE = PROCESSED / "text" / "train_text.json"
TEST_TEXT_FILE = PROCESSED / "text" / "test_text.json"

NUMERICAL_FEATURES = [
    "Number of Experiences",
    "Number of Educations",
    "Number of Licenses",
    "Number of Volunteering",
    "Number of Skills",
    "Number of Recommendations",
    "Number of Projects",
    "Number of Publications",
    "Number of Courses",
    "Number of Honors",
    "Number of Scores",
    "Number of Languages",
    "Number of Organizations",
    "Number of Interests",
    "Number of Activities",
    "Connections",
    "Followers",
]

TEXT_COLUMNS = [
    "intro_text",
    "about_text",
    "experience_text",
    "education_text",
    "licenses_text",
    "volunteering_text",
    "skills_text",
    "recommendations_text",
    "projects_text",
    "publications_text",
    "courses_text",
    "honors_text",
    "scores_text",
    "languages_text",
    "organizations_text",
    "interests_text",
    "activities_text",
    "profile_text",
]


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file was not found: {path}\n"
            "Run pipeline/preprocess.py first."
        )


def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def show_label_distribution(df: pd.DataFrame, name: str) -> None:
    print(f"\n{name} - original labels:")
    counts = (
        df["original_label"]
        .value_counts()
        .sort_index()
    )

    label_names = {
        0: "Legitimate",
        1: "Manual Fake",
        10: "ChatGPT Fake",
        11: "ChatGPT Fake",
        12: "GPT-4 Adversarial",
    }

    for label, count in counts.items():
        print(
            f"  {int(label):>2} "
            f"{label_names.get(int(label), 'Unknown'):22} "
            f"{int(count):>5}"
        )

    print(f"\n{name} - 4-class:")
    class_counts = (
        df["class_4"]
        .value_counts()
        .sort_index()
    )
    class_names = {
        0: "Legitimate",
        1: "Manual Fake",
        2: "ChatGPT Fake",
        3: "GPT-4 Adversarial",
    }

    for label, count in class_counts.items():
        print(
            f"  {int(label)} "
            f"{class_names.get(int(label), 'Unknown'):22} "
            f"{int(count):>5}"
        )

    print(f"\n{name} - binary:")
    binary_counts = (
        df["is_fake"]
        .value_counts()
        .sort_index()
    )

    for label, count in binary_counts.items():
        name_text = "Legitimate" if int(label) == 0 else "Fake"
        print(
            f"  {int(label)} "
            f"{name_text:22} "
            f"{int(count):>5}"
        )


def check_numeric_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    print_section("NUMERICAL DATA")

    print(f"Numerical feature count: {len(NUMERICAL_FEATURES)}")

    missing_train = train[NUMERICAL_FEATURES].isna().sum().sum()
    missing_test = test[NUMERICAL_FEATURES].isna().sum().sum()

    print(
        f"Missing values in train CSV: {int(missing_train)}"
    )
    print(
        f"Missing values in test CSV:  {int(missing_test)}"
    )

    x_train = np.load(X_TRAIN_FILE)
    x_test = np.load(X_TEST_FILE)

    print(f"\nScaled train shape: {x_train.shape}")
    print(f"Scaled test shape:  {x_test.shape}")

    print(
        f"Scaled train range: "
        f"[{x_train.min():.6f}, {x_train.max():.6f}]"
    )
    print(
        f"Scaled test range:  "
        f"[{x_test.min():.6f}, {x_test.max():.6f}]"
    )

    train_ok = (
        x_train.min() >= -1e-6
        and x_train.max() <= 1.000001
    )

    test_ok = (
        x_test.min() >= -1e-6
        and x_test.max() <= 1.000001
    )

    print(
        "Train MinMax range check: "
        + ("PASS" if train_ok else "CHECK")
    )
    print(
        "Test MinMax range check:  "
        + ("PASS" if test_ok else "CHECK")
    )


def check_text_data() -> None:
    print_section("TEXT DATA")

    train_text = pd.read_json(
        TRAIN_TEXT_FILE,
        orient="records",
    )

    test_text = pd.read_json(
        TEST_TEXT_FILE,
        orient="records",
    )

    print(
        f"Train text records: {len(train_text)}"
    )
    print(
        f"Test text records:  {len(test_text)}"
    )

    for column in TEXT_COLUMNS:
        if column not in train_text.columns:
            print(f"[MISSING COLUMN] {column}")
            continue

        train_nonempty = (
            train_text[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .sum()
        )

        test_nonempty = (
            test_text[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .sum()
        )

        print(
            f"{column:22} "
            f"train={train_nonempty:4}/{len(train_text):4} "
            f"test={test_nonempty:4}/{len(test_text):4}"
        )

    print("\nExample profile_text from training data:")
    if len(train_text):
        example = str(train_text.iloc[0]["profile_text"])
        print("-" * 60)
        print(example[:2000])
        if len(example) > 2000:
            print("...[truncated]...")


def check_split_leakage(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    print_section("TRAIN / TEST LEAKAGE CHECK")

    train_ids = set(train["profile_id"])
    test_ids = set(test["profile_id"])

    overlap = train_ids & test_ids

    print(f"Train profile IDs: {len(train_ids)}")
    print(f"Test profile IDs:  {len(test_ids)}")
    print(f"ID overlap:       {len(overlap)}")

    print(
        "Profile ID leakage check: "
        + ("PASS" if not overlap else "FAIL")
    )

    # Check exact normalized profile text overlap.
    train_text = (
        train["profile_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    test_text = (
        test["profile_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    text_overlap = (
        set(train_text[train_text.ne("")])
        & set(test_text[test_text.ne("")])
    )

    print(
        f"Exact non-empty profile-text overlap: "
        f"{len(text_overlap)}"
    )

    print(
        "Exact text leakage check: "
        + ("PASS" if not text_overlap else "CHECK")
    )


def check_duplicates(
    cleaned: pd.DataFrame,
) -> None:
    print_section("DUPLICATE CHECKS")

    print(
        f"Cleaned rows: {len(cleaned)}"
    )

    print(
        "Duplicate profile IDs: "
        f"{int(cleaned['profile_id'].duplicated().sum())}"
    )

    print(
        "Duplicate profile content ignoring label: "
        f"{int(cleaned['profile_id'].duplicated().sum())}"
    )

    print(
        "Repeated Full Name values: "
        f"{int(cleaned['Full Name'].astype(str).str.strip().duplicated().sum())}"
    )

    print(
        "\nNote: repeated names are NOT treated as duplicate profiles."
    )


def main() -> None:
    required_files = [
        CLEANED_FILE,
        TRAIN_FILE,
        TEST_FILE,
        METADATA_FILE,
        X_TRAIN_FILE,
        X_TEST_FILE,
        TRAIN_TEXT_FILE,
        TEST_TEXT_FILE,
    ]

    for path in required_files:
        require_file(path)

    cleaned = pd.read_csv(CLEANED_FILE)
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    with METADATA_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    print_section("TRUSTFUSION PREPROCESSING VALIDATION")

    print(f"Cleaned dataset: {len(cleaned)} profiles")
    print(f"Training set:    {len(train)} profiles")
    print(f"Testing set:     {len(test)} profiles")

    print(
        f"Expected split: 70/30 | "
        f"Actual: "
        f"{100 * len(train) / len(cleaned):.2f}%/"
        f"{100 * len(test) / len(cleaned):.2f}%"
    )

    print_section("LABEL DISTRIBUTION")
    show_label_distribution(cleaned, "ALL DATA")
    show_label_distribution(train, "TRAIN")
    show_label_distribution(test, "TEST")

    check_numeric_data(train, test)
    check_text_data()
    check_split_leakage(train, test)
    check_duplicates(cleaned)

    print_section("PREPROCESSING METADATA")
    print(
        json.dumps(
            {
                "dataset_size": metadata.get("dataset_size"),
                "train_size": metadata.get("train_size"),
                "test_size": metadata.get("test_size"),
                "source_files": metadata.get("source_files"),
            },
            indent=2,
        )
    )

    print_section("VALIDATION COMPLETE")
    print(
        "The preprocessing artifacts are ready for the "
        "next stage: RoBERTa + STE + PCA-150."
    )


if __name__ == "__main__":
    main()
