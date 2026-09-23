
"""
TrustFusion - Phase 1 & Phase 2 preprocessing.

Responsibilities
-----------------
1. Load the real LinkedIn/Ayoobi-Gulati dataset and optional GPT-4 adversarial CSV.
2. Validate the 39-column schema.
3. Normalize numeric fields and structured dictionary/list fields.
4. Build clean section-level text representations.
5. Preserve original labels and derive 4-class + binary labels.
6. Perform duplicate checks.
7. Perform a stratified 70/30 train-test split.
8. Fit MinMaxScaler ONLY on the training set and transform train/test.
9. Save reproducible processed artifacts.

"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


RANDOM_STATE = 42
TEST_SIZE = 0.30

EXPECTED_COLUMNS = [
    "Intro", "Full Name", "Workplace", "Location", "Connections", "Photo",
    "Followers", "About", "Experiences", "Number of Experiences",
    "Educations", "Number of Educations", "Licenses", "Number of Licenses",
    "Volunteering", "Number of Volunteering", "Skills", "Number of Skills",
    "Recommendations", "Number of Recommendations", "Projects",
    "Number of Projects", "Publications", "Number of Publications", "Courses",
    "Number of Courses", "Honors", "Number of Honors", "Scores",
    "Number of Scores", "Languages", "Number of Languages", "Organizations",
    "Number of Organizations", "Interests", "Number of Interests",
    "Activities", "Number of Activities", "Label",
]

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

STRUCTURED_COLUMNS = [
    "Experiences",
    "Educations",
    "Licenses",
    "Volunteering",
    "Skills",
    "Recommendations",
    "Projects",
    "Publications",
    "Courses",
    "Honors",
    "Scores",
    "Languages",
    "Organizations",
    "Interests",
    "Activities",
]

SECTION_COLUMNS = [
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

LABEL_MAP = {
    0: {"class_4": 0, "class_name": "legitimate", "is_fake": 0},
    1: {"class_4": 1, "class_name": "manual_fake", "is_fake": 1},
    10: {"class_4": 2, "class_name": "chatgpt_fake", "is_fake": 1},
    11: {"class_4": 2, "class_name": "chatgpt_fake", "is_fake": 1},
    12: {"class_4": 3, "class_name": "gpt4_adversarial", "is_fake": 1},
}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def validate_schema(df: pd.DataFrame, source_name: str) -> None:
    actual = set(df.columns)
    expected = set(EXPECTED_COLUMNS)

    missing = sorted(expected - actual)
    extra = sorted(actual - expected)

    if missing:
        raise ValueError(
            f"{source_name}: missing required columns: {missing}"
        )

    if extra:
        logging.warning(
            "%s: extra columns will be preserved: %s",
            source_name,
            extra,
        )


def load_pickle(path: Path) -> pd.DataFrame:
    logging.info("Loading pickle dataset: %s", path)
    with path.open("rb") as f:
        obj = pickle.load(f)

    if not isinstance(obj, pd.DataFrame):
        raise TypeError(
            f"{path} contains {type(obj).__name__}, "
            "expected pandas.DataFrame."
        )

    return normalize_column_names(obj)


def load_csv(path: Path) -> pd.DataFrame:
    logging.info("Loading CSV dataset: %s", path)
    return normalize_column_names(pd.read_csv(path))


def parse_structured_value(value: Any) -> Any:
    """Convert serialized dict/list strings back to Python objects when possible."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}

    if isinstance(value, (dict, list, tuple)):
        return value

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return {}

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (dict, list, tuple)):
            return parsed
        return parsed
    except (ValueError, SyntaxError):
        return text


def clean_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def flatten_to_text(value: Any, prefix: str = "") -> str:
    """
    Convert nested profile structures into deterministic readable text.

    URLs and dictionary keys are retained because they can later be useful
    to the relational gate; no identity/detection decision is made here.
    """
    value = parse_structured_value(value)

    parts: list[str] = []

    if isinstance(value, dict):
        for key, item in value.items():
            key_text = clean_scalar(key)
            child = flatten_to_text(item, prefix=key_text)
            if child:
                if prefix:
                    parts.append(f"{prefix}: {key_text}: {child}")
                else:
                    parts.append(f"{key_text}: {child}")

    elif isinstance(value, (list, tuple)):
        for item in value:
            child = flatten_to_text(item, prefix=prefix)
            if child:
                parts.append(child)

    else:
        scalar = clean_scalar(value)
        if scalar:
            parts.append(scalar)

    return " | ".join(parts)


def extract_intro_text(value: Any) -> str:
    """
    Intro is normally a small dict containing the six profile header fields.
    Keep it as a clean section rather than treating the dict representation
    as natural language.
    """
    parsed = parse_structured_value(value)

    if isinstance(parsed, dict):
        ordered_keys = [
            "Full Name",
            "Workplace",
            "Location",
            "Connections",
            "Photo",
            "Followers",
        ]
        chunks = []
        for key in ordered_keys:
            if key in parsed:
                val = clean_scalar(parsed[key])
                if val:
                    chunks.append(f"{key}: {val}")
        return " | ".join(chunks)

    return clean_scalar(parsed)


def build_section_texts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["intro_text"] = out["Intro"].map(extract_intro_text)
    out["about_text"] = out["About"].map(clean_scalar)
    out["experience_text"] = out["Experiences"].map(
        lambda x: flatten_to_text(x, "Experience")
    )
    out["education_text"] = out["Educations"].map(
        lambda x: flatten_to_text(x, "Education")
    )
    out["licenses_text"] = out["Licenses"].map(
        lambda x: flatten_to_text(x, "License")
    )
    out["volunteering_text"] = out["Volunteering"].map(
        lambda x: flatten_to_text(x, "Volunteering")
    )
    out["skills_text"] = out["Skills"].map(
        lambda x: flatten_to_text(x, "Skill")
    )
    out["recommendations_text"] = out["Recommendations"].map(
        lambda x: flatten_to_text(x, "Recommendation")
    )
    out["projects_text"] = out["Projects"].map(
        lambda x: flatten_to_text(x, "Project")
    )
    out["publications_text"] = out["Publications"].map(
        lambda x: flatten_to_text(x, "Publication")
    )
    out["courses_text"] = out["Courses"].map(
        lambda x: flatten_to_text(x, "Course")
    )
    out["honors_text"] = out["Honors"].map(
        lambda x: flatten_to_text(x, "Honor")
    )
    out["scores_text"] = out["Scores"].map(
        lambda x: flatten_to_text(x, "Score")
    )
    out["languages_text"] = out["Languages"].map(
        lambda x: flatten_to_text(x, "Language")
    )
    out["organizations_text"] = out["Organizations"].map(
        lambda x: flatten_to_text(x, "Organization")
    )
    out["interests_text"] = out["Interests"].map(
        lambda x: flatten_to_text(x, "Interest")
    )
    out["activities_text"] = out["Activities"].map(
        lambda x: flatten_to_text(x, "Activity")
    )

    profile_sections = [
        ("Introduction", "intro_text"),
        ("About", "about_text"),
        ("Experiences", "experience_text"),
        ("Education", "education_text"),
        ("Licenses", "licenses_text"),
        ("Volunteering", "volunteering_text"),
        ("Skills", "skills_text"),
        ("Recommendations", "recommendations_text"),
        ("Projects", "projects_text"),
        ("Publications", "publications_text"),
        ("Courses", "courses_text"),
        ("Honors", "honors_text"),
        ("Scores", "scores_text"),
        ("Languages", "languages_text"),
        ("Organizations", "organizations_text"),
        ("Interests", "interests_text"),
        ("Activities", "activities_text"),
    ]

    def join_profile_sections(row: pd.Series) -> str:
        chunks = []
        for section_name, column in profile_sections:
            value = clean_scalar(row[column])
            if value:
                chunks.append(f"{section_name}: {value}")
        return "\n".join(chunks)

    out["profile_text"] = out.apply(join_profile_sections, axis=1)

    return out


def parse_count(value: Any) -> float:
    """Parse values such as 500+, '1,340', 1.2K, 3.4M into numeric values."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip().replace(",", "")
    if not text:
        return np.nan

    multiplier = 1.0
    suffix = text[-1:].upper()

    if suffix == "K":
        multiplier = 1_000.0
        text = text[:-1]
    elif suffix == "M":
        multiplier = 1_000_000.0
        text = text[:-1]

    text = text.rstrip("+").strip()

    try:
        return float(text) * multiplier
    except ValueError:
        return np.nan


def clean_numerical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for column in NUMERICAL_FEATURES:
        out[column] = out[column].map(parse_count)

    for column in NUMERICAL_FEATURES:
        negative_count = int((out[column] < 0).sum())
        if negative_count:
            logging.warning(
                "%s contains %d negative values; replacing them with NaN.",
                column,
                negative_count,
            )
            out.loc[out[column] < 0, column] = np.nan

    return out


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["original_label"] = pd.to_numeric(
        out["Label"], errors="coerce"
    )

    unknown = sorted(
        set(out["original_label"].dropna().astype(int))
        - set(LABEL_MAP)
    )
    if unknown:
        raise ValueError(f"Unknown dataset labels found: {unknown}")

    if out["original_label"].isna().any():
        raise ValueError("Found missing/non-numeric labels.")

    out["original_label"] = out["original_label"].astype(int)

    out["class_4"] = out["original_label"].map(
        lambda x: LABEL_MAP[x]["class_4"]
    ).astype(int)

    out["class_name"] = out["original_label"].map(
        lambda x: LABEL_MAP[x]["class_name"]
    )

    out["is_fake"] = out["original_label"].map(
        lambda x: LABEL_MAP[x]["is_fake"]
    ).astype(int)

    return out


def stable_row_fingerprint(
    row: pd.Series,
    columns: list[str] | None = None,
) -> str:
    """Create a deterministic hash for rows containing nested dict/list values."""
    if columns is None:
        columns = list(row.index)

    normalized = {}

    for column in columns:
        value = row.get(column)

        if isinstance(value, dict):
            normalized[column] = value
        elif isinstance(value, (list, tuple)):
            normalized[column] = list(value)
        elif pd.isna(value):
            normalized[column] = None
        else:
            normalized[column] = value

    payload = json.dumps(
        normalized,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def create_profile_id(row: pd.Series) -> str:
    """
    Create a deterministic internal ID from the complete profile content,
    excluding Label so the same profile cannot receive different IDs merely
    because its label differs.
    """
    profile_columns = [
        column for column in EXPECTED_COLUMNS
        if column != "Label"
    ]

    digest = stable_row_fingerprint(
        row,
        columns=profile_columns,
    )[:16]

    return f"TF_{digest}"


def add_profile_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["profile_id"] = out.apply(create_profile_id, axis=1)

    duplicate_ids = int(
        out["profile_id"].duplicated(keep=False).sum()
    )

    if duplicate_ids:
        logging.warning(
            "%d rows share a deterministic profile_id. "
            "Inspect these before the final experimental run.",
            duplicate_ids,
        )

    return out


def duplicate_report(df: pd.DataFrame) -> dict[str, int]:
    all_columns = [
        column for column in EXPECTED_COLUMNS
        if column in df.columns
    ]

    profile_columns = [
        column for column in all_columns
        if column != "Label"
    ]

    all_fingerprints = df.apply(
        lambda row: stable_row_fingerprint(
            row,
            columns=all_columns,
        ),
        axis=1,
    )

    profile_fingerprints = df.apply(
        lambda row: stable_row_fingerprint(
            row,
            columns=profile_columns,
        ),
        axis=1,
    )

    report = {
        "exact_duplicate_rows": int(
            all_fingerprints.duplicated().sum()
        ),
        "duplicate_profile_ids": int(
            df["profile_id"].duplicated().sum()
        ),
        "duplicate_full_name": int(
            df["Full Name"]
            .astype(str)
            .str.strip()
            .duplicated()
            .sum()
        ),
        "duplicate_profile_content_ignoring_label": int(
            profile_fingerprints.duplicated().sum()
        ),
    }

    return report


def stratified_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    70/30 stratified split matching the current project architecture.

    Important: this is a row-level split. Before the final paper run,
    source-profile linkage between generated and original profiles should
    be investigated to prevent near-duplicate leakage.
    """
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=df["class_4"],
    )

    train = train.sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    test = test.sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    return train, test


def fit_minmax_and_save(
    train: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:

    scaler = MinMaxScaler()

    X_train = train[NUMERICAL_FEATURES].copy()
    X_test = test[NUMERICAL_FEATURES].copy()

    # Fit imputation values on TRAIN ONLY.
    train_medians = X_train.median()

    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    scaler.fit(X_train)

    X_train_scaled = scaler.transform(
        X_train
    ).astype(np.float32)

    X_test_scaled = scaler.transform(
        X_test
    ).astype(np.float32)

    numerical_dir = output_dir / "numerical"
    numerical_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        numerical_dir / "X_train.npy",
        X_train_scaled,
    )

    np.save(
        numerical_dir / "X_test.npy",
        X_test_scaled,
    )

    with (
        numerical_dir / "minmax_scaler.pkl"
    ).open("wb") as f:
        pickle.dump(scaler, f)

    with (
        numerical_dir / "train_medians.pkl"
    ).open("wb") as f:
        pickle.dump(
            train_medians.to_dict(),
            f,
        )

    return {
        "train_shape": list(
            X_train_scaled.shape
        ),
        "test_shape": list(
            X_test_scaled.shape
        ),
        "feature_names": NUMERICAL_FEATURES,
    }


def save_text_arrays(
    train: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path,
) -> None:

    text_dir = output_dir / "text"
    text_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_text = train[
        ["profile_id"] + SECTION_COLUMNS
    ].copy()

    test_text = test[
        ["profile_id"] + SECTION_COLUMNS
    ].copy()

    train_text.to_json(
        text_dir / "train_text.json",
        orient="records",
        force_ascii=False,
        indent=2,
    )

    test_text.to_json(
        text_dir / "test_text.json",
        orient="records",
        force_ascii=False,
        indent=2,
    )


def save_metadata(
    output_dir: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    source_files: list[str],
    numerical_info: dict[str, Any],
    duplicate_info: dict[str, int],
) -> None:

    combined = pd.concat(
        [train, test],
        ignore_index=True,
    )

    metadata = {
        "project": "TrustFusion",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "split_strategy": "stratified row-level split",
        "source_files": source_files,
        "dataset_size": int(len(combined)),
        "train_size": int(len(train)),
        "test_size": int(len(test)),
        "label_counts_all": {
            str(k): int(v)
            for k, v in combined["original_label"]
            .value_counts()
            .sort_index()
            .items()
        },
        "class_4_counts_all": {
            str(k): int(v)
            for k, v in combined["class_4"]
            .value_counts()
            .sort_index()
            .items()
        },
        "binary_counts_all": {
            str(k): int(v)
            for k, v in combined["is_fake"]
            .value_counts()
            .sort_index()
            .items()
        },
        "numerical_features": numerical_info,
        "duplicate_report": duplicate_info,
        "notes": [
            "RoBERTa/STE/PCA is handled by the linguistic gate.",
            "Visual pHash/Milvus is handled by Gate 1.",
            "RGCN/Louvain is handled by Gate 3.",
            "GPT-4 adversarial rows are only those physically present in the source file.",
        ],
    }

    with (
        output_dir / "preprocessing_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )


def select_input_files(
    data_dir: Path,
) -> tuple[Path, Path | None]:

    pcl_candidates = sorted(
        data_dir.rglob("LinkedIn_Dataset.pcl")
    )

    csv_candidates = sorted(
        data_dir.rglob("generated_profiles_NEW.csv")
    )

    if not pcl_candidates:
        raise FileNotFoundError(
            f"LinkedIn_Dataset.pcl was not found under "
            f"{data_dir}. Place the real dataset in data/raw/."
        )

    pcl_path = pcl_candidates[0]
    csv_path = (
        csv_candidates[0]
        if csv_candidates
        else None
    )

    return pcl_path, csv_path


def load_and_merge(
    data_dir: Path,
) -> tuple[pd.DataFrame, list[str]]:

    pcl_path, csv_path = select_input_files(
        data_dir
    )

    pcl_df = load_pickle(pcl_path)
    validate_schema(
        pcl_df,
        pcl_path.name,
    )

    frames = [pcl_df]
    source_files = [str(pcl_path)]

    if csv_path is not None:
        csv_df = load_csv(csv_path)
        validate_schema(
            csv_df,
            csv_path.name,
        )

        frames.append(csv_df)
        source_files.append(str(csv_path))
    else:
        logging.warning(
            "generated_profiles_NEW.csv was not found. "
            "Continuing with the LinkedIn dataset only."
        )

    merged = pd.concat(
        frames,
        ignore_index=True,
    )

    merged = normalize_column_names(
        merged
    )

    # Pandas cannot hash the nested dict/list cells in the raw dataset.
    # Build deterministic fingerprints and remove only truly identical rows.
    before = len(merged)

    fingerprints = merged.apply(
        lambda row: stable_row_fingerprint(row),
        axis=1,
    )

    keep_mask = ~fingerprints.duplicated(keep="first")
    merged = merged.loc[
        keep_mask
    ].reset_index(drop=True)

    removed = before - len(merged)

    if removed:
        logging.info(
            "Removed %d exact duplicate rows.",
            removed,
        )

    return merged, source_files


def run(
    data_dir: Path,
    output_dir: Path,
) -> None:

    setup_logging()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logging.info(
        "=== TrustFusion preprocessing started ==="
    )

    df, source_files = load_and_merge(
        data_dir
    )

    logging.info(
        "Merged dataset size: %d",
        len(df),
    )

    df = clean_numerical_features(df)
    df = build_section_texts(df)
    df = normalize_labels(df)
    df = add_profile_ids(df)

    duplicates = duplicate_report(df)

    logging.info(
        "Duplicate report: %s",
        duplicates,
    )

    cleaned_path = (
        output_dir /
        "cleaned_profiles.csv"
    )

    df.to_csv(
        cleaned_path,
        index=False,
        encoding="utf-8",
    )

    logging.info(
        "Saved cleaned profiles: %s",
        cleaned_path,
    )

    train, test = stratified_split(df)

    train.to_csv(
        output_dir / "train.csv",
        index=False,
        encoding="utf-8",
    )

    test.to_csv(
        output_dir / "test.csv",
        index=False,
        encoding="utf-8",
    )

    logging.info(
        "Train/test split: %d / %d (%.0f%% / %.0f%%)",
        len(train),
        len(test),
        100 * len(train) / len(df),
        100 * len(test) / len(df),
    )

    numerical_info = fit_minmax_and_save(
        train,
        test,
        output_dir,
    )

    save_text_arrays(
        train,
        test,
        output_dir,
    )

    save_metadata(
        output_dir=output_dir,
        train=train,
        test=test,
        source_files=source_files,
        numerical_info=numerical_info,
        duplicate_info=duplicates,
    )

    logging.info(
        "=== TrustFusion preprocessing completed ==="
    )


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "TrustFusion Phase 1/2 "
            "dataset preprocessing"
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help=(
            "Directory containing the real "
            "dataset files."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help=(
            "Directory for processed "
            "outputs."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.data_dir,
        args.output_dir,
    )
