"""
TrustFusion - Visual Asset Curation & Mapping Engine.

Responsibilities:
-----------------
1. Extract base face and logo tiles from raw zip archives:
   - Real human faces from FFHQ (ffhq-dataset-master.zip)
   - AI-generated synthetic faces from StyleGAN (stylegan-master.zip)
   - Brand and institution logos from Logo-2K+ (Logo-2k-plus-Dataset-master.zip)
2. Generate photometric and resolution variations to establish non-overlapping
   Train and Test image pools (strictly preventing cross-split image leakage).
3. Assign visual assets to the 3,619 profiles using Ground-Truth Aligned Sampling:
   - Legitimate profiles (class_4 == 0) with Photo == 'Yes': Real human faces (FFHQ).
   - ChatGPT / GPT-4 synthetic profiles (class_4 in {2, 3}) with Photo == 'Yes':
     AI-generated synthetic faces (StyleGAN).
   - Manual fake profiles (class_4 == 1) with Photo == 'Yes': Realistic mix of
     StyleGAN faces, brand logos (impersonation), and cloned duplicate photos
     (Sybil rings with identical pHash).
   - Profiles with Photo == 'No': None (Cold-Start test cases).
4. Save curated image files to data/processed/visual/images/ and export
   data/processed/visual/profile_image_manifest.json.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter


RANDOM_STATE = 42


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def extract_base_crops(raw_dir: Path) -> dict[str, list[Image.Image]]:
    """Extract individual base face and logo crops from raw zip archives."""
    base_assets: dict[str, list[Image.Image]] = {
        "real_faces": [],
        "ai_faces": [],
        "logos": [],
    }

    # 1. FFHQ Real Faces (1008x512 grid -> 8 cols x 4 rows = 32 faces, 126x128 each)
    ffhq_zip = raw_dir / "ffhq-dataset-master.zip"
    if ffhq_zip.exists():
        with zipfile.ZipFile(ffhq_zip, "r") as z:
            if "ffhq-dataset-master/ffhq-teaser.png" in z.namelist():
                img_data = z.read("ffhq-dataset-master/ffhq-teaser.png")
                teaser = Image.open(io.BytesIO(img_data)).convert("RGB")
                cols, rows = 8, 4
                tw, th = teaser.width // cols, teaser.height // rows
                for r in range(rows):
                    for c in range(cols):
                        crop = teaser.crop((c * tw, r * th, (c + 1) * tw, (r + 1) * th))
                        base_assets["real_faces"].append(crop)
                logging.info("Extracted %d base real faces from FFHQ teaser.", len(base_assets["real_faces"]))

    # 2. StyleGAN AI Faces (1280x768 grid -> 5 cols x 3 rows = 15 faces, 256x256 each)
    stylegan_zip = raw_dir / "stylegan-master.zip"
    if stylegan_zip.exists():
        with zipfile.ZipFile(stylegan_zip, "r") as z:
            if "stylegan-master/stylegan-teaser.png" in z.namelist():
                img_data = z.read("stylegan-master/stylegan-teaser.png")
                teaser = Image.open(io.BytesIO(img_data)).convert("RGB")
                cols, rows = 5, 3
                tw, th = teaser.width // cols, teaser.height // rows
                for r in range(rows):
                    for c in range(cols):
                        crop = teaser.crop((c * tw, r * th, (c + 1) * tw, (r + 1) * th))
                        base_assets["ai_faces"].append(crop)
                logging.info("Extracted %d base synthetic faces from StyleGAN teaser.", len(base_assets["ai_faces"]))

    # 3. Logo-2K+ Brand Logos (793x524 grid -> extract diverse logo crops)
    logo_zip = raw_dir / "Logo-2k-plus-Dataset-master.zip"
    if logo_zip.exists():
        with zipfile.ZipFile(logo_zip, "r") as z:
            if "Logo-2k-plus-Dataset-master/logo/example.png" in z.namelist():
                img_data = z.read("Logo-2k-plus-Dataset-master/logo/example.png")
                teaser = Image.open(io.BytesIO(img_data)).convert("RGB")
                cols, rows = 5, 4
                tw, th = teaser.width // cols, teaser.height // rows
                for r in range(rows):
                    for c in range(cols):
                        crop = teaser.crop((c * tw, r * th, (c + 1) * tw, (r + 1) * th))
                        base_assets["logos"].append(crop)
                logging.info("Extracted %d base logo emblems from Logo-2K+.", len(base_assets["logos"]))

    return base_assets


def generate_augmented_variant(image: Image.Image, rng: np.random.RandomState, variant_idx: int) -> Image.Image:
    """Generate subtle realistic user photo variations (slight crop, color balance, compression)."""
    img = image.copy()

    # Subtle contrast / brightness adjustment
    contrast_factor = 0.9 + 0.2 * rng.uniform()
    brightness_factor = 0.9 + 0.2 * rng.uniform()
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)

    # Slight crop / padding (95% to 100% of original)
    w, h = img.size
    crop_pct = rng.uniform(0.94, 1.0)
    nw, nh = int(w * crop_pct), int(h * crop_pct)
    left = rng.randint(0, max(1, w - nw))
    top = rng.randint(0, max(1, h - nh))
    img = img.crop((left, top, left + nw, top + nh))

    # Standardize to 128x128 resolution for profile display
    img = img.resize((128, 128), Image.Resampling.LANCZOS)
    return img


def build_curated_pools(
    base_assets: dict[str, list[Image.Image]],
    rng: np.random.RandomState,
) -> tuple[dict[str, list[Image.Image]], dict[str, list[Image.Image]]]:
    """
    Build non-overlapping Train and Test pools of images to ensure strict anti-leakage.
    """
    train_pools: dict[str, list[Image.Image]] = {"real_faces": [], "ai_faces": [], "logos": []}
    test_pools: dict[str, list[Image.Image]] = {"real_faces": [], "ai_faces": [], "logos": []}

    for category, base_list in base_assets.items():
        if not base_list:
            continue
        n = len(base_list)
        n_train = max(1, int(n * 0.7))

        # First 70% of base assets for train, remaining 30% for test
        train_bases = base_list[:n_train]
        test_bases = base_list[n_train:]

        # Expand train pool with variants
        for i, b in enumerate(train_bases):
            for v in range(35):
                train_pools[category].append(generate_augmented_variant(b, rng, v))

        # Expand test pool with variants strictly from test bases
        for i, b in enumerate(test_bases):
            for v in range(35):
                test_pools[category].append(generate_augmented_variant(b, rng, v))

    return train_pools, test_pools


def curate_and_map_visual_assets(
    data_dir: Path = Path("data/processed"),
    raw_dir: Path = Path("data/raw"),
    output_dir: Path = Path("data/processed/visual"),
) -> dict[str, Any]:
    setup_logging()
    logging.info("Starting Visual Asset Curation & Mapping...")

    rng = np.random.RandomState(RANDOM_STATE)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    train_file = data_dir / "train.csv"
    test_file = data_dir / "test.csv"
    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError("train.csv and test.csv must exist in data/processed/. Run preprocess.py first.")

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    base_assets = extract_base_crops(raw_dir)
    train_pools, test_pools = build_curated_pools(base_assets, rng)

    manifest: dict[str, dict[str, Any]] = {}
    sybil_clone_image: Image.Image | None = None
    if train_pools["ai_faces"]:
        sybil_clone_image = train_pools["ai_faces"][0].copy()

    # Process Train Split
    train_counts = {"real_faces": 0, "ai_faces": 0, "logos": 0, "clones": 0, "cold_start": 0}
    for idx, row in train_df.iterrows():
        pid = str(row["profile_id"])
        c4 = int(row["class_4"])
        has_photo = str(row.get("Photo", "No")).strip().lower() == "yes"

        if not has_photo:
            manifest[pid] = {
                "profile_id": pid,
                "split": "train",
                "class_4": c4,
                "has_photo": False,
                "image_path": None,
                "visual_source": "none",
                "is_clone": False,
            }
            train_counts["cold_start"] += 1
            continue

        assigned_img: Image.Image
        v_source: str
        is_clone = False

        if c4 == 0:  # Legitimate -> Real human face
            pool = train_pools["real_faces"]
            assigned_img = pool[rng.randint(0, len(pool))]
            v_source = "ffhq_real"
            train_counts["real_faces"] += 1

        elif c4 in {2, 3}:  # ChatGPT / GPT-4 Synthetic -> AI-generated face
            pool = train_pools["ai_faces"]
            assigned_img = pool[rng.randint(0, len(pool))]
            v_source = "stylegan_synthetic"
            train_counts["ai_faces"] += 1

        else:  # Manual fake -> Mix of AI face, Logo, and Sybil clones
            roll = rng.uniform()
            if roll < 0.20 and sybil_clone_image is not None:  # Sybil clone photo
                assigned_img = sybil_clone_image
                v_source = "sybil_reused_photo"
                is_clone = True
                train_counts["clones"] += 1
            elif roll < 0.50 and train_pools["logos"]:  # Brand logo impersonation
                pool = train_pools["logos"]
                assigned_img = pool[rng.randint(0, len(pool))]
                v_source = "logo_impersonation"
                train_counts["logos"] += 1
            else:  # AI Face
                pool = train_pools["ai_faces"]
                assigned_img = pool[rng.randint(0, len(pool))]
                v_source = "stylegan_synthetic"
                train_counts["ai_faces"] += 1

        img_filename = f"{pid}.png"
        img_path = images_dir / img_filename
        assigned_img.save(img_path, format="PNG")

        manifest[pid] = {
            "profile_id": pid,
            "split": "train",
            "class_4": c4,
            "has_photo": True,
            "image_path": str(img_path.relative_to(data_dir.parent)),
            "visual_source": v_source,
            "is_clone": is_clone,
        }

    # Process Test Split (strictly from non-overlapping test_pools)
    test_counts = {"real_faces": 0, "ai_faces": 0, "logos": 0, "clones": 0, "cold_start": 0}
    test_sybil_clone: Image.Image | None = None
    if test_pools["ai_faces"]:
        test_sybil_clone = test_pools["ai_faces"][0].copy()

    for idx, row in test_df.iterrows():
        pid = str(row["profile_id"])
        c4 = int(row["class_4"])
        has_photo = str(row.get("Photo", "No")).strip().lower() == "yes"

        if not has_photo:
            manifest[pid] = {
                "profile_id": pid,
                "split": "test",
                "class_4": c4,
                "has_photo": False,
                "image_path": None,
                "visual_source": "none",
                "is_clone": False,
            }
            test_counts["cold_start"] += 1
            continue

        if c4 == 0:
            pool = test_pools["real_faces"]
            assigned_img = pool[rng.randint(0, len(pool))]
            v_source = "ffhq_real"
            test_counts["real_faces"] += 1
            is_clone = False

        elif c4 in {2, 3}:
            pool = test_pools["ai_faces"]
            assigned_img = pool[rng.randint(0, len(pool))]
            v_source = "stylegan_synthetic"
            test_counts["ai_faces"] += 1
            is_clone = False

        else:
            roll = rng.uniform()
            if roll < 0.20 and test_sybil_clone is not None:
                assigned_img = test_sybil_clone
                v_source = "sybil_reused_photo"
                is_clone = True
                test_counts["clones"] += 1
            elif roll < 0.50 and test_pools["logos"]:
                pool = test_pools["logos"]
                assigned_img = pool[rng.randint(0, len(pool))]
                v_source = "logo_impersonation"
                test_counts["logos"] += 1
                is_clone = False
            else:
                pool = test_pools["ai_faces"]
                assigned_img = pool[rng.randint(0, len(pool))]
                v_source = "stylegan_synthetic"
                test_counts["ai_faces"] += 1
                is_clone = False

        img_filename = f"{pid}.png"
        img_path = images_dir / img_filename
        assigned_img.save(img_path, format="PNG")

        manifest[pid] = {
            "profile_id": pid,
            "split": "test",
            "class_4": c4,
            "has_photo": True,
            "image_path": str(img_path.relative_to(data_dir.parent)),
            "visual_source": v_source,
            "is_clone": is_clone,
        }

    # Save manifest
    manifest_file = output_dir / "profile_image_manifest.json"
    with manifest_file.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    summary = {
        "total_profiles": len(manifest),
        "train_profiles": len(train_df),
        "test_profiles": len(test_df),
        "train_counts": train_counts,
        "test_counts": test_counts,
        "manifest_file": str(manifest_file),
    }

    logging.info("Visual Asset Curation Complete: %s", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TrustFusion Visual Asset Curation & Manifest Generator")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"), help="Processed data directory")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Raw archives directory")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/visual"), help="Visual output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    curate_and_map_visual_assets(
        data_dir=args.data_dir,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
    )
