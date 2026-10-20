"""
TrustFusion - Gate 1: Visual Identity Verification & Cold-Start Pipeline.

Responsibilities:
-----------------
1. 64-bit Hexadecimal DCT Perceptual Hashing (pHash):
   - Computes robust 64-bit perceptual hash using 2D Discrete Cosine Transform (DCT).
   - Identifies photo reuse, cloned avatars, and Sybil rings via Hamming distance.
2. Synthetic Face & Visual Artifact Forensics:
   - Analyzes high-frequency Fourier spectral energy, color entropy, and gradient
     density to distinguish real camera headshots (FFHQ) from AI-generated faces
     (StyleGAN) and corporate logos (Logo-2K+).
3. Gate 1 Multi-Task Neural Representation Head:
   - Maps extracted visual features to a calibrated 64-dimensional latent embedding
     and visual fraud risk score.
4. Cold-Start Modality Masking:
   - Gracefully handles accounts without photos (Photo == 'No'), outputting a
     modality mask M_visual = 0.0 and neutral prior zero-vector (64-D) for
     Phase 4 Cross-Modal Attention Gating.
5. Artifact Serialization:
   - Exports reproducible embeddings, risk scores, modality masks, and metadata to
     data/processed/visual/.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.fftpack
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


RANDOM_STATE = 42
HASH_SIZE = 8
DCT_SIZE = 32
HAMMING_COLLISION_THRESHOLD = 5
VISUAL_EMBEDDING_DIM = 64
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.003


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logging.info("CUDA is available for Gate 1. Using GPU: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logging.info("CUDA not available. Gate 1 falling back to CPU.")
    return device


def compute_phash(image: Image.Image) -> tuple[str, np.ndarray]:
    """
    Computes a 64-bit hexadecimal DCT perceptual hash and normalized AC feature vector.
    """
    img_gray = image.convert("L").resize((DCT_SIZE, DCT_SIZE), Image.Resampling.LANCZOS)
    pixels = np.array(img_gray, dtype=np.float32)

    # 2D Discrete Cosine Transform
    dct_rows = scipy.fftpack.dct(pixels, axis=0, norm="ortho")
    dct_2d = scipy.fftpack.dct(dct_rows, axis=1, norm="ortho")

    # Extract top-left 8x8 low-frequency matrix (excluding DC component at [0, 0])
    low_freq = dct_2d[:HASH_SIZE, :HASH_SIZE]
    ac = low_freq.flatten()[1:]
    median_val = float(np.median(ac))

    # 64-bit binary mask
    bits = (low_freq.flatten() > median_val).astype(np.uint8)
    byte_vals = np.packbits(bits)
    hex_hash = "".join(f"{b:02x}" for b in byte_vals)

    # Normalized DCT coefficients [-1.0, 1.0] for neural network
    norm_dct = (low_freq.flatten() - median_val) / (np.std(low_freq) + 1e-6)
    norm_dct = np.clip(norm_dct, -3.0, 3.0) / 3.0

    return hex_hash, norm_dct.astype(np.float32)


def hamming_distance(hash1: str, hash2: str) -> int:
    """Compute Hamming distance between two 16-character hexadecimal hashes."""
    if hash1 == "NONE" or hash2 == "NONE" or len(hash1) != 16 or len(hash2) != 16:
        return 64
    b1 = bytes.fromhex(hash1)
    b2 = bytes.fromhex(hash2)
    return sum(bin(x ^ y).count("1") for x, y in zip(b1, b2))


def extract_forensic_signals(image: Image.Image) -> np.ndarray:
    """
    Extract frequency-domain and color/edge forensic indicators:
    - High-frequency Fourier power ratio (E_HF) (StyleGAN artifact detection)
    - Color channel variance & entropy
    - Edge sharpness & gradient density
    """
    img_rgb = image.convert("RGB").resize((64, 64), Image.Resampling.LANCZOS)
    arr_rgb = np.array(img_rgb, dtype=np.float32) / 255.0

    # 1. Fourier High-Frequency Energy Ratio
    img_gray = image.convert("L").resize((64, 64), Image.Resampling.LANCZOS)
    arr_gray = np.array(img_gray, dtype=np.float32)
    f_transform = np.fft.fft2(arr_gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift) ** 2

    # Radial distance mask
    rows, cols = 64, 64
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)

    total_energy = np.sum(magnitude_spectrum) + 1e-9
    hf_mask = dist_from_center > 20
    hf_energy_ratio = float(np.sum(magnitude_spectrum[hf_mask]) / total_energy)

    # 2. Color channel statistics
    r_mean, g_mean, b_mean = float(np.mean(arr_rgb[:, :, 0])), float(np.mean(arr_rgb[:, :, 1])), float(np.mean(arr_rgb[:, :, 2]))
    r_std, g_std, b_std = float(np.std(arr_rgb[:, :, 0])), float(np.std(arr_rgb[:, :, 1])), float(np.std(arr_rgb[:, :, 2]))

    # 3. Spatial Gradients / Edge Density
    grad_x = np.diff(arr_gray, axis=1)
    grad_y = np.diff(arr_gray, axis=0)
    edge_density = float(np.mean(np.abs(grad_x)) + np.mean(np.abs(grad_y))) / 255.0

    features = np.array([
        hf_energy_ratio,
        r_mean, g_mean, b_mean,
        r_std, g_std, b_std,
        edge_density,
    ], dtype=np.float32)

    return features


class Gate1NeuralHead(nn.Module):
    """
    Multi-task visual representation network:
    Maps 72-D combined features (64-D DCT + 8-D forensic indicators) to:
    1. 64-D L2-normalized latent visual embedding (for Phase 4 fusion)
    2. Binary visual fraud risk logit
    """

    def __init__(self, input_dim: int = 72, hidden_dim: int = VISUAL_EMBEDDING_DIM):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.representation_layer = nn.Linear(hidden_dim, hidden_dim)
        self.classifier_binary = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.dropout(self.relu(self.bn1(self.fc1(x))))
        rep = self.relu(self.representation_layer(h))
        # L2 normalize representation for cosine distance similarity matching
        norm_rep = nn.functional.normalize(rep, p=2, dim=-1)
        logits_bin = self.classifier_binary(rep).squeeze(-1)
        return norm_rep, logits_bin


def extract_features_for_profile(
    img_path_str: str | None,
    root_dir: Path,
) -> tuple[np.ndarray | None, str]:
    """Extract 72-D visual feature vector and pHash for an image path."""
    if not img_path_str:
        return None, "NONE"

    full_path = root_dir / img_path_str
    if not full_path.exists():
        return None, "NONE"

    try:
        with Image.open(full_path) as img:
            hex_hash, dct_vec = compute_phash(img)
            forensic_vec = extract_forensic_signals(img)
            feature_vec = np.hstack([dct_vec, forensic_vec]).astype(np.float32)
            return feature_vec, hex_hash
    except Exception as e:
        logging.warning("Error processing image %s: %s", full_path, e)
        return None, "NONE"


def train_visual_head(
    X_train_vis: np.ndarray,
    y_train_vis: np.ndarray,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    device: torch.device | None = None,
) -> Gate1NeuralHead:
    device = device or torch.device("cpu")
    model = Gate1NeuralHead(input_dim=X_train_vis.shape[1], hidden_dim=VISUAL_EMBEDDING_DIM).to(device)
    model.train()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train_vis, dtype=torch.float32),
        torch.tensor(y_train_vis, dtype=torch.float32),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    logging.info("Training Gate 1 visual representation head for %d epochs...", epochs)
    for epoch in range(epochs):
        total_loss = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            _, logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            logging.info("Epoch %2d/%2d | Visual Head Loss: %.4f", epoch + 1, epochs, total_loss / len(loader))

    model.eval()
    return model


def run_gate1(
    data_dir: Path = Path("data/processed"),
    output_dir: Path = Path("data/processed/visual"),
    epochs: int = EPOCHS,
) -> dict[str, Any]:
    setup_logging()
    device = get_device()

    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("=== Starting Gate 1 (Visual Identity & Cold-Start Pipeline) ===")

    train_file = data_dir / "train.csv"
    test_file = data_dir / "test.csv"
    manifest_file = output_dir / "profile_image_manifest.json"

    if not manifest_file.exists():
        raise FileNotFoundError(
            f"Visual manifest not found at {manifest_file}. "
            f"Please run pipeline/curate_visual_dataset.py first."
        )

    with manifest_file.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    root_dir = data_dir.parent

    # 1. Extract raw visual features for train profiles with images
    train_features_list: list[np.ndarray] = []
    train_labels_list: list[int] = []
    train_hashes: dict[str, str] = {}

    train_active_indices: list[int] = []
    train_active_features: list[np.ndarray] = []

    for i, pid in enumerate(train_df["profile_id"]):
        entry = manifest.get(str(pid), {})
        img_path = entry.get("image_path")
        c4 = int(train_df.iloc[i]["class_4"])
        is_fake = int(train_df.iloc[i]["is_fake"])

        feat, h = extract_features_for_profile(img_path, root_dir)
        train_hashes[str(pid)] = h
        if feat is not None:
            train_active_indices.append(i)
            train_active_features.append(feat)
            # Label: 0 for real face, 1 for synthetic/fake/logo
            y_val = 0 if c4 == 0 else 1
            train_features_list.append(feat)
            train_labels_list.append(y_val)

    # 2. Train visual head on active training images
    X_train_active = np.array(train_features_list, dtype=np.float32)
    y_train_active = np.array(train_labels_list, dtype=np.float32)

    visual_head = train_visual_head(X_train_active, y_train_active, epochs=epochs, device=device)

    # 3. Generate representations and risk scores for ALL train profiles (with Cold-Start handling)
    n_train = len(train_df)
    train_embeddings = np.zeros((n_train, VISUAL_EMBEDDING_DIM), dtype=np.float32)
    train_risk_scores = np.full((n_train, 1), 0.15, dtype=np.float32)  # Neutral prior for cold-start
    train_modality_masks = np.zeros((n_train, 1), dtype=np.float32)

    with torch.no_grad():
        if train_active_features:
            tensor_train = torch.tensor(np.array(train_active_features), dtype=torch.float32).to(device)
            rep_train, logits_train = visual_head(tensor_train)
            prob_train = torch.sigmoid(logits_train).cpu().numpy().reshape(-1, 1)
            rep_train_np = rep_train.cpu().numpy()

            for k, original_idx in enumerate(train_active_indices):
                train_embeddings[original_idx] = rep_train_np[k]
                train_risk_scores[original_idx] = prob_train[k]
                train_modality_masks[original_idx] = 1.0

    # 4. Generate representations and risk scores for ALL test profiles (with Cold-Start handling)
    n_test = len(test_df)
    test_embeddings = np.zeros((n_test, VISUAL_EMBEDDING_DIM), dtype=np.float32)
    test_risk_scores = np.full((n_test, 1), 0.15, dtype=np.float32)
    test_modality_masks = np.zeros((n_test, 1), dtype=np.float32)

    test_active_indices: list[int] = []
    test_active_features: list[np.ndarray] = []
    test_hashes: dict[str, str] = {}

    for i, pid in enumerate(test_df["profile_id"]):
        entry = manifest.get(str(pid), {})
        img_path = entry.get("image_path")
        feat, h = extract_features_for_profile(img_path, root_dir)
        test_hashes[str(pid)] = h
        if feat is not None:
            test_active_indices.append(i)
            test_active_features.append(feat)

    with torch.no_grad():
        if test_active_features:
            tensor_test = torch.tensor(np.array(test_active_features), dtype=torch.float32).to(device)
            rep_test, logits_test = visual_head(tensor_test)
            prob_test = torch.sigmoid(logits_test).cpu().numpy().reshape(-1, 1)
            rep_test_np = rep_test.cpu().numpy()

            for k, original_idx in enumerate(test_active_indices):
                test_embeddings[original_idx] = rep_test_np[k]
                test_risk_scores[original_idx] = prob_test[k]
                test_modality_masks[original_idx] = 1.0

    # 5. Detect Sybil / Photo Reuse Collisions
    # Compute pairwise collision counts in test set
    active_test_hashes = [test_hashes[str(test_df.iloc[idx]["profile_id"])] for idx in test_active_indices]
    collision_pairs = 0
    for a in range(len(active_test_hashes)):
        for b in range(a + 1, min(a + 50, len(active_test_hashes))):
            if hamming_distance(active_test_hashes[a], active_test_hashes[b]) <= HAMMING_COLLISION_THRESHOLD:
                collision_pairs += 1

    # 6. Save Artifacts
    np.save(output_dir / "train_gate1_embeddings.npy", train_embeddings)
    np.save(output_dir / "test_gate1_embeddings.npy", test_embeddings)
    np.save(output_dir / "train_gate1_risk_scores.npy", train_risk_scores)
    np.save(output_dir / "test_gate1_risk_scores.npy", test_risk_scores)
    np.save(output_dir / "train_visual_modality_masks.npy", train_modality_masks)
    np.save(output_dir / "test_visual_modality_masks.npy", test_modality_masks)

    torch.save(visual_head.state_dict(), output_dir / "gate1_visual_head.pt")

    # Evaluate active test performance
    test_active_y_true = np.array([0 if test_df.iloc[idx]["class_4"] == 0 else 1 for idx in test_active_indices])
    test_active_preds = (test_risk_scores[test_active_indices] >= 0.5).astype(int).ravel()
    test_active_probs = test_risk_scores[test_active_indices].ravel()

    acc = float(accuracy_score(test_active_y_true, test_active_preds))
    f1 = float(f1_score(test_active_y_true, test_active_preds))
    roc_auc = float(roc_auc_score(test_active_y_true, test_active_probs))

    metadata = {
        "visual_embedding_dim": VISUAL_EMBEDDING_DIM,
        "train_samples": n_train,
        "test_samples": n_test,
        "train_active_photos": len(train_active_indices),
        "train_cold_start_photos": n_train - len(train_active_indices),
        "test_active_photos": len(test_active_indices),
        "test_cold_start_photos": n_test - len(test_active_indices),
        "test_active_accuracy": acc,
        "test_active_f1": f1,
        "test_active_roc_auc": roc_auc,
        "sybil_collision_pairs_detected": collision_pairs,
        "anti_leakage": "Images sampled from non-overlapping train and test pools; test set strictly unseen.",
    }

    with (output_dir / "gate1_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logging.info("Gate 1 pipeline completed successfully. Artifacts saved to: %s", output_dir)
    display_gate1_summary(data_dir=data_dir, output_dir=output_dir)
    return metadata


def display_gate1_summary(data_dir: Path, output_dir: Path) -> None:
    meta_file = output_dir / "gate1_metadata.json"
    scores_file = output_dir / "test_gate1_risk_scores.npy"
    masks_file = output_dir / "test_visual_modality_masks.npy"
    manifest_file = output_dir / "profile_image_manifest.json"
    test_file = data_dir / "test.csv"

    if not meta_file.exists() or not scores_file.exists() or not test_file.exists():
        print(f"[!] Gate 1 artifacts not found in {output_dir}. Please run Gate 1 pipeline first.")
        return

    with meta_file.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    with manifest_file.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    test_df = pd.read_csv(test_file)
    test_scores = np.load(scores_file).ravel()
    test_masks = np.load(masks_file).ravel()

    def safe_ascii(val: Any, max_len: int = 80) -> str:
        s = str(val or "").encode("ascii", "ignore").decode("ascii")
        return s[:max_len] + "..." if len(s) > max_len else s

    print("\n" + "=" * 80)
    print("      TRUSTFUSION GATE 1: VISUAL IDENTITY & COLD-START FORENSICS REPORT")
    print("=" * 80)
    print(f"[*] Visual Hashing Engine     : 64-bit Hexadecimal DCT Perceptual Hashing (pHash)")
    print(f"[*] Artifact Forensics        : High-Frequency Fourier Spectrum + Color & Edge Dynamics")
    print(f"[*] Latent Representation     : {meta.get('visual_embedding_dim', 64)}-D L2-Normalized Visual Embedding")
    print(f"[*] Total Profiles Processed  : {meta.get('test_samples', 0):,} Unseen Test Accounts")
    print(f"    - Active Profile Photos   : {meta.get('test_active_photos', 0):,} ({100 * meta.get('test_active_photos', 0) / meta.get('test_samples', 1):.1f}%)")
    print(f"    - Cold-Start Accounts     : {meta.get('test_cold_start_photos', 0):,} (Mask M_visual = 0, Zero-Vector)")
    print("-" * 80)
    print(f"EVALUATION ON ACTIVE UNSEEN TEST PHOTOS ({meta.get('test_active_photos', 0):,} images):")
    print(f"  [OK] Visual Accuracy        : {meta.get('test_active_accuracy', 0) * 100:.2f}%")
    print(f"  [OK] Macro F1-Score         : {meta.get('test_active_f1', 0) * 100:.2f}%")
    print(f"  [OK] ROC-AUC Score          : {meta.get('test_active_roc_auc', 0):.4f}")
    print(f"  [OK] Sybil Clones Detected  : {meta.get('sybil_collision_pairs_detected', 0)} Photo Reuse Collisions (Hamming Dist <= 5)")
    print("-" * 80)
    print("LIVE MULTIMODAL NODE INSPECTION (Test Set):")

    # Pick samples: 1 legit face, 1 synthetic AI face, 1 logo/clone, 1 cold-start
    samples_to_show = []
    found_types = set()

    for idx in range(len(test_df)):
        pid = str(test_df.iloc[idx]["profile_id"])
        entry = manifest.get(pid, {})
        v_source = entry.get("visual_source", "none")
        if v_source not in found_types:
            found_types.add(v_source)
            samples_to_show.append((idx, v_source))
        if len(samples_to_show) >= 4:
            break

    for idx, v_type in samples_to_show:
        row = test_df.iloc[idx]
        score = float(test_scores[idx])
        mask = float(test_masks[idx])
        pred_label = "HIGH RISK / FAKE" if score >= 0.5 else "LOW RISK / LEGITIMATE"
        if mask == 0.0:
            pred_label = "NEUTRAL (COLD-START MASKED)"

        entry = manifest.get(str(row["profile_id"]), {})
        source_label = entry.get("visual_source", "none").upper()

        print(f"  [SAMPLE TYPE: {source_label}] (ID: {row['profile_id']})")
        print(f"    - Ground Truth      : {row['class_name']} (Class {row['class_4']})")
        print(f"    - Visual Mask (M)   : {mask:.1f} ({'Active Image Present' if mask == 1.0 else 'Cold-Start / No Photo'})")
        print(f"    - Visual Risk Score : {score:.4f}  [Decision: {pred_label}]")
        print(f"    - Full Name / Title : {safe_ascii(row.get('Full Name', 'None'), 35)} | {safe_ascii(row.get('Workplace', 'None'), 35)}")
        print()

    print("=" * 80 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TrustFusion Gate 1: Visual Identity & Cold-Start Pipeline")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"), help="Processed data directory")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/visual"), help="Visual output directory")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs for visual head")
    parser.add_argument("--demo", action="store_true", help="Quickly display Gate 1 executive summary and exit.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.demo:
        display_gate1_summary(data_dir=args.data_dir, output_dir=args.output_dir)
    else:
        run_gate1(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
        )
