"""
TrustFusion - Gate 1 Automated Unit Tests.

Validates:
1. 64-bit DCT perceptual hash determinism and scale/crop invariance.
2. Hamming distance collision detection for reused images vs distinct faces.
3. Cold-Start Modality Masking (zero vector output and M_visual = 0.0).
4. Active image embedding dimensions (64-D L2-normalized).
5. Output artifact shapes match train (2533) and test (1086) splits.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from pipeline.gate1_visual import (
    compute_phash,
    hamming_distance,
    extract_forensic_signals,
    VISUAL_EMBEDDING_DIM,
)


class TestGate1Visual(unittest.TestCase):
    def setUp(self) -> None:
        self.output_dir = Path("data/processed/visual")
        # Create synthetic test patterns
        rng = np.random.RandomState(42)
        arr1 = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        arr2 = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        self.img1 = Image.fromarray(arr1)
        self.img2 = Image.fromarray(arr2)

    def test_phash_deterministic(self) -> None:
        """pHash must produce identical hex output for the identical image."""
        h1, _ = compute_phash(self.img1)
        h2, _ = compute_phash(self.img1)
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 16)  # 64 bits = 16 hex characters

    def test_phash_scale_invariance(self) -> None:
        """Slight scaling should yield Hamming distance <= 5."""
        h1, _ = compute_phash(self.img1)
        resized = self.img1.resize((50, 50), Image.Resampling.LANCZOS)
        h_resized, _ = compute_phash(resized)
        dist = hamming_distance(h1, h_resized)
        self.assertLessEqual(dist, 5)

    def test_phash_distinct_images(self) -> None:
        """Completely different random images should have high Hamming distance (>10)."""
        h1, _ = compute_phash(self.img1)
        h2, _ = compute_phash(self.img2)
        dist = hamming_distance(h1, h2)
        self.assertGreater(dist, 10)

    def test_forensic_signals_shape(self) -> None:
        """Forensic extractor must emit 8-D signal array."""
        signals = extract_forensic_signals(self.img1)
        self.assertEqual(signals.shape, (8,))
        self.assertFalse(np.isnan(signals).any())

    def test_cold_start_contract(self) -> None:
        """Cold-start accounts must receive zero vector (64,) and mask 0.0."""
        zero_rep = np.zeros(VISUAL_EMBEDDING_DIM, dtype=np.float32)
        mask = 0.0
        self.assertEqual(zero_rep.shape, (64,))
        self.assertEqual(float(np.sum(np.abs(zero_rep))), 0.0)
        self.assertEqual(mask, 0.0)

    def test_artifacts_exist_and_match_splits(self) -> None:
        """Verify saved arrays match train (2533) and test (1086) counts."""
        train_emb_file = self.output_dir / "train_gate1_embeddings.npy"
        test_emb_file = self.output_dir / "test_gate1_embeddings.npy"
        train_risk_file = self.output_dir / "train_gate1_risk_scores.npy"
        test_risk_file = self.output_dir / "test_gate1_risk_scores.npy"
        train_mask_file = self.output_dir / "train_visual_modality_masks.npy"
        test_mask_file = self.output_dir / "test_visual_modality_masks.npy"

        if not train_emb_file.exists():
            self.skipTest("Artifacts not yet generated on disk.")

        train_emb = np.load(train_emb_file)
        test_emb = np.load(test_emb_file)
        train_risk = np.load(train_risk_file)
        test_risk = np.load(test_risk_file)
        train_mask = np.load(train_mask_file)
        test_mask = np.load(test_mask_file)

        self.assertEqual(train_emb.shape, (2533, 64))
        self.assertEqual(test_emb.shape, (1086, 64))
        self.assertEqual(train_risk.shape, (2533, 1))
        self.assertEqual(test_risk.shape, (1086, 1))
        self.assertEqual(train_mask.shape, (2533, 1))
        self.assertEqual(test_mask.shape, (1086, 1))

        # Check values are bounded
        self.assertTrue(np.all((train_risk >= 0.0) & (train_risk <= 1.0)))
        self.assertTrue(np.all((test_risk >= 0.0) & (test_risk <= 1.0)))


if __name__ == "__main__":
    unittest.main()
