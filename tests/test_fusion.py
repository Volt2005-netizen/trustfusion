"""
TrustFusion - Unit Tests for Step 5: Cross-Modal Attention Gating & Decision Ensemble.

Tests the PyTorch dynamic attention gating module, cold-start modality masking,
multimodal feature vector construction (92-D), and ensemble inference bounds.
"""

from __future__ import annotations

import unittest
from pathlib import Path
import numpy as np
import torch

from pipeline.fusion_classifier import (
    CrossModalAttentionGating,
    construct_multimodal_features,
    get_device,
)


class TestFusionClassifier(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.gating = CrossModalAttentionGating(emb_dim=64).to(self.device)
        self.gating.eval()

    def test_gating_forward_shapes(self):
        batch_size = 16
        h_vis = torch.randn(batch_size, 64)
        h_ling = torch.randn(batch_size, 64)
        h_rel = torch.randn(batch_size, 64)
        mask_vis = torch.ones(batch_size, 1)

        with torch.no_grad():
            h_fused, alphas = self.gating(h_vis, h_ling, h_rel, mask_vis)

        self.assertEqual(h_fused.shape, (batch_size, 64))
        self.assertEqual(alphas.shape, (batch_size, 3))
        # Ensure attention weights sum to 1.0 across modalities
        alpha_sums = alphas.sum(dim=1)
        torch.testing.assert_close(alpha_sums, torch.ones(batch_size), atol=1e-5, rtol=1e-5)

    def test_cold_start_masking_enforcement(self):
        """
        Verify that when visual mask is 0.0 (cold-start / no photo),
        the visual attention weight is strictly 0.0, and attention dynamically
        normalizes between linguistic and relational gates.
        """
        batch_size = 8
        h_vis = torch.randn(batch_size, 64)
        h_ling = torch.randn(batch_size, 64)
        h_rel = torch.randn(batch_size, 64)
        # Force all masks to 0.0 (no photo)
        mask_vis = torch.zeros(batch_size, 1)

        with torch.no_grad():
            h_fused, alphas = self.gating(h_vis, h_ling, h_rel, mask_vis)

        alpha_vis = alphas[:, 0]
        alpha_ling = alphas[:, 1]
        alpha_rel = alphas[:, 2]

        # Visual alpha must be essentially 0.0
        self.assertTrue(torch.all(alpha_vis < 1e-6))
        # Remaining alphas must sum to 1.0
        torch.testing.assert_close(alpha_ling + alpha_rel, torch.ones(batch_size), atol=1e-5, rtol=1e-5)

    def test_multimodal_feature_construction_shape(self):
        n_samples = 10
        h_vis = np.random.randn(n_samples, 64).astype(np.float32)
        h_ling = np.random.randn(n_samples, 64).astype(np.float32)
        h_rel = np.random.randn(n_samples, 64).astype(np.float32)
        mask_vis = np.ones((n_samples, 1), dtype=np.float32)
        risk_vis = np.random.uniform(0, 1, n_samples).astype(np.float32)
        risk_ling = np.random.uniform(0, 1, n_samples).astype(np.float32)
        risk_rel = np.random.uniform(0, 1, n_samples).astype(np.float32)
        comm_feats = np.random.randn(n_samples, 4).astype(np.float32)
        num_feats = np.random.randn(n_samples, 17).astype(np.float32)

        X_fused, X_dec, alphas = construct_multimodal_features(
            h_vis=h_vis,
            h_ling=h_ling,
            h_rel=h_rel,
            mask_vis=mask_vis,
            risk_vis=risk_vis,
            risk_ling=risk_ling,
            risk_rel=risk_rel,
            comm_feats=comm_feats,
            num_feats=num_feats,
            gating_module=self.gating,
            device=self.device,
        )

        # Expected multimodal shape: (n_samples, 92)
        # 64 (fused latent) + 3 (alphas) + 3 (risks) + 1 (mask) + 4 (comm) + 17 (num) = 92
        self.assertEqual(X_fused.shape, (n_samples, 92))
        # Expected decision shape: (n_samples, 28)
        # 3 (risks) + 1 (mask) + 3 (alphas) + 4 (comm) + 17 (num) = 28
        self.assertEqual(X_dec.shape, (n_samples, 28))
        self.assertEqual(alphas.shape, (n_samples, 3))
        self.assertTrue(np.all(np.isfinite(X_fused)))
        self.assertTrue(np.all(np.isfinite(X_dec)))

    def test_device_selection(self):
        dev = get_device()
        self.assertIsInstance(dev, torch.device)
        self.assertIn(dev.type, ["cuda", "cpu"])


if __name__ == "__main__":
    unittest.main()
