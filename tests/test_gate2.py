"""
TrustFusion - Unit Tests for Gate 2: Linguistic & Conversational Forensics.

Tests Section Tag Embedding formatting, off-platforming detection,
temporal IAT & Shannon entropy calculations, PCA fitting, and neural head forward pass.
"""

from __future__ import annotations

import unittest
import numpy as np
import torch

from pipeline.gate2_linguistic import (
    OffPlatformingDetector,
    TemporalEntropyEngine,
    Gate2NeuralHead,
    build_ste_input_text,
    fit_pca_150,
    get_device,
)


class TestGate2Linguistic(unittest.TestCase):

    def test_build_ste_input_text_formatting(self):
        row = {
            "intro_text": "Alice Smith | CyberCorp",
            "about_text": "Experienced security analyst.",
            "skills_text": "Python, Graph Neural Networks",
        }
        formatted = build_ste_input_text(row)
        self.assertIn("[INTRO] Alice Smith | CyberCorp", formatted)
        self.assertIn("[ABOUT] Experienced security analyst.", formatted)
        self.assertIn("[SKILLS] Python, Graph Neural Networks", formatted)
        self.assertNotIn("[EXPERIENCE]", formatted)

    def test_build_ste_input_text_fallback(self):
        row = {"profile_text": "Generic profile content"}
        formatted = build_ste_input_text(row)
        self.assertEqual(formatted, "Generic profile content")

        empty_row = {}
        formatted_empty = build_ste_input_text(empty_row)
        self.assertEqual(formatted_empty, "[EMPTY]")

    def test_off_platforming_detector_signals(self):
        detector = OffPlatformingDetector()
        suspicious_text = (
            "URGENT hiring! Reach me directly on WhatsApp at +1-555-123-4567 "
            "or email recruit@gmail.com. Requires $50 refundable deposit."
        )
        signals = detector.extract_linguistic_signals(suspicious_text)
        self.assertGreater(signals["keyword_density"], 0.0)
        self.assertEqual(signals["email_count"], 1.0)
        self.assertGreater(signals["phone_count"], 0.0)
        self.assertGreater(signals["urgency_count"], 0.0)
        self.assertGreater(signals["off_platform_risk_score"], 0.5)

        benign_text = "Software Engineer with 5 years experience at IBM building scalable microservices."
        benign_signals = detector.extract_linguistic_signals(benign_text)
        self.assertEqual(benign_signals["email_count"], 0.0)
        self.assertEqual(benign_signals["phone_count"], 0.0)
        self.assertEqual(benign_signals["off_platform_risk_score"], 0.0)

    def test_temporal_entropy_static_profile(self):
        # Empty or < 3 timestamps should return neutral cold-start defaults
        result_empty = TemporalEntropyEngine.calculate_iat_entropy([])
        self.assertEqual(result_empty["shannon_entropy"], 0.0)
        self.assertEqual(result_empty["has_temporal_data"], 0.0)

        result_short = TemporalEntropyEngine.calculate_iat_entropy([10.0, 20.0])
        self.assertEqual(result_short["shannon_entropy"], 0.0)
        self.assertEqual(result_short["has_temporal_data"], 0.0)

    def test_temporal_entropy_periodic_bot_vs_human(self):
        # Bot: Exactly periodic intervals (delta = 5.0 seconds every time)
        bot_timestamps = [100.0 + (i * 5.0) for i in range(20)]
        bot_metrics = TemporalEntropyEngine.calculate_iat_entropy(bot_timestamps)
        self.assertEqual(bot_metrics["has_temporal_data"], 1.0)
        self.assertAlmostEqual(bot_metrics["shannon_entropy"], 0.0, places=4)

        # Human: Variable inter-arrival times
        rng = np.random.default_rng(42)
        random_deltas = rng.exponential(scale=15.0, size=30)
        human_timestamps = np.cumsum(random_deltas)
        human_metrics = TemporalEntropyEngine.calculate_iat_entropy(human_timestamps)
        self.assertEqual(human_metrics["has_temporal_data"], 1.0)
        self.assertGreater(human_metrics["shannon_entropy"], 0.0)

    def test_pca_fitting_strictly_on_train(self):
        rng = np.random.default_rng(42)
        X_train = rng.normal(loc=0.0, scale=1.0, size=(200, 768)).astype(np.float32)
        X_test = rng.normal(loc=0.5, scale=1.0, size=(50, 768)).astype(np.float32)

        X_train_pca, X_test_pca, pca, explained_var = fit_pca_150(X_train, X_test, n_components=50)

        self.assertEqual(X_train_pca.shape, (200, 50))
        self.assertEqual(X_test_pca.shape, (50, 50))
        self.assertGreater(explained_var, 0.0)
        self.assertLessEqual(explained_var, 1.0)

    def test_gate2_neural_head_forward(self):
        model = Gate2NeuralHead(input_dim=156, hidden_dim=64, num_classes=4)
        model.eval()

        batch_x = torch.randn(8, 156)
        with torch.no_grad():
            rep, logits4, logitsb = model(batch_x)

        self.assertEqual(rep.shape, (8, 64))
        self.assertEqual(logits4.shape, (8, 4))
        self.assertEqual(logitsb.shape, (8,))

    def test_device_selection(self):
        device = get_device()
        self.assertIsInstance(device, torch.device)
        self.assertIn(device.type, ["cuda", "cpu"])


if __name__ == "__main__":
    unittest.main()
