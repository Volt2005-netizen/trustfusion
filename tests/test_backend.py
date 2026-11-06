"""
TrustFusion - Unit & Integration Tests for Step 7 FastAPI Serving Engine.

Tests:
1. GET  /health (System, GPU, and Model readiness)
2. GET  /api/v1/profiles (Exemplar profiles and test set retrieval)
3. GET  /api/v1/benchmarks (Ablation comparisons and metrics)
4. POST /api/v1/predict (Legitimate profile prediction)
5. POST /api/v1/predict (Fake recruitment bot prediction)
6. POST /api/v1/predict (Cold-start dynamic visual masking alpha_vis == 0.0)
7. POST /api/v1/explain (SHAP feature attribution and GNN paths)
8. POST /api/v1/explain (404 on invalid profile)
9. GET  /dashboard (Interactive HTML dashboard)
10. GET / (Redirect to dashboard)
"""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from fastapi.testclient import TestClient
from backend.main import app


class TestBackendServingAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Using TestClient within lifespan context
        cls.client_cm = TestClient(app)
        cls.client = cls.client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_cm.__exit__(None, None, None)

    def test_01_health_check(self):
        """Verify GET /health reports healthy status, GPU, and all models loaded."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["version"], "1.0.0")
        self.assertIn("gpu_name", data)
        self.assertIn("models_loaded", data)
        self.assertTrue(data["models_loaded"]["catboost_binary"])
        self.assertTrue(data["models_loaded"]["xgboost_binary"])
        self.assertTrue(data["models_loaded"]["attention_gating"])
        self.assertTrue(data["models_loaded"]["shap_tree_explainer"])
        self.assertEqual(data["test_profiles_cached"], 1086)
        self.assertGreater(data["benchmark_accuracy"], 0.98)
        self.assertGreater(data["benchmark_roc_auc"], 0.99)

    def test_02_list_profiles_and_exemplars(self):
        """Verify GET /api/v1/profiles returns the 4 exemplar case studies."""
        response = self.client.get("/api/v1/profiles?limit=5")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("exemplar_case_studies", data)
        self.assertEqual(len(data["exemplar_case_studies"]), 4)
        exemplar_ids = [ex["profile_id"] for ex in data["exemplar_case_studies"]]
        self.assertIn("TF_ac29c8fbb6535468", exemplar_ids)
        self.assertIn("TF_ccd74509dfc347e4", exemplar_ids)
        self.assertEqual(data["total_test_profiles"], 1086)
        self.assertEqual(data["results_returned"], 5)

    def test_03_get_benchmarks(self):
        """Verify GET /api/v1/benchmarks returns gate ablation and SHAP features."""
        response = self.client.get("/api/v1/benchmarks")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("gate_performance_matrix", data)
        matrix = data["gate_performance_matrix"]
        self.assertIn("gate1_visual", matrix)
        self.assertIn("gate2_linguistic", matrix)
        self.assertIn("gate3_relational", matrix)
        self.assertIn("trustfusion_fusion", matrix)
        self.assertGreater(matrix["trustfusion_fusion"]["accuracy"], 0.98)
        self.assertIn("top_10_global_features", data)
        self.assertGreaterEqual(len(data["top_10_global_features"]), 10)

    def test_04_predict_legitimate_profile(self):
        """Verify POST /api/v1/predict correctly identifies authentic executive profile."""
        payload = {"profile_id": "TF_ac29c8fbb6535468"}
        response = self.client.post("/api/v1/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["profile_id"], "TF_ac29c8fbb6535468")
        self.assertFalse(data["is_fake"])
        self.assertEqual(data["triage_verdict"], "SAFE")
        self.assertLess(data["calibrated_risk_score"], 0.15)
        self.assertEqual(data["threat_attribution"]["class_label"], "Legitimate")
        self.assertGreater(data["latency_ms"], 0.0)

    def test_05_predict_fake_recruiter_bot(self):
        """Verify POST /api/v1/predict catches ChatGPT recruitment scam bot."""
        payload = {"profile_id": "TF_ccd74509dfc347e4"}
        response = self.client.post("/api/v1/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["profile_id"], "TF_ccd74509dfc347e4")
        self.assertTrue(data["is_fake"])
        self.assertEqual(data["triage_verdict"], "HIGH_RISK")
        self.assertGreater(data["calibrated_risk_score"], 0.90)
        self.assertEqual(data["threat_attribution"]["class_label"], "ChatGPT Bot")

    def test_06_predict_cold_start_dynamic_masking(self):
        """Verify simulate_cold_start=True zeroes visual attention weight (alpha_vis == 0.0)."""
        payload = {
            "profile_id": "TF_ac29c8fbb6535468",
            "simulate_cold_start": True,
        }
        response = self.client.post("/api/v1/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        gb = data["gate_breakdown"]
        self.assertTrue(gb["is_cold_start"])
        self.assertFalse(gb["visual_photo_present"])
        self.assertEqual(gb["alpha_vis"], 0.0)
        # alpha_ling + alpha_rel should equal 1.0
        self.assertAlmostEqual(gb["alpha_ling"] + gb["alpha_rel"], 1.0, places=3)
        self.assertIn("Cold-Start", data["cold_start_policy_applied"])

    def test_07_explain_profile(self):
        """Verify POST /api/v1/explain returns SHAP drivers, triggers, and GNN paths."""
        payload = {
            "profile_id": "TF_ccd74509dfc347e4",
            "top_k": 5,
        }
        response = self.client.post("/api/v1/explain", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["profile_id"], "TF_ccd74509dfc347e4")
        self.assertIn("top_risk_drivers", data)
        self.assertIn("top_protective_factors", data)
        self.assertIn("linguistic_triggers", data)
        self.assertIn("gnn_subgraph", data)
        self.assertGreaterEqual(len(data["top_risk_drivers"]), 1)

    def test_08_explain_invalid_profile_returns_404(self):
        """Verify POST /api/v1/explain returns 404 for unknown profile ID."""
        payload = {"profile_id": "TF_NON_EXISTENT_PROFILE_99999"}
        response = self.client.post("/api/v1/explain", json=payload)
        self.assertEqual(response.status_code, 404)

    def test_09_dashboard_html_view(self):
        """Verify GET /dashboard returns responsive HTML dashboard."""
        response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        text = response.text
        self.assertIn("TrustFusion", text)
        self.assertIn("SOC LIVE", text)
        self.assertIn("Forensic Triage Selector", text)
        self.assertIn("Cross-Modal Attention Gating", text)
        self.assertIn("SHAP Feature Attribution", text)
        self.assertIn("Relational Graph", text)

    def test_10_root_redirects_to_dashboard(self):
        """Verify GET / redirects to /dashboard."""
        response = self.client.get("/", follow_redirects=False)
        self.assertEqual(response.status_code, 307)
        self.assertEqual(response.headers["location"], "/dashboard")


if __name__ == "__main__":
    unittest.main()
