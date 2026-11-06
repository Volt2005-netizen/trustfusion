"""
TrustFusion - Unit Tests for Step 6: Explainability Suite (XAI).

Tests SHAP TreeExplainer feature attribution, local profile decomposition,
linguistic keyword triggers, and GNNExplainer computational subgraph extraction.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from xai.shap_explainer import (
    DECISION_FEATURE_NAMES,
    MultiModalShapExplainer,
    extract_linguistic_triggers,
)
from xai.gnn_explainer import (
    RelationalGNNExplainer,
    RELATION_NAMES,
)
from pipeline.gate3_relational import (
    Gate3RGCNModel,
    HeteroGraphBuilder,
)


class TestXAISuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_dir = Path("data/processed")
        cls.fusion_dir = Path("data/processed/fusion")
        cls.relational_dir = Path("data/processed/relational")

        # Load small slice of decision features
        cls.X_test_dec = np.load(cls.fusion_dir / "fused_X_test_decision.npy")[:10]

        # Initialize SHAP explainer
        cls.shap_explainer = MultiModalShapExplainer(
            catboost_model_path=cls.fusion_dir / "catboost_binary.cbm",
            xgboost_model_path=cls.fusion_dir / "xgboost_binary.json",
            feature_names=DECISION_FEATURE_NAMES,
        )

    def test_shap_explainer_shapes(self):
        """SHAP value computation must return exact (N, 28) matrix with finite values."""
        shap_vals = self.shap_explainer.explain_matrix(self.X_test_dec)
        self.assertEqual(shap_vals.shape, (10, 28))
        self.assertTrue(np.all(np.isfinite(shap_vals)))

    def test_shap_local_profile_decomposition(self):
        """Single profile explanation must yield drivers, protective factors, and valid risk score."""
        sample_x = self.X_test_dec[0]
        local_exp = self.shap_explainer.explain_single_profile(sample_x, top_k=5)

        self.assertIn("final_risk_score", local_exp)
        self.assertIn("base_value", local_exp)
        self.assertIn("top_risk_drivers", local_exp)
        self.assertIn("top_protective_factors", local_exp)
        self.assertIn("all_feature_attributions", local_exp)

        self.assertGreaterEqual(local_exp["final_risk_score"], 0.0)
        self.assertLessEqual(local_exp["final_risk_score"], 1.0)
        self.assertEqual(len(local_exp["all_feature_attributions"]), 28)

    def test_linguistic_trigger_extraction(self):
        """Keyword scanner must flag off-platforming and urgency triggers."""
        suspicious_text = "Urgent hiring! Send CV to WhatsApp +123456789. No interview needed, guaranteed income."
        benign_text = "Experienced software engineer specializing in distributed systems and cloud architecture."

        suspicious_res = extract_linguistic_triggers(suspicious_text)
        benign_res = extract_linguistic_triggers(benign_text)

        self.assertTrue(suspicious_res["has_high_risk_triggers"])
        self.assertIn("whatsapp", suspicious_res["triggers_found"])
        self.assertIn("urgent hiring", suspicious_res["triggers_found"])

        self.assertFalse(benign_res["has_high_risk_triggers"])
        self.assertEqual(len(benign_res["triggers_found"]), 0)

    def test_gnn_explainer_subgraph_extraction(self):
        """Relational GNNExplainer must extract valid 2-hop ego subgraph with Sybil metrics."""
        # Create a small synthetic heterogeneous graph
        num_users = 4
        user_comps = [{"comp_A"}, {"comp_A"}, {"comp_B"}, {"comp_B"}]
        user_insts = [{"inst_X"}, {"inst_X"}, {"inst_Y"}, {"inst_Y"}]
        user_locs = [{"loc_1"}, {"loc_1"}, {"loc_2"}, {"loc_2"}]

        builder = HeteroGraphBuilder(num_users, user_comps, user_insts, user_locs)
        x_user = np.random.randn(num_users, 17).astype(np.float32)

        df = pd.DataFrame({
            "profile_id": [f"TF_{i}" for i in range(num_users)],
            "Full Name": [f"User_{i}" for i in range(num_users)],
            "Workplace": ["comp_A", "comp_A", "comp_B", "comp_B"],
            "is_fake": [1, 1, 0, 0],
            "class_name": ["manual_fake", "manual_fake", "legitimate", "legitimate"],
        })

        num_entities = builder.num_companies + builder.num_institutes + builder.num_locations
        model = Gate3RGCNModel(
            num_users=num_users,
            num_entities=num_entities,
            user_feat_dim=17,
            hidden_dim=32,
            num_relations=6,
            num_classes=4,
        )

        explainer = RelationalGNNExplainer(
            model=model,
            graph_builder=builder,
            x_user_all=x_user,
            profiles_df=df,
            device=torch.device("cpu"),
        )

        # Explain User 0 (connected to User 1 via comp_A, inst_X, loc_1)
        exp = explainer.explain_user_subgraph(user_idx=0, num_hops=2)

        self.assertEqual(exp["target_user"]["node_idx"], 0)
        self.assertEqual(exp["target_user"]["node_type"], "User")
        self.assertIn("subgraph_stats", exp)
        self.assertIn("top_relational_paths", exp)
        self.assertIn("nodes", exp)
        self.assertIn("edges", exp)

        # Sybil density should be 100% since neighbor User 1 is also fake
        stats = exp["subgraph_stats"]
        self.assertEqual(stats["num_connected_users"], 1)
        self.assertEqual(stats["num_fake_neighbors"], 1)
        self.assertEqual(stats["sybil_density_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
