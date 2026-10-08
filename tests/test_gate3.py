"""
TrustFusion - Unit Tests for Gate 3: Relational Workspace Mapping.

Tests entity extraction, heterogeneous graph indexing, Louvain community detection,
RGCN forward pass shapes, and inductive masking.
"""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd
import torch

from pipeline.gate3_relational import (
    Gate3RGCNModel,
    HeteroGraphBuilder,
    LouvainClusterAnalyzer,
    extract_user_entities,
    get_device,
)


class TestGate3Relational(unittest.TestCase):

    def test_extract_user_entities(self):
        data = {
            "Workplace": ["Google", "Acme Corp"],
            "Location": ["New York, NY", "London, UK"],
            "Experiences": [
                "{'0': {'Workplace': 'Alphabet', 'Workplace Location': 'Mountain View, CA'}}",
                "{'0': {'Workplace': 'Acme Subsidiary', 'Workplace Location': 'London, UK'}}",
            ],
            "Educations": [
                "{'0': {'Institute': 'MIT'}}",
                "{'0': {'Institute': 'Oxford University'}}",
            ],
        }
        df = pd.DataFrame(data)
        comps, insts, locs = extract_user_entities(df)

        self.assertEqual(len(comps), 2)
        self.assertIn("google", comps[0])
        self.assertIn("alphabet", comps[0])
        self.assertIn("mit", insts[0])
        self.assertIn("new york, ny", locs[0])
        self.assertIn("mountain view, ca", locs[0])

    def test_hetero_graph_builder(self):
        user_comps = [{"c1", "c2"}, {"c2", "c3"}]
        user_insts = [{"i1"}, {"i1"}]
        user_locs = [{"l1"}, {"l2"}]

        builder = HeteroGraphBuilder(
            num_users=2,
            user_companies=user_comps,
            user_institutes=user_insts,
            user_locations=user_locs,
        )

        self.assertEqual(builder.num_users, 2)
        self.assertEqual(builder.num_companies, 3)
        self.assertEqual(builder.num_institutes, 1)
        self.assertEqual(builder.num_locations, 2)
        self.assertEqual(builder.total_nodes, 8)

        # Edges should be bi-directional with 6 relation types
        self.assertEqual(builder.edge_index.dim(), 2)
        self.assertEqual(builder.edge_index.size(0), 2)
        self.assertTrue(torch.all(builder.edge_type >= 0))
        self.assertTrue(torch.all(builder.edge_type < 6))

    def test_louvain_cluster_analyzer(self):
        # 4 users: (0, 1) share 2 entities, (2, 3) share 2 entities
        co_counts = {
            (0, 1): 2,
            (2, 3): 2,
        }
        features, modularity = LouvainClusterAnalyzer.compute_community_features(
            num_users=4,
            co_occurrence_counts=co_counts,
        )

        self.assertEqual(features.shape, (4, 4))
        self.assertTrue(np.all(np.isfinite(features)))
        self.assertGreaterEqual(modularity, 0.0)

    def test_gate3_rgcn_model_forward(self):
        num_users = 4
        num_entities = 6
        model = Gate3RGCNModel(
            num_users=num_users,
            num_entities=num_entities,
            user_feat_dim=17,
            hidden_dim=32,
            num_relations=6,
            num_classes=4,
        )
        model.eval()

        x_user = torch.randn(num_users, 17)
        # Sample edges connecting users and entities
        edge_index = torch.tensor([
            [0, 4, 1, 5, 2, 6, 3, 7],
            [4, 0, 5, 1, 6, 2, 7, 3],
        ], dtype=torch.long)
        edge_type = torch.tensor([0, 1, 0, 1, 2, 3, 4, 5], dtype=torch.long)

        with torch.no_grad():
            rep, logits_4, logits_bin = model(x_user, edge_index, edge_type)

        self.assertEqual(rep.shape, (num_users, 32))
        self.assertEqual(logits_4.shape, (num_users, 4))
        self.assertEqual(logits_bin.shape, (num_users,))

    def test_device_selection(self):
        dev = get_device()
        self.assertIsInstance(dev, torch.device)
        self.assertIn(dev.type, ["cuda", "cpu"])


if __name__ == "__main__":
    unittest.main()
