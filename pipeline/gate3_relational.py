"""
TrustFusion - Gate 3: Relational Workspace Mapping Pipeline.

Responsibilities:
-----------------
1. Heterogeneous Information Network (HIN) Construction:
   - Node Types: User (profiles), Company (workplace entities),
     Institute (educational institutions), Location (geographic regions).
   - Relations (6 canonical types):
     - 0: (User, worked_at, Company)
     - 1: (Company, employs, User)
     - 2: (User, studied_at, Institute)
     - 3: (Institute, educated, User)
     - 4: (User, located_in, Location)
     - 5: (Location, contains_user, User)
2. PyG Relational Graph Convolutional Network (RGCN):
   - 2-layer RGCN message passing with relation-specific weight matrices.
   - User node features initialized with numerical structural signals.
   - Entity node features initialized with learned embeddings.
3. Anti-Leakage Inductive Masking:
   - Loss computed strictly on train_mask (labels 0..2532).
   - Test labels strictly masked during training.
4. Louvain Community Detection:
   - Partitions projected User-User co-occurrence graph to identify coordinated
     fraud compounds, emulator rings, and shared infrastructure clusters.
   - Extracts community ID, cluster size, and cluster density signals.
5. Artifact Serialization:
   - Saves reproducible models, embeddings, and community features to
     data/processed/relational/.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
from pathlib import Path
from typing import Any

import community as community_louvain
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv


RANDOM_STATE = 42
HIDDEN_DIM = 64
NUM_RELATIONS = 6
NUM_EPOCHS = 60
LEARNING_RATE = 0.005


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logging.info("CUDA is available. Using GPU: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logging.info("CUDA not available. Falling back to CPU.")
    return device


def safe_parse_structured(value: Any) -> Any:
    """Parse stringified dict/list representations safely."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    if isinstance(value, (dict, list, tuple)):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return {}
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return {}


def extract_user_entities(df: pd.DataFrame) -> tuple[list[set[str]], list[set[str]], list[set[str]]]:
    """
    Extracts associated companies, institutions, and locations for each user.
    """
    user_companies: list[set[str]] = []
    user_institutes: list[set[str]] = []
    user_locations: list[set[str]] = []

    for _, row in df.iterrows():
        c_set: set[str] = set()
        i_set: set[str] = set()
        l_set: set[str] = set()

        # Primary Workplace
        wp = str(row.get("Workplace", "") or "").strip()
        if wp and wp.lower() not in {"nan", "none", ""}:
            c_set.add(wp.lower())

        # Primary Location
        loc = str(row.get("Location", "") or "").strip()
        if loc and loc.lower() not in {"nan", "none", ""}:
            l_set.add(loc.lower())

        # Experiences (Past & Present Workplaces + Workplace Locations)
        exp = safe_parse_structured(row.get("Experiences", {}))
        if isinstance(exp, dict):
            for _, exp_entry in exp.items():
                if isinstance(exp_entry, dict):
                    w = str(exp_entry.get("Workplace", "") or "").strip()
                    if w and w.lower() not in {"nan", "none", ""}:
                        c_set.add(w.lower())
                    wl = str(exp_entry.get("Workplace Location", "") or "").strip()
                    if wl and wl.lower() not in {"nan", "none", ""}:
                        l_set.add(wl.lower())

        # Educations (Institutes)
        edu = safe_parse_structured(row.get("Educations", {}))
        if isinstance(edu, dict):
            for _, edu_entry in edu.items():
                if isinstance(edu_entry, dict):
                    inst = str(edu_entry.get("Institute", "") or "").strip()
                    if inst and inst.lower() not in {"nan", "none", ""}:
                        i_set.add(inst.lower())

        user_companies.append(c_set)
        user_institutes.append(i_set)
        user_locations.append(l_set)

    return user_companies, user_institutes, user_locations


class HeteroGraphBuilder:
    """Builds unified node index, typed relational edges, and User-User projection graph."""

    def __init__(
        self,
        num_users: int,
        user_companies: list[set[str]],
        user_institutes: list[set[str]],
        user_locations: list[set[str]],
    ):
        self.num_users = num_users

        # Collect unique entities
        all_companies = sorted(set().union(*user_companies))
        all_institutes = sorted(set().union(*user_institutes))
        all_locations = sorted(set().union(*user_locations))

        self.company_to_idx = {c: num_users + i for i, c in enumerate(all_companies)}
        offset_inst = num_users + len(all_companies)
        self.institute_to_idx = {inst: offset_inst + i for i, inst in enumerate(all_institutes)}
        offset_loc = offset_inst + len(all_institutes)
        self.location_to_idx = {loc: offset_loc + i for i, loc in enumerate(all_locations)}

        self.num_companies = len(all_companies)
        self.num_institutes = len(all_institutes)
        self.num_locations = len(all_locations)
        self.total_nodes = num_users + self.num_companies + self.num_institutes + self.num_locations

        # Build relational edges
        src_list: list[int] = []
        dst_list: list[int] = []
        type_list: list[int] = []

        # For Louvain projection: count shared entities between users
        self.co_occurrence_counts: dict[tuple[int, int], int] = {}

        entity_to_users: dict[int, list[int]] = {}

        for u in range(num_users):
            # Companies: relation 0 (worked_at), 1 (employs)
            for c in user_companies[u]:
                c_idx = self.company_to_idx[c]
                src_list.extend([u, c_idx])
                dst_list.extend([c_idx, u])
                type_list.extend([0, 1])
                entity_to_users.setdefault(c_idx, []).append(u)

            # Institutes: relation 2 (studied_at), 3 (educated)
            for inst in user_institutes[u]:
                i_idx = self.institute_to_idx[inst]
                src_list.extend([u, i_idx])
                dst_list.extend([i_idx, u])
                type_list.extend([2, 3])
                entity_to_users.setdefault(i_idx, []).append(u)

            # Locations: relation 4 (located_in), 5 (contains_user)
            for loc in user_locations[u]:
                l_idx = self.location_to_idx[loc]
                src_list.extend([u, l_idx])
                dst_list.extend([l_idx, u])
                type_list.extend([4, 5])
                entity_to_users.setdefault(l_idx, []).append(u)

        # Build User-User co-occurrence weights for Louvain
        for e_idx, u_list in entity_to_users.items():
            if len(u_list) > 1 and len(u_list) < 500:  # Avoid overly dense mega-hubs
                for i in range(len(u_list)):
                    for j in range(i + 1, len(u_list)):
                        pair = (u_list[i], u_list[j]) if u_list[i] < u_list[j] else (u_list[j], u_list[i])
                        self.co_occurrence_counts[pair] = self.co_occurrence_counts.get(pair, 0) + 1

        self.edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        self.edge_type = torch.tensor(type_list, dtype=torch.long)

        logging.info(
            "Heterogeneous Graph built: %d Users, %d Companies, %d Institutes, %d Locations (Total %d nodes, %d directed edges)",
            self.num_users,
            self.num_companies,
            self.num_institutes,
            self.num_locations,
            self.total_nodes,
            self.edge_index.size(1),
        )


class LouvainClusterAnalyzer:
    """Partitions user co-occurrence network to expose tightly-coupled fraud rings."""

    @staticmethod
    def compute_community_features(
        num_users: int,
        co_occurrence_counts: dict[tuple[int, int], int],
    ) -> tuple[np.ndarray, float]:
        G = nx.Graph()
        G.add_nodes_from(range(num_users))

        for (u1, u2), w in co_occurrence_counts.items():
            G.add_edge(u1, u2, weight=float(w))

        logging.info("Running Louvain Community Detection on User co-occurrence graph (%d edges)...", G.number_of_edges())
        partition = community_louvain.best_partition(G, random_state=RANDOM_STATE)
        modularity = community_louvain.modularity(partition, G) if G.number_of_edges() > 0 else 0.0

        community_sizes: dict[int, int] = {}
        for node, c_id in partition.items():
            community_sizes[c_id] = community_sizes.get(c_id, 0) + 1

        features = np.zeros((num_users, 4), dtype=np.float32)
        for u in range(num_users):
            c_id = partition[u]
            c_size = community_sizes[c_id]
            degree = G.degree(u)
            weighted_degree = G.degree(u, weight="weight")

            features[u, 0] = float(c_id % 100) / 100.0          # Normalized community bucket
            features[u, 1] = float(np.log1p(c_size))            # Cluster size scale
            features[u, 2] = float(np.log1p(degree))            # Co-occurrence degree
            features[u, 3] = float(np.log1p(weighted_degree))   # Weighted connectivity

        logging.info(
            "Louvain clustering completed. Unique communities: %d | Graph Modularity: %.4f",
            len(community_sizes),
            modularity,
        )
        return features, float(modularity)


class Gate3RGCNModel(nn.Module):
    """
    2-Layer Relational Graph Convolutional Network (RGCN) with
    heterogeneous entity embeddings and classification head.
    """

    def __init__(
        self,
        num_users: int,
        num_entities: int,
        user_feat_dim: int = 17,
        hidden_dim: int = HIDDEN_DIM,
        num_relations: int = NUM_RELATIONS,
        num_classes: int = 4,
    ):
        super().__init__()
        self.num_users = num_users

        # User projection layer
        self.user_proj = nn.Linear(user_feat_dim, hidden_dim)

        # Entity embedding lookup
        self.entity_embedding = nn.Embedding(num_entities, hidden_dim)

        # RGCN Convolutional Layers
        self.rgcn1 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.rgcn2 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # Gate 3 Representation & Classification Heads
        self.representation_layer = nn.Linear(hidden_dim, hidden_dim)
        self.classifier_4class = nn.Linear(hidden_dim, num_classes)
        self.classifier_binary = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x_user: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Project user features
        h_user = self.user_proj(x_user)

        # Entity node embeddings
        entity_indices = torch.arange(
            self.entity_embedding.num_embeddings,
            device=x_user.device,
        )
        h_entities = self.entity_embedding(entity_indices)

        # Concatenate into full graph node representation [N_total, hidden_dim]
        h = torch.cat([h_user, h_entities], dim=0)

        # Layer 1
        h = self.rgcn1(h, edge_index, edge_type)
        h = self.dropout(self.relu(self.bn1(h)))

        # Layer 2
        h = self.rgcn2(h, edge_index, edge_type)
        h = self.relu(self.bn2(h))

        # Extract only User nodes [N_users, hidden_dim]
        h_user_final = h[: self.num_users]

        representation = self.relu(self.representation_layer(h_user_final))
        logits_4class = self.classifier_4class(representation)
        logits_binary = self.classifier_binary(representation).squeeze(-1)

        return representation, logits_4class, logits_binary


def train_gate3_model(
    model: Gate3RGCNModel,
    x_user: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    y_4class: torch.Tensor,
    y_binary: torch.Tensor,
    train_mask: torch.Tensor,
    epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
) -> Gate3RGCNModel:
    """Trains the RGCN strictly evaluating loss on train_mask nodes."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion_4class = nn.CrossEntropyLoss()
    criterion_bin = nn.BCEWithLogitsLoss()

    logging.info("Training Gate 3 PyG RGCN for %d epochs...", epochs)

    for epoch in range(epochs):
        optimizer.zero_grad()
        _, logits_4class, logits_bin = model(x_user, edge_index, edge_type)

        loss_4 = criterion_4class(logits_4class[train_mask], y_4class[train_mask])
        loss_b = criterion_bin(logits_bin[train_mask], y_binary[train_mask])
        loss = loss_4 + loss_b

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 15 == 0 or epoch == epochs - 1:
            logging.info("Epoch %2d/%2d | RGCN Training Loss: %.4f", epoch + 1, epochs, loss.item())

    model.eval()
    return model


def run_gate3(
    data_dir: Path = Path("data/processed"),
    output_dir: Path = Path("data/processed/relational"),
    epochs: int = NUM_EPOCHS,
) -> dict[str, Any]:
    """Execute complete Gate 3 relational workspace mapping pipeline."""
    setup_logging()
    device = get_device()

    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("=== Starting Gate 3 (Relational Workspace Mapping) ===")

    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"
    x_train_num_file = data_dir / "numerical" / "X_train.npy"
    x_test_num_file = data_dir / "numerical" / "X_test.npy"

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    n_train = len(train_df)
    n_test = len(test_df)
    total_users = n_train + n_test

    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    # 1. Extract associated entities
    logging.info("Extracting user entities (companies, institutes, locations)...")
    user_comps, user_insts, user_locs = extract_user_entities(combined_df)

    # 2. Build Heterogeneous Graph
    graph_builder = HeteroGraphBuilder(
        num_users=total_users,
        user_companies=user_comps,
        user_institutes=user_insts,
        user_locations=user_locs,
    )

    # 3. Louvain Community Detection on User co-occurrence network
    community_features, modularity = LouvainClusterAnalyzer.compute_community_features(
        num_users=total_users,
        co_occurrence_counts=graph_builder.co_occurrence_counts,
    )

    train_comm_feats = community_features[:n_train]
    test_comm_feats = community_features[n_train:]

    # 4. Load numerical input features
    x_train_num = np.load(x_train_num_file)
    x_test_num = np.load(x_test_num_file)
    x_user_all = np.vstack([x_train_num, x_test_num]).astype(np.float32)

    # 5. Prepare Tensors & Anti-Leakage Masks
    train_mask = torch.zeros(total_users, dtype=torch.bool)
    train_mask[:n_train] = True

    test_mask = torch.zeros(total_users, dtype=torch.bool)
    test_mask[n_train:] = True

    y_4class = torch.tensor(combined_df["class_4"].to_numpy(), dtype=torch.long).to(device)
    y_binary = torch.tensor(combined_df["is_fake"].to_numpy(), dtype=torch.float32).to(device)
    x_user_tensor = torch.tensor(x_user_all, dtype=torch.float32).to(device)
    edge_index = graph_builder.edge_index.to(device)
    edge_type = graph_builder.edge_type.to(device)

    # 6. Initialize & Train RGCN
    num_entities = graph_builder.num_companies + graph_builder.num_institutes + graph_builder.num_locations
    model = Gate3RGCNModel(
        num_users=total_users,
        num_entities=num_entities,
        user_feat_dim=x_user_all.shape[1],
        hidden_dim=HIDDEN_DIM,
        num_relations=NUM_RELATIONS,
        num_classes=4,
    ).to(device)

    trained_model = train_gate3_model(
        model=model,
        x_user=x_user_tensor,
        edge_index=edge_index,
        edge_type=edge_type,
        y_4class=y_4class,
        y_binary=y_binary,
        train_mask=train_mask,
        epochs=epochs,
    )

    # 7. Generate Relational Embeddings & Risk Probabilities
    with torch.no_grad():
        rep, _, logits_bin = trained_model(x_user_tensor, edge_index, edge_type)
        risk_probs = torch.sigmoid(logits_bin).cpu().numpy()
        rep_numpy = rep.cpu().numpy()

    train_gate3_embeddings = rep_numpy[:n_train]
    test_gate3_embeddings = rep_numpy[n_train:]
    train_gate3_risk = risk_probs[:n_train]
    test_gate3_risk = risk_probs[n_train:]

    # 8. Save Artifacts
    np.save(output_dir / "train_gate3_embeddings.npy", train_gate3_embeddings)
    np.save(output_dir / "test_gate3_embeddings.npy", test_gate3_embeddings)
    np.save(output_dir / "train_gate3_risk_scores.npy", train_gate3_risk)
    np.save(output_dir / "test_gate3_risk_scores.npy", test_gate3_risk)
    np.save(output_dir / "train_community_features.npy", train_comm_feats)
    np.save(output_dir / "test_community_features.npy", test_comm_feats)

    torch.save(trained_model.state_dict(), output_dir / "gate3_rgcn_model.pt")

    metadata = {
        "num_users": total_users,
        "train_users": n_train,
        "test_users": n_test,
        "num_companies": graph_builder.num_companies,
        "num_institutes": graph_builder.num_institutes,
        "num_locations": graph_builder.num_locations,
        "total_graph_nodes": graph_builder.total_nodes,
        "total_directed_edges": int(edge_index.size(1)),
        "num_relations": NUM_RELATIONS,
        "relational_embedding_dim": HIDDEN_DIM,
        "louvain_modularity": modularity,
        "anti_leakage": "Loss evaluated strictly on train_mask; test nodes evaluated without supervision.",
    }

    with (output_dir / "gate3_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logging.info("Gate 3 pipeline completed successfully. Artifacts saved to: %s", output_dir)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TrustFusion Gate 3: Relational Workspace Mapping")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"), help="Processed data directory")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/relational"), help="Relational output directory")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Training epochs for RGCN")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_gate3(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
    )
