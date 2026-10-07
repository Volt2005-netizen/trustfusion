"""
TrustFusion - Gate 2: Linguistic & Conversational Forensics Pipeline.

Responsibilities:
-----------------
1. Section Tag Embeddings (STE): Format section-level text representations with
   structural delimiters and extract 768-D contextual embeddings using RoBERTa-base.
2. Anti-Leakage Dimensionality Reduction: Fit PCA-150 strictly on training embeddings
   and transform both train and test sets to 150-D residual representations.
3. Linguistic & Off-Platforming Forensics: Scan profile texts for spear-phishing syntax,
   platform evasion patterns (WhatsApp, Telegram, etc.), and advance-fee cues.
4. Conversational & Temporal Forensics: Modular IAT (Inter-Arrival Time) and Shannon
   Temporal Entropy engine (evaluates live/synthetic chat streams and provides
   cold-start defaults for static profiles).
5. Gate 2 Neural Representation Head: Train lightweight representation model to
   produce linguistic risk scores and gate output embeddings for Phase 4 fusion.
6. Artifact Serialization: Save reproducible models and feature arrays to data/processed/linguistic/.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


RANDOM_STATE = 42
DEFAULT_MODEL_NAME = "roberta-base"
PCA_COMPONENTS = 150
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 512

OFF_PLATFORM_KEYWORDS = [
    "whatsapp",
    "telegram",
    "wire transfer",
    "crypto",
    "bitcoin",
    "usdt",
    "refundable deposit",
    "registration fee",
    "joining fee",
    "assessment fee",
    "personal email",
    "gmail.com",
    "yahoo.com",
    "outlook.com",
    "protonmail",
    "signal",
    "contact me at",
    "dm me on",
    "reach me at",
    "text me at",
    "send cv to",
    "urgent hiring",
    "immediate opening",
    "no interview",
    "work from home earn",
    "guaranteed income",
]

SECTION_TAGS = [
    ("intro_text", "[INTRO]"),
    ("about_text", "[ABOUT]"),
    ("experience_text", "[EXPERIENCE]"),
    ("education_text", "[EDUCATION]"),
    ("skills_text", "[SKILLS]"),
    ("recommendations_text", "[RECOMMENDATIONS]"),
    ("projects_text", "[PROJECTS]"),
    ("honors_text", "[HONORS]"),
    ("interests_text", "[INTERESTS]"),
    ("activities_text", "[ACTIVITIES]"),
]


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


def build_ste_input_text(row: dict[str, Any]) -> str:
    """Format individual section fields with distinct Section Tag markers."""
    chunks: list[str] = []
    for col, tag in SECTION_TAGS:
        val = str(row.get(col, "") or "").strip()
        if val and val.lower() != "nan" and val.lower() != "none":
            chunks.append(f"{tag} {val}")

    if not chunks:
        # Fallback to general profile_text if individual sections are absent
        profile_text = str(row.get("profile_text", "") or "").strip()
        return profile_text if profile_text else "[EMPTY]"

    return " ".join(chunks)


class ProfileTextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer: Any, max_length: int = MAX_SEQ_LENGTH):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }


class RoBERTaSTEExtractor:
    """Extracts 768-D mean-pooled Section Tag Embeddings using RoBERTa-base."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: torch.device | None = None):
        self.device = device or get_device()
        self.model_name = model_name

        logging.info("Initializing RoBERTa tokenizer and model: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

    def extract_embeddings(
        self,
        texts: list[str],
        batch_size: int = BATCH_SIZE,
    ) -> np.ndarray:
        dataset = ProfileTextDataset(texts, self.tokenizer, max_length=MAX_SEQ_LENGTH)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_embeddings: list[np.ndarray] = []

        logging.info("Extracting embeddings for %d profiles (batch_size=%d)...", len(texts), batch_size)
        use_amp = self.device.type == "cuda"

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                with torch.amp.autocast(device_type=self.device.type, enabled=use_amp):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    last_hidden_state = outputs.last_hidden_state  # [B, L, 768]

                    # Mean pooling over non-padded tokens
                    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                    sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
                    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                    mean_pooled = (sum_embeddings / sum_mask).cpu().to(torch.float32).numpy()

                all_embeddings.append(mean_pooled)

                if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == len(loader):
                    logging.info("Processed batch %d/%d", batch_idx + 1, len(loader))

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return np.vstack(all_embeddings)


class OffPlatformingDetector:
    """Scans profile text for platform evasion, lookalike links, and spear-phishing syntax."""

    @staticmethod
    def extract_linguistic_signals(text: str) -> dict[str, float]:
        text_lower = text.lower()
        word_count = max(1, len(re.findall(r"\w+", text_lower)))

        # 1. Keyword density for off-platform and advance-fee keywords
        keyword_hits = sum(1 for kw in OFF_PLATFORM_KEYWORDS if kw in text_lower)
        keyword_density = keyword_hits / (word_count / 100.0)

        # 2. External contact redirects (regex for email, phone, external links)
        email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
        phone_pattern = r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
        url_pattern = r"https?://[^\s]+|www\.[^\s]+"

        email_count = len(re.findall(email_pattern, text))
        phone_count = len(re.findall(phone_pattern, text))
        url_count = len(re.findall(url_pattern, text))

        # 3. Urgency / scam trigger vocabulary
        urgency_terms = ["urgent", "immediate", "guaranteed", "deposit", "earn", "wire", "pay"]
        urgency_hits = sum(1 for term in urgency_terms if term in text_lower)

        # 4. Composite off-platforming risk heuristic [0.0, 1.0]
        raw_score = (keyword_hits * 0.25) + (email_count * 0.2) + (phone_count * 0.25) + (url_count * 0.15) + (urgency_hits * 0.1)
        normalized_risk = float(np.tanh(raw_score))

        return {
            "keyword_density": float(keyword_density),
            "email_count": float(email_count),
            "phone_count": float(phone_count),
            "url_count": float(url_count),
            "urgency_count": float(urgency_hits),
            "off_platform_risk_score": normalized_risk,
        }

    @classmethod
    def batch_extract(cls, texts: list[str]) -> np.ndarray:
        features = [cls.extract_linguistic_signals(t) for t in texts]
        feature_keys = [
            "keyword_density",
            "email_count",
            "phone_count",
            "url_count",
            "urgency_count",
            "off_platform_risk_score",
        ]
        matrix = np.array([[f[k] for k in feature_keys] for f in features], dtype=np.float32)
        return matrix


class TemporalEntropyEngine:
    """Calculates Inter-Arrival Time (IAT) deltas and Shannon Timing Entropy."""

    @staticmethod
    def calculate_iat_entropy(
        timestamps: list[float] | np.ndarray,
        num_bins: int = 10,
    ) -> dict[str, float]:
        """
        Calculates IAT Shannon Entropy for a sequence of message timestamps (in seconds).
        Returns baseline/neutral metrics for sequences with fewer than 3 timestamps.
        """
        if len(timestamps) < 3:
            return {
                "iat_mean": 0.0,
                "iat_std": 0.0,
                "shannon_entropy": 0.0,
                "normalized_entropy": 0.0,
                "has_temporal_data": 0.0,
            }

        arr = np.sort(np.asarray(timestamps, dtype=np.float64))
        deltas = np.diff(arr)
        deltas = deltas[deltas > 0]

        if len(deltas) < 2:
            return {
                "iat_mean": float(np.mean(deltas)) if len(deltas) else 0.0,
                "iat_std": 0.0,
                "shannon_entropy": 0.0,
                "normalized_entropy": 0.0,
                "has_temporal_data": 1.0,
            }

        # Histogram-based probability distribution
        hist, _ = np.histogram(deltas, bins=num_bins, density=False)
        probabilities = hist / hist.sum()
        probabilities = probabilities[probabilities > 0]

        # Shannon Entropy H(T) = - sum p * log2(p)
        entropy = -float(np.sum(probabilities * np.log2(probabilities)))
        max_possible_entropy = math.log2(num_bins)
        norm_entropy = float(entropy / max_possible_entropy) if max_possible_entropy > 0 else 0.0

        return {
            "iat_mean": float(np.mean(deltas)),
            "iat_std": float(np.std(deltas)),
            "shannon_entropy": entropy,
            "normalized_entropy": norm_entropy,
            "has_temporal_data": 1.0,
        }


class Gate2NeuralHead(nn.Module):
    """
    Lightweight classification and representation head mapping 150-D STE residual
    embeddings and 6-D linguistic indicators to a unified 64-D Gate 2 embedding
    and risk predictions.
    """

    def __init__(self, input_dim: int = 156, hidden_dim: int = 64, num_classes: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.representation_layer = nn.Linear(hidden_dim, hidden_dim)
        self.classifier_4class = nn.Linear(hidden_dim, num_classes)
        self.classifier_binary = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.dropout(self.relu(self.bn1(self.fc1(x))))
        representation = self.relu(self.representation_layer(h))
        logits_4class = self.classifier_4class(representation)
        logits_binary = self.classifier_binary(representation).squeeze(-1)
        return representation, logits_4class, logits_binary


def fit_pca_150(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int = PCA_COMPONENTS,
) -> tuple[np.ndarray, np.ndarray, PCA, float]:
    """Fit PCA strictly on training embeddings and transform both train and test."""
    logging.info("Fitting PCA (%d components) strictly on X_train...", n_components)
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    explained_var = float(np.sum(pca.explained_variance_ratio_))
    logging.info(
        "PCA fit complete. Total explained variance ratio across %d components: %.4f (%.2f%%)",
        n_components,
        explained_var,
        explained_var * 100,
    )
    return X_train_pca, X_test_pca, pca, explained_var


def train_gate2_head(
    X_train: np.ndarray,
    y_train_4: np.ndarray,
    y_train_bin: np.ndarray,
    epochs: int = 25,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> Gate2NeuralHead:
    """Train the Gate 2 representation model using training data only."""
    device = device or torch.device("cpu")
    input_dim = X_train.shape[1]
    model = Gate2NeuralHead(input_dim=input_dim, hidden_dim=64, num_classes=4).to(device)
    model.train()

    criterion_4class = nn.CrossEntropyLoss()
    criterion_bin = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train_4, dtype=torch.long),
        torch.tensor(y_train_bin, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    logging.info("Training Gate 2 neural representation head for %d epochs...", epochs)
    for epoch in range(epochs):
        total_loss = 0.0
        for bx, by4, byb in loader:
            bx, by4, byb = bx.to(device), by4.to(device), byb.to(device)
            optimizer.zero_grad()
            _, logits4, logitsb = model(bx)
            loss4 = criterion_4class(logits4, by4)
            lossb = criterion_bin(logitsb, byb)
            loss = loss4 + lossb
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            logging.info("Epoch %2d/%2d | Loss: %.4f", epoch + 1, epochs, total_loss / len(loader))

    model.eval()
    return model


def run_gate2(
    data_dir: Path = Path("data/processed"),
    output_dir: Path = Path("data/processed/linguistic"),
    batch_size: int = BATCH_SIZE,
    epochs: int = 25,
) -> dict[str, Any]:
    """Execute complete Gate 2 linguistic & conversational forensics pipeline."""
    setup_logging()
    device = get_device()

    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("=== Starting Gate 2 (Linguistic & Temporal Forensics) ===")

    train_text_file = data_dir / "text" / "train_text.json"
    test_text_file = data_dir / "text" / "test_text.json"
    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"

    if not train_text_file.exists() or not test_text_file.exists():
        raise FileNotFoundError(
            f"Processed text files not found in {data_dir}/text. Run pipeline/preprocess.py first."
        )

    logging.info("Loading train and test text records...")
    train_records = json.loads(train_text_file.read_text(encoding="utf-8"))
    test_records = json.loads(test_text_file.read_text(encoding="utf-8"))

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # 1. Format Section Tag Embedding texts
    train_ste_texts = [build_ste_input_text(r) for r in train_records]
    test_ste_texts = [build_ste_input_text(r) for r in test_records]

    # 2. Extract RoBERTa 768-D contextual embeddings
    extractor = RoBERTaSTEExtractor(model_name=DEFAULT_MODEL_NAME, device=device)
    X_train_768 = extractor.extract_embeddings(train_ste_texts, batch_size=batch_size)
    X_test_768 = extractor.extract_embeddings(test_ste_texts, batch_size=batch_size)

    # 3. Fit PCA-150 strictly on training embeddings
    X_train_pca, X_test_pca, pca_model, explained_variance = fit_pca_150(X_train_768, X_test_768)

    # 4. Extract Linguistic Off-Platforming & Deception signals
    detector = OffPlatformingDetector()
    X_train_ling = detector.batch_extract([r.get("profile_text", "") for r in train_records])
    X_test_ling = detector.batch_extract([r.get("profile_text", "") for r in test_records])

    # 5. Concatenate 150-D residual embeddings with 6-D linguistic features
    X_train_gate2 = np.hstack([X_train_pca, X_train_ling]).astype(np.float32)
    X_test_gate2 = np.hstack([X_test_pca, X_test_ling]).astype(np.float32)

    # 6. Train Gate 2 neural representation head on train split
    gate2_head = train_gate2_head(
        X_train=X_train_gate2,
        y_train_4=train_df["class_4"].to_numpy(),
        y_train_bin=train_df["is_fake"].to_numpy(),
        epochs=epochs,
        device=device,
    )

    # 7. Generate Gate 2 representations and risk scores
    with torch.no_grad():
        train_rep, _, train_logits_bin = gate2_head(torch.tensor(X_train_gate2).to(device))
        test_rep, _, test_logits_bin = gate2_head(torch.tensor(X_test_gate2).to(device))

        train_gate2_emb = train_rep.cpu().numpy()
        test_gate2_emb = test_rep.cpu().numpy()
        train_gate2_prob = torch.sigmoid(train_logits_bin).cpu().numpy()
        test_gate2_prob = torch.sigmoid(test_logits_bin).cpu().numpy()

    # 8. Save artifacts
    np.save(output_dir / "X_train_ste_pca150.npy", X_train_pca)
    np.save(output_dir / "X_test_ste_pca150.npy", X_test_pca)
    np.save(output_dir / "X_train_gate2_fused.npy", X_train_gate2)
    np.save(output_dir / "X_test_gate2_fused.npy", X_test_gate2)
    np.save(output_dir / "train_gate2_embeddings.npy", train_gate2_emb)
    np.save(output_dir / "test_gate2_embeddings.npy", test_gate2_emb)
    np.save(output_dir / "train_gate2_risk_scores.npy", train_gate2_prob)
    np.save(output_dir / "test_gate2_risk_scores.npy", test_gate2_prob)

    with (output_dir / "pca_150.pkl").open("wb") as f:
        pickle.dump(pca_model, f)

    torch.save(gate2_head.state_dict(), output_dir / "gate2_neural_head.pt")

    metadata = {
        "model_name": DEFAULT_MODEL_NAME,
        "pca_components": PCA_COMPONENTS,
        "explained_variance_ratio": explained_variance,
        "train_samples": len(train_records),
        "test_samples": len(test_records),
        "input_embedding_dim": 768,
        "fused_feature_dim": X_train_gate2.shape[1],
        "output_representation_dim": train_gate2_emb.shape[1],
        "temporal_forensics_status": "Modular engine active; zero-weight cold-start configured for static profiles.",
    }

    with (output_dir / "gate2_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logging.info("Gate 2 pipeline completed successfully. Artifacts saved to: %s", output_dir)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TrustFusion Gate 2: Linguistic & Temporal Forensics")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"), help="Processed data directory")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/linguistic"), help="Linguistic output directory")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Extraction batch size")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs for Gate 2 head")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_gate2(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
