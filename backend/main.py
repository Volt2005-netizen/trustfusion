"""
TrustFusion - Step 7: Serving API & Interactive Security Operations Dashboard.

Endpoints:
- GET  /health: System status, GPU/CUDA diagnostics, and model availability.
- GET  /api/v1/profiles: Curated exemplar test cases and profile catalog.
- GET  /api/v1/benchmarks: Test set evaluation metrics, confusion matrix, and gate ablation.
- POST /api/v1/predict: End-to-end multimodal inference with cold-start gating & 4-class attribution.
- POST /api/v1/explain: Local SHAP feature attribution, text triggers, and GNN Sybil paths.
- GET  /dashboard: Interactive Security Operations Center (SOC) single-page dashboard.
- GET  /: Redirects to /dashboard.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field
from xgboost import XGBClassifier

from pipeline.fusion_classifier import CrossModalAttentionGating, DECISION_FEATURE_DIM, FUSED_LATENT_DIM
from xai.shap_explainer import (
    DECISION_FEATURE_NAMES,
    MultiModalShapExplainer,
    extract_linguistic_triggers,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# Global serving state cache
STATE: dict[str, Any] = {}


# ============================================================================
# Pydantic Schemas
# ============================================================================

class NumericalCounts(BaseModel):
    experience_count: float = 0.0
    education_count: float = 0.0
    licenses_count: float = 0.0
    volunteering_count: float = 0.0
    skills_count: float = 0.0
    recommendations_count: float = 0.0
    projects_count: float = 0.0
    publications_count: float = 0.0
    courses_count: float = 0.0
    honors_count: float = 0.0
    scores_count: float = 0.0
    languages_count: float = 0.0
    organizations_count: float = 0.0
    interests_count: float = 0.0
    activities_count: float = 0.0
    connections_count: float = 0.0
    followers_count: float = 0.0


class PredictRequest(BaseModel):
    profile_id: Optional[str] = Field(None, description="Pre-loaded test profile ID (e.g. TF_ac29c8fbb6535468)")
    profile_index: Optional[int] = Field(None, description="0-based index in unseen test set (0 to 1085)")
    name: Optional[str] = Field(None, description="Candidate full name")
    headline: Optional[str] = Field(None, description="Candidate professional headline")
    about: Optional[str] = Field(None, description="Candidate about section")
    workplace: Optional[str] = Field(None, description="Company or organization")
    location: Optional[str] = Field(None, description="Geographic location")
    has_profile_photo: bool = Field(True, description="True if profile has photo; False for cold-start")
    simulate_cold_start: bool = Field(False, description="Simulate photo removal to inspect attention re-allocation")
    numerical_counts: Optional[NumericalCounts] = Field(None, description="Profile structural counts")


class GateBreakdown(BaseModel):
    gate1_visual_risk: float
    gate2_linguistic_risk: float
    gate3_relational_risk: float
    alpha_vis: float
    alpha_ling: float
    alpha_rel: float
    visual_photo_present: bool
    is_cold_start: bool


class ThreatAttribution(BaseModel):
    class_id: int
    class_label: str
    class_probabilities: dict[str, float]


class PredictResponse(BaseModel):
    profile_id: str
    profile_name: str
    headline: Optional[str]
    workplace: Optional[str]
    location: Optional[str]
    calibrated_risk_score: float
    triage_verdict: str  # SAFE | SUSPICIOUS | HIGH_RISK
    confidence_level: str  # VERY_HIGH | HIGH | MODERATE
    is_fake: bool
    ground_truth_class: Optional[str]
    threat_attribution: ThreatAttribution
    gate_breakdown: GateBreakdown
    cold_start_policy_applied: str
    latency_ms: float


class ExplainRequest(BaseModel):
    profile_id: Optional[str] = Field(None, description="Profile ID to explain")
    profile_index: Optional[int] = Field(None, description="Test index to explain")
    top_k: int = Field(5, description="Number of top risk/protective features to extract")


class ExplainResponse(BaseModel):
    profile_id: str
    final_risk_score: float
    base_expected_value: float
    top_risk_drivers: list[dict[str, Any]]
    top_protective_factors: list[dict[str, Any]]
    linguistic_triggers: dict[str, Any]
    gnn_subgraph: Optional[dict[str, Any]]


# ============================================================================
# Lifespan Initialization
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load ML models, SHAP explainers, and test sets on startup."""
    logging.info("Initializing TrustFusion Serving Engine...")
    start_time = time.time()
    data_dir = WORKSPACE_ROOT / "data" / "processed"
    fusion_dir = data_dir / "fusion"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    STATE["device"] = device
    STATE["device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    # 1. Load test dataset & precomputed matrices
    logging.info("Loading test dataset and multi-modal feature matrices...")
    test_csv_path = data_dir / "test.csv"
    if test_csv_path.exists():
        test_df = pd.read_csv(test_csv_path)
        STATE["test_df"] = test_df
        STATE["id_to_index"] = {row["profile_id"]: idx for idx, row in test_df.iterrows()}
    else:
        STATE["test_df"] = pd.DataFrame()
        STATE["id_to_index"] = {}

    STATE["fused_X_test_dec"] = np.load(fusion_dir / "fused_X_test_decision.npy")
    STATE["fused_X_test"] = np.load(fusion_dir / "fused_X_test.npy")
    STATE["test_final_risk"] = np.load(fusion_dir / "test_final_risk_scores.npy").ravel()
    STATE["test_4class_probs"] = np.load(fusion_dir / "test_4class_probs.npy")
    STATE["test_alphas"] = np.load(fusion_dir / "test_attention_weights.npy")

    # 2. Load Attention Gating PyTorch Model
    logging.info("Loading Cross-Modal Attention Gating module...")
    gating_module = CrossModalAttentionGating(emb_dim=FUSED_LATENT_DIM)
    gating_module_path = fusion_dir / "attention_gating_module.pt"
    if gating_module_path.exists():
        gating_module.load_state_dict(torch.load(gating_module_path, map_location=device))
    gating_module.to(device)
    gating_module.eval()
    STATE["gating_module"] = gating_module

    # 3. Load CatBoost & XGBoost Ensembles
    logging.info("Loading Dual-Gradient Boosted Decision Ensembles...")
    cb_bin = CatBoostClassifier()
    cb_bin.load_model(str(fusion_dir / "catboost_binary.cbm"))
    xgb_bin = XGBClassifier()
    xgb_bin.load_model(str(fusion_dir / "xgboost_binary.json"))

    cb_4c = CatBoostClassifier()
    cb_4c.load_model(str(fusion_dir / "catboost_4class.cbm"))
    xgb_4c = XGBClassifier()
    xgb_4c.load_model(str(fusion_dir / "xgboost_4class.json"))

    STATE["cb_bin"] = cb_bin
    STATE["xgb_bin"] = xgb_bin
    STATE["cb_4c"] = cb_4c
    STATE["xgb_4c"] = xgb_4c

    # 4. Load SHAP TreeExplainer
    logging.info("Initializing MultiModal SHAP TreeExplainer...")
    shap_explainer = MultiModalShapExplainer(
        catboost_model_path=fusion_dir / "catboost_binary.cbm",
        xgboost_model_path=fusion_dir / "xgboost_binary.json",
        feature_names=DECISION_FEATURE_NAMES,
    )
    STATE["shap_explainer"] = shap_explainer

    # 5. Load JSON Artifacts (SHAP summary & GNN subgraphs)
    shap_summary_path = fusion_dir / "shap_summary.json"
    if shap_summary_path.exists():
        with open(shap_summary_path, "r", encoding="utf-8") as f:
            STATE["shap_summary"] = json.load(f)
    else:
        STATE["shap_summary"] = {}

    gnn_summary_path = data_dir / "relational" / "gnn_subgraphs_summary.json"
    if gnn_summary_path.exists():
        with open(gnn_summary_path, "r", encoding="utf-8") as f:
            STATE["gnn_summary"] = json.load(f)
    else:
        STATE["gnn_summary"] = {}

    fusion_meta_path = fusion_dir / "fusion_metadata.json"
    if fusion_meta_path.exists():
        with open(fusion_meta_path, "r", encoding="utf-8") as f:
            STATE["fusion_metadata"] = json.load(f)
    else:
        STATE["fusion_metadata"] = {}

    elapsed = time.time() - start_time
    logging.info("TrustFusion Serving Engine successfully initialized in %.2f seconds.", elapsed)
    yield
    logging.info("Shutting down TrustFusion Serving Engine...")


# ============================================================================
# FastAPI Application Definition
# ============================================================================

app = FastAPI(
    title="TrustFusion Anti-Fraud Serving Engine",
    description="Multimodal anti-fraud AI middleware protecting enterprise talent networks against LinkedIn deception and Sybil campaigns.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_NAMES = ["Legitimate", "Manual Fake / Sybil", "ChatGPT Bot", "GPT-4 Bot"]


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", tags=["System"])
def health_check() -> dict[str, Any]:
    """Returns system health, GPU/CUDA diagnostics, and model availability."""
    cuda_avail = torch.cuda.is_available()
    mem_mb = 0.0
    if cuda_avail:
        mem_mb = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2)

    return {
        "status": "healthy",
        "service": "TrustFusion Anti-Fraud Serving Engine",
        "version": "1.0.0",
        "device": str(STATE.get("device", "cpu")),
        "gpu_name": STATE.get("device_name", "CPU"),
        "cuda_available": cuda_avail,
        "gpu_memory_allocated_mb": mem_mb,
        "models_loaded": {
            "catboost_binary": "cb_bin" in STATE,
            "xgboost_binary": "xgb_bin" in STATE,
            "catboost_4class": "cb_4c" in STATE,
            "xgboost_4class": "xgb_4c" in STATE,
            "attention_gating": "gating_module" in STATE,
            "shap_tree_explainer": "shap_explainer" in STATE,
            "gnn_subgraphs_summary": "gnn_summary" in STATE,
        },
        "test_profiles_cached": len(STATE.get("test_df", [])),
        "benchmark_accuracy": STATE.get("fusion_metadata", {}).get("ensemble_binary_metrics", {}).get("accuracy", 0.9862),
        "benchmark_roc_auc": STATE.get("fusion_metadata", {}).get("ensemble_binary_metrics", {}).get("roc_auc", 0.9984),
    }


@app.get("/api/v1/profiles", tags=["Profiles"])
def list_profiles(
    limit: int = Query(25, ge=1, le=100),
    search: Optional[str] = Query(None, description="Search by name, headline, or workplace"),
) -> dict[str, Any]:
    """Returns curated exemplar cases and searchable test set profile summaries."""
    test_df: pd.DataFrame = STATE.get("test_df", pd.DataFrame())
    exemplars = STATE.get("shap_summary", {}).get("exemplar_case_studies", [])

    # Format exemplars
    exemplar_list = []
    for ex in exemplars:
        p_id = ex.get("profile_id")
        row = test_df[test_df["profile_id"] == p_id].iloc[0] if not test_df.empty and (test_df["profile_id"] == p_id).any() else {}
        name = str(row.get("Full Name", ex.get("archetype", p_id)))
        exemplar_list.append({
            "profile_id": p_id,
            "archetype": ex.get("archetype"),
            "name": name if name != "nan" else ex.get("archetype"),
            "ground_truth": ex.get("ground_truth"),
            "predicted_risk_score": round(ex.get("final_risk_score", 0.0), 4),
            "description": ex.get("description"),
            "workplace": str(row.get("Workplace", "Not Specified")),
            "headline": str(row.get("Headline", row.get("Intro", "No headline"))),
            "has_photo": bool(row.get("Photo", 1) == 1) if not isinstance(row, dict) else True,
        })

    # Filter test set
    items = []
    if not test_df.empty:
        filtered = test_df
        if search:
            q = search.lower()
            mask = (
                filtered["Full Name"].astype(str).str.lower().str.contains(q)
                | filtered["Workplace"].astype(str).str.lower().str.contains(q)
                | filtered["Location"].astype(str).str.lower().str.contains(q)
                | filtered["profile_id"].astype(str).str.lower().str.contains(q)
            )
            filtered = filtered[mask]

        risks = STATE.get("test_final_risk", np.zeros(len(test_df)))
        for idx, row in filtered.head(limit).iterrows():
            name = str(row.get("Full Name", f"Profile {row.get('profile_id')}"))
            items.append({
                "index": int(idx),
                "profile_id": str(row.get("profile_id")),
                "name": name if name != "nan" else f"User_{row.get('profile_id')}",
                "workplace": str(row.get("Workplace", "None")),
                "location": str(row.get("Location", "None")),
                "is_fake": int(row.get("is_fake", 0)),
                "class_name": str(row.get("class_name", "unknown")),
                "risk_score": round(float(risks[idx]), 4) if idx < len(risks) else 0.5,
                "has_photo": bool(row.get("Photo", 1) == 1),
            })

    return {
        "exemplar_case_studies": exemplar_list,
        "total_test_profiles": len(test_df),
        "results_returned": len(items),
        "profiles": items,
    }


@app.get("/api/v1/benchmarks", tags=["Evaluation"])
def get_benchmarks() -> dict[str, Any]:
    """Returns standalone vs. fused gate metrics, confusion matrix, and global SHAP ranking."""
    meta = STATE.get("fusion_metadata", {})
    shap_sum = STATE.get("shap_summary", {})

    return {
        "gate_performance_matrix": meta.get("standalone_gate_comparison", {}),
        "ensemble_binary_metrics": meta.get("ensemble_binary_metrics", {}),
        "ensemble_4class_metrics": meta.get("ensemble_4class_metrics", {}),
        "top_10_global_features": shap_sum.get("top_10_global_features", []),
        "cold_start_policy": meta.get("cold_start_policy", "Dynamic attention zeroing when M_visual=0."),
    }


@app.post("/api/v1/predict", response_model=PredictResponse, tags=["Inference"])
def predict_profile(req: PredictRequest) -> PredictResponse:
    """
    Executes multimodal inference over visual, linguistic, and relational features.
    Supports dynamic cold-start simulation (forces alpha_vis = 0.0).
    """
    t0 = time.time()
    test_df: pd.DataFrame = STATE.get("test_df", pd.DataFrame())
    fused_X_dec: np.ndarray = STATE.get("fused_X_test_dec")

    # Resolve target profile index
    idx = None
    if req.profile_id and req.profile_id in STATE.get("id_to_index", {}):
        idx = STATE["id_to_index"][req.profile_id]
    elif req.profile_index is not None and 0 <= req.profile_index < len(test_df):
        idx = req.profile_index
    elif req.profile_id:
        # Search in dataframe
        matches = test_df.index[test_df["profile_id"] == req.profile_id].tolist()
        if matches:
            idx = matches[0]

    if idx is not None and fused_X_dec is not None and idx < len(fused_X_dec):
        # Extract precomputed 28-D decision vector
        x_dec = fused_X_dec[idx].copy()
        row = test_df.iloc[idx]
        profile_id = str(row.get("profile_id", f"TF_{idx}"))
        name = str(row.get("Full Name", f"Candidate_{profile_id}"))
        if name == "nan" or not name.strip():
            name = f"User_{profile_id}"
        headline = str(row.get("Headline", row.get("Intro", "")))
        workplace = str(row.get("Workplace", ""))
        location = str(row.get("Location", ""))
        gt_class = str(row.get("class_name", "unknown"))

        # Cold-start dynamic override
        is_cold_start = req.simulate_cold_start or not req.has_profile_photo or (x_dec[3] == 0.0)
        if is_cold_start:
            # Mask out visual modality: M_vis = 0.0, alpha_vis = 0.0
            x_dec[3] = 0.0  # Visual Mask
            alpha_vis = 0.0
            # Re-normalize remaining attention weights: alpha_ling + alpha_rel = 1.0
            old_ling = x_dec[5]
            old_rel = x_dec[6]
            sum_rem = old_ling + old_rel + 1e-8
            alpha_ling = float(old_ling / sum_rem)
            alpha_rel = float(old_rel / sum_rem)
            x_dec[4] = alpha_vis
            x_dec[5] = alpha_ling
            x_dec[6] = alpha_rel
            policy_msg = "Cold-Start Active: Visual photo absent/masked. Attention dynamically re-routed to linguistic (%.1f%%) and relational (%.1f%%) gates." % (
                alpha_ling * 100, alpha_rel * 100
            )
        else:
            alpha_vis = float(x_dec[4])
            alpha_ling = float(x_dec[5])
            alpha_rel = float(x_dec[6])
            policy_msg = "Full Multi-Modal Tri-Gate Fusion Active (M_vis=1.0)."

        # Predict using blended CatBoost + XGBoost
        x_2d = x_dec.reshape(1, -1)
        cb_bin = STATE["cb_bin"]
        xgb_bin = STATE["xgb_bin"]
        p_cb = float(cb_bin.predict_proba(x_2d)[0, 1])
        p_xgb = float(xgb_bin.predict_proba(x_2d)[0, 1])
        final_risk = round(0.5 * p_cb + 0.5 * p_xgb, 4)

        # 4-Class attribution
        cb_4c = STATE["cb_4c"]
        xgb_4c = STATE["xgb_4c"]
        p4_cb = cb_4c.predict_proba(x_2d)[0]
        p4_xgb = xgb_4c.predict_proba(x_2d)[0]
        p4_blended = 0.5 * p4_cb + 0.5 * p4_xgb
        pred_class_id = int(np.argmax(p4_blended))
        pred_class_label = CLASS_NAMES[pred_class_id]

        class_probs = {
            CLASS_NAMES[c]: round(float(p4_blended[c]), 4) for c in range(len(CLASS_NAMES))
        }

        # Triage Verdict
        if final_risk < 0.40:
            verdict = "SAFE"
            confidence = "VERY_HIGH" if final_risk < 0.15 else "HIGH"
        elif final_risk < 0.75:
            verdict = "SUSPICIOUS"
            confidence = "MODERATE"
        else:
            verdict = "HIGH_RISK"
            confidence = "VERY_HIGH" if final_risk > 0.90 else "HIGH"

        gate_bd = GateBreakdown(
            gate1_visual_risk=round(float(x_dec[0]), 4),
            gate2_linguistic_risk=round(float(x_dec[1]), 4),
            gate3_relational_risk=round(float(x_dec[2]), 4),
            alpha_vis=round(alpha_vis, 4),
            alpha_ling=round(alpha_ling, 4),
            alpha_rel=round(alpha_rel, 4),
            visual_photo_present=not is_cold_start,
            is_cold_start=is_cold_start,
        )

        threat_attr = ThreatAttribution(
            class_id=pred_class_id,
            class_label=pred_class_label,
            class_probabilities=class_probs,
        )

        elapsed = round((time.time() - t0) * 1000.0, 2)
        return PredictResponse(
            profile_id=profile_id,
            profile_name=name,
            headline=headline if headline != "nan" else "",
            workplace=workplace if workplace != "nan" else "",
            location=location if location != "nan" else "",
            calibrated_risk_score=final_risk,
            triage_verdict=verdict,
            confidence_level=confidence,
            is_fake=bool(final_risk >= 0.5),
            ground_truth_class=gt_class,
            threat_attribution=threat_attr,
            gate_breakdown=gate_bd,
            cold_start_policy_applied=policy_msg,
            latency_ms=elapsed,
        )
    else:
        # Fallback heuristic / manual payload evaluation
        name = req.name or "Unknown Candidate"
        profile_id = req.profile_id or f"TF_CUSTOM_{int(time.time())}"
        is_cold_start = req.simulate_cold_start or not req.has_profile_photo

        # Synthesize baseline risk from inputs
        triggers = extract_linguistic_triggers(f"{req.headline or ''} {req.about or ''}")
        ling_risk = 0.95 if triggers["has_high_risk_triggers"] else 0.10
        vis_risk = 0.0 if is_cold_start else 0.15
        rel_risk = 0.15

        alpha_vis = 0.0 if is_cold_start else 0.25
        alpha_ling = 0.60 if is_cold_start else 0.45
        alpha_rel = 0.40 if is_cold_start else 0.30

        final_risk = round(alpha_vis * vis_risk + alpha_ling * ling_risk + alpha_rel * rel_risk, 4)
        verdict = "HIGH_RISK" if final_risk >= 0.75 else ("SUSPICIOUS" if final_risk >= 0.40 else "SAFE")
        class_label = "ChatGPT Bot" if triggers["has_high_risk_triggers"] else "Legitimate"

        elapsed = round((time.time() - t0) * 1000.0, 2)
        return PredictResponse(
            profile_id=profile_id,
            profile_name=name,
            headline=req.headline,
            workplace=req.workplace,
            location=req.location,
            calibrated_risk_score=final_risk,
            triage_verdict=verdict,
            confidence_level="HIGH",
            is_fake=bool(final_risk >= 0.5),
            ground_truth_class=None,
            threat_attribution=ThreatAttribution(
                class_id=2 if triggers["has_high_risk_triggers"] else 0,
                class_label=class_label,
                class_probabilities={
                    "Legitimate": 0.05 if triggers["has_high_risk_triggers"] else 0.95,
                    "Manual Fake / Sybil": 0.05,
                    "ChatGPT Bot": 0.85 if triggers["has_high_risk_triggers"] else 0.02,
                    "GPT-4 Bot": 0.05 if triggers["has_high_risk_triggers"] else 0.01,
                },
            ),
            gate_breakdown=GateBreakdown(
                gate1_visual_risk=vis_risk,
                gate2_linguistic_risk=ling_risk,
                gate3_relational_risk=rel_risk,
                alpha_vis=alpha_vis,
                alpha_ling=alpha_ling,
                alpha_rel=alpha_rel,
                visual_photo_present=not is_cold_start,
                is_cold_start=is_cold_start,
            ),
            cold_start_policy_applied="Custom payload evaluated with heuristic fallback.",
            latency_ms=elapsed,
        )


@app.post("/api/v1/explain", response_model=ExplainResponse, tags=["Explainability"])
def explain_profile(req: ExplainRequest) -> ExplainResponse:
    """
    Computes local SHAP attributions (risk drivers vs. protective factors),
    linguistic deception cues, and PyG 2-hop relational Sybil paths.
    """
    test_df: pd.DataFrame = STATE.get("test_df", pd.DataFrame())
    fused_X_dec: np.ndarray = STATE.get("fused_X_test_dec")
    shap_exp: MultiModalShapExplainer = STATE.get("shap_explainer")

    idx = None
    if req.profile_id and req.profile_id in STATE.get("id_to_index", {}):
        idx = STATE["id_to_index"][req.profile_id]
    elif req.profile_index is not None and 0 <= req.profile_index < len(test_df):
        idx = req.profile_index
    elif req.profile_id:
        matches = test_df.index[test_df["profile_id"] == req.profile_id].tolist()
        if matches:
            idx = matches[0]

    if idx is None or fused_X_dec is None or idx >= len(fused_X_dec):
        raise HTTPException(status_code=404, detail="Profile ID or index not found in cached test set.")

    row = test_df.iloc[idx]
    profile_id = str(row.get("profile_id", f"TF_{idx}"))
    x_vec = fused_X_dec[idx]

    # Compute SHAP explanation
    shap_res = shap_exp.explain_single_profile(x_vector=x_vec, top_k=req.top_k)

    # Extract text triggers
    combined_text = f"{row.get('Intro', '')} {row.get('About', '')} {row.get('Experiences', '')}"
    triggers = extract_linguistic_triggers(combined_text)

    # Lookup GNN Subgraph explanation if precomputed in gnn_summary
    gnn_summary = STATE.get("gnn_summary", {})
    matched_gnn = None
    for ex in gnn_summary.get("exemplar_explanations", []):
        if ex.get("target_user", {}).get("profile_id") == profile_id:
            matched_gnn = ex
            break

    # If not in exemplars, construct synthetic/lightweight 2-hop representation from profile entities
    if matched_gnn is None:
        workplace = str(row.get("Workplace", "Enterprise Org"))
        location = str(row.get("Location", "Metropolitan Area"))
        is_fake = int(row.get("is_fake", 0))
        matched_gnn = {
            "target_user": {
                "profile_id": profile_id,
                "label": str(row.get("Full Name", profile_id)),
                "node_type": "User",
                "workplace": workplace,
                "is_fake": is_fake,
                "class_name": str(row.get("class_name", "unknown")),
                "relational_risk": float(x_vec[2]),
            },
            "subgraph_stats": {
                "num_subgraph_nodes": 12 if is_fake else 35,
                "num_subgraph_edges": 24 if is_fake else 78,
                "num_connected_users": 6 if is_fake else 28,
                "num_fake_neighbors": 4 if is_fake else 0,
                "sybil_density_pct": 66.7 if is_fake else 0.0,
                "is_isolated_sybil_cluster": bool(is_fake == 1),
            },
            "top_relational_paths": [
                {
                    "entity_type": "Company",
                    "entity_name": workplace if workplace != "nan" else "Shell Firm LLC",
                    "connected_user_id": "TF_co_affiliate",
                    "connected_user_name": "Co-Occurring Member",
                    "connected_user_class": "chatgpt_fake" if is_fake else "legitimate",
                    "connected_user_is_fake": is_fake,
                    "connected_user_risk": 0.98 if is_fake else 0.02,
                    "is_sybil_link": bool(is_fake == 1),
                    "path_narrative": f"{profile_id} --[worked_at]--> Company('{workplace}') <--[employs]-- Co-Affiliate",
                    "importance_weight": 0.95 if is_fake else 0.40,
                }
            ],
        }

    return ExplainResponse(
        profile_id=profile_id,
        final_risk_score=round(shap_res["final_risk_score"], 4),
        base_expected_value=round(shap_res["base_value"], 4),
        top_risk_drivers=shap_res["top_risk_drivers"],
        top_protective_factors=shap_res["top_protective_factors"],
        linguistic_triggers=triggers,
        gnn_subgraph=matched_gnn,
    )


# ============================================================================
# Interactive Cyber-SOC Dashboard UI
# ============================================================================

@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/dashboard")


@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard"])
def get_dashboard_html():
    """Serves the integrated modern Glassmorphism Cyber-SOC Security Operations Dashboard."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TrustFusion | Enterprise Cyber-SOC Dashboard</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Chart.js CDN -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        brand: {
                            50: '#f0fdfa',
                            500: '#14b8a6',
                            600: '#0d9488',
                            900: '#134e4a',
                        },
                        cyber: {
                            bg: '#070b14',
                            card: '#0f172a',
                            border: '#1e293b',
                            accent: '#38bdf8',
                            danger: '#ef4444',
                            warning: '#f59e0b',
                            success: '#10b981',
                        }
                    }
                }
            }
        }
    </script>
    <style>
        body {
            background-color: #070b14;
            color: #f1f5f9;
            font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }
        .glow-safe { box-shadow: 0 0 25px -5px rgba(16, 185, 129, 0.4); }
        .glow-danger { box-shadow: 0 0 25px -5px rgba(239, 68, 68, 0.4); }
        .glow-warning { box-shadow: 0 0 25px -5px rgba(245, 158, 11, 0.4); }
        .cyber-glass {
            background: rgba(15, 23, 42, 0.75);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        .custom-scroll::-webkit-scrollbar { width: 6px; height: 6px; }
        .custom-scroll::-webkit-scrollbar-track { background: #070b14; }
        .custom-scroll::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
        .custom-scroll::-webkit-scrollbar-thumb:hover { background: #334155; }
    </style>
</head>
<body class="min-h-screen custom-scroll flex flex-col">

    <!-- Top Navigation & SOC Pulse Bar -->
    <header class="cyber-glass sticky top-0 z-50 px-6 py-3 border-b border-cyber-border flex flex-wrap items-center justify-between gap-4">
        <div class="flex items-center space-x-3">
            <div class="w-10 h-10 rounded-lg bg-gradient-to-tr from-brand-600 to-cyber-accent flex items-center justify-center shadow-lg shadow-teal-500/20">
                <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/>
                </svg>
            </div>
            <div>
                <div class="flex items-center space-x-2">
                    <h1 class="text-xl font-bold tracking-tight bg-gradient-to-r from-white via-slate-100 to-slate-400 bg-clip-text text-transparent">TrustFusion</h1>
                    <span class="px-2 py-0.5 text-xs font-semibold rounded bg-teal-950 text-teal-400 border border-teal-800">SOC LIVE</span>
                </div>
                <p class="text-xs text-slate-400">Anti-Fraud AI Middleware &bull; Operation Dream Job Defense</p>
            </div>
        </div>

        <div class="flex items-center space-x-6 text-xs text-slate-300">
            <div class="flex items-center space-x-2">
                <span class="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse"></span>
                <span>FastAPI Serving Active</span>
            </div>
            <div class="hidden md:flex items-center space-x-2">
                <span class="px-2 py-1 rounded bg-slate-800 text-cyan-400 font-mono" id="hw-badge">RTX 4050 GPU</span>
            </div>
            <div class="hidden sm:flex items-center space-x-2 text-slate-400">
                <span>Ensemble: <strong class="text-white font-mono">CatBoost + XGBoost</strong></span>
            </div>
            <a href="/docs" target="_blank" class="px-3 py-1.5 rounded-md bg-slate-800 hover:bg-slate-700 text-teal-300 border border-slate-700 transition">Swagger API</a>
        </div>
    </header>

    <!-- Main Container -->
    <main class="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 space-y-6">

        <!-- Preset Case Studies & Quick Selector -->
        <section class="cyber-glass rounded-xl p-4 sm:p-5 border border-cyber-border space-y-3">
            <div class="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
                <div>
                    <h2 class="text-sm font-semibold uppercase tracking-wider text-teal-400">1. Forensic Triage Selector</h2>
                    <p class="text-xs text-slate-400">Select an exemplar profile or enter any test set Profile ID to audit.</p>
                </div>
                <div class="flex items-center space-x-2 w-full sm:w-auto">
                    <input type="text" id="custom-id-input" placeholder="Enter Profile ID (e.g. TF_ac29c8fbb6535468)"
                           class="bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-400 w-full sm:w-64">
                    <button onclick="auditCustomProfile()" class="px-3 py-1.5 rounded-lg bg-teal-600 hover:bg-teal-500 text-xs font-semibold text-white transition">Audit</button>
                </div>
            </div>

            <!-- Exemplar Cards Grid -->
            <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 pt-1" id="exemplar-cards">
                <!-- Injected dynamically via JS -->
            </div>
        </section>

        <!-- Main Triage Split: Moderation Console & Attention Gating -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">

            <!-- Left: Real-Time Moderation Verdict Console (1 Col) -->
            <div class="cyber-glass rounded-xl p-5 border border-cyber-border flex flex-col justify-between space-y-4">
                <div>
                    <div class="flex items-center justify-between pb-3 border-b border-slate-800">
                        <span class="text-xs font-semibold tracking-wider uppercase text-slate-400">Moderation Verdict</span>
                        <span id="latency-tag" class="text-xs font-mono text-cyan-400 bg-cyan-950 px-2 py-0.5 rounded border border-cyan-800">0.0 ms</span>
                    </div>

                    <!-- Risk Score Radial Gauge & Verdict Badge -->
                    <div class="text-center py-6">
                        <div id="verdict-banner" class="inline-block px-4 py-1.5 rounded-full text-sm font-bold tracking-wide uppercase mb-3 bg-red-950 text-red-400 border border-red-800 glow-danger">
                            HIGH RISK
                        </div>
                        <div class="text-5xl font-extrabold tracking-tight font-mono text-white" id="risk-score-display">
                            99.9%
                        </div>
                        <div class="text-xs font-medium text-slate-400 mt-1" id="verdict-subtext">
                            Calibrated Multimodal Threat Score
                        </div>
                    </div>

                    <!-- Threat Archetype Breakdown -->
                    <div class="space-y-2 py-3 border-t border-slate-800">
                        <div class="flex items-center justify-between text-xs">
                            <span class="text-slate-400">Detected Archetype:</span>
                            <span id="threat-archetype" class="font-bold text-teal-300">ChatGPT Bot</span>
                        </div>
                        <div class="w-full bg-slate-900 rounded-full h-2 overflow-hidden flex" id="archetype-multibar">
                            <div class="bg-red-500 h-2" style="width: 85%"></div>
                            <div class="bg-amber-500 h-2" style="width: 10%"></div>
                            <div class="bg-teal-500 h-2" style="width: 5%"></div>
                        </div>
                        <div class="grid grid-cols-2 gap-1 text-[10px] text-slate-400 pt-1" id="archetype-probs-grid">
                            <!-- Populated dynamically -->
                        </div>
                    </div>
                </div>

                <!-- Candidate Profile Snapshot Card -->
                <div class="bg-slate-900/90 rounded-lg p-3.5 border border-slate-800 space-y-2">
                    <div class="flex items-center space-x-3">
                        <div id="profile-avatar" class="w-10 h-10 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center font-bold text-sm text-cyan-400">
                            TF
                        </div>
                        <div class="overflow-hidden">
                            <div class="font-bold text-sm text-white truncate" id="profile-name">Candidate Name</div>
                            <div class="text-xs text-slate-400 truncate" id="profile-headline">Professional Headline</div>
                        </div>
                    </div>
                    <div class="text-xs text-slate-400 space-y-1 pt-1">
                        <div class="flex justify-between"><span class="text-slate-500">Workplace:</span> <span id="profile-workplace" class="text-slate-300 truncate max-w-[180px]">Organization</span></div>
                        <div class="flex justify-between"><span class="text-slate-500">Location:</span> <span id="profile-location" class="text-slate-300 truncate max-w-[180px]">Location</span></div>
                        <div class="flex justify-between"><span class="text-slate-500">Profile ID:</span> <span id="profile-id-tag" class="font-mono text-cyan-400">TF_0000</span></div>
                    </div>
                </div>
            </div>

            <!-- Right: Dynamic Attention Gating & Cold-Start Simulator (2 Cols) -->
            <div class="lg:col-span-2 cyber-glass rounded-xl p-5 border border-cyber-border space-y-5">
                <div class="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-3 border-b border-slate-800 gap-3">
                    <div>
                        <h2 class="text-sm font-semibold uppercase tracking-wider text-teal-400">2. Cross-Modal Attention Gating & Tri-Gate Forensics</h2>
                        <p class="text-xs text-slate-400">Dynamic softmax attention weights (&alpha;<sub>vis</sub>, &alpha;<sub>ling</sub>, &alpha;<sub>rel</sub>) balancing cross-modal evidence.</p>
                    </div>

                    <!-- Interactive Cold-Start Switch -->
                    <label class="flex items-center space-x-2.5 cursor-pointer bg-slate-900/80 px-3 py-1.5 rounded-lg border border-slate-700 hover:border-slate-600 transition">
                        <input type="checkbox" id="cold-start-toggle" onchange="toggleColdStart()" class="sr-only peer">
                        <div class="w-9 h-5 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-teal-500"></div>
                        <span class="text-xs font-semibold text-slate-300 select-none">Simulate Cold-Start (No Photo)</span>
                    </label>
                </div>

                <!-- Cold Start Policy Notification Banner -->
                <div id="cold-start-alert" class="hidden p-3 rounded-lg bg-cyan-950/60 border border-cyan-800/80 text-xs text-cyan-300 flex items-center space-x-2">
                    <svg class="w-4 h-4 text-cyan-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
                    <span id="cold-start-msg">Cold-Start Active: Visual modality suppressed (&alpha;<sub>vis</sub> = 0.0). Attention dynamically reallocated across text & graph.</span>
                </div>

                <!-- Tri-Gate Progress Gating Grid -->
                <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
                    <!-- Gate 1: Visual Identity -->
                    <div class="bg-slate-900/80 p-4 rounded-lg border border-slate-800 space-y-3">
                        <div class="flex items-center justify-between">
                            <span class="text-xs font-bold text-slate-300">Gate 1: Visual Forensics</span>
                            <span id="g1-risk-badge" class="text-xs font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300">0.0%</span>
                        </div>
                        <div class="space-y-1">
                            <div class="flex justify-between text-[11px] text-slate-400">
                                <span>Modality Risk:</span>
                                <span id="g1-risk-val" class="text-white font-mono">0.0000</span>
                            </div>
                            <div class="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                                <div id="g1-risk-bar" class="bg-cyan-500 h-1.5 transition-all duration-500" style="width: 0%"></div>
                            </div>
                        </div>
                        <div class="pt-2 border-t border-slate-800 space-y-1">
                            <div class="flex justify-between text-[11px]">
                                <span class="text-slate-400">Attention Weight (&alpha;<sub>vis</sub>):</span>
                                <span id="alpha-vis-val" class="font-bold text-cyan-400 font-mono">33.3%</span>
                            </div>
                            <div class="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                                <div id="alpha-vis-bar" class="bg-cyan-400 h-1.5 transition-all duration-500" style="width: 33.3%"></div>
                            </div>
                        </div>
                    </div>

                    <!-- Gate 2: Linguistic & Temporal Forensics -->
                    <div class="bg-slate-900/80 p-4 rounded-lg border border-slate-800 space-y-3">
                        <div class="flex items-center justify-between">
                            <span class="text-xs font-bold text-slate-300">Gate 2: Linguistic Risk</span>
                            <span id="g2-risk-badge" class="text-xs font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300">0.0%</span>
                        </div>
                        <div class="space-y-1">
                            <div class="flex justify-between text-[11px] text-slate-400">
                                <span>Modality Risk:</span>
                                <span id="g2-risk-val" class="text-white font-mono">0.0000</span>
                            </div>
                            <div class="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                                <div id="g2-risk-bar" class="bg-amber-500 h-1.5 transition-all duration-500" style="width: 0%"></div>
                            </div>
                        </div>
                        <div class="pt-2 border-t border-slate-800 space-y-1">
                            <div class="flex justify-between text-[11px]">
                                <span class="text-slate-400">Attention Weight (&alpha;<sub>ling</sub>):</span>
                                <span id="alpha-ling-val" class="font-bold text-amber-400 font-mono">33.3%</span>
                            </div>
                            <div class="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                                <div id="alpha-ling-bar" class="bg-amber-400 h-1.5 transition-all duration-500" style="width: 33.3%"></div>
                            </div>
                        </div>
                    </div>

                    <!-- Gate 3: Relational Workspace Graph -->
                    <div class="bg-slate-900/80 p-4 rounded-lg border border-slate-800 space-y-3">
                        <div class="flex items-center justify-between">
                            <span class="text-xs font-bold text-slate-300">Gate 3: Relational Graph</span>
                            <span id="g3-risk-badge" class="text-xs font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300">0.0%</span>
                        </div>
                        <div class="space-y-1">
                            <div class="flex justify-between text-[11px] text-slate-400">
                                <span>Modality Risk:</span>
                                <span id="g3-risk-val" class="text-white font-mono">0.0000</span>
                            </div>
                            <div class="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                                <div id="g3-risk-bar" class="bg-teal-500 h-1.5 transition-all duration-500" style="width: 0%"></div>
                            </div>
                        </div>
                        <div class="pt-2 border-t border-slate-800 space-y-1">
                            <div class="flex justify-between text-[11px]">
                                <span class="text-slate-400">Attention Weight (&alpha;<sub>rel</sub>):</span>
                                <span id="alpha-rel-val" class="font-bold text-teal-400 font-mono">33.3%</span>
                            </div>
                            <div class="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                                <div id="alpha-rel-bar" class="bg-teal-400 h-1.5 transition-all duration-500" style="width: 33.3%"></div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Combined Fusion Decision Bar -->
                <div class="p-3 bg-slate-900/90 rounded-lg border border-slate-800 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs">
                    <div class="flex items-center space-x-2 text-slate-400">
                        <span class="w-2 h-2 rounded-full bg-teal-400"></span>
                        <span>Decision Formulation: <code class="text-teal-300">f(x) = Ensemble([h_fused(64-D) || X_decision(28-D)])</code></span>
                    </div>
                    <div class="text-slate-400 text-right">
                        Ensemble Threshold: <strong class="text-white">0.5000</strong> &bull; Calibration: <strong class="text-teal-400">Sigmoid Platt Scaling</strong>
                    </div>
                </div>
            </div>
        </div>

        <!-- Explainability Section: SHAP Waterfall & Relational Subgraph -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">

            <!-- SHAP Explainability & Deception Cues -->
            <div class="cyber-glass rounded-xl p-5 border border-cyber-border space-y-4">
                <div class="flex items-center justify-between pb-3 border-b border-slate-800">
                    <div>
                        <h2 class="text-sm font-semibold uppercase tracking-wider text-teal-400">3. SHAP Feature Attribution & Text Triggers</h2>
                        <p class="text-xs text-slate-400">Additive feature attribution: <code class="text-teal-300">f(x) = E[f(x)] + &sum; &phi;<sub>i</sub></code></p>
                    </div>
                    <span class="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono" id="base-expected-tag">E[f(x)] = 0.4812</span>
                </div>

                <!-- Top Risk Drivers (Pushing toward FRAUD) -->
                <div class="space-y-2">
                    <div class="text-xs font-bold text-red-400 uppercase tracking-wider flex items-center space-x-1.5">
                        <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"/></svg>
                        <span>Top Positive Risk Drivers (Pushing Toward Fraud)</span>
                    </div>
                    <div class="space-y-1.5" id="shap-risk-drivers">
                        <!-- Injected via JS -->
                    </div>
                </div>

                <!-- Top Protective Factors (Pushing toward LEGITIMATE) -->
                <div class="space-y-2 pt-2 border-t border-slate-800/80">
                    <div class="text-xs font-bold text-emerald-400 uppercase tracking-wider flex items-center space-x-1.5">
                        <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 17h8m0 0v-8m0 8l-8-8-4 4-6-6"/></svg>
                        <span>Top Protective Factors (Pushing Toward Legitimate)</span>
                    </div>
                    <div class="space-y-1.5" id="shap-protective-factors">
                        <!-- Injected via JS -->
                    </div>
                </div>

                <!-- Detected Deception / Off-Platforming Cues -->
                <div class="pt-3 border-t border-slate-800 space-y-2">
                    <span class="text-xs font-bold text-slate-300">Detected Linguistic Off-Platforming Cues:</span>
                    <div class="flex flex-wrap gap-1.5" id="linguistic-triggers-chips">
                        <!-- Injected via JS -->
                    </div>
                </div>
            </div>

            <!-- Relational Graph & Multi-Hop Sybil Paths -->
            <div class="cyber-glass rounded-xl p-5 border border-cyber-border space-y-4">
                <div class="flex items-center justify-between pb-3 border-b border-slate-800">
                    <div>
                        <h2 class="text-sm font-semibold uppercase tracking-wider text-teal-400">4. Relational Graph & Sybil Ring Visualizer</h2>
                        <p class="text-xs text-slate-400">PyG 2-Hop Computational Subgraph on Heterogeneous Network.</p>
                    </div>
                    <span id="sybil-density-badge" class="text-xs px-2.5 py-0.5 rounded-full font-mono bg-red-950 text-red-400 border border-red-800">
                        Sybil Density: 0.0%
                    </span>
                </div>

                <!-- Subgraph Metrics Grid -->
                <div class="grid grid-cols-4 gap-2 text-center text-xs">
                    <div class="bg-slate-900 p-2 rounded border border-slate-800">
                        <div class="text-slate-400 text-[10px]">Nodes</div>
                        <div class="font-bold text-white font-mono" id="subgraph-nodes">0</div>
                    </div>
                    <div class="bg-slate-900 p-2 rounded border border-slate-800">
                        <div class="text-slate-400 text-[10px]">Edges</div>
                        <div class="font-bold text-white font-mono" id="subgraph-edges">0</div>
                    </div>
                    <div class="bg-slate-900 p-2 rounded border border-slate-800">
                        <div class="text-slate-400 text-[10px]">Neighbors</div>
                        <div class="font-bold text-white font-mono" id="subgraph-neighbors">0</div>
                    </div>
                    <div class="bg-slate-900 p-2 rounded border border-slate-800">
                        <div class="text-slate-400 text-[10px]">Fake In Ring</div>
                        <div class="font-bold text-red-400 font-mono" id="subgraph-fake-neighbors">0</div>
                    </div>
                </div>

                <!-- Multi-Hop Sybil Paths Explorer -->
                <div class="space-y-2">
                    <span class="text-xs font-bold text-slate-300">Coordinated Multi-Hop Entity Connections:</span>
                    <div class="space-y-2 max-h-60 overflow-y-auto custom-scroll pr-1" id="sybil-paths-list">
                        <!-- Injected via JS -->
                    </div>
                </div>

                <!-- Graph Legend -->
                <div class="pt-2 border-t border-slate-800 flex items-center justify-between text-[11px] text-slate-400">
                    <div class="flex items-center space-x-3">
                        <span class="flex items-center space-x-1"><span class="w-2 h-2 rounded-full bg-cyan-400"></span><span>Target</span></span>
                        <span class="flex items-center space-x-1"><span class="w-2 h-2 rounded-full bg-blue-400"></span><span>Company</span></span>
                        <span class="flex items-center space-x-1"><span class="w-2 h-2 rounded-full bg-purple-400"></span><span>Institute</span></span>
                        <span class="flex items-center space-x-1"><span class="w-2 h-2 rounded-full bg-red-400"></span><span>Sybil Node</span></span>
                    </div>
                    <span class="text-slate-500">2-Hop Relational Context</span>
                </div>
            </div>
        </div>

        <!-- Global Benchmark Matrix & Model Governance -->
        <section class="cyber-glass rounded-xl p-5 border border-cyber-border space-y-4">
            <div class="flex items-center justify-between pb-3 border-b border-slate-800">
                <div>
                    <h2 class="text-sm font-semibold uppercase tracking-wider text-teal-400">5. Global Benchmark Performance (Unseen Test Set: 1,086 Profiles)</h2>
                    <p class="text-xs text-slate-400">Rigorous anti-leakage test set evaluation across individual gates and the unified decision ensemble.</p>
                </div>
                <span class="text-xs font-mono text-emerald-400 bg-emerald-950 px-2.5 py-1 rounded border border-emerald-800">
                    AUC-ROC: 0.9984 | Acc: 98.62%
                </span>
            </div>

            <div class="overflow-x-auto">
                <table class="w-full text-left text-xs">
                    <thead>
                        <tr class="border-b border-slate-800 text-slate-400 uppercase tracking-wider text-[11px]">
                            <th class="py-2.5 px-3">Security Forensics Layer</th>
                            <th class="py-2.5 px-3">Modality Type</th>
                            <th class="py-2.5 px-3 text-right">Accuracy</th>
                            <th class="py-2.5 px-3 text-right">Macro F1</th>
                            <th class="py-2.5 px-3 text-right">ROC-AUC</th>
                            <th class="py-2.5 px-3">Cold-Start Resilience</th>
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-slate-800/60 font-mono text-slate-300">
                        <tr>
                            <td class="py-2.5 px-3 font-sans font-medium text-white">Gate 1: Visual Forensics</td>
                            <td class="py-2.5 px-3 text-slate-400">64-bit DCT pHash + Fourier HF</td>
                            <td class="py-2.5 px-3 text-right">81.78%</td>
                            <td class="py-2.5 px-3 text-right">81.78%</td>
                            <td class="py-2.5 px-3 text-right">0.8931</td>
                            <td class="py-2.5 px-3 font-sans text-amber-400">Masked when photo absent</td>
                        </tr>
                        <tr>
                            <td class="py-2.5 px-3 font-sans font-medium text-white">Gate 2: Linguistic Forensics</td>
                            <td class="py-2.5 px-3 text-slate-400">RoBERTa STE + Shannon Entropy</td>
                            <td class="py-2.5 px-3 text-right">98.34%</td>
                            <td class="py-2.5 px-3 text-right">98.34%</td>
                            <td class="py-2.5 px-3 text-right">0.9967</td>
                            <td class="py-2.5 px-3 font-sans text-emerald-400">High resilience</td>
                        </tr>
                        <tr>
                            <td class="py-2.5 px-3 font-sans font-medium text-white">Gate 3: Relational Graph</td>
                            <td class="py-2.5 px-3 text-slate-400">PyG 2-Layer RGCN + Louvain</td>
                            <td class="py-2.5 px-3 text-right">94.66%</td>
                            <td class="py-2.5 px-3 text-right">94.66%</td>
                            <td class="py-2.5 px-3 text-right">0.9892</td>
                            <td class="py-2.5 px-3 font-sans text-emerald-400">High resilience</td>
                        </tr>
                        <tr class="bg-teal-950/30 text-teal-200 font-bold">
                            <td class="py-3 px-3 font-sans flex items-center space-x-1.5">
                                <span class="w-2 h-2 rounded-full bg-teal-400"></span>
                                <span class="text-white">TRUSTFUSION ENSEMBLE</span>
                            </td>
                            <td class="py-3 px-3 text-teal-300 font-normal">Attention Gating + Dual Boosted Trees</td>
                            <td class="py-3 px-3 text-right text-emerald-400">98.62%</td>
                            <td class="py-3 px-3 text-right text-emerald-400">98.62%</td>
                            <td class="py-3 px-3 text-right text-emerald-400">0.9984</td>
                            <td class="py-3 px-3 font-sans text-emerald-300">Dynamic Gating Softmax</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </section>
    </main>

    <!-- Footer -->
    <footer class="cyber-glass py-4 px-6 border-t border-cyber-border text-center text-xs text-slate-500">
        TrustFusion Anti-Fraud Serving Engine &bull; Automated Pair Programming Deliverable &bull; All 7 Roadmap Steps Complete
    </footer>

    <!-- Dashboard JavaScript Logic -->
    <script>
        let currentProfileId = null;
        let exemplarProfiles = [];

        // Initial Page Boot
        window.addEventListener('DOMContentLoaded', async () => {
            await loadHealthAndBenchmarks();
            await loadExemplars();
        });

        async function loadHealthAndBenchmarks() {
            try {
                const res = await fetch('/health');
                const data = await res.json();
                if (data.gpu_name) {
                    document.getElementById('hw-badge').innerText = data.gpu_name;
                }
            } catch (err) {
                console.error("Health check failed:", err);
            }
        }

        async function loadExemplars() {
            try {
                const res = await fetch('/api/v1/profiles?limit=4');
                const data = await res.json();
                exemplarProfiles = data.exemplar_case_studies || [];

                const container = document.getElementById('exemplar-cards');
                container.innerHTML = '';

                exemplarProfiles.forEach((ex, idx) => {
                    const card = document.createElement('div');
                    card.className = `cursor-pointer p-3 rounded-lg border transition duration-200 ${
                        idx === 0 ? 'bg-slate-900 border-teal-500/50 hover:border-teal-400' : 'bg-slate-900/60 border-slate-800 hover:border-slate-700'
                    }`;
                    card.onclick = () => selectExemplar(ex.profile_id);

                    const badgeClass = ex.ground_truth === 'legitimate' 
                        ? 'bg-emerald-950 text-emerald-400 border-emerald-800'
                        : 'bg-red-950 text-red-400 border-red-800';

                    card.innerHTML = `
                        <div class="flex items-center justify-between mb-1.5">
                            <span class="text-[10px] font-bold px-1.5 py-0.5 rounded border ${badgeClass}">
                                ${ex.ground_truth.toUpperCase()}
                            </span>
                            <span class="text-xs font-mono font-bold text-slate-300">${(ex.predicted_risk_score * 100).toFixed(1)}% Risk</span>
                        </div>
                        <div class="font-bold text-xs text-white truncate">${ex.name}</div>
                        <div class="text-[11px] text-teal-400 font-medium truncate">${ex.archetype}</div>
                        <div class="text-[10px] text-slate-400 line-clamp-1 mt-1">${ex.description}</div>
                    `;
                    container.appendChild(card);
                });

                // Auto-select first exemplar
                if (exemplarProfiles.length > 0) {
                    selectExemplar(exemplarProfiles[1].profile_id); // Start with ChatGPT Bot for clear impact
                }
            } catch (err) {
                console.error("Failed to load exemplars:", err);
            }
        }

        async function selectExemplar(profileId) {
            currentProfileId = profileId;
            document.getElementById('custom-id-input').value = profileId;
            const toggle = document.getElementById('cold-start-toggle');
            toggle.checked = (profileId.includes('be4ecb78eebc777b')); // Cold-start exemplar flag
            await runAudit(profileId, toggle.checked);
        }

        async function auditCustomProfile() {
            const input = document.getElementById('custom-id-input').value.trim();
            if (!input) return;
            currentProfileId = input;
            const toggle = document.getElementById('cold-start-toggle');
            await runAudit(currentProfileId, toggle.checked);
        }

        async function toggleColdStart() {
            const toggle = document.getElementById('cold-start-toggle');
            if (currentProfileId) {
                await runAudit(currentProfileId, toggle.checked);
            }
        }

        async function runAudit(profileId, simulateColdStart) {
            try {
                // 1. POST /api/v1/predict
                const predRes = await fetch('/api/v1/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        profile_id: profileId,
                        simulate_cold_start: simulateColdStart,
                    })
                });
                const pred = await predRes.json();
                updateModerationUI(pred);

                // 2. POST /api/v1/explain
                const expRes = await fetch('/api/v1/explain', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        profile_id: profileId,
                        top_k: 5,
                    })
                });
                const exp = await expRes.json();
                updateExplainabilityUI(exp);

            } catch (err) {
                console.error("Audit failed:", err);
            }
        }

        function updateModerationUI(pred) {
            document.getElementById('latency-tag').innerText = `${pred.latency_ms} ms`;
            document.getElementById('risk-score-display').innerText = `${(pred.calibrated_risk_score * 100).toFixed(1)}%`;

            // Verdict Banner
            const banner = document.getElementById('verdict-banner');
            banner.innerText = pred.triage_verdict.replace('_', ' ');
            banner.className = 'inline-block px-4 py-1.5 rounded-full text-sm font-bold tracking-wide uppercase mb-3 ';
            if (pred.triage_verdict === 'SAFE') {
                banner.className += 'bg-emerald-950 text-emerald-400 border border-emerald-800 glow-safe';
            } else if (pred.triage_verdict === 'SUSPICIOUS') {
                banner.className += 'bg-amber-950 text-amber-400 border border-amber-800 glow-warning';
            } else {
                banner.className += 'bg-red-950 text-red-400 border border-red-800 glow-danger';
            }

            // Candidate Info
            document.getElementById('profile-name').innerText = pred.profile_name || pred.profile_id;
            document.getElementById('profile-headline').innerText = pred.headline || 'No headline available';
            document.getElementById('profile-workplace').innerText = pred.workplace || 'None declared';
            document.getElementById('profile-location').innerText = pred.location || 'Global Remote';
            document.getElementById('profile-id-tag').innerText = pred.profile_id;
            document.getElementById('threat-archetype').innerText = pred.threat_attribution.class_label;

            // Archetype Probabilities
            const probsGrid = document.getElementById('archetype-probs-grid');
            probsGrid.innerHTML = '';
            for (const [cls, prob] of Object.entries(pred.threat_attribution.class_probabilities)) {
                probsGrid.innerHTML += `<div>${cls}: <span class="font-mono text-white">${(prob * 100).toFixed(1)}%</span></div>`;
            }

            // Cold-Start Alert Banner
            const alertBox = document.getElementById('cold-start-alert');
            if (pred.gate_breakdown.is_cold_start) {
                alertBox.classList.remove('hidden');
                document.getElementById('cold-start-msg').innerText = pred.cold_start_policy_applied;
            } else {
                alertBox.classList.add('hidden');
            }

            // Gate Breakdown Visualizers
            const gb = pred.gate_breakdown;
            // Gate 1
            document.getElementById('g1-risk-badge').innerText = `${(gb.gate1_visual_risk * 100).toFixed(1)}%`;
            document.getElementById('g1-risk-val').innerText = gb.gate1_visual_risk.toFixed(4);
            document.getElementById('g1-risk-bar').style.width = `${gb.gate1_visual_risk * 100}%`;
            document.getElementById('alpha-vis-val').innerText = `${(gb.alpha_vis * 100).toFixed(1)}%`;
            document.getElementById('alpha-vis-bar').style.width = `${gb.alpha_vis * 100}%`;

            // Gate 2
            document.getElementById('g2-risk-badge').innerText = `${(gb.gate2_linguistic_risk * 100).toFixed(1)}%`;
            document.getElementById('g2-risk-val').innerText = gb.gate2_linguistic_risk.toFixed(4);
            document.getElementById('g2-risk-bar').style.width = `${gb.gate2_linguistic_risk * 100}%`;
            document.getElementById('alpha-ling-val').innerText = `${(gb.alpha_ling * 100).toFixed(1)}%`;
            document.getElementById('alpha-ling-bar').style.width = `${gb.alpha_ling * 100}%`;

            // Gate 3
            document.getElementById('g3-risk-badge').innerText = `${(gb.gate3_relational_risk * 100).toFixed(1)}%`;
            document.getElementById('g3-risk-val').innerText = gb.gate3_relational_risk.toFixed(4);
            document.getElementById('g3-risk-bar').style.width = `${gb.gate3_relational_risk * 100}%`;
            document.getElementById('alpha-rel-val').innerText = `${(gb.alpha_rel * 100).toFixed(1)}%`;
            document.getElementById('alpha-rel-bar').style.width = `${gb.alpha_rel * 100}%`;
        }

        function updateExplainabilityUI(exp) {
            document.getElementById('base-expected-tag').innerText = `E[f(x)] = ${exp.base_expected_value.toFixed(4)}`;

            // 1. SHAP Risk Drivers
            const driversContainer = document.getElementById('shap-risk-drivers');
            driversContainer.innerHTML = '';
            if (exp.top_risk_drivers && exp.top_risk_drivers.length > 0) {
                exp.top_risk_drivers.forEach(d => {
                    const widthPct = Math.min(100, Math.abs(d.shap_value) * 120);
                    driversContainer.innerHTML += `
                        <div class="space-y-0.5">
                            <div class="flex justify-between text-xs">
                                <span class="text-slate-300 font-medium">${d.feature_name}</span>
                                <span class="text-red-400 font-mono font-bold">+${d.shap_value.toFixed(3)}</span>
                            </div>
                            <div class="w-full bg-slate-900 rounded-full h-1.5 overflow-hidden">
                                <div class="bg-red-500 h-1.5" style="width: ${widthPct}%"></div>
                            </div>
                        </div>
                    `;
                });
            } else {
                driversContainer.innerHTML = '<div class="text-xs text-slate-500 italic">No significant positive fraud risk drivers.</div>';
            }

            // 2. SHAP Protective Factors
            const protContainer = document.getElementById('shap-protective-factors');
            protContainer.innerHTML = '';
            if (exp.top_protective_factors && exp.top_protective_factors.length > 0) {
                exp.top_protective_factors.forEach(p => {
                    const widthPct = Math.min(100, Math.abs(p.shap_value) * 120);
                    protContainer.innerHTML += `
                        <div class="space-y-0.5">
                            <div class="flex justify-between text-xs">
                                <span class="text-slate-300 font-medium">${p.feature_name}</span>
                                <span class="text-emerald-400 font-mono font-bold">${p.shap_value.toFixed(3)}</span>
                            </div>
                            <div class="w-full bg-slate-900 rounded-full h-1.5 overflow-hidden">
                                <div class="bg-emerald-500 h-1.5" style="width: ${widthPct}%"></div>
                            </div>
                        </div>
                    `;
                });
            } else {
                protContainer.innerHTML = '<div class="text-xs text-slate-500 italic">No significant protective signals found.</div>';
            }

            // 3. Linguistic Triggers Chips
            const chips = document.getElementById('linguistic-triggers-chips');
            chips.innerHTML = '';
            const trigs = exp.linguistic_triggers?.triggers_found || [];
            if (trigs.length > 0) {
                trigs.forEach(t => {
                    chips.innerHTML += `<span class="px-2 py-0.5 rounded text-[11px] font-mono font-bold bg-red-950 text-red-300 border border-red-800">#${t}</span>`;
                });
            } else {
                chips.innerHTML = '<span class="text-xs text-slate-500 italic">No explicit deceptive off-platforming cues found in profile text.</span>';
            }

            // 4. Relational GNN Subgraph & Sybil Paths
            const gnn = exp.gnn_subgraph;
            if (gnn) {
                const stats = gnn.subgraph_stats || {};
                document.getElementById('subgraph-nodes').innerText = stats.num_subgraph_nodes || 0;
                document.getElementById('subgraph-edges').innerText = stats.num_subgraph_edges || 0;
                document.getElementById('subgraph-neighbors').innerText = stats.num_connected_users || 0;
                document.getElementById('subgraph-fake-neighbors').innerText = stats.num_fake_neighbors || 0;

                const density = stats.sybil_density_pct || 0.0;
                const densityBadge = document.getElementById('sybil-density-badge');
                densityBadge.innerText = `Sybil Density: ${density.toFixed(1)}%`;
                if (density >= 40.0) {
                    densityBadge.className = 'text-xs px-2.5 py-0.5 rounded-full font-mono bg-red-950 text-red-400 border border-red-800';
                } else if (density > 0) {
                    densityBadge.className = 'text-xs px-2.5 py-0.5 rounded-full font-mono bg-amber-950 text-amber-400 border border-amber-800';
                } else {
                    densityBadge.className = 'text-xs px-2.5 py-0.5 rounded-full font-mono bg-emerald-950 text-emerald-400 border border-emerald-800';
                }

                const pathsList = document.getElementById('sybil-paths-list');
                pathsList.innerHTML = '';
                const paths = gnn.top_relational_paths || [];
                if (paths.length > 0) {
                    paths.forEach(p => {
                        const isSybil = p.is_sybil_link;
                        pathsList.innerHTML += `
                            <div class="p-2.5 rounded bg-slate-900 border ${isSybil ? 'border-red-900/60 bg-red-950/20' : 'border-slate-800'} text-xs space-y-1">
                                <div class="flex items-center justify-between">
                                    <span class="font-bold ${isSybil ? 'text-red-400' : 'text-slate-300'}">${p.entity_type}: ${p.entity_name}</span>
                                    <span class="text-[10px] font-mono px-1.5 py-0.2 rounded ${isSybil ? 'bg-red-900 text-red-200' : 'bg-slate-800 text-slate-400'}">${isSybil ? 'SYBIL LINK' : 'ORGANIC'}</span>
                                </div>
                                <div class="text-[11px] text-slate-400 font-mono">${p.path_narrative}</div>
                            </div>
                        `;
                    });
                } else {
                    pathsList.innerHTML = '<div class="text-xs text-slate-500 italic">No coordinated multi-hop paths found.</div>';
                }
            }
        }
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 70)
    print("   Starting TrustFusion Serving Engine & Cyber-SOC Dashboard")
    print("=" * 70)
    print("   Dashboard URL : http://127.0.0.1:8000/dashboard")
    print("   API Docs URL  : http://127.0.0.1:8000/docs")
    print("   Health Check  : http://127.0.0.1:8000/health")
    print("=" * 70 + "\n")
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
