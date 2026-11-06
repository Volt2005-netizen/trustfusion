"""
TrustFusion - Step 6: Multi-Modal SHAP Explainability Module.

Responsibilities:
-----------------
1. Decision Ensemble Feature Attribution:
   - Uses TreeExplainer to compute exact SHAP values for CatBoost and XGBoost
     over the 28-D decision feature representations (Gate risks, attention weights,
     visual mask, Louvain topology, and structural profile counts).
   - Blends SHAP values 50/50 across both gradient boosted models to provide
     unbiased additive explanations: f(x) = E[f(x)] + sum(phi_i).
2. Global Feature Importance:
   - Computes mean absolute SHAP values across all 1,086 unseen test profiles.
   - Generates ranked forensic importance tables and saves metadata.
3. Local Profile Drilldowns & Waterfall Attributions:
   - Decomposes individual profile fraud risk into positive risk drivers
     (pushing toward FRAUD) and negative protective factors (pushing toward LEGITIMATE).
4. Linguistic / Text-Level Forensic Highlighting:
   - Extracts token-level triggers and off-platforming cues (WhatsApp, Telegram,
     crypto, urgency) from profile text to complement tabular attribution.
5. Serialization & Examiner Demonstration:
   - Saves SHAP matrices and explanation summaries to data/processed/fusion/.
   - Supports instant examiner demonstration via --demo flag (< 1s).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from xgboost import XGBClassifier


DECISION_FEATURE_NAMES = [
    "Visual Risk Score (Gate 1)",
    "Linguistic Risk Score (Gate 2)",
    "Relational Risk Score (Gate 3)",
    "Visual Modality Mask (Photo Available)",
    "Visual Attention Weight (alpha_vis)",
    "Linguistic Attention Weight (alpha_ling)",
    "Relational Attention Weight (alpha_rel)",
    "Louvain Community Bucket",
    "Louvain Cluster Size (log1p)",
    "Louvain Co-occurrence Degree (log1p)",
    "Louvain Weighted Connectivity (log1p)",
    "Number of Experiences",
    "Number of Educations",
    "Number of Licenses",
    "Number of Volunteering",
    "Number of Skills",
    "Number of Recommendations",
    "Number of Projects",
    "Number of Publications",
    "Number of Courses",
    "Number of Honors",
    "Number of Scores",
    "Number of Languages",
    "Number of Organizations",
    "Number of Interests",
    "Number of Activities",
    "Connections Count",
    "Followers Count",
]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


class MultiModalShapExplainer:
    """Computes and interprets blended SHAP values for the TrustFusion decision ensemble."""

    def __init__(
        self,
        catboost_model_path: Path | str,
        xgboost_model_path: Path | str,
        feature_names: list[str] | None = None,
    ):
        self.feature_names = feature_names or DECISION_FEATURE_NAMES

        logging.info("Loading CatBoost and XGBoost decision ensemble models...")
        self.cb = CatBoostClassifier()
        self.cb.load_model(str(catboost_model_path))

        self.xgb = XGBClassifier()
        self.xgb.load_model(str(xgboost_model_path))

        logging.info("Initializing TreeExplainer instances for CatBoost and XGBoost...")
        self.cb_explainer = shap.TreeExplainer(self.cb)
        self.xgb_explainer = shap.TreeExplainer(self.xgb)

        # Baseline expected values (margins)
        self.cb_expected = float(np.ravel(self.cb_explainer.expected_value)[0])
        self.xgb_expected = float(np.ravel(self.xgb_explainer.expected_value)[0])
        self.blended_expected = 0.5 * self.cb_expected + 0.5 * self.xgb_expected

    def explain_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Computes 50/50 blended SHAP values for a feature matrix.
        Args:
            X: [N, 28] feature matrix
        Returns:
            blended_shap: [N, 28] blended SHAP attribution matrix
        """
        cb_shap = np.array(self.cb_explainer.shap_values(X))
        xgb_shap = np.array(self.xgb_explainer.shap_values(X))

        if cb_shap.ndim == 3:  # In case of binary 2-class output [N, D, 2]
            cb_shap = cb_shap[:, :, 1]
        if xgb_shap.ndim == 3:
            xgb_shap = xgb_shap[:, :, 1]

        blended_shap = 0.5 * cb_shap + 0.5 * xgb_shap
        return blended_shap.astype(np.float32)

    def compute_global_importance(self, shap_values: np.ndarray) -> list[dict[str, Any]]:
        """
        Ranks features by mean absolute SHAP value.
        """
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        sorted_indices = np.argsort(mean_abs)[::-1]

        ranking = []
        for rank, idx in enumerate(sorted_indices, start=1):
            ranking.append({
                "rank": rank,
                "feature_index": int(idx),
                "feature_name": self.feature_names[idx],
                "mean_abs_shap": float(mean_abs[idx]),
                "relative_importance_pct": float(mean_abs[idx] / np.sum(mean_abs) * 100.0),
            })
        return ranking

    def explain_single_profile(
        self,
        x_vector: np.ndarray,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        Decomposes a single profile's risk into driving and mitigating factors.
        """
        x_2d = x_vector.reshape(1, -1)
        shap_vals = self.explain_matrix(x_2d)[0]

        # Model probability
        p_cb = float(self.cb.predict_proba(x_2d)[0, 1])
        p_xgb = float(self.xgb.predict_proba(x_2d)[0, 1])
        final_risk = 0.5 * p_cb + 0.5 * p_xgb

        factors = []
        for i, val in enumerate(shap_vals):
            factors.append({
                "feature_name": self.feature_names[i],
                "feature_value": float(x_vector[i]),
                "shap_value": float(val),
                "impact": "INCREASES_FRAUD_RISK" if val > 0 else "DECREASES_FRAUD_RISK",
            })

        # Sort by magnitude of contribution
        factors_sorted = sorted(factors, key=lambda f: abs(f["shap_value"]), reverse=True)
        top_risk_drivers = [f for f in factors_sorted if f["shap_value"] > 0][:top_k]
        top_protective = [f for f in factors_sorted if f["shap_value"] < 0][:top_k]

        return {
            "final_risk_score": final_risk,
            "base_value": self.blended_expected,
            "top_risk_drivers": top_risk_drivers,
            "top_protective_factors": top_protective,
            "all_feature_attributions": factors,
        }


def extract_linguistic_triggers(text: str) -> dict[str, Any]:
    """Extracts explicit high-risk keywords and deception triggers from profile text."""
    text_lower = (text or "").lower()
    keywords = [
        "whatsapp", "telegram", "crypto", "bitcoin", "usdt",
        "wire transfer", "western union", "cashapp", "zelle",
        "send cv to", "urgent hiring", "immediate opening",
        "no interview", "work from home earn", "guaranteed income",
        "deposit", "pay fee", "confidential project",
    ]
    detected = [kw for kw in keywords if kw in text_lower]
    return {
        "text_sample": text[:120] + "..." if len(text) > 120 else text,
        "triggers_found": detected,
        "trigger_count": len(detected),
        "has_high_risk_triggers": len(detected) > 0,
    }


def run_shap_analysis(
    data_dir: Path = Path("data/processed"),
    fusion_dir: Path = Path("data/processed/fusion"),
    output_dir: Path = Path("data/processed/fusion"),
) -> dict[str, Any]:
    setup_logging()
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("=== Starting Step 6: Multi-Modal SHAP Explainability Suite ===")

    cb_path = fusion_dir / "catboost_binary.cbm"
    xgb_path = fusion_dir / "xgboost_binary.json"
    X_train_dec = np.load(fusion_dir / "fused_X_train_decision.npy")
    X_test_dec = np.load(fusion_dir / "fused_X_test_decision.npy")
    test_df = pd.read_csv(data_dir / "test.csv")

    explainer = MultiModalShapExplainer(
        catboost_model_path=cb_path,
        xgboost_model_path=xgb_path,
        feature_names=DECISION_FEATURE_NAMES,
    )

    logging.info("Computing blended SHAP values for all %d test profiles...", len(X_test_dec))
    test_shap = explainer.explain_matrix(X_test_dec)

    logging.info("Computing global feature importance ranking...")
    global_importance = explainer.compute_global_importance(test_shap)

    # Save SHAP matrices
    np.save(output_dir / "test_shap_values.npy", test_shap)

    # Compile exemplar case studies
    exemplar_specs = [
        ("Legitimate Executive", 0, "Real face, clean linguistic tone, organic network"),
        ("ChatGPT Recruiter Fake", 2, "StyleGAN headshot, synthetic prose, isolated graph"),
        ("Manual Phishing Impersonator", 1, "Logo impersonation, urgency cues, low history"),
    ]

    exemplar_explanations = []
    for label, target_class, desc in exemplar_specs:
        match_idx = int(np.where(test_df["class_4"] == target_class)[0][0])
        row = test_df.iloc[match_idx]
        local_exp = explainer.explain_single_profile(X_test_dec[match_idx], top_k=5)
        text_triggers = extract_linguistic_triggers(str(row.get("About", "")) + " " + str(row.get("Headline", "")))

        exemplar_explanations.append({
            "archetype": label,
            "profile_id": str(row["profile_id"]),
            "ground_truth": str(row["class_name"]),
            "description": desc,
            "final_risk_score": local_exp["final_risk_score"],
            "top_risk_drivers": local_exp["top_risk_drivers"],
            "top_protective_factors": local_exp["top_protective_factors"],
            "linguistic_triggers": text_triggers,
        })

    # Cold-Start Exemplar (No Photo)
    masks = np.load(data_dir / "visual" / "test_visual_modality_masks.npy").ravel()
    cold_indices = np.where(masks == 0.0)[0]
    cold_idx = int(cold_indices[0]) if len(cold_indices) else 0
    cold_row = test_df.iloc[cold_idx]
    cold_exp = explainer.explain_single_profile(X_test_dec[cold_idx], top_k=5)
    cold_text_triggers = extract_linguistic_triggers(str(cold_row.get("About", "")) + " " + str(cold_row.get("Headline", "")))

    exemplar_explanations.append({
        "archetype": "Cold-Start Account (No Photo)",
        "profile_id": str(cold_row["profile_id"]),
        "ground_truth": str(cold_row["class_name"]),
        "description": "Visual alpha strictly 0.0, attention redistributed to language and graph",
        "final_risk_score": cold_exp["final_risk_score"],
        "top_risk_drivers": cold_exp["top_risk_drivers"],
        "top_protective_factors": cold_exp["top_protective_factors"],
        "linguistic_triggers": cold_text_triggers,
    })

    metadata = {
        "project": "TrustFusion Step 6: Multi-Modal SHAP Explainability Suite",
        "num_decision_features": len(DECISION_FEATURE_NAMES),
        "num_test_profiles_explained": len(X_test_dec),
        "base_expected_value": explainer.blended_expected,
        "top_10_global_features": global_importance[:10],
        "exemplar_case_studies": exemplar_explanations,
    }

    with (output_dir / "shap_summary.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Save clean CSV ranking for examiner audits
    df_rank = pd.DataFrame(global_importance)
    df_rank.to_csv(output_dir / "shap_global_feature_ranking.csv", index=False)

    logging.info("SHAP analysis completed successfully. Artifacts saved to: %s", output_dir)
    display_shap_summary(output_dir)
    return metadata


def display_shap_summary(output_dir: Path = Path("data/processed/fusion")) -> None:
    summary_file = output_dir / "shap_summary.json"
    if not summary_file.exists():
        print(f"[!] SHAP summary file not found at {summary_file}. Please run analysis first.")
        return

    with summary_file.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    def safe_ascii(val: Any, max_len: int = 75) -> str:
        s = str(val or "").encode("ascii", "ignore").decode("ascii")
        return s[:max_len] + "..." if len(s) > max_len else s

    print("\n" + "=" * 82)
    print("         TRUSTFUSION STEP 6: MULTI-MODAL SHAP EXPLAINABILITY SUITE")
    print("=" * 82)
    print(f"[*] Explainer Method           : Blended TreeExplainer (CatBoost 350 + XGBoost 300)")
    print(f"[*] Decision Feature Space     : {meta.get('num_decision_features', 28)}-D Multi-Modal Vector")
    print(f"[*] Unseen Profiles Analyzed   : {meta.get('num_test_profiles_explained', 1086):,} profiles")
    print(f"[*] Baseline Expected Margin   : {meta.get('base_expected_value', 0.0):.4f}")
    print("-" * 82)
    print("GLOBAL FEATURE IMPORTANCE MATRIX (Mean Absolute SHAP Values across Test Set):")
    print(f"  {'Rank':<4} | {'Forensic Feature Name':<38} | {'Mean |SHAP|':>11} | {'Rel Importance':>14}")
    print(f"  {'-' * 4}-+-{'-' * 38}-+-{'-' * 11}-+-{'-' * 14}")

    for item in meta.get("top_10_global_features", []):
        r = item["rank"]
        name = item["feature_name"]
        val = item["mean_abs_shap"]
        pct = item["relative_importance_pct"]
        bar = "#" * int(pct / 2.5)
        print(f"  #{r:<3} | {name:<38} | {val:>11.4f} | {pct:>13.2f}% {bar}")

    print("-" * 82)
    print("LOCAL FORENSIC AUDIT TRAIL (Exemplar Profile Explanations):")

    for case in meta.get("exemplar_case_studies", []):
        arch = case["archetype"]
        pid = case["profile_id"]
        gt = case["ground_truth"]
        risk = case["final_risk_score"]
        verdict = "HIGH RISK / BLOCKED" if risk >= 0.75 else ("SUSPICIOUS / REVIEW" if risk >= 0.4 else "LEGITIMATE / CLEARED")

        print(f"\n  [{arch.upper()}] (ID: {pid} | GT: {gt})")
        print(f"    - Unified Fraud Risk    : {risk:.4f}  [Verdict: {verdict}]")
        print(f"    - Top Risk Drivers (+)  :")
        for d in case.get("top_risk_drivers", [])[:3]:
            print(f"      * {d['feature_name']:<35} (Value: {d['feature_value']:>7.3f}) -> SHAP: +{d['shap_value']:.4f}")
        if not case.get("top_risk_drivers"):
            print("      * (None - Zero high-risk indicators)")

        print(f"    - Protective Factors (-):")
        for p in case.get("top_protective_factors", [])[:3]:
            print(f"      * {p['feature_name']:<35} (Value: {p['feature_value']:>7.3f}) -> SHAP: {p['shap_value']:.4f}")

        trig = case.get("linguistic_triggers", {})
        if trig.get("triggers_found"):
            print(f"    - Text Triggers Flagged : {', '.join(trig['triggers_found'])}")
        else:
            print(f"    - Text Triggers Flagged : None (Clean professional language)")

    print("\n" + "=" * 82 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TrustFusion Step 6: Multi-Modal SHAP Explainer")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"), help="Processed data directory")
    parser.add_argument("--fusion-dir", type=Path, default=Path("data/processed/fusion"), help="Fusion artifacts directory")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/fusion"), help="Output directory")
    parser.add_argument("--demo", action="store_true", help="Quickly display SHAP executive summary and exit.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.demo:
        display_shap_summary(args.output_dir)
    else:
        run_shap_analysis(data_dir=args.data_dir, fusion_dir=args.fusion_dir, output_dir=args.output_dir)
