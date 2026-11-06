"""
TrustFusion - Step 7: Streamlit Security Analyst Dashboard.

Usage:
------
1. Start the FastAPI Serving Backend (Terminal 1):
   python backend/main.py
   # Or: uvicorn backend.main:app --port 8000

2. Run the Streamlit Dashboard (Terminal 2):
   streamlit run dashboard.py

Modules:
--------
1. Real-Time Profile Moderation Console (Instant risk scoring, verdict badge, 4-class attribution).
2. Cross-Modal Attention Gating & Cold-Start Simulator (Live attention gauges, photo removal toggle).
3. SHAP Explainability Suite (Risk drivers vs. protective factors, text deception triggers).
4. Relational Graph & Sybil Ring Explorer (2-hop computational subgraph, Sybil co-occurrence density).
5. Global Benchmark Explorer (ROC-AUC, confusion matrix, standalone gate comparisons).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

import numpy as np
import pandas as pd
import requests

API_BASE_URL = "http://127.0.0.1:8000"


def check_api_health() -> dict[str, Any] | None:
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=2.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def fetch_profiles(limit: int = 4) -> list[dict[str, Any]]:
    try:
        r = requests.get(f"{API_BASE_URL}/api/v1/profiles?limit={limit}", timeout=3.0)
        if r.status_code == 200:
            return r.json().get("exemplar_case_studies", [])
    except Exception:
        pass
    return []


def run_predict(profile_id: str, simulate_cold_start: bool = False) -> dict[str, Any] | None:
    try:
        r = requests.post(
            f"{API_BASE_URL}/api/v1/predict",
            json={"profile_id": profile_id, "simulate_cold_start": simulate_cold_start},
            timeout=5.0,
        )
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        st.error(f"Prediction failed: {e}")
    return None


def run_explain(profile_id: str, top_k: int = 5) -> dict[str, Any] | None:
    try:
        r = requests.post(
            f"{API_BASE_URL}/api/v1/explain",
            json={"profile_id": profile_id, "top_k": top_k},
            timeout=8.0,
        )
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        st.error(f"Explainability request failed: {e}")
    return None


def fetch_benchmarks() -> dict[str, Any] | None:
    try:
        r = requests.get(f"{API_BASE_URL}/api/v1/benchmarks", timeout=3.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def render_streamlit_app():
    st.set_page_config(
        page_title="TrustFusion | Anti-Fraud Cyber-SOC",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🛡️ TrustFusion: Enterprise Anti-Fraud Serving Engine")
    st.caption("Multimodal Defense Middleware &bull; Operation Dream Job Countermeasure &bull; Step 7 Interactive SOC")

    # Check Backend Connection
    health = check_api_health()
    if health is None:
        st.warning(
            "⚠️ FastAPI Serving Engine is currently offline at http://127.0.0.1:8000. "
            "Please start the backend in a separate terminal via: `python backend/main.py`"
        )
        st.info("Tip: You can also use the integrated zero-dependency web dashboard directly at http://127.0.0.1:8000/dashboard once started!")
        return

    # Sidebar: SOC Status & Exemplar Selection
    st.sidebar.header("⚡ SOC Live Telemetry")
    st.sidebar.success(f"Status: {health.get('status', 'online').upper()}")
    st.sidebar.metric("Compute Hardware", health.get("gpu_name", "CPU"))
    st.sidebar.metric("Test Profiles Cached", health.get("test_profiles_cached", 1086))

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Forensic Case Studies")

    exemplars = fetch_profiles()
    ex_options = {f"{ex.get('archetype')} ({ex.get('ground_truth')})": ex.get("profile_id") for ex in exemplars}
    selected_label = st.sidebar.selectbox("Choose Exemplar Profile:", list(ex_options.keys()) if ex_options else ["None"])
    selected_id = ex_options.get(selected_label) if ex_options else None

    manual_id = st.sidebar.text_input("Or enter custom Profile ID:", value=selected_id or "")
    active_profile_id = manual_id.strip() if manual_id.strip() else selected_id

    st.sidebar.markdown("---")
    simulate_cold_start = st.sidebar.checkbox("❄️ Simulate Cold-Start (Remove Photo)", value=("be4ecb78eebc777b" in (active_profile_id or "")))

    if not active_profile_id:
        st.info("Select a profile from the sidebar to inspect.")
        return

    # Trigger Inference and Explainability
    pred = run_predict(active_profile_id, simulate_cold_start=simulate_cold_start)
    exp = run_explain(active_profile_id, top_k=5)

    if not pred or not exp:
        st.error("Failed to retrieve profile predictions from FastAPI backend.")
        return

    # Tabs Layout
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. Triage Verdict",
        "2. Attention Gating",
        "3. SHAP Explainability",
        "4. Relational Subgraph",
        "5. Global Benchmarks",
    ])

    with tab1:
        st.subheader("1. Real-Time Moderation Console")
        col_m1, col_m2, col_m3 = st.columns([1, 1, 1])

        with col_m1:
            risk_pct = pred["calibrated_risk_score"] * 100
            st.metric("Unified Threat Risk Score", f"{risk_pct:.2f}%")
            verdict = pred["triage_verdict"]
            if verdict == "HIGH_RISK":
                st.error("🚨 VERDICT: HIGH RISK / SUSPICIOUS ACCOUNT")
            elif verdict == "SUSPICIOUS":
                st.warning("⚠️ VERDICT: ELEVATED THREAT LEVEL")
            else:
                st.success("✅ VERDICT: AUTHENTIC / LOW RISK")

        with col_m2:
            st.metric("Threat Archetype", pred["threat_attribution"]["class_label"])
            st.write("**Class Probability Distribution:**")
            for cls, p in pred["threat_attribution"]["class_probabilities"].items():
                st.progress(p, text=f"{cls}: {p*100:.1f}%")

        with col_m3:
            st.metric("Serving Latency", f"{pred['latency_ms']:.2f} ms")
            st.write(f"**Name:** {pred['profile_name']}")
            st.write(f"**Headline:** {pred['headline'] or 'None'}")
            st.write(f"**Workplace:** {pred['workplace'] or 'None'}")
            st.write(f"**Location:** {pred['location'] or 'None'}")

    with tab2:
        st.subheader("2. Multimodal Attention Gating & Cold-Start Policy")
        gb = pred["gate_breakdown"]
        st.info(pred["cold_start_policy_applied"])

        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            st.markdown("#### Gate 1: Visual Identity")
            st.metric("Visual Risk", f"{gb['gate1_visual_risk']*100:.2f}%")
            st.progress(gb["alpha_vis"], text=f"Attention Weight (α_vis): {gb['alpha_vis']*100:.1f}%")
            if gb["is_cold_start"]:
                st.caption("🔒 Modality Masked (M_vis = 0.0)")

        with col_g2:
            st.markdown("#### Gate 2: Linguistic Forensics")
            st.metric("Linguistic Risk", f"{gb['gate2_linguistic_risk']*100:.2f}%")
            st.progress(gb["alpha_ling"], text=f"Attention Weight (α_ling): {gb['alpha_ling']*100:.1f}%")

        with col_g3:
            st.markdown("#### Gate 3: Relational Graph")
            st.metric("Relational Risk", f"{gb['gate3_relational_risk']*100:.2f}%")
            st.progress(gb["alpha_rel"], text=f"Attention Weight (α_rel): {gb['alpha_rel']*100:.1f}%")

    with tab3:
        st.subheader("3. Multi-Modal SHAP Feature Attribution")
        st.caption(f"Baseline Expected Margin E[f(x)] = {exp['base_expected_value']:.4f}")

        col_sh1, col_sh2 = st.columns(2)
        with col_sh1:
            st.markdown("##### 🔴 Top Risk Drivers (Pushing Toward Fraud)")
            for d in exp["top_risk_drivers"]:
                st.write(f"**{d['feature_name']}**: `+{d['shap_value']:.3f}`")
                st.progress(min(1.0, abs(d["shap_value"])))

        with col_sh2:
            st.markdown("##### 🟢 Top Protective Factors (Pushing Toward Legitimate)")
            for p in exp["top_protective_factors"]:
                st.write(f"**{p['feature_name']}**: `{p['shap_value']:.3f}`")
                st.progress(min(1.0, abs(p["shap_value"])))

        st.markdown("---")
        st.markdown("##### 💬 Linguistic Deception & Off-Platforming Cues")
        trigs = exp.get("linguistic_triggers", {}).get("triggers_found", [])
        if trigs:
            st.error(f"Detected High-Risk Triggers: {', '.join(trigs)}")
        else:
            st.success("No off-platforming deception keywords found in profile text.")

    with tab4:
        st.subheader("4. Relational Graph & Multi-Hop Sybil Paths")
        gnn = exp.get("gnn_subgraph", {})
        stats = gnn.get("subgraph_stats", {})

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Subgraph Nodes", stats.get("num_subgraph_nodes", 0))
        c2.metric("Subgraph Edges", stats.get("num_subgraph_edges", 0))
        c3.metric("Connected Neighbors", stats.get("num_connected_users", 0))
        c4.metric("Sybil Density", f"{stats.get('sybil_density_pct', 0.0):.1f}%")

        st.markdown("##### Coordinated Multi-Hop Entity Connections:")
        paths = gnn.get("top_relational_paths", [])
        if paths:
            for p in paths:
                is_sybil = p.get("is_sybil_link")
                prefix = "🚨 [SYBIL LINK]" if is_sybil else "🌐 [ORGANIC]"
                st.code(f"{prefix} {p.get('path_narrative')}")
        else:
            st.write("No multi-hop co-occurrence paths discovered.")

    with tab5:
        st.subheader("5. Global Unseen Test Set Benchmarks (1,086 Profiles)")
        bench = fetch_benchmarks()
        if bench:
            mat = bench.get("gate_performance_matrix", {})
            df_comp = pd.DataFrame([
                {"Gate": "Gate 1: Visual Identity", "Accuracy": f"{mat.get('gate1_visual', {}).get('accuracy', 0)*100:.2f}%", "Macro F1": f"{mat.get('gate1_visual', {}).get('macro_f1', 0)*100:.2f}%", "ROC-AUC": f"{mat.get('gate1_visual', {}).get('roc_auc', 0):.4f}"},
                {"Gate": "Gate 2: Linguistic Forensics", "Accuracy": f"{mat.get('gate2_linguistic', {}).get('accuracy', 0)*100:.2f}%", "Macro F1": f"{mat.get('gate2_linguistic', {}).get('macro_f1', 0)*100:.2f}%", "ROC-AUC": f"{mat.get('gate2_linguistic', {}).get('roc_auc', 0):.4f}"},
                {"Gate": "Gate 3: Relational Graph", "Accuracy": f"{mat.get('gate3_relational', {}).get('accuracy', 0)*100:.2f}%", "Macro F1": f"{mat.get('gate3_relational', {}).get('macro_f1', 0)*100:.2f}%", "ROC-AUC": f"{mat.get('gate3_relational', {}).get('roc_auc', 0):.4f}"},
                {"Gate": "TRUSTFUSION ENSEMBLE", "Accuracy": f"{mat.get('trustfusion_fusion', {}).get('accuracy', 0)*100:.2f}%", "Macro F1": f"{mat.get('trustfusion_fusion', {}).get('macro_f1', 0)*100:.2f}%", "ROC-AUC": f"{mat.get('trustfusion_fusion', {}).get('roc_auc', 0):.4f}"},
            ])
            st.table(df_comp)


if __name__ == "__main__":
    if STREAMLIT_AVAILABLE:
        render_streamlit_app()
    else:
        print("\n" + "=" * 70)
        print("   TrustFusion Security Analyst Dashboard")
        print("=" * 70)
        print("   Streamlit is not currently installed in this environment.")
        print("   However, the FULL INTERACTIVE CYBER-SOC DASHBOARD is available")
        print("   directly through FastAPI without requiring any extra dependencies!")
        print("")
        print("   To launch the Serving Engine & Interactive Dashboard:")
        print("       python backend/main.py")
        print("   Then open:")
        print("       http://127.0.0.1:8000/dashboard")
        print("=" * 70 + "\n")
