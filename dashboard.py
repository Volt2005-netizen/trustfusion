"""
TrustFusion - Step 7: Streamlit Security Analyst Dashboard.

Usage:
------
1. Start the FastAPI Serving Backend (Terminal 1):
   python backend/main.py

2. Run the Streamlit Dashboard (Terminal 2):
   streamlit run dashboard.py
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

# Self-healing virtualenv launcher: If executed via system Python without dependencies,
# automatically relaunch using the project's dedicated virtual environment.
_venv_python = WORKSPACE_ROOT / "venv" / "Scripts" / "python.exe"
if not _venv_python.exists():
    _venv_python = WORKSPACE_ROOT / "venv" / "bin" / "python"

if _venv_python.exists() and Path(sys.executable).resolve() != _venv_python.resolve():
    try:
        import requests
    except ModuleNotFoundError:
        import subprocess
        print(f"[*] Detected unactivated Python ({sys.executable}).")
        print(f"[*] Seamlessly relaunching using project virtual environment: {_venv_python}\n")
        sys.exit(subprocess.call([str(_venv_python)] + sys.argv))

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


def run_predict(profile_id: str) -> dict[str, Any] | None:
    try:
        r = requests.post(
            f"{API_BASE_URL}/api/v1/predict",
            json={"profile_id": profile_id},
            timeout=5.0,
        )
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        st.error(f"Prediction failed: {e}")
    return None


def run_explain(profile_id: str, top_k: int = 4) -> dict[str, Any] | None:
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


def fetch_dataset_analysis() -> dict[str, Any] | None:
    try:
        r = requests.get(f"{API_BASE_URL}/api/v1/dataset-analysis", timeout=3.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def render_streamlit_app():
    st.set_page_config(
        page_title="TrustFusion | Identity & Threat Intelligence",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🛡️ TrustFusion: Candidate Identity & Threat Intelligence")
    st.caption("Automated Recruitment Anti-Fraud Platform &bull; Real-Time Candidate Verification")

    # Check Backend Connection
    health = check_api_health()
    if health is None:
        st.warning(
            "⚠️ Verification Engine is currently offline at http://127.0.0.1:8000. "
            "Please start the engine in your terminal via: `python backend/main.py`"
        )
        st.info("Tip: You can also use the integrated zero-dependency web dashboard directly at http://127.0.0.1:8000/dashboard once started!")
        return

    # Sidebar: Live Telemetry & Candidate Selection
    st.sidebar.header("⚡ System Telemetry")
    st.sidebar.success(f"Status: {health.get('engine_status', 'ONLINE')}")
    st.sidebar.metric("Verification Engine", health.get("gpu_name", "GPU Accelerated"))
    st.sidebar.metric("Candidates in Database", health.get("test_profiles_cached", 1086))

    st.sidebar.markdown("---")
    st.sidebar.subheader("👤 Candidate Selection")

    exemplars = fetch_profiles()
    ex_options = {f"{ex.get('candidate_name')} - {ex.get('category')}": ex.get("profile_id") for ex in exemplars}
    selected_label = st.sidebar.selectbox("Choose Candidate to Audit:", list(ex_options.keys()) if ex_options else ["None"])
    selected_id = ex_options.get(selected_label) if ex_options else None

    manual_query = st.sidebar.text_input("Or search candidate by name:", value="")
    active_profile_id = selected_id
    if manual_query.strip():
        # Search via API
        try:
            res = requests.get(f"{API_BASE_URL}/api/v1/profiles?limit=1&search={manual_query.strip()}").json()
            if res.get("profiles"):
                active_profile_id = res["profiles"][0]["profile_id"]
        except Exception:
            pass

    if not active_profile_id:
        st.info("Select a candidate from the sidebar to inspect.")
        return

    # Trigger Verification & Explainability
    pred = run_predict(active_profile_id)
    exp = run_explain(active_profile_id, top_k=4)

    if not pred or not exp:
        st.error("Failed to retrieve candidate verification data.")
        return

    # Main Page Navigation Tabs
    main_tab1, main_tab2 = st.tabs([
        "🔍 Candidate Verification & Audit",
        "📊 Dataset Intelligence & Threat Landscape",
    ])

    with main_tab1:
        # Candidate Header Banner
        col_c1, col_c2, col_c3 = st.columns([1.2, 1, 1])

        with col_c1:
            st.subheader(pred["profile_name"])
            st.write(f"**Role:** {pred['headline']}")
            st.write(f"**Organization:** {pred['workplace']}")
            st.write(f"**Location:** {pred['location']}")

        with col_c2:
            st.metric("Threat Likelihood Score", f"{pred['threat_risk_score']*100:.1f}%")
            verdict = pred["triage_verdict"]
            if verdict == "HIGH_RISK":
                st.error("🚨 VERDICT: HIGH RISK / FRAUDULENT")
            elif verdict == "SUSPICIOUS":
                st.warning("⚠️ VERDICT: ELEVATED THREAT LEVEL")
            else:
                st.success("✅ VERDICT: AUTHENTIC / LOW RISK")

        with col_c3:
            st.metric("Assessed Category", pred["category"])
            st.info(pred["summary_insight"])

        st.markdown("---")
        st.subheader("Multi-Signal Identity Verification")

        vs = pred["verification_signals"]
        col_v1, col_v2, col_v3 = st.columns(3)

        with col_v1:
            st.markdown("#### 1. Profile Photo Analysis")
            st.metric("Photo Authenticity", f"{vs['photo_verification_score']*100:.0f}%")
            st.caption(f"Status: {vs['photo_status']}")
            st.progress(vs["photo_verification_score"])

        with col_v2:
            st.markdown("#### 2. Text & Communication")
            st.metric("Language Integrity", f"{vs['language_integrity_score']*100:.0f}%")
            st.caption(f"Status: {vs['language_status']}")
            st.progress(vs["language_integrity_score"])

        with col_v3:
            st.markdown("#### 3. Workplace & Network")
            st.metric("Network Authenticity", f"{vs['network_authenticity_score']*100:.0f}%")
            st.caption(f"Status: {vs['network_status']}")
            st.progress(vs["network_authenticity_score"])

        st.markdown("---")
        col_e1, col_e2 = st.columns(2)

        with col_e1:
            st.subheader("Why was this assessment made?")
            st.markdown("##### 🔴 Identified Risk Signals")
            risks = exp.get("identified_threat_signals", [])
            if risks:
                for r in risks:
                    st.error(f"**{r['signal_title']}** ({r['impact_level']} Risk): {r['description']}")
            else:
                st.write("No major deceptive risk signals detected.")

            st.markdown("##### 🟢 Trust & Authenticity Factors")
            trusts = exp.get("trust_and_authenticity_signals", [])
            if trusts:
                for t in trusts:
                    st.success(f"**{t['signal_title']}** ({t['impact_level']} Trust): {t['description']}")
            else:
                st.write("No strong positive career tenure factors found.")

            keywords = exp.get("detected_deceptive_keywords", [])
            if keywords:
                st.warning(f"Detected Keywords: {', '.join(['#' + w for w in keywords])}")

        with col_e2:
            st.subheader("Colleague & Organization Network")
            net = exp.get("network_analysis", {})
            st.metric("Network Health", net.get("workplace_network_health", "Clean"))

            paths = net.get("co_affiliated_paths", [])
            if paths:
                st.write("**Observed Network Affiliations:**")
                for p in paths:
                    is_sybil = p.get("is_sybil_link")
                    tag = "🚨 [FLAGGED LINK]" if is_sybil else "🌐 [VERIFIED]"
                    st.code(f"{tag} {p.get('path_narrative')}")
            else:
                st.write("No multi-hop co-affiliations recorded.")

    with main_tab2:
        st.subheader("📊 LinkedIn Identity & Threat Dataset Analysis")
        ds = fetch_dataset_analysis()
        if ds:
            ov = ds.get("overview", {})
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Analyzed Profiles", f"{ov.get('total_profiles_analyzed', 3619):,}")
            m2.metric("Authentic Industry Profiles", "1,800 (49.7%)")
            m3.metric("Fraudulent Profiles", "1,819 (50.3%)")
            m4.metric("Network Entities Mapped", "16,970 Entities")

            st.markdown("---")
            st.markdown("#### Threat Category Composition")
            for cat in ds.get("category_distribution", []):
                st.write(f"**{cat['category']}**: `{cat['count']:,} accounts ({cat['percentage']}%)`")
                st.caption(cat["threat_nature"])
                st.progress(cat["percentage"] / 100.0)

            st.markdown("---")
            st.markdown("#### Key Threat Patterns Discovered in Dataset")
            for f in ds.get("key_forensic_discoveries", []):
                st.info(f"**{f['finding']}**: {f['detail']}")


if __name__ == "__main__":
    if STREAMLIT_AVAILABLE:
        render_streamlit_app()
    else:
        print("\n" + "=" * 70)
        print("   TrustFusion Security Analyst Dashboard")
        print("=" * 70)
        print("   The FULL INTERACTIVE CYBER-SOC DASHBOARD is available")
        print("   directly through FastAPI without requiring Streamlit!")
        print("")
        print("   To launch:")
        print("       python backend/main.py")
        print("   Then open:")
        print("       http://127.0.0.1:8000/dashboard")
        print("=" * 70 + "\n")
