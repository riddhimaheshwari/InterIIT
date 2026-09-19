import streamlit as st
import pandas as pd
import numpy as np
import json
import sqlite3
from pathlib import Path

from src.eval.report import EvalReportGenerator
from src.online.infer import ComplianceInferenceEngine
from src.audit.log import AuditLogger
from src.audit.registry import AdapterRegistry
from src.eval.confusion_set import get_confusion_benchmark

st.set_page_config(
    page_title="Continual-Counsel | Legal AI Compliance System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0px; }
    .sub-title { font-size: 1.05rem; color: #4B5563; margin-bottom: 20px; }
    .metric-card { background-color: #F3F4F6; border-radius: 8px; padding: 15px; border-left: 5px solid #2563EB; }
    .badge-verified { background-color: #DEF7EC; color: #03543F; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 0.85rem; }
    .badge-blocked { background-color: #FDE8E8; color: #9B1C1C; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 0.85rem; }
    .audit-box { background-color: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 8px; padding: 15px; font-family: monospace; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# Initialize engines
report_gen = EvalReportGenerator()
registry = AdapterRegistry()
audit_logger = AuditLogger()

@st.cache_resource
def get_engine():
    return ComplianceInferenceEngine()

engine = get_engine()

# Sidebar Navigation
st.sidebar.title("⚖️ Continual-Counsel")
st.sidebar.markdown("**Edge-Deployable Compliance AI**")
st.sidebar.markdown("---")
view_option = st.sidebar.radio(
    "Navigation Mode",
    ["📊 Continual Learning Curves", "🔍 Live Edge Query Console", "📜 Tamper-Evident Audit Ledger", "🧩 Adversarial Confusion Set", "🏗️ Adapter Registry & Lineage"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔒 Zero-Network Status")
st.sidebar.success("Strict Offline Mode: ACTIVE")
st.sidebar.caption("Outbound Egress: Blocked\nLocal FAISS: Loaded\nQuantized Weights: Bundled")

# VIEW 1: Continual Learning Curves
if view_option == "📊 Continual Learning Curves":
    st.markdown('<p class="main-title">Quarterly Continual Learning & Stability Curves</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Tracks Backward Transfer (BWT), Forward Transfer (FWT), Sawtooth Basis Rank, and Memory Footprint across Q1→Q4 updates.</p>', unsafe_allow_html=True)

    history = report_gen.load_history()
    quarters = history.get("quarters", ["Q1", "Q2", "Q3", "Q4"])

    if not quarters:
        st.warning("No update history found. Run the offline update pipeline first.")
    else:
        # Top KPI row
        col1, col2, col3, col4 = st.columns(4)
        latest_acc = history["avg_accuracy"][-1] if history["avg_accuracy"] else 0.0
        latest_bwt = history["bwt"][-1] if history["bwt"] else 0.0
        latest_rank = history["basis_rank"][-1] if history["basis_rank"] else 0
        latest_footprint = history["adapter_footprint_mb"][-1] if history["adapter_footprint_mb"] else 0.0

        col1.metric("Average Accuracy", f"{latest_acc * 100:.1f}%", f"{latest_bwt * 100:.2f}% BWT")
        col2.metric("Backward Transfer (BWT)", f"{latest_bwt:.4f}", "Gate: > -0.03")
        col3.metric("Orthogonal Basis Rank", f"{latest_rank}", "Sawtooth Reset at Merge")
        col4.metric("Adapter Footprint", f"{latest_footprint:.1f} MB", "Bounded via TIES")

        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["📈 Transfer & Accuracy Curves", "📐 Sawtooth Basis Rank & Footprint", "🛡️ Hallucination & Drift Metrics"])

        n_q = len(quarters)
        with tab1:
            st.subheader("Continual Transfer Progression")
            df_perf = pd.DataFrame({
                "Quarter": quarters,
                "Average Accuracy": history.get("avg_accuracy", [])[:n_q],
                "BWT (Backward Transfer)": history.get("bwt", [])[:n_q],
                "FWT (Forward Transfer)": history.get("fwt", [])[:n_q]
            })
            st.line_chart(df_perf.set_index("Quarter"))

            st.markdown("#### Per-Regime Retention Matrix $R_{t,i}$")
            matrix_data = history.get("regime_accuracy_matrix", {})
            if matrix_data:
                df_matrix = pd.DataFrame(matrix_data).T
                st.dataframe(df_matrix.style.highlight_max(axis=0, color="#D1FAE5"), use_container_width=True)

        with tab2:
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Orthogonal Basis Rank (Sawtooth Diagnostic)")
                st.caption("Demonstrates the growth of orthonormal basis Q during O-LoRA learning, followed by reset at TIES-merge.")
                df_rank = pd.DataFrame({
                    "Quarter": quarters,
                    "Basis Rank": history.get("basis_rank", [])[:n_q]
                })
                st.bar_chart(df_rank.set_index("Quarter"))

            with col_b:
                st.subheader("Adapter Storage Footprint (MB)")
                st.caption("Consolidation via TIES-merge bounds memory growth and keeps edge deployment lightweight.")
                df_fp = pd.DataFrame({
                    "Quarter": quarters,
                    "Footprint (MB)": history.get("adapter_footprint_mb", [])[:n_q]
                })
                st.area_chart(df_fp.set_index("Quarter"))

        with tab3:
            st.subheader("Citation Hallucination Rate & Golden Set Drift")
            col_c, col_d = st.columns(2)
            with col_c:
                st.markdown("##### Precedent-Citation Hallucination Rate")
                df_hal = pd.DataFrame({
                    "Quarter": quarters,
                    "Hallucination Rate": history.get("hallucination_rate", [])[:n_q]
                })
                st.line_chart(df_hal.set_index("Quarter"))

            with col_d:
                st.markdown("##### Golden Set Refresh Drift Diff")
                st.caption("Fraction of key conclusions altered post-TIES merge refresh.")
                df_drift = pd.DataFrame({
                    "Quarter": quarters,
                    "Drift Fraction": history.get("drift_diff", [])[:n_q]
                })
                st.bar_chart(df_drift.set_index("Quarter"))

# VIEW 2: Live Edge Query Console
elif view_option == "🔍 Live Edge Query Console":
    st.markdown('<p class="main-title">Offline Edge Query Console</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Interactive evaluation of local retrieval, confidence gating, self-verification, and cryptographic provenance logging.</p>', unsafe_allow_html=True)

    col_q1, col_q2 = st.columns([2, 1])
    with col_q1:
        user_query = st.text_area(
            "Enter Compliance Query / Statutory Scenario:",
            value="What is the mandatory reporting timeframe for critical AI malfunctions under INC-301?",
            height=100
        )
    with col_q2:
        regime_filter = st.selectbox("Temporal Regime Filter (Optional):", ["All Regimes", "Q1", "Q2", "Q3", "Q4"])
        target_regime = None if regime_filter == "All Regimes" else regime_filter

    if st.button("🚀 Process Query (Local Offline Edge Loop)", type="primary"):
        with st.spinner("Processing through local FAISS retrieval, confidence gate, verification, and audit chain..."):
            result = engine.process_query(user_query, target_regime=target_regime)

        if result["status"] == "blocked":
            st.error(f"⛔ Query Blocked by Confidence Gate: **{result['answer']}**")
            st.caption("Top similarity score fell below threshold (0.65). Generation halted to prevent hallucinations.")
        else:
            st.success("✅ Compliance Synthesis Generated & Verified")
            st.markdown(f"### Legal Opinion & Compliance Notice\n> {result['answer']}")

        st.markdown("---")
        st.subheader("🔍 Execution Trace & Provenance Details")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Verification Status", result["verification_verdict"])
        c2.metric("Confidence Gate", "PASSED" if result["confidence_passed"] else "BLOCKED")
        c3.metric("Active Adapter", result.get("adapter_id", "genesis"))

        with st.expander("📚 Retrieved Grounding Context (FAISS Chunks)", expanded=True):
            for idx, chunk in enumerate(result.get("retrieved_chunks", [])):
                st.markdown(f"**[{chunk.get('clause_id')}]** - *{chunk.get('title')}* (Regime: {chunk.get('regime_id')}, Score: `{chunk.get('similarity_score')}`)")
                st.caption(chunk.get("text"))

        with st.expander("🔐 Tamper-Evident Cryptographic Audit Block", expanded=True):
            st.json(result["audit_record"])

# VIEW 3: Tamper-Evident Audit Ledger
elif view_option == "📜 Tamper-Evident Audit Ledger":
    st.markdown('<p class="main-title">Cryptographic Audit Ledger & Chain Inspector</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Inspect immutable hash-chained audit trails, verify ledger cryptographic integrity, and query past answers.</p>', unsafe_allow_html=True)

    is_valid, err_msg = audit_logger.verify_integrity()
    if is_valid:
        st.success("🛡️ **Cryptographic Ledger Integrity: VERIFIED (Zero Tampering Detected)**")
    else:
        st.error(f"🚨 **LEDGER TAMPERING DETECTED**: {err_msg}")

    st.markdown("---")
    st.subheader("Recent Audit Ledger Entries")
    records = audit_logger.list_records(limit=20)
    
    if not records:
        st.info("No queries logged yet. Submit queries in the Live Query Console.")
    else:
        for rec in records:
            with st.expander(f"Query ID: {rec['query_id']} | Time: {rec['timestamp']} | Verdict: {rec['verification_verdict']}"):
                st.markdown(f"**Query:** {rec['query_text']}")
                st.markdown(f"**Final Answer:** {rec['final_answer']}")
                st.markdown(f"**Retrieval Index Hash:** `{rec['retrieval_index_hash']}`")
                st.markdown(f"**Adapter Version Hash:** `{rec['adapter_hash']}`")
                st.markdown(f"**Previous Hash ($H_{{n-1}}$):** `{rec['previous_hash']}`")
                st.markdown(f"**Record Hash ($H_n$):** `{rec['record_hash']}`")

# VIEW 4: Adversarial Confusion Set
elif view_option == "🧩 Adversarial Confusion Set":
    st.markdown('<p class="main-title">Cross-Regime Adversarial Confusion Benchmark</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Evaluates queries requiring complex synthesis and reconciliation of overlapping or superseding rules across multiple quarters.</p>', unsafe_allow_html=True)

    confusion_items = get_confusion_benchmark()
    for item in confusion_items:
        with st.expander(f"⚠️ {item['id']}: {item['title']} (Cross-Regimes: {', '.join(item['cross_regimes'])})", expanded=True):
            st.markdown(f"**Scenario / Query:**\n> {item['query']}")
            st.markdown(f"**Governing Authority / Regime:** `{item['governing_regime']}`")
            st.markdown(f"**Expected Legal Synthesis:**\n> {item['expected_synthesis']}")
            st.markdown(f"**Key Statutory Requirements:** {', '.join(item['key_requirements'])}")
            st.caption(f"Confusing Elements: {', '.join(item['confusing_elements'])}")

# VIEW 5: Adapter Registry & Lineage
elif view_option == "🏗️ Adapter Registry & Lineage":
    st.markdown('<p class="main-title">Adapter Version Registry & Merge Lineage</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Tracks training metadata, hyperparameters, validation scores, and TIES merge lineage trees.</p>', unsafe_allow_html=True)

    all_adapters = registry.list_all_adapters()
    if not all_adapters:
        st.info("No adapters registered yet.")
    else:
        for a in all_adapters:
            is_m = a.get("is_merged", False)
            title_prefix = "🧬 [TIES-MERGED]" if is_m else "🔹 [ORTHOGONAL LORA]"
            with st.expander(f"{title_prefix} {a['adapter_id']} (Regime: {a['regime_id']}) - Created: {a['created_at']}"):
                st.markdown(f"**Adapter Hash:** `{a['adapter_hash']}`")
                st.markdown(f"**Base Model:** `{a['base_model']}`")
                st.markdown(f"**Training Data Hash:** `{a['training_data_hash']}`")
                st.markdown(f"**Merge Lineage Parents:** `{a['merge_lineage']}`")
                st.markdown("##### Hyperparameters")
                st.json(a["hyperparameters"])
                st.markdown("##### Acceptance Validation Metrics")
                st.json(a["validation_metrics"])
