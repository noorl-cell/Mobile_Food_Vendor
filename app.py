"""
Streamlit UI — Suffolk County Mobile Food Vendor Compliance Assistant
Run: streamlit run app.py
"""
import json
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# Auto-load .env so ANTHROPIC_API_KEY is available before RAG imports
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)
except ImportError:
    pass

from importlib import import_module
predict_mod = import_module("06_predict")
score_vendor = predict_mod.score_vendor
rag_mod = import_module("07_rag")

PROC = ROOT / "data" / "processed"
RULES = ROOT / "rules"
OUT = ROOT / "outputs"

st.set_page_config(page_title="Mobile Food Vendor Compliance", layout="wide")


@st.cache_data
def load_violations():
    df = pd.read_csv(PROC / "suffolk_violations_clean.csv", parse_dates=["inspection_date"])
    df["facility_name"] = df["facility_name"].fillna("")
    df["city"] = df["city"].fillna("")
    return df


@st.cache_data
def load_features():
    return pd.read_csv(PROC / "features.csv", parse_dates=["inspection_date"])


@st.cache_data
def load_rules():
    return json.loads((RULES / "extracted_rules.json").read_text())


@st.cache_data
def load_model_report():
    return json.loads((OUT / "stage5_model_report.json").read_text())


@st.cache_data
def load_permits():
    return json.loads((RULES / "required_permits.json").read_text())


@st.cache_resource
def load_rag():
    return rag_mod.RAG()


violations = load_violations()
features = load_features()
rules = load_rules()
model_report = load_model_report()
permits_doc = load_permits()
rag = load_rag()

st.title("🥪 Suffolk County Mobile Food Vendor Compliance Assistant")
st.caption("Real inspection data (2024-2025) · NY Sanitary Code Subpart 14-4 · NOAA weather · GBC risk model")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["💬 Ask the Assistant", "🔍 Search Vendors", "📊 Vendor Profile",
     "⚖️ Score a Scenario", "📋 Required Permits", "📜 Regulations", "🧠 Model Info"]
)

# ─────────────────────────── TAB 1: CHATBOT ───────────────────────────
with tab1:
    st.subheader("Ask a question about permits, rules, or inspections")
    import os as _os
    _llm_on = bool(_os.environ.get("ANTHROPIC_API_KEY"))
    st.caption(
        f"RAG over {len(rag.passages)} passages · "
        f"{'🟢 LLM mode (claude-haiku-4-5)' if _llm_on else '🟡 Retrieval-only (set ANTHROPIC_API_KEY for LLM)'}"
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    examples = [
        "What permits do I need to start a food truck in Suffolk County?",
        "How cold should I keep cold food?",
        "Do I need a commissary agreement?",
        "What fees should I expect in my first year?",
        "What are the most common violations?",
    ]
    cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        if cols[i].button(ex, key=f"ex_{i}"):
            st.session_state.pending_q = ex

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📚 {len(msg['sources'])} sources"):
                    for s in msg["sources"]:
                        st.markdown(
                            f"**{s['title']}** · _{s['kind']}_ · score {s['score']}  \n"
                            f"{s['text'][:300]}{'...' if len(s['text'])>300 else ''}  \n"
                            f"<small>Source: {s['source']}</small>",
                            unsafe_allow_html=True,
                        )

    user_q = st.chat_input("Ask about permits, rules, inspections...")
    if not user_q and st.session_state.get("pending_q"):
        user_q = st.session_state.pop("pending_q")

    if user_q:
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                result = rag.answer(user_q, k=5)
            mode_badge = "🤖 LLM" if result.get("mode") == "llm" else "🔎 Retrieval-only"
            st.markdown(f"_{mode_badge}_\n\n{result['answer']}")
            with st.expander(f"📚 {len(result['sources'])} sources"):
                for s in result["sources"]:
                    st.markdown(
                        f"**{s['title']}** · _{s['kind']}_ · score {s['score']}  \n"
                        f"{s['text'][:300]}{'...' if len(s['text'])>300 else ''}  \n"
                        f"<small>Source: {s['source']}</small>",
                        unsafe_allow_html=True,
                    )
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        })

    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat"):
            st.session_state.chat_history = []
            st.rerun()

# ─────────────────────────── TAB 2: SEARCH ───────────────────────────
with tab2:
    st.subheader("Search inspection records")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        q = st.text_input("Search by name / address / violation text", "")
    with c2:
        cities = ["(All)"] + sorted(violations["city"].dropna().unique().tolist())
        city = st.selectbox("City", cities)
    with c3:
        years = ["(All)", "2024", "2025"]
        year = st.selectbox("Year", years)

    df = violations.copy()
    if q:
        mask = (
            df["facility_name"].str.contains(q, case=False, na=False)
            | df["address"].fillna("").str.contains(q, case=False, na=False)
            | df["violation_text"].fillna("").str.contains(q, case=False, na=False)
        )
        df = df[mask]
    if city != "(All)":
        df = df[df["city"] == city]
    if year != "(All)":
        df = df[df["inspection_date"].dt.year == int(year)]

    st.write(f"**{len(df):,}** records matched · **{df['facility_id'].nunique():,}** unique facilities")
    st.dataframe(
        df[["facility_name", "city", "zip", "inspection_date", "code_section", "violation_text"]]
        .sort_values("inspection_date", ascending=False)
        .head(500),
        use_container_width=True, hide_index=True,
    )

# ─────────────────────────── TAB 3: PROFILE ───────────────────────────
with tab3:
    st.subheader("Facility profile + risk score")
    names = sorted(violations["facility_name"].dropna().unique())
    name = st.selectbox("Pick a facility", names, index=0 if names else None)

    if name:
        sub = violations[violations["facility_name"] == name].sort_values("inspection_date")
        st.write(f"**{len(sub)}** violations across **{sub['inspection_date'].nunique()}** visits")

        fc1, fc2, fc3 = st.columns(3)
        fc1.metric("Total violations", len(sub))
        fc2.metric("Distinct visits", sub["inspection_date"].nunique())
        fc3.metric("Cities", sub["city"].nunique())

        # Score via model using this facility's actual history
        feat_row = features[features["facility_name"] == name].tail(1)
        if not feat_row.empty:
            scenario = {
                "vendor_name": name,
                "permit_current": True,
                "has_handwash_station": True,
                "handles_hot_cold_foods": bool(feat_row["has_temperature"].iloc[0]),
                "unapproved_sources": False,
                "pest_history": bool(feat_row["has_pests"].iloc[0]),
                "prior_visits": int(feat_row["prior_visits"].iloc[0]),
                "prior_violations_total": int(feat_row["prior_viol_sum"].iloc[0]),
                "prior_critical_total": int(feat_row["prior_crit_sum"].iloc[0]),
                "prior_viol_avg": float(feat_row["prior_viol_avg"].iloc[0]),
                "days_since_last_inspection": int(feat_row["days_since_last"].iloc[0]),
                "month": int(feat_row["month"].iloc[0]),
            }
            result = score_vendor(scenario)
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Compliance score", f"{result['compliance_score']}/100")
            sc2.metric("Risk band", result["risk_band"].upper())
            sc3.metric("Model confidence", result["model_confidence"])

        st.markdown("**Visit history**")
        st.dataframe(
            sub[["inspection_date", "city", "code_section", "violation_text"]]
            .sort_values("inspection_date", ascending=False),
            use_container_width=True, hide_index=True,
        )

# ─────────────────────────── TAB 4: SCORE SCENARIO ───────────────────────────
with tab4:
    st.subheader("Score a what-if vendor scenario")
    col1, col2 = st.columns(2)
    with col1:
        vname = st.text_input("Vendor name", "My Food Truck")
        permit = st.checkbox("Permit current", True)
        handwash = st.checkbox("Has handwash station", True)
        hotcold = st.checkbox("Handles hot/cold foods", True)
        pests = st.checkbox("Pest history", False)
        unapproved = st.checkbox("Uses unapproved sources", False)
    with col2:
        prior_visits = st.number_input("Prior inspections", 0, 50, 3)
        prior_tot = st.number_input("Prior violations total", 0, 500, 4)
        prior_crit = st.number_input("Prior critical violations", 0, 200, 0)
        days_since = st.number_input("Days since last inspection", -1, 2000, 180)
        month = st.slider("Month of inspection", 1, 12, 7)

    if st.button("🔮 Score vendor", type="primary"):
        scenario = {
            "vendor_name": vname,
            "permit_current": permit,
            "has_handwash_station": handwash,
            "handles_hot_cold_foods": hotcold,
            "pest_history": pests,
            "unapproved_sources": unapproved,
            "prior_visits": prior_visits,
            "prior_violations_total": prior_tot,
            "prior_critical_total": prior_crit,
            "prior_viol_avg": prior_tot / max(prior_visits, 1),
            "days_since_last_inspection": days_since,
            "month": month,
        }
        result = score_vendor(scenario)
        m1, m2, m3 = st.columns(3)
        m1.metric("Compliance score", f"{result['compliance_score']}/100")
        m2.metric("Risk band", result["risk_band"].upper())
        m3.metric("Model confidence", result["model_confidence"])

        st.markdown("### Recommended actions")
        if result["recommended_actions"]:
            for a in result["recommended_actions"]:
                icon = "🔴" if a["priority"] == "critical" else "🟠"
                st.markdown(f"{icon} **[{a['priority'].upper()}]** {a['action']}  \n<small>_{a['grounded_in']}_</small>", unsafe_allow_html=True)
        else:
            st.success("No critical actions — maintain current practices.")

        st.markdown("### Top risk contributors")
        st.dataframe(pd.DataFrame(result["top_risk_contributors"]), hide_index=True)

        st.markdown("### Permits required for this operation")
        core_required = {"sc_mobile_food_permit", "food_protection_cert",
                         "commissary_agreement", "general_liability_insurance"}
        if not permit:
            core_required.add("ny_sales_tax_cert")  # likely missing if permit lapsed
        if hotcold:
            core_required.add("fire_suppression_inspection")
        quick = [p for p in permits_doc["permits"] if p["id"] in core_required]
        st.dataframe(
            pd.DataFrame([{
                "Permit": p["name"],
                "Issuer": p["issuer"],
                "Renewal": p["renewal"],
                "Typical fee": f"${p['typical_fee_usd']}",
            } for p in quick]),
            hide_index=True, use_container_width=True,
        )
        st.caption("See the **📋 Required Permits** tab for full checklist and legal citations.")

        with st.expander("Full traceability (derived_from)"):
            st.json(result["derived_from"])

# ─────────────────────────── TAB 5: REQUIRED PERMITS ───────────────────────────
with tab5:
    st.subheader("Required permits to operate a mobile food vendor in Suffolk County")
    st.caption(
        f"Jurisdiction: **{permits_doc['jurisdiction']}** · "
        f"Last updated: {permits_doc['last_updated']}"
    )

    # Operation profile → filtered checklist
    st.markdown("### 1. Tell us about your operation")
    c1, c2, c3 = st.columns(3)
    with c1:
        op_cook = st.checkbox("Cooks with open flame / fryer / grill", True, key="p_cook")
        op_truck = st.checkbox("Motor vehicle (truck / trailer)", True, key="p_truck")
    with c2:
        op_town = st.checkbox("Operates on town streets / public property", True, key="p_town")
        op_parks = st.checkbox("Operates in parks or beaches", False, key="p_parks")
    with c3:
        op_events = st.checkbox("Only at events / festivals (≤14 days)", False, key="p_event")
        op_taxable = st.checkbox("Sells prepared food (collects sales tax)", True, key="p_tax")

    # Always required
    always = {
        "sc_mobile_food_permit", "food_protection_cert",
        "commissary_agreement", "general_liability_insurance",
    }
    if op_taxable:
        always.add("ny_sales_tax_cert")
    if op_truck:
        always.add("dmv_commercial_reg")
    if op_town:
        always.add("town_vending_permit")
    if op_cook:
        always.add("fire_suppression_inspection")

    required = [p for p in permits_doc["permits"] if p["id"] in always]
    optional = [p for p in permits_doc["permits"] if p["id"] not in always]

    st.markdown(f"### 2. Your permit checklist — **{len(required)} required**")
    total_low = total_high = 0
    for p in required:
        with st.expander(f"✅ {p['name']}  —  {p['issuer']}"):
            st.markdown(f"**Required for:** {p['required_for']}")
            st.markdown(f"**Legal authority:** {p['authority']}")
            st.markdown(f"**Typical fee:** ${p['typical_fee_usd']}")
            st.markdown(f"**Renewal:** {p['renewal']}")
            st.markdown("**Prerequisites:**")
            for pre in p["prerequisites"]:
                st.markdown(f"- {pre}")
            st.caption(f"Source: {p['source']}")

    # Rough fee estimate
    def fee_range(s):
        import re
        nums = [int(x.replace(",", "")) for x in re.findall(r"\d[\d,]*", str(s))]
        return (nums[0], nums[-1]) if nums else (0, 0)

    lows, highs = zip(*(fee_range(p["typical_fee_usd"]) for p in required)) if required else ((0,), (0,))
    st.info(
        f"💰 **Estimated annual fee range:** "
        f"${sum(lows):,} — ${sum(highs):,} "
        f"(excludes insurance deductibles and commissary rent)"
    )

    st.markdown("### 3. Conditional / situational permits")
    for cp in permits_doc["conditional_permits"]:
        st.markdown(f"- **{cp['name']}** — {cp['when']}  \n  _Issuer: {cp['issuer']}_")

    with st.expander("See all permit types (including ones not required for your profile)"):
        for p in optional:
            st.markdown(f"- **{p['name']}** — {p['issuer']}")

# ─────────────────────────── TAB 6: REGULATIONS ───────────────────────────
with tab6:
    st.subheader("Extracted regulations (grounded in source)")
    rq = st.text_input("Filter rules", "")
    for r in rules:
        blob = json.dumps(r).lower()
        if rq and rq.lower() not in blob:
            continue
        with st.expander(f"[{r.get('confidence','?').upper()}] {r['rule_id']} · {r.get('risk_type','')}"):
            st.write(r.get("requirement_summary", ""))
            st.caption(f"Source: {r['source']}")
            if r.get("section"):
                st.caption(f"Section: {r['section']}")

# ─────────────────────────── TAB 7: MODEL INFO ───────────────────────────
with tab7:
    st.subheader("Model performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC AUC", f"{model_report['roc_auc']:.3f}")
    m2.metric("PR AUC", f"{model_report['pr_auc']:.3f}")
    m3.metric("Train rows", f"{model_report['train_rows']:,}")
    m4.metric("Test rows", f"{model_report['test_rows']:,}")
    st.markdown("### Feature importances")
    st.bar_chart(
        pd.DataFrame(model_report["feature_importances"]).set_index("feature")
    )
    st.markdown("### Confusion matrix (test set)")
    cm = pd.DataFrame(
        model_report["confusion_matrix"],
        index=["actual_low", "actual_high"],
        columns=["pred_low", "pred_high"],
    )
    st.dataframe(cm)
