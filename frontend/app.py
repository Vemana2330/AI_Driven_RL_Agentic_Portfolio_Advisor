# frontend/app.py
import streamlit as st
import requests
import json

API_URL = "http://127.0.0.1:8000/run-portfolio-analysis"

st.set_page_config(
    page_title="RL Agentic Portfolio System",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI-Driven RL Agentic Portfolio Advisor")
st.caption("Reinforcement Learning + Multi-Agent Market Intelligence")

# ----------------------------------------
# Sidebar — User Inputs
# ----------------------------------------
st.sidebar.header("Investor Profile")

investment_amount = st.sidebar.number_input(
    "💰 Investment Amount ($)",
    min_value=100,
    max_value=1_000_000,
    value=10_000
)

risk_level = st.sidebar.selectbox(
    "⚖ Risk Appetite",
    ["low", "medium", "high"],
    index=1
)

duration_months = st.sidebar.selectbox(
    "⏳ Investment Duration (Months)",
    [1, 3, 6, 12],
    index=2
)

# ----------------------------------------
# Execute Workflow — Backend Call
# ----------------------------------------
if st.button("🚀 Run Portfolio Intelligence"):

    payload = {
        "investment_amount": investment_amount,
        "risk_level": risk_level,
        "duration_months": duration_months
    }

    st.info("⏳ Running multi-agent RL and market analysis... this may take a moment")

    try:
        response = requests.post(API_URL, json=payload)
        result = response.json()
    except Exception as e:
        st.error("❌ Unable to reach backend API — Is FastAPI running?")
        st.exception(e)
        st.stop()

    if result.get("status") != "success":
        st.error("⚠ Error running analysis")
        st.json(result)
        st.stop()

    output = result.get("response", {})  # Because backend wraps it

    st.success("✔ Portfolio Analysis Complete")

    # ----------------------------------------
    # Final Investment Report
    # ----------------------------------------
    st.subheader("📄 Final Investment Report")
    report_text = output if isinstance(output, str) else str(output)
    st.write(report_text)

    st.divider()
    st.header("🔬 Detailed AI Outputs")

    def safe_show(section_title, key):
        st.subheader(section_title)
        val = result.get("response", {}).get(key, {})
        try:
            st.json(val)
        except:
            st.write(val)

    # RL & Multi-Agent Insights
    safe_show("🧠 DQN Allocation Output", "dqn_allocation")
    safe_show("📊 PPO Phase-Based Allocation", "policy_phases")
    safe_show("📰 News & Sentiment", "sentiment")
    safe_show("⚠ Risk Evaluation", "risk")
    safe_show("📈 Technical Indicators", "technical")
    safe_show("💰 Financial Performance", "performance")
    safe_show("🌍 Macro Economic Regime", "macro")
