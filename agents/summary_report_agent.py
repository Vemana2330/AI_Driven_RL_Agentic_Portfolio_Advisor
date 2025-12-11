"""
summary_report_agent.py
Agent that synthesizes all intelligence outputs into a single
professional investment report for UI display.
"""

from crewai import Agent
from textwrap import dedent


class SummaryReportAgent:
    """
    FINAL report writer — consumes all aggregated results
    (RL allocation, sentiment, fundamentals, technicals, macro, risk)
    and produces a hedge-fund-grade structured investment report.
    """

    def __init__(self, verbose=True):

        # No tools used — agent is purely generative
        self.tools = []

        self.agent = Agent(
            role="Senior AI Portfolio Research Strategist",
            goal=(
                "Produce a hedge-fund-grade investment report unifying reinforcement "
                "learning allocation, technical momentum, macroeconomic regime, "
                "sentiment intelligence and fundamental scoring."
            ),
            backstory=(
                "You are a seasoned investment strategist with expertise spanning "
                "macroeconomics, fundamental equity evaluation, technical momentum "
                "analysis, and quantitative portfolio optimization."
            ),
            tools=self.tools,
            allow_delegation=False,
            verbose=verbose
        )

    # ----------------------------------------------------------------------
    # Public method called from Controller / API
    # ----------------------------------------------------------------------
    def generate_report(self, aggregated_input: dict):

        # Safe extract with fallbacks to prevent KeyErrors during debugging
        user_inputs = aggregated_input.get("user_inputs", {})
        risk_level = user_inputs.get("risk_level", "medium")
        investment_amount = user_inputs.get("investment_amount", "Unknown")
        duration_months = user_inputs.get("duration_months", "Unknown")

        prompt = dedent(f"""
        You are generating the FINAL investment report for the client.

        USER PROFILE
        ------------
        Risk Level: {risk_level}
        Investment Amount: ${investment_amount}
        Duration: {duration_months} months


        === DATA INSIGHTS FROM MULTI-AGENT AI SYSTEM ===

        REINFORCEMENT LEARNING RECOMMENDATION
        --------------------------------------
        Allocation Recommendation:
        {aggregated_input.get('dqn_allocation', {})}

        Policy-Based Phase Allocation:
        {aggregated_input.get('policy_phases', {})}


        PERFORMANCE (Fundamentals)
        ---------------------------
        {aggregated_input.get('performance', {})}


        SENTIMENT ANALYSIS
        -------------------
        {aggregated_input.get('sentiment', {})}


        RISK ANALYSIS
        --------------
        {aggregated_input.get('risk', {})}


        TECHNICAL ANALYSIS SIGNALS
        ---------------------------
        {aggregated_input.get('technical', {})}


        MACROECONOMIC REGIME EVALUATION
        --------------------------------
        {aggregated_input.get('macro', {})}


        === YOUR TASK ===

        Create a structured institutional-grade investment report containing:

        1️⃣ Executive Summary (5–8 sentences)
        2️⃣ Recommended Portfolio Allocation (quantified and justified)
        3️⃣ Phase-wise Investment Approach (timing guidance)
        4️⃣ Company Breakdown:
              - Sentiment
              - Technical stance (entry / hold / reduce)
              - Risk classification
              - Fundamental strength score
        5️⃣ Macro Regime Impact — explain how economic conditions modify this strategy
        6️⃣ Risk Management & Capital Protection Rules
        7️⃣ Scenario Forecast (Best / Base / Worst case)
        8️⃣ Final Action Plan: bullet format, directly actionable next steps

        MOST IMPORTANT:
        After the report, output a final compact JSON block:

        {{
          "recommendation": "...",
          "warnings": [...],
          "best_case_return": "...",
          "worst_case_risk": "...",
          "top_picks": [...],
          "avoid_or_reduce": [...]
        }}

        Style:
        - Institutional
        - Neutral tone
        - Evidence-based
        - No emojis
        - No hype language
        - Clear investment rationale
        """)

        return self.agent.run(prompt)


# Local debug
if __name__ == "__main__":
    dummy = {
        "user_inputs": {
            "risk_level": "medium",
            "investment_amount": 10000,
            "duration_months": 12
        },
        "dqn_allocation": {"NVDA": 40, "MSFT": 30, "GOOG": 20, "TSLA": 10},
        "policy_phases": {},
        "performance": {},
        "sentiment": {},
        "risk": {},
        "technical": {},
        "macro": {}
    }
    sr = SummaryReportAgent(verbose=False)
    out = sr.generate_report(dummy)
    print(out)
