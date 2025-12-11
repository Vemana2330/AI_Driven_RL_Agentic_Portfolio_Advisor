"""
performance_agent.py
Evaluates company fundamentals using financial data & scoring.
"""

from crewai import Agent

# Import the function-style tools (not classes)
from tools.built_in_tools import fetch_yfinance_company_data
from tools.custom_tools import extract_financial_metrics, score_fundamentals


class PerformanceAgent:
    def __init__(self, verbose=True):

        # CrewAI-compatible tool registry
        self.tools = [
            {
                "name": "fetch_yfinance_company_data",
                "description": "Retrieve balance sheet, cash flow, valuations and KPIs from Yahoo Finance.",
                "tool": fetch_yfinance_company_data
            },
            {
                "name": "extract_financial_metrics",
                "description": "Calculate ratios like profit margin, revenue growth, ROE, PEG, D/E, FCF trend.",
                "tool": extract_financial_metrics
            },
            {
                "name": "fundamental_health_scoring",
                "description": "Convert financial metrics into normalized fundamental score (0–100).",
                "tool": score_fundamentals
            }
        ]

        # LLM Agent
        self.agent = Agent(
            role="AI Company Performance Analyst",
            goal=(
                "Evaluate the company’s fundamental financial performance and assign "
                "a health score using profitability, growth, valuation, and leverage metrics."
            ),
            backstory=(
                "You specialize in financial statement analysis and valuation modeling, "
                "identifying strong outperformers in the AI sector."
            ),
            tools=self.tools,
            allow_delegation=False,
            verbose=verbose,
        )

    # Backend direct call — No LLM required to compute values
    def evaluate(self, ticker: str):
        # Step 1: Fetch financial results
        financials = fetch_yfinance_company_data(ticker)

        # Step 2: Compute ratios + KPIs
        metrics = extract_financial_metrics(financials)

        # Step 3: Produce scoring
        score = score_fundamentals(metrics)

        return {
            "ticker": ticker,
            "financial_metrics": metrics,
            "fundamental_score": score.get("fundamental_health_score", None),
            "analysis_details": score
        }


# Debug
if __name__ == "__main__":
    agent = PerformanceAgent(verbose=False)
    print(agent.evaluate("NVDA"))
