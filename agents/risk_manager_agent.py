"""
risk_manager_agent.py
Agent that evaluates portfolio risk per ticker based on user risk preference.
"""

from crewai import Agent
from tools.custom_tools import compute_risk_metrics


class RiskManagerAgent:
    """
    Wrapper class used by ControllerAgent and backend routes.
    Provides:
      - self.agent : CrewAI Agent usable in LLM task workflow
      - evaluate() : direct backend-safe function call (no LLM)
    """

    def __init__(self, verbose: bool = True):

        # CrewAI-compatible tool registration (function-based)
        self.tools = [
            {
                "name": "compute_risk_metrics",
                "description": (
                    "Compute volatility, max drawdown, and detect risk level "
                    "for each stock. Matches results against user risk preference: low/medium/high."
                ),
                "tool": compute_risk_metrics
            }
        ]

        # LLM Agent
        self.agent = Agent(
            role="AI Portfolio Risk Manager",
            goal=(
                "Evaluate which AI stocks align with the investor's risk tolerance "
                "using volatility, drawdown, and risk-band classification."
            ),
            backstory=(
                "You are a financial risk specialist trained in market psychology "
                "and volatility pattern analysis, focused on protecting investor capital."
            ),
            tools=self.tools,
            allow_delegation=False,
            verbose=verbose,
        )

    # Direct backend call — NON-LLM
    def evaluate(self, risk_preference: str = "medium"):
        """
        Direct backend call — bypasses LLM for fast inference.
        """
        return compute_risk_metrics(risk_preference)
        

# Debug test
if __name__ == "__main__":
    rm = RiskManagerAgent(verbose=False)
    result = rm.evaluate("low")
    print(result)
