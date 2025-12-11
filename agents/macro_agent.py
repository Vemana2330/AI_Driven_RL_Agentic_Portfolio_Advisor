"""
macro_agent.py
Agent that evaluates macroeconomic regime and risk environment.
"""

from crewai import Agent

# Import function-based tools
from tools.built_in_tools import fetch_macro_indicators         # function replacing MacroEconomicFetcherTool.run
from tools.custom_tools import score_macro_conditions             # function replacing MacroScoringTool.run


class MacroEconomicAgent:
    """
    Provides:
      - self.agent : CrewAI agent for LLM-driven macro reasoning
      - run()     : backend direct macro score + regime
    """

    def __init__(self, verbose=True):

        # CrewAI Tool Mapping (function-based)
        self.tools = [
            {
                "name": "fetch_macro_indicators",
                "description": "Retrieve inflation, interest rates, unemployment, VIX, oil prices, GDP data.",
                "tool": fetch_macro_indicators
            },
            {
                "name": "macro_regime_scoring",
                "description": "Score economic regime (Bullish, Neutral, High-Risk) using weighted macro metrics.",
                "tool": score_macro_conditions
            }
        ]

        # LLM Agent
        self.agent = Agent(
            role="Macro Economic Intelligence Analyst",
            goal=(
                "Interpret macroeconomic indicators to evaluate investment climate "
                "and determine if markets are entering Bullish, Neutral, or High-Risk phases."
            ),
            backstory=(
                "You are an expert at connecting inflation trends, interest rates, "
                "employment health, market volatility and growth cycles to generate "
                "regime-based portfolio strategy recommendations."
            ),
            tools=self.tools,
            allow_delegation=False,
            verbose=verbose
        )

    # ---------------------------------------------------------------------
    # Direct backend call (no LLM)
    # ---------------------------------------------------------------------
    def run(self):
        macro_data = fetch_macro_indicators()          # fetch metrics
        scored = score_macro_conditions(macro_data)      # score regime
        return scored


# Debug
if __name__ == "__main__":
    agent = MacroEconomicAgent(verbose=False)
    print(agent.run())
