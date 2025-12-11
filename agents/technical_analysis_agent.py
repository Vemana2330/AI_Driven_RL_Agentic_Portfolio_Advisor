"""
technical_analysis_agent.py
Agent that performs technical analysis on the AI sector stock universe.
"""

import os
import json
import numpy as np
import pandas as pd
from crewai import Agent

# Import function-based tools
from tools.custom_tools import (
    compute_technical_indicators,         # replaces TechnicalIndicatorsTool.run
    compute_support_resistance,    # replaces SupportResistanceTool.run
    compute_volume_trend,              # replaces VolumeAnalysisTool.run
    compute_technical_score      # replaces TechnicalScoringTool.run
)


class TechnicalAnalysisAgent:
    """
    Provides:
      - self.agent    : CrewAI LLM-driven technical analysis evaluator
      - analyze(...) : Backend computation for UI and pipeline
    """

    TICKERS = [
        "AMD", "AMZN", "AVGO", "CRM", "GOOG",
        "INTC", "META", "MSFT", "NVDA", "ORCL", "TSLA"
    ]

    def __init__(
        self,
        rl_model_dir: str = "models/rl_model_v1",
        verbose: bool = True,
    ):
        self.rl_model_dir = rl_model_dir

        # Load price_cols.json
        price_cols_path = os.path.join(self.rl_model_dir, "price_cols.json")
        with open(price_cols_path, "r") as f:
            self.price_cols = json.load(f)

        # Load the RL inference CSV (historical market prices)
        csv_path = os.path.join(self.rl_model_dir, "inference_market_252days.csv")
        self.df = pd.read_csv(csv_path)

        # Register functional tools for CrewAI agent
        self.tools = [
            {
                "name": "compute_technical_indicators",
                "description": "Calculate RSI, MACD, Bollinger Bands, SMA crossovers.",
                "tool": compute_technical_indicators
            },
            {
                "name": "compute_support_resistance",
                "description": "Estimate price-based support and resistance ranges.",
                "tool": compute_support_resistance
            },
            {
                "name": "compute_volume_trend",
                "description": "Analyze volatility of trading volume for confidence signals.",
                "tool": compute_volume_trend
            },
            {
                "name": "compute_technical_score",
                "description": "Generate final technical score (0–100) with reasoning.",
                "tool": compute_technical_score
            }
        ]

        # CrewAI LLM Agent
        self.agent = Agent(
            role="AI Sector Technical Analyst",
            goal=(
                "Analyze price trends, momentum and volatility to determine "
                "technical strength, entry timing and trend reliability."
            ),
            backstory=(
                "You are a systematic technical analyst specializing in AI sector "
                "pattern recognition and momentum strategy generation."
            ),
            tools=self.tools,
            allow_delegation=False,
            verbose=verbose,
        )


    # ---------------------------------------------------------------------
    # INTERNAL price + volume extractor
    # ---------------------------------------------------------------------
    def _get_series(self, ticker: str, time_window_days: int):
        close_col = f"{ticker}_Close"
        vol_col = f"{ticker}_Volume"

        if close_col not in self.df.columns:
            return None, None

        sub = self.df.tail(time_window_days)
        close_series = sub[close_col].dropna().tolist()

        if vol_col in sub.columns:
            volume_series = sub[vol_col].fillna(0).tolist()
        else:
            volume_series = [1.0] * len(close_series)

        return close_series, volume_series


    # ---------------------------------------------------------------------
    # DIRECT TECHNICAL ANALYSIS (NO LLM)
    # ---------------------------------------------------------------------
    def analyze(
        self,
        time_window_days: int = 180,
        sensitivity: str = "medium",
    ):
        results = []

        time_window_days = min(time_window_days, len(self.df))

        for ticker in self.TICKERS:
            close_series, volume_series = self._get_series(ticker, time_window_days)
            if not close_series:
                continue

            # 1️⃣ Indicators
            ind = compute_technical_indicators(close_prices=close_series)

            # 2️⃣ Support / Resistance
            sr = compute_support_resistance(close_prices=close_series)

            # 3️⃣ Volume momentum
            vol = compute_volume_trend(volume_series=volume_series)

            # 4️⃣ Score output
            score = compute_technical_score(
                indicators=ind,
                support_resistance=sr,
                volume_info=vol,
                sensitivity=sensitivity,
            )

            results.append({
                "ticker": ticker,
                "technical_score": score["technical_score"],
                "trend": score["trend"],
                "rsi_status": score["rsi_status"],
                "momentum": score["momentum"],
                "entry_zone": score["entry_zone"],
                "volume_trend": score["volume_trend"],
                "volume_strength": score["volume_strength"],
                "support_levels": sr.get("support_levels", []),
                "resistance_levels": sr.get("resistance_levels", []),
                "summary": score["summary"],
            })

        return {
            "time_window_days": time_window_days,
            "sensitivity": sensitivity,
            "results": results,
        }


# Local Debug
if __name__ == "__main__":
    ta = TechnicalAnalysisAgent(verbose=False)
    output = ta.analyze(180, "medium")
    print("Tickers analyzed:", len(output["results"]))
    print("Sample:", output["results"][0] if output["results"] else "No output")
