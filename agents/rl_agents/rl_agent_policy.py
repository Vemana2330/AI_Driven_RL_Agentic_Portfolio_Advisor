"""
rl_agent_policy.py
Phase-based PPO / Policy inference for AI Portfolio Allocation
"""

import os
import numpy as np
from tools.phase_policy_inference import PhasePolicyInference
import pandas as pd
import json
import pickle


class PhasePolicyAgent:
    def __init__(
        self,
        agent_dir="models/rl_policy_model",
        dqn_market_dir="models/rl_model_v1"
    ):
        # Load metadata
        with open(os.path.join(dqn_market_dir, "state_cols.json")) as f:
            self.state_cols = json.load(f)
        with open(os.path.join(dqn_market_dir, "price_cols.json")) as f:
            self.price_cols = json.load(f)

        # Load RL inference history (used as state input for policy)
        df = pd.read_csv(os.path.join(dqn_market_dir, "inference_market_252days.csv"))

        # Initialize policy inference engine
        self.policy = PhasePolicyInference(
            model_dir=agent_dir,
            df=df,
            state_cols=self.state_cols,
            price_cols=self.price_cols
        )

        # Extract tickers (Close columns)
        self.tickers = [col.replace("_Close", "") for col in self.price_cols[:-1]]


    def run_phase_plan(
        self,
        dqn_weights,
        horizon_months=6,
        investment_amount=5000
    ):
        """
        dqn_weights (list or np.array): allocation output from DQN inference
        """
        # Ensure numpy float32
        dqn_weights = np.array(dqn_weights, dtype=np.float32)

        # ✨ SAFETY: Check for mismatch with # tickers
        if len(dqn_weights) != len(self.tickers):
            raise ValueError(
                f"DQN weight size mismatch: expected {len(self.tickers)}, got {len(dqn_weights)}"
            )

        result = self.policy.run(
            dqn_weights=dqn_weights,
            horizon_months=horizon_months,
            investment_amount=investment_amount
        )

        # ✨ Return JSON-safe values + tickers
        return {
            **result,
            "tickers": self.tickers,
            "phase_allocation_plan": result.get("phase_allocation_plan", []),
            "expected_final_value": float(result.get("expected_final_value", 0)),
        }


# Debug test
if __name__ == "__main__":
    agent = PhasePolicyAgent()
    output = agent.run_phase_plan(dqn_weights=[0.2]*11, horizon_months=6, investment_amount=5000)
    print(output)
