"""
rl_agent_dqn.py
Reusable inference module for our trained DQN portfolio allocator
"""

import os
import json
import torch
import pickle
import pandas as pd
import numpy as np
from models.model_definition import DQN


class PortfolioAllocatorDQN:
    def __init__(
        self,
        agent_dir="models/rl_model_v1",
        device="cpu"
    ):
        self.agent_dir = agent_dir
        self.device = torch.device(device)

        # Load state columns
        with open(os.path.join(agent_dir, "state_cols.json"), "r") as f:
            self.state_cols = json.load(f)

        # Load price columns
        with open(os.path.join(agent_dir, "price_cols.json"), "r") as f:
            self.price_cols = json.load(f)

        # Load scaler
        with open(os.path.join(agent_dir, "scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)

        # Load market input
        self.market_csv_path = os.path.join(agent_dir, "inference_market_252days.csv")
        self.market_df = pd.read_csv(self.market_csv_path)

        # Load trained DQN model
        state_dim = len(self.state_cols)
        action_dim = 5  # discrete actions
        self.model = DQN(state_dim, action_dim)
        self.model.load_state_dict(torch.load(os.path.join(agent_dir, "dqn_model.pth"), map_location=device))
        self.model.eval()

        # Extract tickers from price columns
        self.tickers = [col.replace("_Close", "") for col in self.price_cols[:-1]]

    # ---------------------------------------------------------
    # Prepare state
    # ---------------------------------------------------------
    def _prepare_state(self, row):
        state_vector = row[self.state_cols].values.astype(np.float32)
        return torch.tensor(state_vector, dtype=torch.float32, device=self.device).unsqueeze(0)

    # ---------------------------------------------------------
    # Action application
    # ---------------------------------------------------------
    def _apply_action(self, action, weights, row):
        momentum_cols = [c for c in row.index if "_mom20" in c and "QQQ" not in c]
        vol_cols = [c for c in row.index if "_vol20" in c and "QQQ" not in c]

        new_weights = weights.copy()

        # Action definitions
        if action == 0:
            pass  # Hold

        elif action == 1:  # Momentum tilt
            rank = np.argsort(-row[momentum_cols].values)
            new_weights[:] = 0
            if len(rank) > 0:
                alloc = np.array([0.6, 0.4][:min(2, len(rank))])
                new_weights[rank[:2]] = alloc

        elif action == 2:  # Low volatility tilt
            rank = np.argsort(row[vol_cols].values)
            new_weights[:] = 0
            if len(rank) > 0:
                alloc = np.array([0.5, 0.5][:min(2, len(rank))])
                new_weights[rank[:2]] = alloc

        elif action == 3:  # Defensive reduction
            new_weights = new_weights * 0.4

        elif action == 4:  # Aggressive AI bet
            picks = ["NVDA", "META", "MSFT"]
            chosen = []
            for idx, col in enumerate(self.price_cols[:-1]):
                if any(x in col for x in picks):
                    chosen.append(idx)
            new_weights[:] = 0
            if len(chosen) > 0:
                new_weights[chosen] = 1.0 / len(chosen)

        # Normalize — ALWAYS
        if new_weights.sum() > 0:
            new_weights = new_weights / new_weights.sum()

        return new_weights

    # ---------------------------------------------------------
    # RUN INFERENCE
    # ---------------------------------------------------------
    def run_inference(
        self,
        investment_amount: float,
        risk_level: str = "medium",
        custom_market_csv=None
    ):
        df = pd.read_csv(custom_market_csv) if custom_market_csv else self.market_df.copy()

        # Scale state features
        df[self.state_cols] = self.scaler.transform(df[self.state_cols])

        portfolio_value = float(investment_amount)
        weights = np.zeros(len(self.price_cols) - 1)
        pv_curve = []
        weight_history = []
        actions_taken = []

        risk_multiplier = {"low": 0.7, "medium": 1.0, "high": 1.4}.get(risk_level.lower(), 1.0)

        for _, row in df.iterrows():
            state_t = self._prepare_state(row)

            with torch.no_grad():
                action = int(self.model(state_t).argmax().item())
            actions_taken.append(action)

            weights = self._apply_action(action, weights, row)

            # Portfolio returns
            ret_cols = [c + "_ret" for c in self.price_cols[:-1]]
            if all(col in row for col in ret_cols):
                returns = row[ret_cols].values
                portfolio_value *= (1 + np.dot(weights, returns) * risk_multiplier)

            pv_curve.append(portfolio_value)
            weight_history.append(weights.copy())

        # Average portfolio weights (last 50 steps)
        recent_weights = np.mean(weight_history[-50:], axis=0) if weight_history else weights

        return {
            "initial_investment": investment_amount,
            "final_value": portfolio_value,
            "profit": portfolio_value - investment_amount,
            "return_%": ((portfolio_value / investment_amount) - 1) * 100,
            "risk_level": risk_level,
            "tickers": self.tickers,
            "average_weights": np.round(recent_weights, 4).tolist(),
            "portfolio_growth_curve": pv_curve,
            "max_value": max(pv_curve),
            "min_value": min(pv_curve),
            "actions_taken": actions_taken,  # NEW explainability feature
        }


if __name__ == "__main__":
    agent = PortfolioAllocatorDQN()
    output = agent.run_inference(investment_amount=5000, risk_level="high")
    print(output)
