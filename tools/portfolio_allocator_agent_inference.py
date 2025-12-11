"""
portfolio_allocator_agent_inference.py
Reusable inference module for DQN allocation — lightweight alternative
to full rl_agent_dqn.py when only allocation preview is needed.
"""

import torch
import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path

# Import Model Class (Must match training)
from models.model_definition import DQN  


class PortfolioAllocatorInference:
    def __init__(
        self,
        agent_dir="models/rl_model_v1",
        device="cpu"
    ):
        """
        Loads trained DQN model, scaler, metadata and 252-day market data.
        """

        BASE_DIR = Path(__file__).resolve().parents[1]
        self.agent_dir = BASE_DIR / agent_dir
        self.device = torch.device(device)

        # Paths
        self.model_path = self.agent_dir / "dqn_model.pth"
        self.scaler_path = self.agent_dir / "scaler.pkl"
        self.state_cols_path = self.agent_dir / "state_cols.json"
        self.price_cols_path = self.agent_dir / "price_cols.json"
        self.market_data_path = self.agent_dir / "inference_market_252days.csv"

        # ---------------- Metadata ----------------
        with open(self.state_cols_path) as f:
            self.state_cols = json.load(f)

        with open(self.price_cols_path) as f:
            self.price_cols = json.load(f)

        # Extract tickers from close columns
        self.tickers = [col.replace("_Close", "") for col in self.price_cols[:-1]]

        # ---------------- Scaler ------------------
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        # ---------------- Market Data --------------
        self.df = pd.read_csv(self.market_data_path)

        # Protect against missing or re-ordered columns
        for col in self.state_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing state column in CSV: {col}")

        # Scale state variables
        self.df[self.state_cols] = self.scaler.transform(self.df[self.state_cols])

        # ---------------- Model -------------------
        state_dim = len(self.state_cols)
        action_dim = 5
        self.model = DQN(state_dim, action_dim).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

    # -----------------------------------------------------
    # INFERENCE
    # -----------------------------------------------------
    def run(self, investment_amount: float, risk_level: str = "medium"):
        """
        Runs lightweight allocation simulation, used for quick portfolio preview.
        """

        df = self.df.copy()

        portfolio_value = float(investment_amount)
        num_stocks = len(self.price_cols) - 1
        weights = np.zeros(num_stocks)

        pv_curve = []
        weight_history = []

        # Risk multiplier
        risk_multiplier = {"low": 0.7, "medium": 1.0, "high": 1.3}.get(risk_level.lower(), 1.0)

        for _, row in df.iterrows():

            # Prepare state
            state = row[self.state_cols].values.astype(np.float32)
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            # Select action
            with torch.no_grad():
                action = int(self.model(state_t).argmax().item())

            # Simulate returns
            returns = row[[col + "_ret" for col in self.price_cols[:-1]]].values
            portfolio_value *= (1 + np.dot(weights, returns) * risk_multiplier)

            # Soft rebalance logic
            if action < num_stocks:
                target = np.zeros(num_stocks)
                target[action] = 1.0
                weights = 0.85 * weights + 0.15 * target
                if weights.sum() > 0:
                    weights = weights / weights.sum()

            pv_curve.append(float(portfolio_value))
            weight_history.append(weights.copy())

        # Average allocation based on final 40 periods
        recent_weights = np.mean(weight_history[-40:], axis=0) if len(weight_history) else weights

        return {
            "initial_investment": investment_amount,
            "final_value": float(portfolio_value),
            "profit": float(portfolio_value - investment_amount),
            "return_%": float(((portfolio_value / investment_amount) - 1) * 100),
            "risk_level": risk_level,
            "tickers": self.tickers,
            "average_weights": np.round(recent_weights, 4).tolist(),
            "portfolio_growth_curve": [float(v) for v in pv_curve],
            "max_value": float(max(pv_curve)),
            "min_value": float(min(pv_curve)),
        }


# ---------------------- Local Debug ----------------------
if __name__ == "__main__":
    allocator = PortfolioAllocatorInference()
    result = allocator.run(5000, risk_level="medium")
    print(result)
