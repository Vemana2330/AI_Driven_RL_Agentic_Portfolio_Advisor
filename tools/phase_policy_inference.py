"""
phase_policy_inference.py
Reusable inference module for Phase PPO Policy Allocation
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.distributions import Normal
import torch.nn as nn


# =========================================
# PPO Actor Model
# =========================================
class PPOActorPhase(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = self.shared(state)
        mu = self.mu_head(x)
        std = torch.exp(self.log_std).expand_as(mu)
        return mu, std


# =========================================
# PHASE ENVIRONMENT
# =========================================
class PhaseDurationEnv:
    def __init__(self, df, state_cols, price_cols, num_phases=4, max_horizon_days=252):
        self.df = df.reset_index(drop=True)
        self.state_cols = state_cols
        self.price_cols = price_cols
        self.num_stocks = len(price_cols) - 1
        self.num_phases = num_phases
        self.max_horizon_days = max_horizon_days

        # runtime state
        self.current = None
        self.t = None
        self.horizon_days = None
        self.phase_length = None
        self.phase_idx = None
        self.base_weights = None
        self.weights = None

    def reset(self, base_weights, horizon_days):
        self.horizon_days = horizon_days
        self.t = 0
        self.phase_length = max(1, horizon_days // self.num_phases)
        self.phase_idx = 0

        w = np.asarray(base_weights, dtype=np.float32)
        w = w / (w.sum() + 1e-8)

        self.base_weights = w
        self.weights = w.copy()
        return self._get_state()

    def _get_state(self):
        """State = zeros + time fraction + phase one-hot + base weights"""
        feats = np.zeros(len(self.state_cols), dtype=np.float32)
        time_frac = np.array([self.t / max(1, self.horizon_days)], dtype=np.float32)
        phase_one_hot = np.eye(self.num_phases)[self.phase_idx].astype(np.float32)
        return np.concatenate([feats, time_frac, phase_one_hot, self.base_weights], axis=0)

    def step(self, action_logits):
        # Softmax over action logits to get valid probability distribution
        alloc = np.exp(action_logits - np.max(action_logits))
        alloc = alloc / (alloc.sum() + 1e-8)

        # Blend base weight × action emphasis
        raw = alloc * self.base_weights
        self.weights = raw / (raw.sum() + 1e-8)

        self.t += 1
        if self.t % self.phase_length == 0:
            self.phase_idx = min(self.num_phases - 1, self.phase_idx + 1)

        done = self.t >= self.horizon_days
        return self._get_state(), 0, done, {"phase_idx": self.phase_idx, "weights": self.weights.copy()}


# =========================================
# POLICY INFERENCE MODULE
# =========================================
class PhasePolicyInference:
    HORIZON_MAP = {1: 21, 3: 63, 6: 126, 12: 252}
    PHASE_NAMES = ["Entry", "Ramp-up", "Peak", "Exit"]

    def __init__(self, model_dir="models/rl_policy_model", df=None, state_cols=None, price_cols=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load JSON metadata
        with open(os.path.join(model_dir, "model_dims.json")) as f:
            meta = json.load(f)

        self.state_dim = meta["state_dim"]
        self.action_dim = meta["action_dim"]
        self.NUM_PHASES = meta["NUM_PHASES"]
        self.state_cols = state_cols or meta["STATE_COLS"]
        self.price_cols = price_cols or meta["PRICE_COLS"]

        # PPO ACTOR
        actor_path = os.path.join(model_dir, "ppo_phase_actor.pth")
        self.actor = PPOActorPhase(self.state_dim, self.action_dim).to(self.device)
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.actor.eval()

        # PHASE ENV
        self.env = PhaseDurationEnv(df, self.state_cols, self.price_cols)

        # Extract final tickers
        self.tickers = [c.replace("_Close", "") for c in self.price_cols[:-1]]


    def run(self, dqn_weights, horizon_months, investment_amount):
        # Validate input horizon
        if horizon_months not in self.HORIZON_MAP:
            raise ValueError(f"horizon_months must be one of {list(self.HORIZON_MAP.keys())}")

        horizon_days = self.HORIZON_MAP[horizon_months]
        state = self.env.reset(base_weights=dqn_weights, horizon_days=horizon_days)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device)

        all_phases = [[] for _ in range(self.NUM_PHASES)]

        # Inference loop
        while True:
            with torch.no_grad():
                mu, std = self.actor(state_t.unsqueeze(0))
                action = Normal(mu, std).sample().squeeze(0).cpu().numpy()

            next_state, _, done, info = self.env.step(action)
            all_phases[info["phase_idx"]].append(info["weights"])

            state_t = torch.tensor(next_state, dtype=torch.float32, device=self.device)

            if done:
                break

        # Build output formatted plan
        plan = []
        for i in range(self.NUM_PHASES):
            if len(all_phases[i]) == 0:
                avg = np.zeros(len(self.tickers))
            else:
                avg = np.mean(all_phases[i], axis=0)
                avg = np.nan_to_num(avg, nan=0.0, posinf=0.0, neginf=0.0)

            plan.append({
                "phase_index": i,
                "phase_name": self.PHASE_NAMES[i],
                "allocations": {self.tickers[j]: float(avg[j]) for j in range(len(self.tickers))}
            })

        return {
            "investment_amount": investment_amount,
            "horizon_months": horizon_months,
            "phase_allocation_plan": plan
        }
