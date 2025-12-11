"""
feedback_manager.py
Handles reinforcement learning feedback signals for Portfolio Agents.
Implements:
1️⃣ Rule-based reward/penalty using output.get("status")
2️⃣ Optional LLM-based evaluation (+1, 0, -1)
"""

import json
import os
from langchain_openai import ChatOpenAI

from config import config
from logging_utils import logger


class FeedbackManager:
    """
    Minimal reinforcement feedback manager similar to AI Travel Planner.
    Tracks + rewards / penalties for each agent output.
    """

    def __init__(self):
        # RL memory file path
        self.memory_file = config.paths.rl_memory_file
        self.memory = self._load_memory()

        # Optional LLM-based evaluator
        try:
            self.llm = ChatOpenAI(
                model=config.llm.model_name,
                temperature=config.llm.temperature,
                api_key=config.api_keys.openai_api_key,
            )
        except Exception:
            self.llm = None
            logger.warning("[FeedbackManager] LLM evaluator unavailable.")

    # ---------------------------------------------------------
    # Memory Helpers
    # ---------------------------------------------------------
    def _load_memory(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r") as f:
                    return json.load(f)
            except:
                logger.error("[FeedbackManager] Failed to load RL memory file.")
        return {}

    def _save_memory(self):
        try:
            with open(self.memory_file, "w") as f:
                json.dump(self.memory, f, indent=4)
        except:
            logger.error("[FeedbackManager] Failed to save RL memory.")

    # ---------------------------------------------------------
    # Reward / Penalize
    # ---------------------------------------------------------
    def reward(self, agent_name):
        self.memory[agent_name] = self.memory.get(agent_name, 0) + 1
        self._save_memory()
        logger.info(f"[Feedback] ✔ REWARD applied → {agent_name} | Score = {self.memory[agent_name]}")

    def penalize(self, agent_name):
        self.memory[agent_name] = self.memory.get(agent_name, 0) - 1
        self._save_memory()
        logger.info(f"[Feedback] ✖ PENALTY applied → {agent_name} | Score = {self.memory[agent_name]}")

    # ---------------------------------------------------------
    # LLM Output Quality Scoring (Optional)
    # ---------------------------------------------------------
    def evaluate_output(self, agent_name: str, output: dict) -> int:
        """
        LLM evaluates the quality:
        +1 = useful & correct
         0 = acceptable
        -1 = unclear, incorrect, useless
        """
        if not self.llm:
            return 0
        
        try:
            prompt = f"""
Evaluate this agent output for clarity, correctness and usefulness.

Return ONLY one value:
+1 (good) | 0 (neutral) | -1 (bad)

Agent Name: {agent_name}
Output JSON:
{output}
"""

            # Using `.invoke()` — updated method
            response = self.llm.invoke(prompt).content.strip()

            if response.startswith("+1"):
                return +1
            if response.startswith("-1"):
                return -1
            return 0

        except Exception as e:
            logger.error(f"[FeedbackManager] LLM evaluation failed: {e}")
            return 0

    # ---------------------------------------------------------
    # Main Feedback Application Pipeline
    # ---------------------------------------------------------
    def apply_feedback(self, agent_name: str, output: dict):
        """
        AI Travel Planner Style Logic:
        1️⃣ If status="error" → penalty
        2️⃣ else → reward
        3️⃣ LLM evaluation → adjust score
        """

        status = output.get("status", "success")  # default success if not provided
        if status == "error":
            self.penalize(agent_name)
        else:
            self.reward(agent_name)

        # LLM reinforcement
        llm_score = self.evaluate_output(agent_name, output)
        if llm_score == +1:
            self.reward(agent_name)
        elif llm_score == -1:
            self.penalize(agent_name)

        logger.info(
            f"[FeedbackManager] Final Reinforcement Score → {agent_name} = {self.memory.get(agent_name, 0)}"
        )

        return self.memory.get(agent_name, 0)
