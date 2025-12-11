"""
controller_agent.py
Master Orchestrator for RL + Multi-Agent Intelligence + Summary Report.
"""

from crewai import Agent
from langchain_openai import ChatOpenAI

from config import config
from logging_utils import logger

# RL Inference Agents
from agents.rl_agents.rl_agent_dqn import PortfolioAllocatorDQN
from agents.rl_agents.rl_agent_policy import PhasePolicyAgent

# Intelligence Agents
from agents.sentiment_agent import SentimentAgent
from agents.risk_manager_agent import RiskManagerAgent
from agents.technical_analysis_agent import TechnicalAnalysisAgent
from agents.performance_agent import PerformanceAgent
from agents.macro_agent import MacroEconomicAgent
from agents.summary_report_agent import SummaryReportAgent

# Feedback
from reinforcement_learning.feedback_manager import FeedbackManager


class ControllerAgent:
    """
    Controls full RL + Market Intelligence + Summary Report flow.
    """

    DEFAULT_TICKERS = ["NVDA", "MSFT", "META", "GOOG", "AMZN"]

    def __init__(self):
        logger.info("[Controller] Initializing portfolio controller...")

        self.llm = ChatOpenAI(
            model=config.llm.model_name,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            api_key=config.api_keys.openai_api_key,
        )

        self.agent = Agent(
            role="Master Portfolio Orchestrator",
            goal="Coordinate RL models and market intelligence for portfolio allocation decisions.",
            backstory="You combine AI reinforcement learning with macro, sentiment, risk and technical signals.",
            verbose=True,
            llm=self.llm,
            memory=True,
        )

        # Sub-agents
        self.dqn_agent = PortfolioAllocatorDQN()
        self.ppo_agent = PhasePolicyAgent()
        self.sentiment = SentimentAgent()
        self.risk = RiskManagerAgent()
        self.technical = TechnicalAnalysisAgent()
        self.performance = PerformanceAgent()
        self.macro = MacroEconomicAgent()
        self.summary = SummaryReportAgent()

        self.feedback = FeedbackManager()

    # -----------------------------------------------------------------
    def run_full_portfolio_analysis(
        self,
        investment_amount: float,
        risk_level: str,
        duration_months: int,
        tickers: list = None
    ):
        logger.info("[Controller] Running full intelligence pipeline...")

        tickers = tickers or self.DEFAULT_TICKERS

        # -----------------------------------------------------------------
        # RL — DQN
        # -----------------------------------------------------------------
        try:
            dqn_output = self.dqn_agent.run_inference(
                investment_amount=investment_amount,
                risk_level=risk_level
            )
            dqn_output["status"] = "success"
        except Exception as e:
            dqn_output = {"status": "error", "message": str(e), "average_weights": []}
        self.feedback.apply_feedback("DQNAllocator", dqn_output)

        # -----------------------------------------------------------------
        # RL — PPO Phase Allocation
        # -----------------------------------------------------------------
        try:
            ppo_output = self.ppo_agent.run_phase_plan(
                dqn_weights=dqn_output.get("average_weights", []),
                horizon_months=duration_months,
                investment_amount=investment_amount
            )
            ppo_output["status"] = "success"
        except Exception as e:
            ppo_output = {"status": "error", "message": str(e)}
        self.feedback.apply_feedback("PPOPolicy", ppo_output)

        # -----------------------------------------------------------------
        # SENTIMENT PER TICKER
        # -----------------------------------------------------------------
        sentiment_results = {}
        for t in tickers:
            try:
                out = self.sentiment.run(t)
                sentiment_results[t] = {"status": "success", "data": out}
            except Exception as e:
                sentiment_results[t] = {"status": "error", "message": str(e)}
            self.feedback.apply_feedback("SentimentAgent", sentiment_results[t])

        # -----------------------------------------------------------------
        # RISK
        # -----------------------------------------------------------------
        try:
            risk_output = self.risk.evaluate(risk_preference=risk_level)
            risk_output["status"] = "success"
        except Exception as e:
            risk_output = {"status": "error", "message": str(e)}
        self.feedback.apply_feedback("RiskAgent", risk_output)

        # -----------------------------------------------------------------
        # TECHNICAL ANALYSIS
        # -----------------------------------------------------------------
        try:
            tech_output = self.technical.analyze(time_window_days=180, sensitivity="medium")
            tech_output["status"] = "success"
        except Exception as e:
            tech_output = {"status": "error", "message": str(e)}
        self.feedback.apply_feedback("TechnicalAgent", tech_output)

        # -----------------------------------------------------------------
        # PERFORMANCE / FUNDAMENTALS
        # -----------------------------------------------------------------
        performance_output = {}
        for t in tickers:
            try:
                perf = self.performance.evaluate(ticker=t)
                performance_output[t] = {"status": "success", "data": perf}
            except Exception as e:
                performance_output[t] = {"status": "error", "message": str(e)}
            self.feedback.apply_feedback("PerformanceAgent", performance_output[t])

        # -----------------------------------------------------------------
        # MACRO ECONOMICS
        # -----------------------------------------------------------------
        try:
            macro_output = self.macro.run()
            macro_output["status"] = "success"
        except Exception as e:
            macro_output = {"status": "error", "message": str(e)}
        self.feedback.apply_feedback("MacroAgent", macro_output)

        # -----------------------------------------------------------------
        # AGGREGATE DATA FOR SUMMARY REPORT
        # -----------------------------------------------------------------
        AGGREGATED = {
            "user_inputs": {
                "investment_amount": investment_amount,
                "risk_level": risk_level,
                "duration_months": duration_months,
            },
            "dqn_allocation": dqn_output,
            "policy_phases": ppo_output.get("phase_allocation_plan", []),
            "sentiment": sentiment_results,
            "risk": risk_output,
            "technical": tech_output,
            "performance": performance_output,
            "macro": macro_output,
        }

        # -----------------------------------------------------------------
        # FINAL SUMMARY REPORT (LLM)
        # -----------------------------------------------------------------
        final_report_text = self.summary.generate_report(AGGREGATED)

        logger.info("[Controller] Pipeline complete.")

        return {
            "report": final_report_text,
            "dqn_allocation": dqn_output,
            "policy_phases": ppo_output.get("phase_allocation_plan", []),
            "sentiment": sentiment_results,
            "risk": risk_output,
            "technical": tech_output,
            "performance": performance_output,
            "macro": macro_output,
            "user_inputs": AGGREGATED["user_inputs"],
        }
