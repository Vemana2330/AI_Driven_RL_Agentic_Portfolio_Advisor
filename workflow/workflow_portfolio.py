"""
workflow_portfolio.py
Complete sequential workflow that connects API/UI → ControllerAgent → Final Report
"""

import traceback
from datetime import datetime

from logging_utils import logger
from agents.controller_agent import ControllerAgent


class PortfolioWorkflow:
    """
    Orchestrates the end-to-end portfolio intelligence pipeline.
    ControllerAgent internally handles:
    - DQN RL agent
    - PPO Phase agent
    - Sentiment agent
    - Risk manager agent
    - Technical analysis agent
    - Performance fundamentals agent
    - Macro economic agent
    - Final Summary report agent
    """

    def __init__(self):
        logger.info("[Workflow] Initializing PortfolioWorkflow...")
        self.controller = ControllerAgent()

    def run(
        self,
        investment_amount: float,
        risk_level: str = "medium",
        duration_months: int = 6,
        sectors_filter=None,                # currently unused but kept for future agent
        max_loss=None,
        rebalancing_frequency=None
    ):
        """
        API / Streamlit entry point.
        Executes the FULL portfolio workflow and returns final output.
        """

        logger.info("[Workflow] Starting Portfolio Pipeline...")

        try:
            start_time = datetime.now()

            # ⚡ Execute Controller Pipeline
            final_output = self.controller.run_full_portfolio_analysis(
                investment_amount=investment_amount,
                risk_level=risk_level,
                duration_months=duration_months
            )

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            logger.info(f"[Workflow] Completed in {duration:.2f} seconds")

            return {
                "status": "success",
                "started_at": str(start_time),
                "completed_at": str(end_time),
                "duration_seconds": duration,
                **final_output  # Unpacks: report, risk, sentiment, technical, etc.
            }

        except Exception as e:
            error_message = f"Workflow execution failed: {str(e)}"
            logger.error(error_message)
            logger.error(traceback.format_exc())

            return {
                "status": "error",
                "message": error_message,
                "traceback": traceback.format_exc(),
            }


# Manual test
if __name__ == "__main__":
    wf = PortfolioWorkflow()
    out = wf.run(
        investment_amount=10000,
        risk_level="medium",
        duration_months=6,
        sectors_filter=["AI", "Cloud"],
        max_loss=10,
        rebalancing_frequency="quarterly"
    )
    print(out)
