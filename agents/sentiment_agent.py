# agents/sentiment_agent.py

from crewai import Agent
from tools.built_in_tools import tavily_search_news
from tools.custom_tools import compute_sentiment_score


class SentimentAgent:
    def __init__(self):

        # FUNCTION Tools — must use "func"
        self.tools = [
            {
                "name": "tavily_search_news",
                "description": "Fetch recent news articles for a specified stock ticker",
                "func": tavily_search_news
            },
            {
                "name": "compute_sentiment_score",
                "description": "Compute sentiment polarity (-1 to +1) and return explanation",
                "func": compute_sentiment_score
            }
        ]

        self.agent = Agent(
            role="AI Sector Sentiment Analyst",
            goal="Analyze market sentiment for AI companies based on real-time news.",
            backstory=(
                "You translate market psychology and global news into early trend signals "
                "for institutional investors."
            ),
            tools=self.tools,
            allow_delegation=True,
            verbose=True,
            max_iterations=2
        )

    def run(self, ticker: str):
        return self.agent.run(
            f"""
            Analyze real-time sentiment for: {ticker}

            - Call tavily_search_news(query="{ticker} stock") to retrieve headlines.
            - Use compute_sentiment_score on joined summaries.
            - Output JSON strictly in this structure:
            {{
              "ticker": "{ticker}",
              "sentiment_label": "",
              "sentiment_score": float,
              "sample_headlines": [],
              "core_risk_flag": "",
              "opportunity_flag": "",
              "short_explanation": ""
            }}

            Rules:
            - If score < -0.2 → core_risk_flag = "Negative sentiment trend"
            - If score > +0.3 → opportunity_flag = "Bullish acceleration"
            - DO NOT hallucinate headlines.
            - Use only results Tavily provides.
            """
        )
