"""
config.py
Global configuration for RL-Agentic-Portfolio-System
"""

import os
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables
load_dotenv()


# ---------------------------------------------------------
# LLM CONFIG
# ---------------------------------------------------------
class LLMConfig(BaseModel):
    model_name: str = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")  # default
    temperature: float = float(os.getenv("LLM_TEMPERATURE", 0.2))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", 2000))


# ---------------------------------------------------------
# API KEYS CONFIG
# ---------------------------------------------------------
class APIKeys(BaseModel):
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    fred_api_key: str = os.getenv("FRED_API_KEY", "")  # for Macro Agent


# ---------------------------------------------------------
# FILE PATHS & RL SETTINGS
# ---------------------------------------------------------
class Paths(BaseModel):
    rl_memory_file: str = os.getenv("RL_MEMORY_FILE", "rl_memory_portfolio.json")
    logs_dir: str = os.getenv("LOGS_DIR", "logs")


# ---------------------------------------------------------
# APPLICATION CONFIG
# ---------------------------------------------------------
class AppConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    api_keys: APIKeys = APIKeys()
    paths: Paths = Paths()

    project_name: str = "RL-Agentic-Portfolio-System"
    default_currency: str = "USD"
    debug_mode: bool = True


# ---------------------------------------------------------
# EXPORT CONFIG
# ---------------------------------------------------------
config = AppConfig()
