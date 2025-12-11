"""
logging_utils.py
Centralized logger for RL-Agentic-Portfolio-System
"""

import logging
import os
from datetime import datetime
from config import config

# -------------------------------------------------------------
# Ensure logs directory exists
# -------------------------------------------------------------
LOGS_DIR = config.paths.logs_dir or "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# -------------------------------------------------------------
# Dynamic log file naming
# -------------------------------------------------------------
log_file_path = os.path.join(
    LOGS_DIR,
    f"portfolio_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

# -------------------------------------------------------------
# Logging format
# -------------------------------------------------------------
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    "%Y-%m-%d %H:%M:%S"
)

# File Handler
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Console Handler with color highlight
class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[96m",
        "INFO": "\033[92m",
        "WARNING": "\033[93m",
        "ERROR": "\033[91m",
        "CRITICAL": "\033[95m",
    }

    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(ColorFormatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    "%Y-%m-%d %H:%M:%S"
))

# -------------------------------------------------------------
# Main Logger Instance
# -------------------------------------------------------------
logger = logging.getLogger("RL_Agentic_Portfolio")
logger.setLevel(logging.INFO)

# Avoid duplicate handler assignment
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
