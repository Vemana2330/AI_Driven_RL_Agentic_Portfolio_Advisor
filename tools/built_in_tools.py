# tools/built_in_tools.py

import os
import requests
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred

load_dotenv()

# ---------------------------------------------------------
# Tavily Search (Function-Based Tool)
# ---------------------------------------------------------
def tavily_search_news(query: str):
    """
    Fetch real-time AI sector news/articles using Tavily API.
    Returns JSON: { results: [...] }
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {"error": "TAVILY_API_KEY missing in .env"}

    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",
        "max_results": 10
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    except Exception as e:
        return {"error": f"Tavily request failed: {str(e)}"}

# ---------------------------------------------------------
# Yahoo Finance Fundamental Data
# ---------------------------------------------------------
def fetch_yfinance_company_data(ticker: str):
    """
    Returns financial info, balance sheet, cash flow, earnings for a stock.
    """
    try:
        stock = yf.Ticker(ticker)

        return {
            "ticker": ticker,
            "info": stock.info if hasattr(stock, "info") else {},
            "financials": stock.financials.to_dict() if hasattr(stock, "financials") else {},
            "balance_sheet": stock.balance_sheet.to_dict() if hasattr(stock, "balance_sheet") else {},
            "cashflow": stock.cashflow.to_dict() if hasattr(stock, "cashflow") else {}
        }

    except Exception as e:
        return {"error": f"Yahoo Finance fetch failed: {str(e)}", "ticker": ticker}

# ---------------------------------------------------------
# FRED Macro-Economic Indicators Data
# ---------------------------------------------------------
def fetch_macro_indicators(indicators: list = None):
    """
    Fetch CPI, Fed Rate, GDP, VIX, Unemployment from FRED API.
    """
    fred_key = os.getenv("FRED_API_KEY")
    if not fred_key:
        return {"error": "FRED_API_KEY missing in .env"}

    fred = Fred(api_key=fred_key)

    default_indicators = {
        "CPIAUCSL": "Inflation (CPI)",
        "UNRATE": "Unemployment Rate",
        "FEDFUNDS": "Federal Funds Rate",
        "GDPC1": "Real GDP Growth",
        "DCOILWTICO": "Crude Oil Prices",
        "VIXCLS": "VIX Fear Index"
    }

    if indicators is None:
        indicators = list(default_indicators.keys())

    results = {}

    for code in indicators:
        try:
            series = fred.get_series_latest_release(code)
            latest_value = float(series[-1])
            results[default_indicators.get(code, code)] = latest_value
        except Exception:
            results[default_indicators.get(code, code)] = "Unavailable"

    return {
        "status": "success",
        "macro_indicators": results
    }
