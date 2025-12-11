import yfinance as yf
import pandas as pd
import os

AI_TICKERS = [
    "NVDA","MSFT","GOOG","META","AMZN",
    "AMD","AVGO","TSLA","CRM","ORCL","INTC"
]

START_DATE = "2015-01-01"
END_DATE = "2024-12-31"

def ensure_folders():
    os.makedirs("data/raw", exist_ok=True)

def download_ai_stock_data():
    print("Downloading AI tech stock OHLCV data...")
    data = yf.download(AI_TICKERS, start=START_DATE, end=END_DATE)
    data.to_csv("data/raw/ai_stocks.csv")
    print("Saved to data/raw/ai_stocks.csv")

def download_benchmark_data():
    print("Downloading Benchmark data (QQQ)...")
    bench = yf.download("QQQ", start=START_DATE, end=END_DATE)
    bench.to_csv("data/raw/benchmark_qqq.csv")
    print("Saved to data/raw/benchmark_qqq.csv")

def download_risk_free_rate():
    print("Downloading Risk-Free rate (10Y Treasury Yield)...")
    risk_free = yf.download("^TNX", start=START_DATE, end=END_DATE, auto_adjust=True)
    risk_free["RF"] = risk_free["Close"] / 100
    risk_free[["RF"]].to_csv("data/raw/risk_free_rate.csv")
    print("Saved to data/raw/risk_free_rate.csv")



if __name__ == "__main__":
    ensure_folders()
    #download_ai_stock_data()
    #download_benchmark_data()
    download_risk_free_rate()
    print("\n AI Stock dataset download completed.\n")


