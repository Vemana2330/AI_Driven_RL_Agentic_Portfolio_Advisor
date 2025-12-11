# tools/custom_tools.py

from textblob import TextBlob
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd

# ======================================================================
# 1️⃣ Sentiment Score Function
# ======================================================================

def compute_sentiment_score(text: str):
    """
    Computes sentiment polarity (-1 bearish to +1 bullish)
    Returns: { sentiment_score, sentiment_label }
    """
    analysis = TextBlob(text)
    polarity = float(analysis.sentiment.polarity)

    if polarity > 0.2:
        label = "bullish"
    elif polarity < -0.2:
        label = "bearish"
    else:
        label = "neutral"

    return {
        "sentiment_score": polarity,
        "sentiment_label": label
    }


# ======================================================================
# 2️⃣ Risk Metrics Function
# ======================================================================

def compute_risk_metrics(risk_preference: str = "medium"):
    """
    Computes:
    - Volatility
    - Max drawdown
    - Risk tier match vs user preference
    """
    risk_pref = str(risk_preference).strip().lower()
    if risk_pref not in {"low", "medium", "high"}:
        risk_pref = "medium"

    try:
        BASE_DIR = Path(__file__).resolve().parents[1]
        model_dir = BASE_DIR / "models" / "rl_model_v1"
        price_cols_path = model_dir / "price_cols.json"
        market_csv_path = model_dir / "inference_market_252days.csv"

        if not price_cols_path.exists() or not market_csv_path.exists():
            return {
                "error": "Historical dataset for risk metrics not found.",
                "missing_price_cols_json": not price_cols_path.exists(),
                "missing_market_csv": not market_csv_path.exists()
            }

        with open(price_cols_path, "r") as f:
            price_cols = json.load(f)

        df = pd.read_csv(market_csv_path)
        metrics = []

        for col in price_cols[:-1]:
            ticker = col.replace("_Close", "")
            if col not in df.columns:
                continue

            prices = df[col].astype(float)
            returns = prices.pct_change().dropna()

            if returns.empty:
                continue

            # volatility and drawdown
            vol = float(returns.std())
            cum = (1 + returns).cumprod()
            running_max = cum.cummax()
            drawdowns = cum / running_max - 1.0
            max_dd = float(drawdowns.min())

            metrics.append({
                "ticker": ticker,
                "volatility": vol,
                "max_drawdown": max_dd
            })

        if not metrics:
            return {"error": "Unable to compute risk metrics. CSV mismatch."}

        vols = np.array([m["volatility"] for m in metrics])
        low_thr = float(np.percentile(vols, 33))
        high_thr = float(np.percentile(vols, 66))

        for m in metrics:
            v = m["volatility"]
            if v <= low_thr:
                level = "Low"
            elif v <= high_thr:
                level = "Medium"
            else:
                level = "High"

            if risk_pref == "low":
                allowed = {"Low"}
            elif risk_pref == "medium":
                allowed = {"Low", "Medium"}
            else:
                allowed = {"Low", "Medium", "High"}

            m["risk_level_detected"] = level
            m["matches_user_profile"] = level in allowed

        return {
            "risk_preference": risk_pref,
            "universe_size": len(metrics),
            "metrics": metrics
        }

    except Exception as e:
        return {"error": f"Risk metrics exception: {str(e)}"}


# ======================================================================
# 3️⃣ Technical Indicators Function (SMA / MACD / RSI / Bollinger)
# ======================================================================

def compute_technical_indicators(close_prices: list):

    prices = np.array(close_prices, dtype=np.float64)

    if len(prices) < 5:
        last_price = float(prices[-1]) if len(prices) else 0.0
        return {
            "last_price": last_price,
            "sma20": last_price,
            "sma50": last_price,
            "sma200": last_price,
            "rsi": 50.0,
            "macd": 0.0,
            "macd_signal": 0.0,
            "bb_middle": last_price,
            "bb_upper": last_price,
            "bb_lower": last_price,
        }

    last_price = float(prices[-1])

    def sma(p, window):
        return float(p[-window:].mean()) if len(p) >= window else float(p.mean())

    sma20 = sma(prices, 20)
    sma50 = sma(prices, 50)
    sma200 = sma(prices, 200)

    def ema(p, period):
        alpha = 2 / (period + 1.0)
        ema_arr = np.zeros_like(p)
        ema_arr[0] = p[0]
        for i in range(1, len(p)):
            ema_arr[i] = alpha * p[i] + (1 - alpha) * ema_arr[i - 1]
        return ema_arr

    def rsi_calc(p, period=14):
        if len(p) < period + 1:
            return 50.0
        deltas = np.diff(p)
        ups = np.where(deltas > 0, deltas, 0.0)
        downs = np.where(deltas < 0, -deltas, 0.0)
        rs = ups[-period:].mean() / (downs[-period:].mean() + 1e-8)
        return float(100 - (100 / (1 + rs)))

    rsi = rsi_calc(prices)
    ema12 = ema(prices, 12)
    ema26 = ema(prices, 26)
    macd_series = ema12 - ema26
    macd_signal_series = ema(macd_series, 9)
    macd = float(macd_series[-1])
    macd_signal = float(macd_signal_series[-1])

    if len(prices) < 20:
        bb_middle = prices.mean()
        bb_std = prices.std()
    else:
        window = prices[-20:]
        bb_middle = window.mean()
        bb_std = window.std(ddof=0)

    return {
        "last_price": last_price,
        "sma20": float(sma20),
        "sma50": float(sma50),
        "sma200": float(sma200),
        "rsi": float(rsi),
        "macd": macd,
        "macd_signal": macd_signal,
        "bb_middle": float(bb_middle),
        "bb_upper": float(bb_middle + 2 * bb_std),
        "bb_lower": float(bb_middle - 2 * bb_std),
    }


# ======================================================================
# 4️⃣ Support & Resistance Detection
# ======================================================================

def compute_support_resistance(close_prices: list):
    prices = np.array(close_prices, dtype=np.float64)

    if len(prices) < 10:
        return {
            "support_levels": [float(prices.min())] if len(prices) else [],
            "resistance_levels": [float(prices.max())] if len(prices) else [],
        }

    recent = prices[-60:] if len(prices) > 60 else prices
    low_level = float(np.percentile(recent, 10))
    high_level = float(np.percentile(recent, 90))

    return {
        "support_levels": [low_level],
        "resistance_levels": [high_level],
    }

# ======================================================================
# 5️⃣ Volume Trend Analysis
# ======================================================================

def compute_volume_trend(volume_series: list):
    vols = np.array(volume_series, dtype=np.float64)

    if len(vols) == 0:
        return {"volume_trend": "unknown", "volume_strength": 0.0}

    long_window = min(60, len(vols))
    short_window = min(10, len(vols))

    long_avg = vols[-long_window:].mean()
    short_avg = vols[-short_window:].mean()

    ratio = short_avg / long_avg if long_avg > 0 else 1.0

    if ratio > 1.2:
        trend = "increasing"
    elif ratio < 0.8:
        trend = "decreasing"
    else:
        trend = "flat"

    return {
        "volume_trend": trend,
        "volume_strength": float(ratio)
    }


# ======================================================================
# 6️⃣ Technical Scoring (Produces 0–100 score + NL summary)
# ======================================================================

def compute_technical_score(
        indicators: dict, 
        support_resistance: dict, 
        volume_info: dict, 
        sensitivity: str = "medium"
    ):

    price = indicators["last_price"]
    sma20 = indicators["sma20"]
    sma50 = indicators["sma50"]
    sma200 = indicators["sma200"]
    rsi = indicators["rsi"]
    macd = indicators["macd"]
    macd_signal = indicators["macd_signal"]
    bb_upper = indicators["bb_upper"]
    bb_lower = indicators["bb_lower"]
    bb_mid = indicators["bb_middle"]

    volume_trend = volume_info.get("volume_trend", "unknown")
    volume_strength = volume_info.get("volume_strength", 1.0)

    # Sensitivity adjustments for RSI
    sensitivity = sensitivity.lower()
    if sensitivity == "low":
        rsi_ob, rsi_os = 75, 25
    elif sensitivity == "high":
        rsi_ob, rsi_os = 65, 35
    else:
        rsi_ob, rsi_os = 70, 30

    # Trend score
    if sma20 > sma50 > sma200:
        trend = "uptrend"
        trend_score = 25
    elif sma20 < sma50 < sma200:
        trend = "downtrend"
        trend_score = 5
    else:
        trend = "sideways"
        trend_score = 15

    # RSI score
    if rsi > rsi_ob:
        rsi_status = "overbought"
        rsi_score = 5
    elif rsi < rsi_os:
        rsi_status = "oversold"
        rsi_score = 20
    else:
        rsi_status = "neutral"
        rsi_score = 15

    # MACD
    if macd > macd_signal:
        momentum_label = "bullish"
        macd_score = 20
    elif macd < macd_signal:
        momentum_label = "bearish"
        macd_score = 5
    else:
        momentum_label = "neutral"
        macd_score = 10

    # Bollinger Zone (entry opportunity)
    if price <= bb_lower:
        entry_zone = "good_buy_zone"
        bb_score = 20
    elif price >= bb_upper:
        entry_zone = "sell_or_reduce_zone"
        bb_score = 5
    else:
        if abs(price - bb_mid) / (bb_upper - bb_lower + 1e-8) < 0.25:
            entry_zone = "neutral_zone"
            bb_score = 15
        else:
            entry_zone = "moderate_zone"
            bb_score = 12

    # Volume Effect
    if volume_trend == "increasing":
        vol_score = min(10, 5 * volume_strength)
    elif volume_trend == "decreasing":
        vol_score = max(0, 5 * (2 - volume_strength))
    else:
        vol_score = 5

    raw_score = trend_score + rsi_score + macd_score + bb_score + vol_score
    technical_score = int(max(0, min(100, raw_score)))

    summary_parts = [
        f"Trend: {trend}",
        f"RSI status: {rsi_status} (RSI={rsi:.1f})",
        f"Momentum: {momentum_label} (MACD trend)",
        f"Entry zone: {entry_zone}",
        f"Volume trend: {volume_trend} (strength={volume_strength:.2f})"
    ]
    summary = "; ".join(summary_parts)

    return {
        "technical_score": technical_score,
        "trend": trend,
        "rsi_status": rsi_status,
        "momentum": momentum_label,
        "entry_zone": entry_zone,
        "volume_trend": volume_trend,
        "summary": summary,
    }


# ======================================================================
# 7️⃣ Financial Metrics Extraction
# ======================================================================

def extract_financial_metrics(data: dict):
    try:
        info = data.get("info", {})
        financials = data.get("financials", {})
        balance_sheet = data.get("balance_sheet", {})
        cashflow = data.get("cashflow", {})

        pe_ratio = info.get("trailingPE")
        peg_ratio = info.get("pegRatio")
        profit_margin = info.get("profitMargins") * 100 if info.get("profitMargins") else None
        roe = info.get("returnOnEquity") * 100 if info.get("returnOnEquity") else None
        debt_to_equity = info.get("debtToEquity")

        revenue_growth = None
        revenue_cols = list(financials.keys())
        if len(revenue_cols) >= 2:
            r1 = financials[revenue_cols[0]].get("Total Revenue", None)
            r2 = financials[revenue_cols[1]].get("Total Revenue", None)
            if r1 and r2:
                revenue_growth = ((r1 - r2) / r2) * 100

        fcf = None
        try:
            fcf = cashflow[list(cashflow.keys())[0]].get("Free Cash Flow")
        except:
            pass

        return {
            "pe_ratio": pe_ratio,
            "peg_ratio": peg_ratio,
            "profit_margin": profit_margin,
            "return_on_equity": roe,
            "revenue_growth_2y": revenue_growth,
            "debt_to_equity": debt_to_equity,
            "free_cash_flow": "positive" if fcf and fcf > 0 else "negative"
        }

    except Exception as e:
        return {"error": str(e)}


# ======================================================================
# 8️⃣ Fundamental Scoring
# ======================================================================

def score_fundamentals(metrics: dict):
    score = 0
    max_score = 6 * 20

    if metrics.get("pe_ratio") and metrics["pe_ratio"] < 40:
        score += 15

    if metrics.get("peg_ratio") and metrics["peg_ratio"] < 2:
        score += 20

    if metrics.get("profit_margin") and metrics["profit_margin"] > 15:
        score += 20

    if metrics.get("return_on_equity") and metrics["return_on_equity"] > 20:
        score += 20

    if metrics.get("revenue_growth_2y") and metrics["revenue_growth_2y"] > 8:
        score += 15

    if metrics.get("debt_to_equity") and metrics["debt_to_equity"] < 1:
        score += 10

    return {
        "fundamental_health_score": int((score / max_score) * 100),
        "metrics_used": metrics
    }


# ======================================================================
# 9️⃣ Macro Scoring
# ======================================================================

def score_macro_conditions(metrics: dict):

    score = 50

    inflation = metrics.get("Inflation (CPI)")
    if inflation and inflation > 4:
        score -= 15
    elif inflation and inflation < 2.5:
        score += 10

    unemployment = metrics.get("Unemployment Rate")
    if unemployment and unemployment > 6:
        score -= 20
    elif unemployment and unemployment < 4:
        score += 10

    fed_rate = metrics.get("Federal Funds Rate")
    if fed_rate and fed_rate > 4:
        score -= 15
    elif fed_rate and fed_rate < 1:
        score += 10

    vix = metrics.get("VIX Fear Index")
    if vix and vix > 25:
        score -= 20
    elif vix and vix < 15:
        score += 10

    if score >= 65:
        regime = "Bullish Expansion"
    elif score >= 45:
        regime = "Neutral / Transition"
    else:
        regime = "Recessionary / High-Risk"

    return {
        "macro_score": score,
        "macro_regime": regime,
        "metrics_used": metrics
    }
