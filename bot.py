import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os

from pypfopt import risk_models
from pypfopt import EfficientFrontier
from pypfopt.black_litterman import (
    BlackLittermanModel,
    market_implied_prior_returns
)

# =====================================================
# TELEGRAM SETTINGS
# =====================================================

TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# =====================================================
# DOWNLOAD DATA
# =====================================================

tickers = {
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Platinum": "PL=F",
    "Palladium": "PA=F",
    "IMOEX": "IMOEX.ME"
}

data = pd.DataFrame()

for asset, ticker in tickers.items():

    df = yf.download(
        ticker,
        start="2018-01-01",
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    data[asset] = df["Close"]

data = data.dropna()

# =====================================================
# RETURNS
# =====================================================

returns = data.pct_change().dropna()

# =====================================================
# RECENT DATA (1 YEAR)
# =====================================================

lookback_days = 252

recent_prices = data.tail(lookback_days)

# =====================================================
# COVARIANCE MATRIX
# =====================================================

S = risk_models.exp_cov(
    recent_prices,
    frequency=252
)

# =====================================================
# MARKET WEIGHTS
# =====================================================

market_weights = {
    "Gold": 0.30,
    "Silver": 0.20,
    "Platinum": 0.15,
    "Palladium": 0.15,
    "IMOEX": 0.20
}

# =====================================================
# IMPLIED RETURNS
# =====================================================

delta = 2.5

prior = market_implied_prior_returns(
    market_weights,
    delta,
    S
)

# =====================================================
# DYNAMIC VIEWS
# =====================================================

momentum_20 = returns.tail(20).mean() * 252
momentum_60 = returns.tail(60).mean() * 252

views = (
    0.7 * momentum_20 +
    0.3 * momentum_60
)

views = views.clip(
    lower=-0.30,
    upper=0.30
)

views = views.to_dict()

print("\n========== PRIOR RETURNS ==========")
print(prior)

print("\n========== VIEWS ==========")
print(pd.Series(views))

# =====================================================
# BLACK-LITTERMAN
# =====================================================

bl = BlackLittermanModel(
    S,
    pi=prior,
    absolute_views=views,
    tau=0.05
)

bl_returns = bl.bl_returns()
bl_cov = bl.bl_cov()

print("\n========== BL RETURNS ==========")
print(bl_returns)

# =====================================================
# BENCHMARK
# =====================================================

equal_weights = np.array([0.2] * 5)

benchmark_return = np.dot(
    bl_returns.values,
    equal_weights
)

benchmark_volatility = np.sqrt(
    equal_weights.T @ bl_cov.values @ equal_weights
)

benchmark_sharpe = (
    benchmark_return - 0.05
) / benchmark_volatility

# =====================================================
# OPTIMIZATION
# =====================================================

ef = EfficientFrontier(
    bl_returns,
    bl_cov,
    weight_bounds=(0.00, 0.60)
)

ef.max_sharpe(
    risk_free_rate=0.05
)

cleaned_weights = ef.clean_weights()

# =====================================================
# PERFORMANCE
# =====================================================

expected_return, volatility, sharpe = ef.portfolio_performance(
    risk_free_rate=0.05
)

# =====================================================
# SIGNALS
# =====================================================

signals = {}

for asset in cleaned_weights:

    diff = cleaned_weights[asset] - market_weights[asset]

    if diff > 0.03:
        signals[asset] = f"BUY (+{diff*100:.1f}%)"

    elif diff < -0.03:
        signals[asset] = f"SELL ({diff*100:.1f}%)"

    else:
        signals[asset] = "HOLD"

# =====================================================
# ANALYTICS
# =====================================================

dominant_asset = max(
    cleaned_weights,
    key=cleaned_weights.get
)

if cleaned_weights["Gold"] >= 0.40:
    regime = "Risk-Off"

elif cleaned_weights["IMOEX"] >= 0.35:
    regime = "Risk-On"

else:
    regime = "Balanced"

if volatility < 0.15:
    stability = "Low Risk"

elif volatility < 0.25:
    stability = "Moderate Risk"

else:
    stability = "High Risk"

effective_assets = sum(
    1 for w in cleaned_weights.values()
    if w > 0.05
)

if effective_assets >= 4:
    diversification = "High"

elif effective_assets >= 3:
    diversification = "Medium"

else:
    diversification = "Low"

# =====================================================
# MESSAGE
# =====================================================

weights_text = "\n".join(
    [
        f"{asset}: {weight*100:.2f}%"
        for asset, weight in cleaned_weights.items()
    ]
)

signals_text = "\n".join(
    [
        f"{asset}: {signal}"
        for asset, signal in signals.items()
    ]
)

message = f"""
BLACK-LITTERMAN PORTFOLIO UPDATE

Date: {pd.Timestamp.today().date()}

PORTFOLIO WEIGHTS

{weights_text}

BUY / SELL SIGNALS

{signals_text}

ANALYTICS

Market Regime: {regime}
Dominant Asset: {dominant_asset}
Portfolio Stability: {stability}
Diversification: {diversification}

PERFORMANCE

Expected Return: {expected_return*100:.2f}%
Volatility: {volatility*100:.2f}%
Sharpe Ratio: {sharpe:.2f}

BENCHMARK (20% EACH)

Expected Return: {benchmark_return*100:.2f}%
Volatility: {benchmark_volatility*100:.2f}%
Sharpe Ratio: {benchmark_sharpe:.2f}

ADVANTAGE

Return Delta: {(expected_return-benchmark_return)*100:.2f}%

Sharpe Delta: {(sharpe-benchmark_sharpe):.2f}
"""

print(message)

# =====================================================
# TELEGRAM
# =====================================================

url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

payload = {
    "chat_id": CHAT_ID,
    "text": message
}

requests.post(url, data=payload)

# =====================================================
# SAVE FILES
# =====================================================

pd.DataFrame.from_dict(
    cleaned_weights,
    orient="index",
    columns=["Weight"]
).to_csv("last_weights.csv")

comparison = pd.DataFrame({
    "Portfolio": ["Equal Weight", "Black-Litterman"],
    "Return": [
        benchmark_return,
        expected_return
    ],
    "Volatility": [
        benchmark_volatility,
        volatility
    ],
    "Sharpe": [
        benchmark_sharpe,
        sharpe
    ]
})

comparison.to_csv(
    "portfolio_comparison.csv",
    index=False
)
