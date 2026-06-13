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
    'Gold': 'GC=F',
    'Silver': 'SI=F',
    'Platinum': 'PL=F',
    'Palladium': 'PA=F',
    'IMOEX': 'IMOEX.ME'
}

data = pd.DataFrame()

for asset, ticker in tickers.items():

    df = yf.download(
        ticker,
        start='2010-01-01',
        interval='1d',
        auto_adjust=True
    )

    data[asset] = df['Close']

data = data.dropna()

# =====================================================
# RETURNS
# =====================================================

returns = data.pct_change().dropna()

# =====================================================
# COVARIANCE MATRIX
# =====================================================

S = risk_models.sample_cov(
    data,
    frequency=252
)

# =====================================================
# MARKET WEIGHTS
# =====================================================

market_weights = {
    'Gold': 0.30,
    'Silver': 0.20,
    'Platinum': 0.15,
    'Palladium': 0.15,
    'IMOEX': 0.20
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
# DYNAMIC VIEWS (Momentum-based)
# =====================================================

recent_returns = (
    returns
    .tail(60)
    .mean()
    * 252
)

views = recent_returns.clip(
    lower=0.03,
    upper=0.25
).to_dict()

print("Dynamic Views:")
print(views)

# =====================================================
# BLACK-LITTERMAN MODEL
# =====================================================

bl = BlackLittermanModel(
    S,
    pi=prior,
    absolute_views=views
)

bl_returns = bl.bl_returns()

bl_cov = bl.bl_cov()

# =====================================================
# EQUAL WEIGHT BENCHMARK
# =====================================================

equal_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

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
    weight_bounds=(0.05, 0.40)
)

weights = ef.max_sharpe(
    risk_free_rate=0.05
)

cleaned_weights = ef.clean_weights()

# =====================================================
# PORTFOLIO PERFORMANCE
# =====================================================

expected_return, volatility, sharpe = ef.portfolio_performance(
    risk_free_rate=0.05
)

# =====================================================
# BUY / SELL SIGNALS
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

if cleaned_weights['Gold'] > 0.35:
    regime = "Risk-Off"
else:
    regime = "Balanced"

if volatility < 0.20:
    stability = "Stable"
else:
    stability = "Moderate Risk"

div_score = "High"

# =====================================================
# TELEGRAM MESSAGE
# =====================================================

message = f"""
BLACK-LITTERMAN PORTFOLIO UPDATE

Date: {pd.Timestamp.today().date()}

PORTFOLIO WEIGHTS

Gold: {cleaned_weights['Gold']*100:.2f}%
Silver: {cleaned_weights['Silver']*100:.2f}%
Platinum: {cleaned_weights['Platinum']*100:.2f}%
Palladium: {cleaned_weights['Palladium']*100:.2f}%
IMOEX: {cleaned_weights['IMOEX']*100:.2f}%

BUY / SELL SIGNALS

Gold: {signals['Gold']}
Silver: {signals['Silver']}
Platinum: {signals['Platinum']}
Palladium: {signals['Palladium']}
IMOEX: {signals['IMOEX']}

ANALYTICS

Market Regime: {regime}
Dominant Asset: {dominant_asset}
Portfolio Stability: {stability}
Diversification: {div_score}

PERFORMANCE

PERFORMANCE

BLACK-LITTERMAN

Expected Return: {expected_return*100:.2f}%
Volatility: {volatility*100:.2f}%
Sharpe Ratio: {sharpe:.2f}

BENCHMARK (20% EACH)

Expected Return: {benchmark_return*100:.2f}%
Volatility: {benchmark_volatility*100:.2f}%
Sharpe Ratio: {benchmark_sharpe:.2f}

ADVANTAGE

Return Delta:
{(expected_return-benchmark_return)*100:.2f}%

Sharpe Delta:
{(sharpe-benchmark_sharpe):.2f}
"""
# =====================================================
# SEND MESSAGE
# =====================================================

url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

payload = {
    "chat_id": CHAT_ID,
    "text": message
}

requests.post(url, data=payload)

print(message)

weights_df = pd.DataFrame.from_dict(
    cleaned_weights,
    orient='index',
    columns=['Weight']
)

weights_df.to_csv("last_weights.csv")

dominant_asset = max(
    cleaned_weights,
    key=cleaned_weights.get
)

if cleaned_weights['Gold'] > 0.35:
    regime = "Risk-Off"
else:
    regime = "Balanced"

if volatility < 0.2:
    stability = "Stable"
else:
    stability = "Moderate Risk"

div_score = "High"

message = f"""
BLACK-LITTERMAN PORTFOLIO UPDATE

Date: {pd.Timestamp.today().date()}

PORTFOLIO WEIGHTS

Gold: {cleaned_weights['Gold']*100:.2f}%
Silver: {cleaned_weights['Silver']*100:.2f}%
Platinum: {cleaned_weights['Platinum']*100:.2f}%
Palladium: {cleaned_weights['Palladium']*100:.2f}%
IMOEX: {cleaned_weights['IMOEX']*100:.2f}%

BUY / SELL SIGNALS

Gold: {signals['Gold']}
Silver: {signals['Silver']}
Platinum: {signals['Platinum']}
Palladium: {signals['Palladium']}
IMOEX: {signals['IMOEX']}

ANALYTICS

Market Regime: {regime}
Dominant Asset: {dominant_asset}
Portfolio Stability: {stability}
Diversification: {div_score}

PERFORMANCE

Expected Return: {expected_return*100:.2f}%
Volatility: {volatility*100:.2f}%
Sharpe Ratio: {sharpe:.2f}
"""

if cleaned_weights['Gold'] + cleaned_weights['Silver'] > 0.7:
    regime = "Defensive Metals Dominance"
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
