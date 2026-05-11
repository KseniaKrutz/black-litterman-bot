import yfinance as yf
import pandas as pd
import numpy as np
import requests

from pypfopt import risk_models
from pypfopt import EfficientFrontier
from pypfopt.black_litterman import (
    BlackLittermanModel,
    market_implied_prior_returns
)

# =====================================================
# TELEGRAM SETTINGS
# =====================================================

TOKEN = "YOUR_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"

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
        interval='1mo',
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
    frequency=12
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
# BLACK-LITTERMAN VIEWS
# =====================================================

views = {
    'Gold': 0.14,
    'Silver': 0.11,
    'Platinum': 0.07,
    'Palladium': 0.09,
    'IMOEX': 0.08
}

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

expected_return, volatility, sharpe = ef.portfolio_performance()

# =====================================================
# TELEGRAM MESSAGE
# =====================================================

message = f"""
BLACK-LITTERMAN PORTFOLIO UPDATE

Gold: {cleaned_weights['Gold']*100:.2f}%
Silver: {cleaned_weights['Silver']*100:.2f}%
Platinum: {cleaned_weights['Platinum']*100:.2f}%
Palladium: {cleaned_weights['Palladium']*100:.2f}%
IMOEX: {cleaned_weights['IMOEX']*100:.2f}%

Expected Return: {expected_return*100:.2f}%
Volatility: {volatility*100:.2f}%
Sharpe Ratio: {sharpe:.2f}
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
