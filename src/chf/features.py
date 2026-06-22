"""Turn raw prices into features and targets for ML agents.

All features use only PAST information (rolling windows), so there is no
look-ahead. The target uses the *next* day's return — but it is only ever used
as a training label, never as an input.
"""
from __future__ import annotations

import pandas as pd


def make_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Build a table of predictive features from an OHLCV frame."""
    close = prices["close"]
    ret = close.pct_change()

    feat = pd.DataFrame(index=prices.index)
    feat["ret_1"] = ret                              # yesterday's return
    feat["ret_5"] = close.pct_change(5)              # 1-week momentum
    feat["ret_10"] = close.pct_change(10)            # 2-week momentum
    feat["vol_10"] = ret.rolling(10).std()           # short-term volatility
    feat["vol_30"] = ret.rolling(30).std()           # medium-term volatility
    feat["mom_20"] = close / close.rolling(20).mean() - 1  # distance from 20d mean
    feat["mom_50"] = close / close.rolling(50).mean() - 1  # distance from 50d mean

    # RSI-style oscillator (0..100): how one-sided recent moves have been.
    up = ret.clip(lower=0).rolling(14).mean()
    down = (-ret.clip(upper=0)).rolling(14).mean()
    feat["rsi_14"] = 100 - 100 / (1 + up / (down + 1e-9))

    if "volume" in prices.columns:
        feat["vol_change_5"] = prices["volume"].pct_change(5)

    return feat


def make_target(prices: pd.DataFrame) -> pd.Series:
    """Label = 1 if the NEXT day's close is higher, else 0."""
    next_return = prices["close"].pct_change().shift(-1)
    return (next_return > 0).astype(int)
