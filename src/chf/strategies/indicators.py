"""A few classic technical-indicator strategies, for an honest comparison.

Each returns a 0/1 long-or-flat signal from textbook default parameters (no
tuning on the test set). The point of the comparison is to show that popular
indicators are roughly interchangeable and rarely beat buy-and-hold out-of-
sample after costs — not to crown the lucky winner of a single horse race.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from chf.strategies.base import Strategy


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def _stateful(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    """Hold long from an entry signal until an exit signal fires."""
    e, x = entry.fillna(False).values, exit_.fillna(False).values
    pos = np.zeros(len(e))
    holding = 0
    for i in range(len(e)):
        if holding == 0 and e[i]:
            holding = 1
        elif holding == 1 and x[i]:
            holding = 0
        pos[i] = holding
    return pd.Series(pos, index=entry.index)


class MACDCross(Strategy):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        self.fast, self.slow, self.signal = fast, slow, signal
        self.name = "MACD(12,26,9)"

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        macd = close.ewm(span=self.fast).mean() - close.ewm(span=self.slow).mean()
        signal = macd.ewm(span=self.signal).mean()
        return (macd > signal).astype(float).fillna(0.0)


class RSIReversion(Strategy):
    def __init__(self, window: int = 14, low: int = 30, high: int = 70) -> None:
        self.window, self.low, self.high = window, low, high
        self.name = "RSI(14) возврат"

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        rsi = _rsi(prices["close"], self.window)
        return _stateful(rsi < self.low, rsi > self.high)


class BollingerReversion(Strategy):
    def __init__(self, window: int = 20, n_std: float = 2.0) -> None:
        self.window, self.n_std = window, n_std
        self.name = "Bollinger(20,2)"

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        mid = close.rolling(self.window).mean()
        std = close.rolling(self.window).std()
        lower = mid - self.n_std * std
        return _stateful(close < lower, close > mid)


class DonchianBreakout(Strategy):
    def __init__(self, window: int = 20) -> None:
        self.window = window
        self.name = "Donchian(20) пробой"

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        upper = close.rolling(self.window).max().shift(1)   # prior N-day high
        lower = close.rolling(self.window).min().shift(1)
        return _stateful(close > upper, close < lower)


class MomentumTS(Strategy):
    def __init__(self, lookback: int = 90) -> None:
        self.lookback = lookback
        self.name = "Моментум(90)"

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        past_return = close / close.shift(self.lookback) - 1
        return (past_return > 0).astype(float).fillna(0.0)
