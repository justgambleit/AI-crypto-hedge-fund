"""Moving-average crossover — the Level 1 baseline agent.

Go long when the fast moving average is above the slow one, flat otherwise.
Simple, transparent, and a fair benchmark every smarter agent must beat.
"""
from __future__ import annotations

import pandas as pd

from chf.strategies.base import Strategy


class MovingAverageCross(Strategy):
    def __init__(self, fast: int = 20, slow: int = 50) -> None:
        if fast >= slow:
            raise ValueError("fast window must be shorter than slow window")
        self.fast = fast
        self.slow = slow
        self.name = f"MA({fast},{slow})"

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        fast_ma = close.rolling(self.fast).mean()
        slow_ma = close.rolling(self.slow).mean()
        long_when = (fast_ma > slow_ma).astype(float)  # 1.0 long, 0.0 flat
        return long_when.fillna(0.0)
