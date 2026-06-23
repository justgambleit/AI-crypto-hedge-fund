"""Moving-average crossover — long/short variant (Level 1 baseline + shorts).

Same crossover logic as MovingAverageCross, but instead of going flat
when the fast MA is below the slow MA, it goes SHORT (-1). This is the
classic golden-cross / death-cross rule, always in the market.
"""
from __future__ import annotations

import pandas as pd

from chf.strategies.base import Strategy


class MovingAverageCrossLS(Strategy):
    def __init__(self, fast: int = 50, slow: int = 200) -> None:
        if fast >= slow:
            raise ValueError("fast window must be shorter than slow window")
        self.fast = fast
        self.slow = slow
        self.name = f"MA-LS({fast},{slow})"

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        fast_ma = close.rolling(self.fast).mean()
        slow_ma = close.rolling(self.slow).mean()
        # +1.0 long when fast above slow (golden cross),
        # -1.0 short when fast below slow (death cross).
        signal = pd.Series(1.0, index=close.index)
        signal[fast_ma <= slow_ma] = -1.0
        # During the warm-up period the MAs are NaN -> stay flat (0).
        signal[fast_ma.isna() | slow_ma.isna()] = 0.0
        return signal