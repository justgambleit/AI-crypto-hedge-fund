"""Position sizing — turning a signal into a *bet size*.

A 0/1 signal says *whether* to be in; sizing says *how much*. We try two classic
risk-based rules on top of the moving-average signal:

* volatility targeting — scale exposure to a constant target volatility
  (lever up when calm, cut when wild);
* fractional Kelly — bet a fraction of the Kelly optimum mu / sigma^2.

Both are computed from trailing data only; run_backtest then applies the usual
one-bar lag, so there is no look-ahead. Sizing is a *multiplier on edge*, not a
source of edge: with no positive edge the Kelly fraction collapses toward zero.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from chf.strategies.base import Strategy

PERIODS_PER_YEAR = 365


def vol_target_size(prices: pd.DataFrame, target_ann_vol: float = 0.5,
                    window: int = 30, max_leverage: float = 2.0) -> pd.Series:
    ret = prices["close"].pct_change()
    realized = ret.rolling(window).std() * np.sqrt(PERIODS_PER_YEAR)
    return (target_ann_vol / realized).clip(upper=max_leverage).fillna(0.0)


def fractional_kelly_size(prices: pd.DataFrame, fraction: float = 0.5,
                          window: int = 90, max_leverage: float = 2.0) -> pd.Series:
    ret = prices["close"].pct_change()
    mu = ret.rolling(window).mean()
    var = ret.rolling(window).var()
    kelly = mu / var                       # full Kelly fraction (daily)
    return (fraction * kelly).clip(lower=0.0, upper=max_leverage).fillna(0.0)


class SizedStrategy(Strategy):
    """Wrap a base strategy and scale its 0/1 signal by a sizing rule."""

    def __init__(self, base: Strategy,
                 size_fn: Callable[[pd.DataFrame], pd.Series],
                 name: str | None = None) -> None:
        self.base = base
        self.size_fn = size_fn
        self.name = name or f"{base.name} sized"

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        signal = self.base.generate_signals(prices)
        size = self.size_fn(prices)
        return (signal * size).fillna(0.0)
