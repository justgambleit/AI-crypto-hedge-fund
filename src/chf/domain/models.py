"""Plain data structures passed between layers.

Keeping these here (separate from the logic that produces them) mirrors the
'domain' layer in a clean architecture: the rest of the code depends on these
shapes, not on how they were computed.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    """The output of one backtest run."""

    equity_curve: pd.Series  # portfolio value over time, starts at 1.0
    returns: pd.Series       # strategy return each period
    positions: pd.Series     # position held each period (1 long / 0 flat / -1 short)
    metrics: dict[str, float]  # ROI, Sharpe, max drawdown, ...
