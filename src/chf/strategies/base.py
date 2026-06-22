"""Common interface for every signal agent (strategy).

Every agent — moving averages, ML, econometric — exposes the same method, so
the backtester can treat them identically. This single interface is what makes
'swap one agent for another' trivial, and it is the heart of the agent-based
design.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Turns price data into target positions.

    The returned Series is aligned to the price index. Each value is the
    desired position for that bar: 1 = fully long, 0 = flat, -1 = fully short.
    """

    name: str = "strategy"

    @abstractmethod
    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        """Return a position Series given an OHLCV DataFrame."""
        raise NotImplementedError
