"""Unit tests for the metrics module. Run with: uv run pytest"""
from __future__ import annotations

import pandas as pd

from chf.services import metrics


def test_total_return():
    r = pd.Series([0.10, 0.0, -0.05])
    expected = 1.10 * 1.0 * 0.95 - 1
    assert round(metrics.total_return(r), 6) == round(expected, 6)


def test_max_drawdown():
    equity = pd.Series([1.0, 1.2, 0.9, 1.0])
    expected = 0.9 / 1.2 - 1  # -25%
    assert round(metrics.max_drawdown(equity), 6) == round(expected, 6)


def test_sharpe_zero_volatility_is_zero():
    flat = pd.Series([0.0, 0.0, 0.0])
    assert metrics.sharpe_ratio(flat) == 0.0
