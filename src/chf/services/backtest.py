"""A small, transparent vectorised backtester.

It takes a strategy's target positions and the asset's returns, applies a
one-bar execution lag (you act on a signal on the *next* bar, never the same
one — that avoids look-ahead bias), subtracts trading costs whenever the
position changes, and returns the equity curve plus metrics.

Writing this ourselves instead of using a black-box library means every number
is explainable — which matters when someone asks 'where does this Sharpe come
from?'.
"""
from __future__ import annotations

import pandas as pd

from chf.domain.models import BacktestResult
from chf.services import metrics
from chf.strategies.base import Strategy


def run_backtest(
    prices: pd.DataFrame,
    strategy: Strategy,
    cost: float = 0.001,  # 0.1% per trade (fees + slippage)
) -> BacktestResult:
    close = prices["close"]
    asset_returns = close.pct_change().fillna(0.0)

    # A signal seen on bar t is executed on bar t+1 -> shift by one.
    positions = strategy.generate_signals(prices).shift(1).fillna(0.0)

    # Cost is charged on the size of each position change (turnover).
    turnover = positions.diff().abs().fillna(0.0)
    strat_returns = positions * asset_returns - turnover * cost

    equity = (1 + strat_returns).cumprod()
    return BacktestResult(
        equity_curve=equity,
        returns=strat_returns,
        positions=positions,
        metrics=metrics.summary(strat_returns, equity),
    )
