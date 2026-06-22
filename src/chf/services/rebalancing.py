"""Dynamic (walk-forward) portfolio rebalancing.

Instead of fixing weights once, we step through time and periodically recompute
the target weights using ONLY a trailing window of past data — so the strategy
adapts to changing markets with no look-ahead. Trading costs are charged on
every weight change, which is what makes over-frequent rebalancing expensive.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from chf.services.portfolio import annualized_stats, max_sharpe, min_variance

WeightFn = Callable[[pd.DataFrame], pd.Series]


def equal_weights(window: pd.DataFrame) -> pd.Series:
    cols = window.columns
    return pd.Series(1.0 / len(cols), index=cols)


def min_var_weights(window: pd.DataFrame) -> pd.Series:
    mean, cov = annualized_stats(window)
    return min_variance(mean, cov)


def max_sharpe_weights(window: pd.DataFrame) -> pd.Series:
    mean, cov = annualized_stats(window)
    return max_sharpe(mean, cov)


def rebalance_backtest(
    returns: pd.DataFrame,
    weight_fn: WeightFn,
    lookback: int = 180,
    rebalance_every: int = 30,
    cost: float = 0.001,
):
    """Walk forward; recompute target weights every `rebalance_every` days.

    Returns (net_returns, weights_history, n_rebalances).
    """
    n = len(returns)
    weights_history = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)

    rebalance_days = range(lookback, n, rebalance_every)
    for d in rebalance_days:
        window = returns.iloc[d - lookback:d]            # strictly past data
        w = weight_fn(window).reindex(returns.columns).fillna(0.0)
        weights_history.iloc[d] = w.values

    # Hold weights between rebalances; sit in cash before the first one.
    weights_history = weights_history.ffill().fillna(0.0)

    turnover = weights_history.diff().abs().sum(axis=1).fillna(0.0)
    gross = (weights_history * returns).sum(axis=1)
    net = gross - turnover * cost
    return net, weights_history, len(list(rebalance_days))
