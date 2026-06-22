"""Portfolio optimisation — classic mean-variance (Markowitz).

Everything is written explicitly with scipy so each step is transparent:
- annualise the mean returns and covariance from daily data,
- find the minimum-variance and maximum-Sharpe long-only portfolios,
- generate random portfolios to draw the efficient-frontier cloud.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

PERIODS_PER_YEAR = 365


def annualized_stats(daily_returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Annualised mean-return vector and covariance matrix from daily returns."""
    mean = daily_returns.mean() * PERIODS_PER_YEAR
    cov = daily_returns.cov() * PERIODS_PER_YEAR
    return mean, cov


def portfolio_performance(weights, mean, cov) -> tuple[float, float, float]:
    """Return (annual return, annual volatility, Sharpe) for given weights."""
    w = np.asarray(weights, dtype=float)
    mu = np.asarray(mean, dtype=float)
    sigma = np.asarray(cov, dtype=float)
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ sigma @ w))
    sharpe = ret / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def _optimize(objective: Callable[[np.ndarray], float], n: int) -> np.ndarray:
    bounds = [(0.0, 1.0)] * n                       # long-only, no leverage
    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}  # fully invested
    x0 = np.repeat(1.0 / n, n)
    res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    return res.x


def min_variance(mean: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Long-only portfolio with the smallest possible volatility."""
    sigma = cov.values
    w = _optimize(lambda w: w @ sigma @ w, len(mean))
    return pd.Series(w, index=mean.index)


def max_sharpe(mean: pd.Series, cov: pd.DataFrame, risk_free: float = 0.0) -> pd.Series:
    """Long-only portfolio with the best risk-adjusted return (tangency)."""
    mu = mean.values
    sigma = cov.values

    def neg_sharpe(w: np.ndarray) -> float:
        ret = w @ mu
        vol = np.sqrt(w @ sigma @ w)
        return -(ret - risk_free) / vol if vol > 0 else 0.0

    w = _optimize(neg_sharpe, len(mean))
    return pd.Series(w, index=mean.index)


def random_portfolios(n_portfolios: int, mean: pd.Series, cov: pd.DataFrame, seed: int = 0):
    """Sample random long-only weights; return arrays of (return, vol, sharpe)."""
    rng = np.random.default_rng(seed)
    n = len(mean)
    rets, vols, sharpes = [], [], []
    for _ in range(n_portfolios):
        w = rng.random(n)
        w /= w.sum()
        r, v, s = portfolio_performance(w, mean, cov)
        rets.append(r)
        vols.append(v)
        sharpes.append(s)
    return np.array(rets), np.array(vols), np.array(sharpes)


def portfolio_returns(weights: pd.Series, daily_returns: pd.DataFrame) -> pd.Series:
    """Daily portfolio returns for fixed weights (rebalanced to target each day)."""
    cols = list(weights.index)
    return daily_returns[cols].mul(weights.values, axis=1).sum(axis=1)
