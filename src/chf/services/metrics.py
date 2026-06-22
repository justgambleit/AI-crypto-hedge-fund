"""Performance and risk metrics computed from a return series.

Daily crypto data trades 365 days a year, so we annualise with 365 rather than
the 252 used for stock markets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 365


def total_return(returns: pd.Series) -> float:
    """Cumulative return over the whole period (ROI)."""
    return float((1 + returns).prod() - 1)


def cagr(returns: pd.Series) -> float:
    """Compound annual growth rate."""
    n = len(returns)
    if n == 0:
        return 0.0
    growth = (1 + returns).prod()
    return float(growth ** (PERIODS_PER_YEAR / n) - 1)


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Annualised return per unit of volatility."""
    excess = returns - risk_free / PERIODS_PER_YEAR
    sd = excess.std()
    if sd == 0:
        return 0.0
    return float(np.sqrt(PERIODS_PER_YEAR) * excess.mean() / sd)


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Annualised return per unit of *downside* volatility.

    Like Sharpe, but the denominator only counts losing days, so a strategy is
    not punished for big *upside* swings. This is the 'Sortino' objective used
    when optimising strategies in practice.
    """
    excess = returns - risk_free / PERIODS_PER_YEAR
    downside = np.minimum(excess, 0.0)
    dd = float(np.sqrt((downside ** 2).mean()))   # downside deviation
    if dd == 0 or np.isnan(dd):
        return 0.0
    return float(np.sqrt(PERIODS_PER_YEAR) * excess.mean() / dd)


def max_drawdown(equity_curve: pd.Series) -> float:
    """Largest peak-to-trough drop of the equity curve (negative number)."""
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return float(drawdown.min())


def summary(returns: pd.Series, equity_curve: pd.Series) -> dict[str, float]:
    return {
        "total_return": total_return(returns),
        "CAGR": cagr(returns),
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "max_drawdown": max_drawdown(equity_curve),
    }
