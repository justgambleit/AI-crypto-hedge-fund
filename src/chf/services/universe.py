"""Level 5 — scaling to a large universe of coins.

With 100+ coins of different ages we cannot align on a common window. Instead we
select coins *point-in-time*: at each rebalance only coins that already existed
and traded long enough are eligible. We rank them by momentum, take the top N,
equal-weight them, and a risk agent moves us to cash in down-regimes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from chf.data.loader import DATA_DIR

UNIVERSE_DIR = DATA_DIR / "universe"


def load_universe_prices() -> pd.DataFrame:
    """Wide close-price matrix (dates x coins); NaN where a coin didn't trade."""
    series = {}
    for f in sorted(UNIVERSE_DIR.glob("*.parquet")):
        sym = f.stem.replace("-", "/")
        series[sym] = pd.read_parquet(f)["close"]
    return pd.DataFrame(series).sort_index()


def universe_backtest(
    prices: pd.DataFrame,
    top_n: int = 15,
    mom_lookback: int = 90,
    min_history: int = 180,
    rebalance_every: int = 30,
    cost: float = 0.001,
    use_risk_agent: bool = True,
    risk_ma: int = 100,
    btc: str = "BTC/USDT",
):
    """Walk-forward backtest over a large, ragged universe.

    Returns (net_returns, weights_history, info).
    """
    returns = prices.pct_change()
    n = len(prices)
    W = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)

    # Risk agent: down-regime when BTC is below its own moving average (point-in-time).
    btc_sma = prices[btc].rolling(risk_ma).mean()
    risk_off = prices[btc] < btc_sma

    start = max(min_history, mom_lookback, risk_ma)
    rebal_days = list(range(start, n, rebalance_every))
    n_cash = 0

    for d in rebal_days:
        if use_risk_agent and bool(risk_off.iloc[d]):
            W.iloc[d] = 0.0          # to cash
            n_cash += 1
            continue
        window = prices.iloc[d - min_history:d + 1]
        eligible = window.columns[window.notna().all()]
        if len(eligible) == 0:
            W.iloc[d] = 0.0
            continue
        mom = prices[eligible].iloc[d] / prices[eligible].iloc[d - mom_lookback] - 1
        chosen = mom.sort_values(ascending=False).head(top_n).index
        w = pd.Series(0.0, index=prices.columns)
        w[chosen] = 1.0 / len(chosen)
        W.iloc[d] = w.values

    W = W.ffill().fillna(0.0)
    # Decisions are made using close[d]; execute them from the NEXT bar so we
    # never earn a return that ended before the decision -> no one-day look-ahead.
    W_exec = W.shift(1).fillna(0.0)
    turnover = W_exec.diff().abs().sum(axis=1).fillna(0.0)
    gross = (W_exec * returns).sum(axis=1).fillna(0.0)   # NaN returns on non-held coins are skipped
    net = gross - turnover * cost

    active = W.index >= prices.index[start]
    info = {
        "n_coins": prices.shape[1],
        "n_rebalances": len(rebal_days),
        "avg_holdings": float((W > 0).sum(axis=1)[active].mean()),
        "pct_in_cash": n_cash / max(1, len(rebal_days)),
    }
    return net, W, info
