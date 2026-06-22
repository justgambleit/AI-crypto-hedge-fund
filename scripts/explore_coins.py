"""Quick check: how much history exists for a candidate coin list.

    uv run python scripts/explore_coins.py

Downloads nothing permanent — just prints how many daily candles each symbol
has and what the common (aligned) window would be. Handy for spotting coins
that are too new for a proper train/test backtest.
"""
from __future__ import annotations

import time

import ccxt
import pandas as pd

EXCHANGES = ["binance", "kraken", "kucoin", "bybit"]
CANDIDATES = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ZEC/USDT", "PUMP/USDT", "HYPE/USDT"]
TIMEFRAME = "1d"
SINCE = "2021-01-01T00:00:00Z"


def get_exchange():
    for name in EXCHANGES:
        try:
            ex = getattr(ccxt, name)({"enableRateLimit": True})
            ex.load_markets()
            print(f"Exchange: {name}\n")
            return ex
        except Exception as e:  # noqa: BLE001
            print(f"  {name} unavailable: {e}")
    raise RuntimeError("No exchange reachable")


def fetch_all(ex, sym, since):
    rows = ex.fetch_ohlcv(sym, TIMEFRAME, since=since, limit=1000)
    out = rows[:]
    while rows and len(rows) == 1000:
        rows = ex.fetch_ohlcv(sym, TIMEFRAME, since=rows[-1][0] + 1, limit=1000)
        out += rows
        time.sleep(ex.rateLimit / 1000)
    return out


def main():
    ex = get_exchange()
    since = ex.parse8601(SINCE)
    series = {}
    for sym in CANDIDATES:
        if sym not in ex.markets:
            print(f"{sym:12s}  НЕ ТОРГУЕТСЯ на {ex.id}")
            continue
        rows = fetch_all(ex, sym, since)
        if not rows:
            print(f"{sym:12s}  нет данных")
            continue
        idx = pd.to_datetime([r[0] for r in rows], unit="ms", utc=True)
        s = pd.Series([r[4] for r in rows], index=idx)
        series[sym] = s
        print(f"{sym:12s}  {len(s):5d} дней   {s.index.min().date()} → {s.index.max().date()}")

    if series:
        aligned = pd.DataFrame(series).dropna()
        print(f"\nОбщий период по ВСЕМ монетам сразу: {len(aligned)} дней")
        if len(aligned):
            print(f"  ({aligned.index.min().date()} → {aligned.index.max().date()})")
        verdict = "достаточно" if len(aligned) > 500 else "МАЛО для честного train/test"
        print(f"Вывод: {verdict}")


if __name__ == "__main__":
    main()
