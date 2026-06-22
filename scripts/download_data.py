"""Download historical OHLCV data and save it to data/.

Uses ccxt to fetch daily candles from a public exchange. Only public market
data is requested, so no API key is needed. Run this ONCE: the resulting
files are committed to the repo so the notebook stays fully reproducible.

    uv run python scripts/download_data.py
"""
from __future__ import annotations

import time
from pathlib import Path

import ccxt
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)

# Tried in order until one responds. Some exchanges are geo-blocked in
# certain regions, so we fall back automatically.
EXCHANGES = ["binance", "kraken", "kucoin", "bybit"]

SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
    "ZEC/USDT", "PEPE/USDT", "DOGE/USDT",
]
TIMEFRAME = "1d"
SINCE = "2021-01-01T00:00:00Z"


def get_exchange():
    last_err = None
    for name in EXCHANGES:
        try:
            ex = getattr(ccxt, name)({"enableRateLimit": True})
            ex.load_markets()
            print(f"Using exchange: {name}")
            return ex
        except Exception as e:  # noqa: BLE001
            print(f"  {name} unavailable: {e}")
            last_err = e
    raise RuntimeError(f"No exchange reachable. Last error: {last_err}")


def fetch_ohlcv(ex, symbol, timeframe, since_ms, limit=1000):
    rows, since = [], since_ms
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + 1
        if len(batch) < limit:
            break
        time.sleep(ex.rateLimit / 1000)
    return rows


def to_frame(rows):
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("date").drop(columns="ts").sort_index()


def main():
    ex = get_exchange()
    since_ms = ex.parse8601(SINCE)
    for symbol in SYMBOLS:
        if symbol not in ex.markets:
            print(f"  skip {symbol}: not listed on {ex.id}")
            continue
        print(f"Downloading {symbol} ...")
        df = to_frame(fetch_ohlcv(ex, symbol, TIMEFRAME, since_ms))
        fname = symbol.replace("/", "-") + ".parquet"
        df.to_parquet(DATA_DIR / fname)
        print(f"  saved {len(df)} rows -> data/{fname}")
    print("Done.")


if __name__ == "__main__":
    main()
