"""Download a large universe of coins for Level 5 (100+ pairs).

    uv run python scripts/download_universe.py

Picks the most liquid USDT spot pairs by 24h quote volume (excluding
stablecoins and leveraged tokens), then saves daily OHLCV for each into
data/universe/. Coins have different history lengths — that's expected and
handled at backtest time (point-in-time selection, no common-window dropna).
"""
from __future__ import annotations

import time
from pathlib import Path

import ccxt
import pandas as pd

EXCHANGES = ["binance", "kraken", "kucoin", "bybit"]
N_COINS = 120
TIMEFRAME = "1d"
SINCE = "2021-01-01T00:00:00Z"
CORE = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ZEC/USDT", "DOGE/USDT"]

# Skip stablecoins and leveraged/wrapped tokens.
SKIP_BASES = {"USDC", "FDUSD", "TUSD", "DAI", "USDD", "USDP", "EUR", "GBP",
              "USTC", "BUSD", "PAX", "SUSD", "USD1", "XUSD", "WBTC", "WBETH"}
SKIP_FRAGMENTS = ("UP", "DOWN", "BULL", "BEAR")

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "universe"


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


def pick_symbols(ex) -> list[str]:
    tickers = ex.fetch_tickers()
    rows = []
    for sym, t in tickers.items():
        if not sym.endswith("/USDT"):
            continue
        mkt = ex.markets.get(sym, {})
        if not mkt.get("spot", False) or not mkt.get("active", True):
            continue
        base = sym.split("/")[0]
        if base in SKIP_BASES or any(f in base for f in SKIP_FRAGMENTS):
            continue
        vol = t.get("quoteVolume") or 0
        rows.append((sym, vol))
    rows.sort(key=lambda r: r[1], reverse=True)
    chosen = [s for s, _ in rows[:N_COINS]]
    for c in CORE:                       # always include our core coins
        if c not in chosen and c in ex.markets:
            chosen.append(c)
    return chosen


def fetch_all(ex, sym, since):
    rows = ex.fetch_ohlcv(sym, TIMEFRAME, since=since, limit=1000)
    out = rows[:]
    while rows and len(rows) == 1000:
        rows = ex.fetch_ohlcv(sym, TIMEFRAME, since=rows[-1][0] + 1, limit=1000)
        out += rows
        time.sleep(ex.rateLimit / 1000)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ex = get_exchange()
    since = ex.parse8601(SINCE)
    symbols = pick_symbols(ex)
    print(f"Downloading {len(symbols)} coins -> {OUT_DIR}\n")

    saved = 0
    for i, sym in enumerate(symbols, 1):
        try:
            rows = fetch_all(ex, sym, since)
        except Exception as e:  # noqa: BLE001
            print(f"[{i:3d}/{len(symbols)}] {sym:14s} ошибка: {e}")
            continue
        if not rows:
            print(f"[{i:3d}/{len(symbols)}] {sym:14s} нет данных")
            continue
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.drop(columns="ts").set_index("date")
        fname = sym.replace("/", "-") + ".parquet"
        df.to_parquet(OUT_DIR / fname)
        saved += 1
        print(f"[{i:3d}/{len(symbols)}] {sym:14s} {len(df):5d} дней -> {fname}")

    print(f"\nГотово: сохранено {saved} монет в {OUT_DIR}")


if __name__ == "__main__":
    main()
