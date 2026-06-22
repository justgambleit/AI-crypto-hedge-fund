"""Load the OHLCV data produced by scripts/download_data.py.

This is the 'data adapter' layer: the only place that knows where files live
and what format they are in. The rest of the code just asks for prices.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[3] / "data"


def load_ohlcv(symbol: str) -> pd.DataFrame:
    """Return the full OHLCV frame for one symbol (e.g. 'BTC/USDT')."""
    fname = symbol.replace("/", "-") + ".parquet"
    path = DATA_DIR / fname
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: uv run python scripts/download_data.py"
        )
    return pd.read_parquet(path)


def load_close_prices(symbols: list[str]) -> pd.DataFrame:
    """Return a date-indexed DataFrame of close prices, one column per symbol."""
    return pd.DataFrame({s: load_ohlcv(s)["close"] for s in symbols}).dropna(how="all")
