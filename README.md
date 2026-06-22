# Crypto Hedge Fund (MVP)

An AI agent-based system for automated cryptocurrency trading and risk
management. Built incrementally across five levels, from a single-pair
baseline to a dynamically rebalanced portfolio of 100+ pairs.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for a reproducible
environment.

```bash
# 1. install uv (once)   macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. build the environment
uv sync
# 3. download historical data (once)
uv run python scripts/download_data.py
# 4. open the notebook
uv run jupyter lab
# (optional) lint and test
uv run ruff check .
uv run pytest
```

## Architecture

Clean, layered design — each folder has one job:

```
src/chf/
├── domain/        data structures shared across the system (models.py)
├── data/          loading market data (loader.py)
├── strategies/    the agents: base.py interface + concrete strategies
│   ├── base.py            common Strategy interface
│   └── moving_average.py  Level 1 baseline
└── services/      the engine room
    ├── backtest.py        runs a strategy, returns equity + metrics
    └── metrics.py         ROI, Sharpe, max drawdown
```

The single deliverable notebook imports from `chf` and walks through the five
levels. The library stays modular; the notebook tells the story.

## Reproducibility

Clone -> `uv sync` -> run the notebook top to bottom. `uv.lock` pins exact
versions; the committed data makes results deterministic.

## Live trading

Backtesting needs no API keys. For future live mode, copy `.env.example` to
`.env` and fill it in. `.env` is git-ignored and must never be committed.
