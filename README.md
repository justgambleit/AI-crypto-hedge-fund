# AI Crypto Hedge Fund

An MVP of an AI-agent-based cryptocurrency trading and risk-management system,
built as a take-home assignment. It pairs a conceptual design (presentation)
with a reproducible research notebook that backtests five levels of increasing
complexity — all out-of-sample and after costs.

## Quick start

Requires [uv](https://docs.astral.sh/uv/). All data is already included in `data/`,
so nothing needs to be downloaded to reproduce the results.

```bash
uv sync                # create the environment from uv.lock
uv run jupyter lab     # open notebook.ipynb -> Kernel -> Restart & Run All
```

Data can be re-fetched (optional) via the helpers in `scripts/`.

## Project structure

```
crypto-hedge-fund/
├─ notebook.ipynb      # deliverable: 5 levels + bonuses, end-to-end
├─ pyproject.toml      # pinned, reproducible environment (uv.lock)
├─ data/               # all OHLCV data (6 core coins + 120-coin universe)
├─ scripts/            # data download helpers (ccxt / Binance)
├─ tests/              # pytest unit tests
└─ src/chf/            # the `chf` package (modular, importable)
   ├─ features.py          # ML features & target (look-ahead free)
   ├─ strategies/          # base.py, moving_average.py, ml.py, indicators.py
   └─ services/            # backtest, metrics, portfolio, rebalancing,
                           #   universe, sizing
```

A GARCH(1,1) econometric agent is defined in the notebook (Level 2).

## The five levels

1. **Baseline** — MA crossover on BTC. Cuts drawdown but trails buy & hold on return.
2. **Econometrics + ML + agents** — GARCH(1,1), RandomForest and rule-based agents
   compared. ML shows no out-of-sample edge; the econometric vol-filter modestly
   beats passive holding; the simplest rule wins.
3. **Portfolio (static)** — 6 coins, Markowitz vs equal weight. Equal weight (1/N)
   +33.7% beat max-Sharpe −43.8% out-of-sample (DeMiguel, 2009).
4. **Dynamic rebalancing** — walk-forward weights; monthly rebalancing optimal by Sharpe.
5. **Large universe** — 120 coins, point-in-time momentum + a risk agent.
   Underperforms BTC out-of-sample; the risk agent sat out the FTX collapse (0% vs −16%).

Bonuses: position sizing (fractional Kelly gave the best risk-adjusted return) and a
classical-indicator comparison.

## Methodology & honesty

- **Out-of-sample only** — chronological train/test split; metrics on the test period.
- **No look-ahead** — signals executed with a one-bar lag; trailing features only.
  A one-bar look-ahead bug in the universe backtester was found and fixed.
- **Robustness** — time-series cross-validation (5 expanding folds) for the ML agent;
  transaction-cost sensitivity up to 0.4%; crisis stress-tests (LUNA, FTX, Oct-2025).
- **No survivorship bias** — point-in-time coin selection in the large universe.
- **Reproducible** — pinned environment, all data included, single self-contained notebook.

## Key takeaway

Across every level, added complexity did not pay: simple, risk-aware approaches beat
optimization and machine learning out-of-sample. The defensible edge is disciplined
risk management and honest validation.
