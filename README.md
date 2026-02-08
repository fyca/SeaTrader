# SeaTrader (tradebot)

Paper-first, swing/long-term **long-only** automated trading bot for **US equities + crypto** using **Alpaca US**.

This repo is designed to be:
- **Safe by default** (paper trading + dry-run guards)
- **Unattended-capable** (CLI commands suitable for cron/schedulers)
- **Explainable** (dashboards, artifacts, trade logs, backtest detail)
- **Extensible** (pluggable strategies + GUI Strategy Builder)

> Disclaimer: This project is for educational/personal use. Trading involves risk. Nothing here is financial advice.

---

## Where we started → where we are

### Initial goal
Build a safe, unattended swing/long-term trading bot that:
- trades **both equities and crypto**
- paper trades first, then can be switched to live via config
- includes a local **HTML dashboard** for monitoring + config
- includes **backtesting** (including parameter sweeps)
- supports **pluggable strategies**, including a **GUI Strategy Builder** to define rules and weighted scoring

### Current state (high level)
✅ Working end-to-end pipeline:
- Universe → bars → signals → target weights → **order plan** → optional **paper order placement**
- Daily risk-check with drawdown tracking and exit signals
- Dashboard with monitoring + backtesting UI + Strategy Builder
- Strategy registry supports built-ins + user-defined strategies
- Backtest engine supports stocks/crypto, slippage model, portfolio DD stop behavior A, per-asset stop loss, and many toggles

---

## Key constraints / design decisions
- **Swing/long-term** focus (default: weekly rebalance + daily risk-check)
- **Long-only** (no shorts)
- **No margin**
- **No leveraged/inverse ETFs**
- **No penny stocks** (default min price: $5)
- Target allocation: **50% equities / 50% crypto** (cash allowed when nothing qualifies)
- Max positions: **10 stocks** and **5 crypto**
- Portfolio drawdown limit: **20%**
  - Live behavior chosen: **A = freeze** (no new entries; exits allowed)

---

## Safety defaults and guardrails

### Dry-run / order placement gating
- Default is **DRY_RUN** behavior: compute plans and print what would be done.
- Orders are only sent when you explicitly add `--place-orders`.

### Order guardrails (execution limits)
Configurable hard limits to prevent surprises:
- `execution.max_orders_per_run`
- `execution.max_single_order_notional_usd`
- `execution.max_total_notional_usd`

### Drawdown freeze
- Drawdown peak tracking stored locally in `data/state.json`
- When drawdown exceeds configured limit, bot can enter **FROZEN** mode
  - freezes new entries
  - allows exits/reductions

---

## Project layout

- `src/tradebot/` – core package
  - `cli.py` – main CLI entrypoint
  - `commands/` – CLI commands (`rebalance`, `risk-check`, `dashboard`)
  - `adapters/` – Alpaca clients + bar fetching
  - `strategies/` – built-in and user strategies
  - `backtest/` – backtest engine + job runner + cache
  - `dashboard/` – FastAPI app + HTML UI (dashboard + builder)
  - `risk/` – drawdown + exit signal helpers
  - `util/` – state, artifacts, equity curve utilities

- `config/config.yaml` – main configuration
- `data/` – local runtime output (artifacts, caches, backtests)

---

## Setup

### 1) Create a venv and install

```bash
cd tradebot
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### 2) Configure environment

Create a local `.env` (never commit secrets):

- Alpaca paper keys
- dashboard token settings

A `.env.example` exists as a template.

> Important: `.env` is gitignored. Do not commit credentials.

### 3) Verify Alpaca connectivity

Run dashboard or a dry-run command (below). If you hit an Alpaca market data error about SIP subscription, see **Data feed**.

---

## Data feed note (Alpaca)

For equities, this repo uses the **IEX feed** for Alpaca data access to avoid the common error:
> `subscription does not permit querying recent SIP data`

Crypto bars are fetched via Alpaca crypto data.

---

## Running the bot (CLI)

### Rebalance (plan only)
Computes a rebalance plan and prints it; does not place orders.

```bash
tradebot rebalance --config config/config.yaml
```

### Rebalance (place paper orders)
Places orders only if:
- `DRY_RUN=false` and
- you pass `--place-orders`

```bash
tradebot rebalance --config config/config.yaml --place-orders
```

### Daily risk check
Runs drawdown checks, generates exit signals, and writes artifacts.

```bash
tradebot risk-check --config config/config.yaml
```

---

## Dashboard

Start the local dashboard:

```bash
./scripts/start_dashboard.sh
```

Default URL:
- http://127.0.0.1:8008

Dashboard highlights:
- Account overview, positions, open orders, fills
- Exposure view including **pending orders** and estimated quantities
- Run artifacts (last rebalance / last risk-check / last placed orders)
- Backtesting UI (run + history + charts)
- Strategy dropdown and a button to open the Strategy Builder

### Dashboard auth token
Token gating can be enabled/disabled via env/config:
- `DASHBOARD_REQUIRE_TOKEN=false` (default in this project)

If you enable it, endpoints that mutate state (start backtest, save config, etc.) will require a token.

---

## Strategies

### Built-in strategies
Currently included (names may evolve):
- `baseline_trendvol`
- `regime_filtered_trendvol` (SPY/BTC trend gating)
- `breakout_trend`
- `pullback_in_trend`

### User strategies (Strategy Builder)
The Strategy Builder lets you create rule-based strategies with:
- **Entry rules**: nested AND/OR groups
- **Exit rules**: nested AND/OR groups
- **Score factors**: weighted sum of numeric indicators and boolean conditions

User strategies are stored as JSON under:
- `tradebot/strategies/user/<id>.json`

They appear automatically in the strategy dropdown.

### Rule engine indicators
Supported indicator kinds in rule-based strategies include:
- `CLOSE`
- `SMA(n)`
- `EMA(n)`
- `RSI(n)`
- `HIGHEST(n)`
- `LOWEST(n)`
- `ROC(n)`
- `RET_1D`
- `ANN_VOL(n)`

Derived/preset indicators:
- `DIST_SMA(n)`
- `BREAKOUT(n)`
- `CROSS_ABOVE(fast, slow)` (exposed via presets)
- `SMA_SLOPE(n, lookback)` (exposed via presets)

---

## Backtesting

Backtests run in the background via the dashboard or programmatically.

### Features
- Stocks + crypto support
- Slippage model (bps)
- Weekly or daily rebalance
- Liquidation modes:
  - liquidate non-selected
  - hold until exit
- Per-asset stop loss (daily check)
- Portfolio drawdown stop (behavior A: liquidate-to-cash, resume next rebalance)
- Strategy selection in the backtest UI
- Benchmarks overlay (S&P 500 index line; SPY comparison line can be toggled/removed)
- Interactive charts (Plotly): hover, zoom, range slider, unified x-hover
- Portfolio RSI plotted under the equity curve
- Optional exclusion rule: stop trading a symbol if its P/L falls below a floor
  - optionally includes unrealized P/L
  - optionally liquidates immediately when excluded

### Backtest artifacts
Each backtest job writes:
- `data/backtests/<job_id>/status.json`
- `data/backtests/<job_id>/result.json`

Backtest history in the UI shows key metrics for quick scanning.

### Caching
Bars are cached to parquet under:
- `data/cache/bars/`

Cache keys include the date range to avoid stale-range bugs.

---

## Artifacts (runtime outputs)

The bot writes these “last run” artifacts into `data/`:
- `last_account.json`
- `last_rebalance.json`
- `last_risk_check.json`
- `last_placed_orders.json`

And maintains:
- `state.json` (drawdown peak and freeze tracking)
- `equity_curve.jsonl` (for dashboard charting)

---

## Troubleshooting

### Alpaca SIP subscription error
If you see:
- `subscription does not permit querying recent SIP data`

Use IEX feed for equities (this repo already defaults to IEX in the adapter).

### Backtest doesn’t reflect selected strategy
Make sure dashboard is restarted after code changes and that backtest UI strategy selection is set (the backtest strategy dropdown is intentionally not overwritten by auto-refresh).

---

## Roadmap / future improvements

High-impact next steps:

### Strategy Builder
- Expand indicator library further (including multi-input indicators like ATR requires OHLC data)
- Better score-factor editing (more presets, validation, templates)
- “Explain why entry/exit passed/failed” (per-condition evaluation output)

### Backtesting
- Extend benchmark options and robust comparisons
- Performance refactor: precompute rolling indicators per symbol to speed large sweeps
- Better walk-forward testing / cross-validation

### Live execution enhancements
- Broker-native stop orders (requires order management/cancel-replace)
- Improved order types and sizing policies
- Scheduling hooks (cron templates, “run every weekday at X”, etc.)

### Data
- Improve/validate long-history cache coverage for meaningful 5y+ results

---

## Quick commands (copy/paste)

```bash
# start dashboard
./scripts/start_dashboard.sh

# plan rebalance (no orders)
tradebot rebalance --config config/config.yaml

# place paper orders (requires DRY_RUN=false and --place-orders)
tradebot rebalance --config config/config.yaml --place-orders

# daily risk check
tradebot risk-check --config config/config.yaml
```
