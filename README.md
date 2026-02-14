# SeaTrader (tradebot)

Paper-first, long-only automation for **US equities + crypto** on Alpaca.

SeaTrader includes:
- Live/paper rebalance and risk-check commands
- Backtesting with per-asset controls (stocks vs crypto)
- Dashboard UI + Strategy Builder
- Safety guardrails and heartbeat monitoring

> Educational/personal use only. Not financial advice.

---

## Quick start

```bash
cd tradebot
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Create `.env` from `.env.example` and set Alpaca keys.

Run dashboard:

```bash
./scripts/start_dashboard.sh
```

Open: `http://127.0.0.1:8008`

---

## Operator docs

- **Step-by-step usage guide:** [`docs/INSTRUCTIONS.md`](docs/INSTRUCTIONS.md)
- Screenshot callouts were removed for clarity; instructions are text-first.

---

## Core safety model

- Long-only
- No margin/shorts
- Per-asset stop loss support
- Drawdown freeze controls
- Order guardrails (`max_orders_per_run`, `max_single_order_notional_usd`, `max_total_notional_usd`)
- Paper-first workflow encouraged

---

## Configuration

Primary config: `config/config.yaml`

Key blocks:
- `allocation`
- `limits`
- `risk`
- `execution`
- `scheduling`
- `rebalance`

### Important risk setting

`risk.execute_exit_liquidations`
- `false` (default): risk-check is signal-only
- `true`: risk-check submits market sell orders for exit signals

For real order placement, also ensure:
- `dry_run: false`

---

## Running from CLI

### Rebalance (plan only)

```bash
tradebot rebalance --config config/config.yaml
```

### Rebalance with orders

```bash
tradebot rebalance --config config/config.yaml --place-orders
```

### Risk-check

```bash
tradebot risk-check --config config/config.yaml
```

---

## Dashboard highlights

- Live/paper controls with per-asset settings
- Per-asset schedules (rebalance + risk-check)
- Backtest runner + iteration sweeps
- Daily ledger, trade history, open positions at end
- Strategy Builder integration

### UX note

Legacy/global fallback controls were removed from the dashboard.  
Fallback behavior is configured per asset class only.

---

## Backtesting

Backtests write artifacts to:
- `data/backtests/<job_id>/status.json`
- `data/backtests/<job_id>/result.json`

Useful runtime artifacts:
- `data/last_rebalance.json`
- `data/last_risk_check.json`
- `data/last_placed_orders.json`
- `data/state.json`

---

## Heartbeat monitoring (SeaTrader-focused)

Heartbeat checks are defined in workspace `HEARTBEAT.md` and currently watch:
- New backtest completion/failure
- New risk-check liquidation/exit activity

State is tracked in:
- `memory/heartbeat-state.json`

---

## Troubleshooting

### Risk-check not liquidating
- Verify `risk.execute_exit_liquidations: true`
- Verify `dry_run: false`
- Check `data/last_risk_check.json -> executed_liquidations`

### Alpaca SIP subscription error
Use IEX feed for equities (already defaulted in this project).

---

## Repo layout

- `src/tradebot/cli.py` – CLI entrypoint
- `src/tradebot/commands/` – rebalance, risk-check, dashboard actions
- `src/tradebot/backtest/` – engine + jobs
- `src/tradebot/dashboard/` – FastAPI app + frontend
- `src/tradebot/strategies/` – built-in and user strategies
- `config/` – config and presets
- `data/` – runtime artifacts/cache/backtest outputs
