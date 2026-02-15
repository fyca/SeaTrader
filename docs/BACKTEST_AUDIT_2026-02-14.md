# SeaTrader Backtest Audit (2026-02-14)

Scope reviewed:
- `src/tradebot/backtest/engine.py`
- `src/tradebot/backtest/job.py`
- `src/tradebot/backtest/intraday.py`
- dashboard iteration wiring in `src/tradebot/dashboard/index.html`
- recent backtest artifacts in `data/backtests/*`

## Executive summary

I found several issues that can materially affect trust in metrics:

1. **Sharpe is inflated by construction** (calendar-day annualization + weekend flat marks).
2. **Backtest selection can differ from live** because strategy input bars in engine drop non-close columns (e.g., volume filters silently bypassed).
3. **Global `use_limit_orders` is effectively ignored** in backtest order decisioning unless per-asset order type is explicitly set.
4. **Intraday pricing callbacks are gated by global rebalance schedule, not per-asset schedule** (can mis-price per-asset schedule runs).
5. **Time iteration UX can still mislead users** when execution mode is `daily` (time iteration fields are irrelevant, but UI allows it).

Also: crypto iteration showing almost identical max drawdown is plausible with current implementation because drawdown is measured from daily close-marked equity curve and may not change even when fill timing changes.

---

## Findings

### 1) Sharpe annualization likely too high (High)

- In `engine.py`, Sharpe and volatility are annualized using `sqrt(365)` for all runs.
- Equity curve is evaluated on calendar days (`pd.date_range(..., freq="D")`) and marked with last available close on non-trading days for equities.

Impact:
- Calendar-day treatment + flat weekend returns can reduce measured daily volatility, while annualization stays aggressive.
- This can push Sharpe up vs standard trading-day convention.

Observed from current artifacts:
- Reported Sharpe around **6.3**.
- Same curve recomputed at `sqrt(252)` is materially lower (~5.2), and weekday-only approximation lower still (~4.9).

Recommendation:
- Use asset-aware annualization:
  - equities/both: 252
  - crypto-only: 365
- Consider computing returns only on days with market movement for equities (or use trading-calendar index).

---

### 2) Backtest strategy input drops volume/other fields (High)

- In `run_backtest`, bars passed to strategies are reconstructed as DataFrames with only:
  - `{"close": s2.values}`
- Strategies like `breakout_trend` check optional volume/liquidity filters when `volume` exists.

Impact:
- Backtest may skip liquidity gates that live selection uses.
- Can materially change symbol selection and performance realism.

Recommendation:
- Slice and pass full source OHLCV frame up to day (not just close), preserving columns used by strategies.

---

### 3) Global `use_limit_orders` not honored in `_use_limit_for` (High)

- `_use_limit_for(sym)` currently:
  - returns per-asset order type if set (`order_type_*`)
  - else returns `False`
- It does **not** fall back to `params.use_limit_orders`.

Impact:
- Users may enable global limit orders and still get market behavior unless per-asset fields are set.

Recommendation:
- `_use_limit_for` should return `bool(params.use_limit_orders)` when per-asset order type is unset.

---

### 4) Intraday pricing gating uses global rebalance schedule (Medium)

- In `job.py`, intraday callback is allowed only on `reb_days` derived from global `p.rebalance/p.rebalance_day`.
- Engine itself supports per-asset schedules (`eq_rebal_days`, `cr_rebal_days`).

Impact:
- If per-asset schedule diverges from global schedule, intraday pricing can be suppressed on valid asset rebalance days.

Recommendation:
- Gate intraday callback by symbol class + per-asset rebalance day set.

---

### 5) Iteration UI can appear to test times while times are irrelevant (Medium)

- If execution mode is `daily`, `execution_time_local` iteration won’t affect pricing.
- UI currently allows iteration regardless of mode.

Impact:
- Users can run many iterations and get near-identical results with no obvious warning.

Recommendation:
- Block or warn when iterating time axes while mode is `daily`.
- Auto-switch to `intraday` on time-axis iteration (with explicit UI notice).

---

### 6) Why crypto max drawdown can stay constant across time iterations (Informational)

- Drawdown is computed from daily equity marks (`portfolio_value(day)` using close-based `px`).
- Timing changes alter fill prices, but max DD can remain at same magnitude/date if path shape is similar.

Implication:
- Unchanged max DD alone is not definitive proof of a bug.
- Sharpe/return changing while DD remains constant can happen.

---

## Already-fixed related issue

- Iteration axis synchronization bug for per-asset times was fixed previously:
  - commit `620374a`
  - aligns `execution_time_local_*` and `risk_check_time_local_*` during iteration runs.

---

## Suggested fix order (fastest impact)

1. Fix `_use_limit_for` global fallback.
2. Pass full OHLCV slices to strategy selectors in backtest.
3. Fix Sharpe/vol annualization conventions (252 vs 365 + trading-day handling).
4. Fix intraday callback gating to per-asset schedule.
5. Add dashboard warnings for invalid/meaningless time-iteration modes.

---

## Validation checklist after fixes

- Compare 3 known jobs before/after with identical seed params.
- Verify symbol universe/selection parity against live dry-run logs.
- Re-run crypto-only time iteration; check:
  - entries/exits count differences,
  - execution price distribution,
  - DD sensitivity,
  - Sharpe range plausibility.
- Add unit tests for:
  - `_use_limit_for` fallback,
  - per-asset intraday day gating,
  - strategy input column preservation,
  - sharpe annualization path.
