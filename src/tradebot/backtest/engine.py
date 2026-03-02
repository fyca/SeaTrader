from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Literal

import numpy as np
import pandas as pd

from tradebot.risk.exits import trend_break_exit
from tradebot.strategies.registry import get_strategy
from tradebot.strategies.rule_engine import EvalContext, eval_rule, eval_indicator


@dataclass(frozen=True)
class BacktestParams:
    start: str  # YYYY-MM-DD
    end: str    # YYYY-MM-DD
    initial_equity: float = 100000.0
    slippage_bps: float = 10.0
    # Deprecated global knobs retained only for backward-compat preset loading.
    use_limit_orders: bool = False
    limit_offset_bps: float = 10.0
    # Per-asset execution options
    order_type_equities: Literal["market", "limit"] | None = None
    order_type_crypto: Literal["market", "limit"] | None = None
    limit_offset_bps_equities: float | None = None
    limit_offset_bps_crypto: float | None = None
    # Optional parity with live: if limit not fillable by open, convert to market-at-open.
    limit_fallback_to_market_open: bool = False
    limit_fallback_to_market_open_equities: bool | None = None
    limit_fallback_to_market_open_crypto: bool | None = None
    limit_fallback_time_local: str = "06:30"
    rebalance: Literal["weekly", "daily"] = "weekly"
    rebalance_day: Literal["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"] = "MON"
    # Per-asset rebalance schedule (fallback to global rebalance/rebalance_day)
    rebalance_frequency_equities: Literal["weekly", "daily"] | None = None
    rebalance_day_equities: Literal["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"] | None = None
    rebalance_frequency_crypto: Literal["weekly", "daily"] | None = None
    rebalance_day_crypto: Literal["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"] | None = None

    # Execution pricing options
    # - daily mode: uses daily open/close as before
    # - intraday mode: fetches minute bars on rebalance days and prices at execution_time_local
    execution_time_mode: Literal["daily", "intraday"] = "intraday"
    execution_time: Literal["open", "close"] = "close"  # daily mode only
    execution_time_local: str = "15:55"  # intraday mode only
    execution_time_local_equities: str | None = None
    execution_time_local_crypto: str | None = None
    execution_tz: str = "America/Los_Angeles"
    # Exit/risk check time (intraday mode): used for stop/exclusion/dd exits
    risk_check_time_local: str = "12:30"
    risk_check_time_local_equities: str | None = None
    risk_check_time_local_crypto: str | None = None
    # Per-asset risk schedule (fallback to daily at risk_check_time_local)
    risk_check_frequency_equities: Literal["weekly", "daily", "hourly"] | None = None
    risk_check_day_equities: Literal["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"] | None = None
    risk_check_minute_of_hour_equities: int | None = None
    risk_check_hourly_checks_equities: list[Literal["stop_loss", "dd_stop", "strategy_exit"]] | None = None
    risk_check_frequency_crypto: Literal["weekly", "daily", "hourly"] | None = None
    risk_check_day_crypto: Literal["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"] | None = None
    risk_check_minute_of_hour_crypto: int | None = None
    risk_check_hourly_checks_crypto: list[Literal["stop_loss", "dd_stop", "strategy_exit"]] | None = None

    strategy_id: str = "baseline_trendvol"
    # Optional per-asset entry strategy overrides (fallback to strategy_id)
    strategy_id_equities: str | None = None
    strategy_id_crypto: str | None = None
    # Optional per-asset exit strategy references (reserved for strategy-builder exits)
    exit_strategy_id_equities: str | None = None
    exit_strategy_id_crypto: str | None = None
    exit_enabled_equities: bool = False
    exit_enabled_crypto: bool = False
    asset_mode: Literal["both", "equities", "crypto"] = "both"
    rebalance_mode: Literal["target_notional", "no_add_to_losers"] = "target_notional"
    liquidation_mode: Literal["liquidate_non_selected", "hold_until_exit"] = "liquidate_non_selected"
    per_asset_stop_loss_pct: float | None = None
    trailing_stop_stocks_enabled: bool = False
    trailing_stop_crypto_enabled: bool = False
    trailing_stop_stocks_start_gain_pct: float | None = 0.05
    trailing_stop_crypto_start_gain_pct: float | None = 0.05
    trailing_stop_stocks_pct: float | None = 0.02
    trailing_stop_crypto_pct: float | None = 0.02
    trailing_stop_anchor: Literal["highest_since_entry", "highest_close_since_entry"] = "highest_since_entry"
    # When False, block same-day buy->sell roundtrips in backtest (PDT-safe behavior).
    allow_same_day_roundtrip: bool = False

    # If (realized + optional unrealized) P/L percent for a symbol falls below this threshold,
    # permanently exclude it from further trading for the rest of the backtest.
    #
    # Percent is computed relative to current cost basis when a position is held:
    #   (realized_pnl + unrealized_pnl) / (avg_cost * qty)
    #
    # Example: -0.005 = -0.5%
    symbol_pnl_floor_pct: float | None = None

    # If True, immediately liquidate any currently-held position when a symbol hits the P/L floor.
    # If False, the symbol is excluded from new entries but existing holdings are not force-sold.
    symbol_pnl_floor_liquidate: bool = True

    # If True, apply the floor to (realized + unrealized) P/L for currently held positions.
    # If False, floor applies to realized P/L only.
    symbol_pnl_floor_include_unrealized: bool = True

    # Optional portfolio-level drawdown stop (behavior A):
    # - if equity drawdown from peak >= portfolio_dd_stop, liquidate ALL positions to cash
    # - stay in cash until the NEXT scheduled rebalance day
    portfolio_dd_stop: float | None = None

    universe_mode: Literal["full", "single"] = "full"
    symbol: str | None = None
    min_crypto_price: float | None = None
    # Optional backtest-only overrides for universe sizing/liquidity gates.
    max_equity_positions: int | None = None
    max_crypto_positions: int | None = None
    min_avg_crypto_dollar_volume_20d: float | None = None


class BacktestStopped(Exception):
    pass


@dataclass(frozen=True)
class BacktestResult:
    params: dict
    equity_curve: list[dict]
    metrics: dict
    trades: list[dict]               # realized (sells/trims/exits)
    events: list[dict]               # buys + sells + other lifecycle events
    open_positions: list[dict]
    realized_pnl_by_symbol: dict
    excluded_symbols: list[str]


def _date_range(start: str, end: str) -> pd.DatetimeIndex:
    # Use tz-naive dates consistently
    s = pd.to_datetime(start).tz_localize(None)
    e = pd.to_datetime(end).tz_localize(None)
    idx = pd.date_range(s, e, freq="D")
    return idx


def _rebalance_days(days: pd.DatetimeIndex, mode: str, weekly_day: str = "MON") -> set[pd.Timestamp]:
    if mode in ("daily", "hourly"):
        return set(days)
    day_map = {"MON":0, "TUE":1, "WED":2, "THU":3, "FRI":4, "SAT":5, "SUN":6}
    wd = day_map.get(str(weekly_day).upper(), 0)
    return set([d for d in days if d.weekday() == wd])


def run_backtest(
    *,
    stock_bars: dict[str, pd.DataFrame],
    crypto_bars: dict[str, pd.DataFrame],
    stock_universe: list[str],
    crypto_universe: list[str],
    cfg,
    params: BacktestParams,
    progress_cb=None,
    debug_cb=None,
    stop_cb=None,
    debug_verbose: bool = False,
    intraday_price_cb: Callable[[str, pd.Timestamp], float | None] | None = None,
    intraday_limit_touch_cb: Callable[[str, pd.Timestamp, str, float], bool] | None = None,
    risk_intraday_price_cb: Callable[[str, pd.Timestamp], float | None] | None = None,
) -> BacktestResult:
    """Simple long-only backtest using daily closes.

    Assumptions:
    - Rebalance at close on rebalance day.
    - Trades executed at close +/- slippage.
    - Equal weight within sleeve; cash otherwise.
    """

    start = pd.to_datetime(params.start)
    end = pd.to_datetime(params.end)
    days = _date_range(params.start, params.end)
    rebal_days = _rebalance_days(days, params.rebalance, params.rebalance_day)
    eq_rebal_days = _rebalance_days(days, params.rebalance_frequency_equities or params.rebalance, params.rebalance_day_equities or params.rebalance_day)
    cr_rebal_days = _rebalance_days(days, params.rebalance_frequency_crypto or params.rebalance, params.rebalance_day_crypto or params.rebalance_day)
    eq_risk_freq = str(params.risk_check_frequency_equities or "daily")
    cr_risk_freq = str(params.risk_check_frequency_crypto or "daily")
    eq_risk_days = _rebalance_days(days, eq_risk_freq, params.risk_check_day_equities or params.rebalance_day)
    cr_risk_days = _rebalance_days(days, cr_risk_freq, params.risk_check_day_crypto or params.rebalance_day)
    eq_risk_minute = int(params.risk_check_minute_of_hour_equities if params.risk_check_minute_of_hour_equities is not None else 5)
    cr_risk_minute = int(params.risk_check_minute_of_hour_crypto if params.risk_check_minute_of_hour_crypto is not None else 5)
    eq_hourly_checks = set(params.risk_check_hourly_checks_equities or ["stop_loss", "dd_stop"])
    cr_hourly_checks = set(params.risk_check_hourly_checks_crypto or ["stop_loss", "dd_stop"])

    use_hourly_progress = bool(eq_risk_freq == "hourly" or cr_risk_freq == "hourly")
    hourly_minutes_set: set[int] = set()
    if eq_risk_freq == "hourly":
        hourly_minutes_set.add(int(eq_risk_minute))
    if cr_risk_freq == "hourly":
        hourly_minutes_set.add(int(cr_risk_minute))
    hourly_slots_per_day = max(1, 24 * max(1, len(hourly_minutes_set))) if use_hourly_progress else 1
    progress_total_steps = int(len(days) * hourly_slots_per_day) if use_hourly_progress else int(len(days))
    progress_steps_done = 0

    # Equity trading sessions from available stock bars (captures weekends/holidays closure).
    eq_trade_days: set[pd.Timestamp] = set()
    try:
        for _sym, _df in (stock_bars or {}).items():
            if _df is None or len(_df) == 0:
                continue
            _idx = pd.to_datetime(_df.index)
            try:
                if getattr(_idx, "tz", None) is not None:
                    _idx = _idx.tz_convert(None)
            except Exception:
                pass
            try:
                _idx = _idx.tz_localize(None)
            except Exception:
                pass
            for d in _idx:
                eq_trade_days.add(pd.Timestamp(d).normalize())
        if not eq_trade_days:
            # fallback at least to weekdays if bars unavailable
            eq_trade_days = set([d for d in days if d.weekday() < 5])
    except Exception:
        eq_trade_days = set([d for d in days if d.weekday() < 5])

    eq_rebal_days = set([d for d in eq_rebal_days if d in eq_trade_days])
    eq_risk_days = set([d for d in eq_risk_days if d in eq_trade_days])

    equity = float(params.initial_equity)
    cash = equity
    positions_qty: dict[str, float] = {}
    positions_avg_cost: dict[str, float] = {}
    positions_entry_date: dict[str, str] = {}
    positions_peak_mark: dict[str, float] = {}
    pending_limits: list[dict] = []  # simulated working limit orders
    trades: list[dict] = []
    events: list[dict] = []
    realized_pnl_by_symbol: dict[str, float] = {}
    excluded: set[str] = set()

    peak_equity = equity
    stopped_until_next_rebalance = False
    dd_stop_events = 0
    max_observed_dd = 0.0
    dd_stop_trigger_day: pd.Timestamp | None = None

    hourly_debug = {
        "slots_considered": 0,
        "symbol_checks": 0,
        "price_points_found": 0,
        "strategy_exit_selected": 0,
        "strategy_exit_no_rule": 0,
        "strategy_exit_evaluated": 0,
        "strategy_exit_triggered": 0,
        "strategy_exit_cache_hits": 0,
        "price_cache_hits": 0,
        "price_cache_misses": 0,
    }
    strategy_eval_cache: dict[tuple[str, str], bool] = {}
    hourly_price_cache: dict[tuple[str, str], float | None] = {}

    # Resolve per-asset strategy ids (with legacy fallback)
    eq_strategy_id = params.strategy_id_equities or params.strategy_id
    cr_strategy_id = params.strategy_id_crypto or params.strategy_id
    eq_strat = get_strategy(eq_strategy_id)
    cr_strat = get_strategy(cr_strategy_id)

    # Optional per-asset exit rule specs from rule-based user strategies.
    def _load_exit_rule(strategy_id: str, enabled: bool):
        if not enabled:
            return None
        try:
            s = get_strategy(strategy_id)
            if hasattr(s, "spec"):
                return (getattr(s, "spec", {}) or {}).get("exit")
        except Exception:
            return None
        return None

    eq_exit_rule = _load_exit_rule(params.exit_strategy_id_equities or eq_strategy_id, bool(params.exit_enabled_equities))
    cr_exit_rule = _load_exit_rule(params.exit_strategy_id_crypto or cr_strategy_id, bool(params.exit_enabled_crypto))

    # Precompute normalized OHLCV frames and close/open series once.
    # This avoids repeated copy/sort/tz work during each rebalance slice.
    bars_all_src = {**stock_bars, **crypto_bars}
    bars_all: dict[str, pd.DataFrame] = {}
    closes: dict[str, pd.Series] = {}
    opens: dict[str, pd.Series] = {}
    highs: dict[str, pd.Series] = {}
    lows: dict[str, pd.Series] = {}
    def _naive_utc_index(idx: pd.Index) -> pd.DatetimeIndex:
        di = pd.to_datetime(idx)
        # If tz-aware, drop tz to compare with naive backtest dates
        try:
            if getattr(di, "tz", None) is not None:
                di = di.tz_convert(None)
        except Exception:
            pass
        # Sometimes it's tz-aware per-element
        if hasattr(di, "tz_localize"):
            try:
                di = di.tz_localize(None)
            except Exception:
                pass
        return di

    for sym, df in bars_all_src.items():
        if df is not None and len(df) and "close" in df.columns:
            dfx = df.copy()
            dfx.index = _naive_utc_index(dfx.index)
            dfx = dfx.sort_index()
            bars_all[sym] = dfx
            closes[sym] = dfx["close"].astype(float)
            if "open" in dfx.columns:
                opens[sym] = dfx["open"].astype(float)
            else:
                opens[sym] = pd.Series(dtype=float)
            highs[sym] = dfx["high"].astype(float) if "high" in dfx.columns else pd.Series(dtype=float)
            lows[sym] = dfx["low"].astype(float) if "low" in dfx.columns else pd.Series(dtype=float)
        else:
            closes[sym] = pd.Series(dtype=float)
            opens[sym] = pd.Series(dtype=float)
            highs[sym] = pd.Series(dtype=float)
            lows[sym] = pd.Series(dtype=float)

    px_cache: dict[tuple[str, str], float | None] = {}
    px_open_cache: dict[tuple[str, str], float | None] = {}
    px_col_cache: dict[tuple[str, str, str], float | None] = {}

    def px(sym: str, day: pd.Timestamp) -> float | None:
        dkey = pd.Timestamp(day).strftime("%Y-%m-%d")
        key = (str(sym), dkey)
        if key in px_cache:
            return px_cache[key]
        s = closes.get(sym)
        if s is None or len(s) == 0:
            px_cache[key] = None
            return None
        # use last valid close on/before day (skip NaN/zero/invalid tails)
        sub = s.loc[:day]
        if len(sub) == 0:
            px_cache[key] = None
            return None
        vals = sub.dropna().astype(float)
        if len(vals) == 0:
            px_cache[key] = None
            return None
        for v in reversed(vals.values.tolist()):
            if np.isfinite(v) and float(v) > 0:
                out = float(v)
                px_cache[key] = out
                return out
        px_cache[key] = None
        return None

    def px_open(sym: str, day: pd.Timestamp) -> float | None:
        dkey = pd.Timestamp(day).strftime("%Y-%m-%d")
        key = (str(sym), dkey)
        if key in px_open_cache:
            return px_open_cache[key]
        s = opens.get(sym)
        if s is None or len(s) == 0:
            px_open_cache[key] = None
            return None
        sub = s.loc[:day]
        if len(sub) == 0:
            px_open_cache[key] = None
            return None
        vals = sub.dropna().astype(float)
        if len(vals) == 0:
            px_open_cache[key] = None
            return None
        for v in reversed(vals.values.tolist()):
            if np.isfinite(v) and float(v) > 0:
                out = float(v)
                px_open_cache[key] = out
                return out
        px_open_cache[key] = None
        return None

    def px_col(sym: str, day: pd.Timestamp, col: str) -> float | None:
        dkey = pd.Timestamp(day).strftime("%Y-%m-%d")
        key = (str(sym), dkey, str(col))
        if key in px_col_cache:
            return px_col_cache[key]

        s: pd.Series
        if col == "high":
            s = highs.get(sym, pd.Series(dtype=float))
        elif col == "low":
            s = lows.get(sym, pd.Series(dtype=float))
        elif col == "open":
            s = opens.get(sym, pd.Series(dtype=float))
        elif col == "close":
            s = closes.get(sym, pd.Series(dtype=float))
        else:
            df = bars_all.get(sym)
            if df is None or len(df) == 0 or col not in df.columns:
                px_col_cache[key] = None
                return None
            try:
                s = df[col].astype(float)
            except Exception:
                px_col_cache[key] = None
                return None

        if s is None or len(s) == 0:
            px_col_cache[key] = None
            return None
        try:
            sub = s.loc[:day]
            if len(sub) == 0:
                px_col_cache[key] = None
                return None
            v = float(sub.iloc[-1])
            out = v if np.isfinite(v) and v > 0 else None
            px_col_cache[key] = out
            return out
        except Exception:
            px_col_cache[key] = None
            return None

    def exec_px(sym: str, day: pd.Timestamp) -> float | None:
        # Intraday execution pricing (rebalance only)
        if params.execution_time_mode == "intraday" and intraday_price_cb is not None:
            v = intraday_price_cb(sym, day)
            if v is not None:
                return v

        # Daily execution pricing
        if params.execution_time == "open":
            v = px_open(sym, day)
            if v is not None:
                return v
        return px(sym, day)

    def risk_px(sym: str, day: pd.Timestamp) -> float | None:
        # Intraday execution pricing for risk exits
        if params.execution_time_mode == "intraday" and risk_intraday_price_cb is not None:
            v = risk_intraday_price_cb(sym, day)
            if v is not None:
                return v
        # default to close for daily risk checks
        return px(sym, day)

    def risk_px_at_ts(sym: str, ts_local: pd.Timestamp) -> float | None:
        key = (str(sym), pd.Timestamp(ts_local).strftime("%Y-%m-%d %H:%M"))
        if key in hourly_price_cache:
            hourly_debug["price_cache_hits"] += 1
            return hourly_price_cache[key]
        hourly_debug["price_cache_misses"] += 1
        if params.execution_time_mode == "intraday" and risk_intraday_price_cb is not None:
            v = risk_intraday_price_cb(sym, ts_local)
            if v is not None:
                hourly_price_cache[key] = v
                return v
        v = px(sym, pd.Timestamp(ts_local).normalize())
        hourly_price_cache[key] = v
        return v

    def _clamp_fill_px(sym: str, day: pd.Timestamp, px_in: float, side: str) -> float:
        """Clamp simulated fills to daily bar envelope for realism.

        - buy fills cannot exceed day high
        - sell fills cannot go below day low
        """
        try:
            p = float(px_in)
            lo = px_low(sym, day)
            hi = px_high(sym, day)
            if (lo is None) or (hi is None):
                return p
            lo = float(lo); hi = float(hi)
            if side == "buy":
                return float(min(max(p, lo), hi))
            if side == "sell":
                return float(max(min(p, hi), lo))
            return float(min(max(p, lo), hi))
        except Exception:
            return float(px_in)

    def _trail_cfg(sym: str) -> tuple[bool, float, float | None]:
        is_crypto = "/" in sym
        if is_crypto:
            start = float(params.trailing_stop_crypto_start_gain_pct if params.trailing_stop_crypto_start_gain_pct is not None else 0.05)
            return bool(params.trailing_stop_crypto_enabled), start, (float(params.trailing_stop_crypto_pct) if params.trailing_stop_crypto_pct is not None else None)
        start = float(params.trailing_stop_stocks_start_gain_pct if params.trailing_stop_stocks_start_gain_pct is not None else 0.05)
        return bool(params.trailing_stop_stocks_enabled), start, (float(params.trailing_stop_stocks_pct) if params.trailing_stop_stocks_pct is not None else None)

    def _trail_peak(sym: str, day: pd.Timestamp, mark_px: float) -> float:
        if str(params.trailing_stop_anchor) == "highest_close_since_entry":
            s = closes.get(sym)
            if s is not None and len(s) > 0:
                entry_s = positions_entry_date.get(sym)
                start_d = pd.to_datetime(entry_s) if entry_s else day
                sub = s.loc[start_d:day].dropna().astype(float)
                if len(sub) > 0:
                    return float(np.max(sub.values))
        prev = float(positions_peak_mark.get(sym, 0.0) or 0.0)
        return max(prev, float(mark_px))

    def _equity_slot_open(ts_local: pd.Timestamp) -> bool:
        # Equities: prune obvious closed windows in hourly mode.
        # Keep pre/post-market (04:00-20:00 in exchange-local time), weekdays only.
        wd = int(pd.Timestamp(ts_local).weekday())
        if wd >= 5:
            return False
        h = int(pd.Timestamp(ts_local).hour)
        return 4 <= h <= 20

    def closes_until_day(sym: str, day: pd.Timestamp) -> pd.Series:
        s = closes.get(sym)
        if s is None or len(s) == 0:
            return pd.Series(dtype=float)
        try:
            return s.loc[:day].dropna().astype(float)
        except Exception:
            return pd.Series(dtype=float)

    def _use_limit_for(sym: str) -> bool:
        is_crypto = "/" in sym
        ot = (params.order_type_crypto if is_crypto else params.order_type_equities)
        # Global limit toggle intentionally ignored; per-asset order_type is authoritative.
        if ot in ("market", "limit"):
            return ot == "limit"
        return False

    def _limit_off_bps_for(sym: str) -> float:
        is_crypto = "/" in sym
        v = params.limit_offset_bps_crypto if is_crypto else params.limit_offset_bps_equities
        return float(v if v is not None else 10.0)

    def _fallback_for(sym: str) -> bool:
        is_crypto = "/" in sym
        v = params.limit_fallback_to_market_open_crypto if is_crypto else params.limit_fallback_to_market_open_equities
        if v is None:
            return bool(params.limit_fallback_to_market_open)
        return bool(v)

    def portfolio_value(day: pd.Timestamp) -> float:
        total = cash
        for sym, q in positions_qty.items():
            p = px(sym, day)
            if p is None:
                # fallback valuation if mark is temporarily missing
                p = float(positions_avg_cost.get(sym, 0.0) or 0.0)
            if p <= 0:
                continue
            total += q * p
        return float(total)

    curve: list[dict] = []

    def _record_trade(t: dict) -> None:
        """Record a realized trade (typically sells/trims/exits)."""
        trades.append(t)
        sym = str(t.get("symbol") or "").strip()
        pnl = float(t.get("pnl") or 0.0)
        if sym:
            realized_pnl_by_symbol[sym] = float(realized_pnl_by_symbol.get(sym, 0.0) + pnl)
        if debug_verbose:
            try:
                _dbg(
                    "trade_recorded",
                    symbol=sym,
                    reason=t.get("reason"),
                    qty=t.get("qty"),
                    entry_price=t.get("entry_price"),
                    exit_price=t.get("exit_price"),
                    pnl=t.get("pnl"),
                    pnl_pct=t.get("pnl_pct"),
                    entry_date=t.get("entry_date"),
                    exit_date=t.get("exit_date"),
                )
            except Exception:
                pass

    def _event(e: dict) -> None:
        # Attach forensic bar + timing snapshot so fills can be audited later.
        try:
            sym = str(e.get("symbol") or "").strip()
            day_s = str(e.get("date") or "").strip()
            is_crypto = "/" in sym
            if sym and day_s:
                # Fill timing metadata (inferred from execution/risk schedules).
                reason = str(e.get("reason") or "")
                risk_reasons = {
                    "strategy_exit_rule", "per_asset_stop_loss", "portfolio_dd_stop",
                    "trailing_stop_stocks", "trailing_stop_crypto", "symbol_pnl_floor_exclude",
                }
                if reason in risk_reasons:
                    t_local = (params.risk_check_time_local_crypto if is_crypto else params.risk_check_time_local_equities) or params.risk_check_time_local
                    e.setdefault("fill_source", "risk_check")
                else:
                    if params.execution_time_mode == "intraday":
                        t_local = (params.execution_time_local_crypto if is_crypto else params.execution_time_local_equities) or params.execution_time_local
                        e.setdefault("fill_source", "intraday_exec")
                    else:
                        t_local = "09:30" if str(params.execution_time) == "open" else "16:00"
                        e.setdefault("fill_source", f"daily_{params.execution_time}")
                e.setdefault("fill_ts_local", f"{day_s} {t_local}")

                # Attach bar envelope used for forensic checks.
                df0 = bars_all.get(sym)
                if df0 is not None and len(df0) > 0:
                    dkey = pd.Timestamp(day_s)
                    row = None
                    if dkey in df0.index:
                        row = df0.loc[dkey]
                        if isinstance(row, pd.DataFrame):
                            row = row.iloc[-1]
                    else:
                        day_df = df0.loc[df0.index.normalize() == dkey]
                        if len(day_df):
                            row = day_df.iloc[-1]
                    if row is not None:
                        e.setdefault("bar_open", float(row.get("open")) if "open" in row else None)
                        e.setdefault("bar_high", float(row.get("high")) if "high" in row else None)
                        e.setdefault("bar_low", float(row.get("low")) if "low" in row else None)
                        e.setdefault("bar_close", float(row.get("close")) if "close" in row else None)
                        e.setdefault("bar_volume", float(row.get("volume")) if "volume" in row else None)
                        e.setdefault("bar_source", "daily_backtest_bars")
                        e.setdefault("bar_date", day_s)
        except Exception:
            pass

        events.append(e)
        if debug_verbose:
            try:
                _dbg(
                    "event",
                    event_type=e.get("type"),
                    symbol=e.get("symbol"),
                    reason=e.get("reason"),
                    date=e.get("date"),
                    qty=e.get("qty"),
                    price=e.get("price"),
                    notional=e.get("notional"),
                    side=e.get("side"),
                    limit_px=e.get("limit_px"),
                )
            except Exception:
                pass

    def _liquidate_excluded(day: pd.Timestamp, *, reason: str) -> None:
        """If a symbol is excluded, immediately sell any remaining position."""
        nonlocal cash
        if not excluded:
            return
        for sym in list(positions_qty.keys()):
            if sym not in excluded:
                continue
            p0 = px(sym, day)
            if p0 is None:
                continue
            base_px = risk_px(sym, day) or p0
            sell_px = base_px * (1 - params.slippage_bps / 10000.0)
            sell_px = _clamp_fill_px(sym, day, sell_px, "sell")
            q = positions_qty.get(sym, 0.0)
            if (not bool(params.allow_same_day_roundtrip)) and str(positions_entry_date.get(sym) or "") == day.strftime("%Y-%m-%d"):
                continue
            cash += q * sell_px
            avg_cost = positions_avg_cost.get(sym, p0)
            entry_date = positions_entry_date.get(sym)
            pnl = (sell_px - avg_cost) * q
            rec = {
                "symbol": sym,
                "entry_date": entry_date,
                "exit_date": day.strftime("%Y-%m-%d"),
                "qty": q,
                "entry_price": avg_cost,
                "exit_price": sell_px,
                "pnl": pnl,
                "pnl_pct": (sell_px / avg_cost - 1.0) if avg_cost else None,
                "reason": reason,
            }
            _record_trade(rec)
            _event({
                "type": "sell",
                "symbol": sym,
                "date": day.strftime("%Y-%m-%d"),
                "qty": float(q),
                "price": float(sell_px),
                "notional": float(q * sell_px),
                "new_qty": 0.0,
                "reason": rec["reason"],
                "pnl": float(pnl),
            })
            positions_qty.pop(sym, None)
            positions_avg_cost.pop(sym, None)
            positions_entry_date.pop(sym, None)

    def _apply_pending_fill(po: dict, *, day: pd.Timestamp, fill_px: float, reason: str) -> None:
        nonlocal cash
        sym = str(po.get("symbol"))
        side = str(po.get("side"))
        if side == "buy":
            notional = float(po.get("notional") or 0.0)
            q_add = (notional / fill_px) if fill_px > 0 else 0.0
            if q_add <= 0:
                return
            # cap by cash
            max_q = cash / fill_px if fill_px > 0 else 0.0
            q_add = min(q_add, max_q)
            cost = q_add * fill_px
            if q_add <= 0:
                return
            cash -= cost
            prevQ = positions_qty.get(sym, 0.0)
            newQ = prevQ + q_add
            prevCost = positions_avg_cost.get(sym, fill_px)
            if prevQ <= 0:
                positions_entry_date[sym] = day.strftime("%Y-%m-%d")
                positions_avg_cost[sym] = fill_px
                positions_peak_mark[sym] = float(fill_px)
            else:
                positions_avg_cost[sym] = (prevQ * prevCost + q_add * fill_px) / (prevQ + q_add)
            positions_qty[sym] = newQ
            _event({"type":"buy", "symbol":sym, "date":day.strftime("%Y-%m-%d"), "qty":float(q_add), "price":float(fill_px), "expected_price": float(po.get("limit_px")) if po.get("limit_px") is not None else None, "notional":float(cost), "new_qty":float(newQ), "reason":reason})
        else:
            q_sub = min(float(po.get("qty") or 0.0), positions_qty.get(sym, 0.0))
            if q_sub <= 0:
                return
            proceeds = q_sub * fill_px
            cash += proceeds
            p0 = px(sym, day) or fill_px
            avg_cost = positions_avg_cost.get(sym, p0)
            entry_date = positions_entry_date.get(sym)
            pnl = (fill_px - avg_cost) * q_sub
            rec = {
                "symbol": sym,
                "entry_date": entry_date,
                "exit_date": day.strftime("%Y-%m-%d"),
                "qty": q_sub,
                "entry_price": avg_cost,
                "exit_price": fill_px,
                "pnl": pnl,
                "pnl_pct": (fill_px / avg_cost - 1.0) if avg_cost else None,
                "reason": reason,
            }
            _record_trade(rec)
            newQ = max(0.0, positions_qty.get(sym, 0.0) - q_sub)
            if newQ <= 1e-12:
                positions_qty.pop(sym, None)
                positions_avg_cost.pop(sym, None)
                positions_entry_date.pop(sym, None)
                newQ = 0.0
            else:
                positions_qty[sym] = newQ
            _event({"type":"sell", "symbol":sym, "date":day.strftime("%Y-%m-%d"), "qty":float(q_sub), "price":float(fill_px), "expected_price": float(po.get("limit_px")) if po.get("limit_px") is not None else None, "notional":float(proceeds), "new_qty":float(newQ), "reason":reason, "pnl":float(pnl)})

    def _limit_touched(sym: str, day: pd.Timestamp, side: str, limit_px: float) -> bool:
        if intraday_limit_touch_cb is not None:
            try:
                return bool(intraday_limit_touch_cb(sym, day, side, limit_px))
            except Exception:
                pass
        lo = px_col(sym, day, "low")
        hi = px_col(sym, day, "high")
        if side == "buy":
            return (lo is not None) and (lo <= limit_px)
        return (hi is not None) and (hi >= limit_px)

    def _process_pending_limits(day: pd.Timestamp) -> None:
        if not pending_limits:
            return
        for po in list(pending_limits):
            placed_day = pd.to_datetime(po.get("placed_day"))
            sym = str(po.get("symbol"))
            side = str(po.get("side"))
            limit_px = float(po.get("limit_px") or 0.0)

            # cancel any still-open order at next rebalance boundary (per asset class)
            is_crypto = "/" in sym
            on_rebal_boundary = (day in (cr_rebal_days if is_crypto else eq_rebal_days))
            if on_rebal_boundary and day > placed_day:
                _event({"type":"cancel", "symbol":sym, "date":day.strftime("%Y-%m-%d"), "reason":"limit_cancel_next_rebalance", "side":side})
                pending_limits.remove(po)
                continue

            if day < placed_day:
                continue

            if _limit_touched(sym, day, side, limit_px):
                _apply_pending_fill(po, day=day, fill_px=limit_px, reason="limit_fill")
                pending_limits.remove(po)
                continue

            if bool(po.get("fallback", False)):
                op = px_open(sym, day) or px(sym, day)
                if op is not None:
                    mkt_px = op * (1 + params.slippage_bps / 10000.0) if side == "buy" else op * (1 - params.slippage_bps / 10000.0)
                    _apply_pending_fill(po, day=day, fill_px=float(mkt_px), reason="limit_fallback_market_open")
                    pending_limits.remove(po)

    def _sma(series: pd.Series, n: int, day: pd.Timestamp) -> float | None:
        if series is None or len(series) == 0:
            return None
        sub = series.loc[:day]
        if len(sub) < n:
            return None
        v = float(sub.tail(n).mean())
        return v if np.isfinite(v) else None

    def _regime(day: pd.Timestamp) -> dict:
        """Simple regime flags used for visualization.

        equity_risk_on: SPY close > SMA200
        crypto_risk_on: BTC/USD close > SMA200
        """
        out = {"equity_risk_on": None, "crypto_risk_on": None}
        spy = closes.get("SPY")
        btc = closes.get("BTC/USD")
        spy_px = px("SPY", day) if spy is not None else None
        btc_px = px("BTC/USD", day) if btc is not None else None
        spy_ma = _sma(spy, 200, day) if spy is not None else None
        btc_ma = _sma(btc, 200, day) if btc is not None else None
        if spy_px is not None and spy_ma is not None:
            out["equity_risk_on"] = bool(spy_px > spy_ma)
        if btc_px is not None and btc_ma is not None:
            out["crypto_risk_on"] = bool(btc_px > btc_ma)
        return out

    def _dbg(msg: str, **fields):
        if debug_cb is None:
            return
        try:
            debug_cb(msg, **fields)
        except Exception:
            pass

    def _collect_indicator_specs(rule_obj, out: list[dict]) -> None:
        if isinstance(rule_obj, dict):
            if "kind" in rule_obj:
                out.append(rule_obj)
            for v in rule_obj.values():
                _collect_indicator_specs(v, out)
        elif isinstance(rule_obj, list):
            for it in rule_obj:
                _collect_indicator_specs(it, out)

    def _eval_ctx_until_day(sym: str, day: pd.Timestamp, ann_factor: float) -> EvalContext | None:
        df = bars_all.get(sym)
        if df is None or len(df) == 0:
            return None
        try:
            dfx = df.loc[:day]
        except Exception:
            return None
        if dfx is None or len(dfx) == 0 or "close" not in dfx.columns:
            return None
        cls = dfx["close"].dropna().astype(float)
        if len(cls) == 0:
            return None
        highs = dfx["high"].dropna().astype(float) if "high" in dfx.columns else None
        lows = dfx["low"].dropna().astype(float) if "low" in dfx.columns else None
        opens = dfx["open"].dropna().astype(float) if "open" in dfx.columns else None
        vols = dfx["volume"].dropna().astype(float) if "volume" in dfx.columns else None
        return EvalContext(closes=cls, ann_factor=ann_factor, highs=highs, lows=lows, opens=opens, volumes=vols)

    def _indicator_snapshot(rule_obj, ctx: EvalContext) -> dict[str, float | None]:
        specs: list[dict] = []
        _collect_indicator_specs(rule_obj, specs)
        seen: set[str] = set()
        snap: dict[str, float | None] = {}
        for sp in specs:
            kind = str(sp.get("kind") or "").lower()
            if not kind:
                continue
            parts = [kind]
            for k in sorted([k for k in sp.keys() if k != "kind"]):
                parts.append(f"{k}={sp.get(k)}")
            key = "|".join(parts)
            if key in seen:
                continue
            seen.add(key)
            try:
                snap[key] = eval_indicator(ctx, sp)
            except Exception:
                snap[key] = None
        return snap

    for i, day in enumerate(days):
        if stop_cb is not None:
            try:
                if bool(stop_cb()):
                    raise BacktestStopped("stopped_by_user")
            except BacktestStopped:
                raise
            except Exception:
                pass
        day_s = day.strftime("%Y-%m-%d")
        if debug_verbose:
            _dbg("day_start", day=day_s, idx=i + 1, total=len(days), positions=len(positions_qty), pending_limits=len(pending_limits))

        def _can_sell_today(sym: str) -> bool:
            if bool(params.allow_same_day_roundtrip):
                return True
            return str(positions_entry_date.get(sym) or "") != day_s

        # Process pending simulated limit orders first
        if params.order_type_equities == "limit" or params.order_type_crypto == "limit":
            _process_pending_limits(day)

        # Mark-to-market
        equity = portfolio_value(day)
        peak_equity = max(peak_equity, equity)

        # Evaluate symbol P/L floor (optionally include unrealized) and exclude+liquidate
        floor = params.symbol_pnl_floor_pct
        if floor is not None:
            fl = float(floor)
            # We can only compute a meaningful percent while a position is held
            for sym, q in list(positions_qty.items()):
                p0 = px(sym, day)
                if p0 is None:
                    continue
                avg_cost = positions_avg_cost.get(sym)
                if avg_cost is None or avg_cost <= 0:
                    continue
                basis = float(avg_cost * q)
                if basis == 0:
                    continue
                unreal = float((p0 - avg_cost) * q)
                realized = float(realized_pnl_by_symbol.get(sym, 0.0))
                tot = realized + (unreal if params.symbol_pnl_floor_include_unrealized else 0.0)
                pct = float(tot / basis)
                if pct <= fl:
                    excluded.add(sym)

        # If symbol is excluded already, optionally liquidate at start of day
        if params.symbol_pnl_floor_liquidate:
            _liquidate_excluded(day, reason="symbol_pnl_floor_exclude")

        # Hourly risk checks (intraday): run selected checks at configured minute; skip if no matching bar.
        if params.execution_time_mode == "intraday" and (eq_risk_freq == "hourly" or cr_risk_freq == "hourly"):
            hourly_minutes = sorted(hourly_minutes_set)

            for hr in range(24):
                for mm in hourly_minutes:
                    ts_local = pd.Timestamp(day).replace(hour=hr, minute=int(mm), second=0, microsecond=0)
                    hourly_debug["slots_considered"] += 1
                    if debug_verbose:
                        _dbg("hourly_slot", day=day_s, ts=str(ts_local), positions=len(positions_qty), minute=int(mm))

                    if not positions_qty:
                        continue

                    eq_slot_open = _equity_slot_open(ts_local)

                    for sym in list(positions_qty.keys()):
                        is_crypto = "/" in sym
                        if is_crypto:
                            if cr_risk_freq != "hourly" or ts_local.minute != cr_risk_minute:
                                continue
                            checks = cr_hourly_checks
                        else:
                            if eq_risk_freq != "hourly" or ts_local.minute != eq_risk_minute or (not eq_slot_open):
                                continue
                            checks = eq_hourly_checks

                        hourly_debug["symbol_checks"] += 1
                        p_intraday = risk_px_at_ts(sym, ts_local)
                        if debug_verbose:
                            _dbg("hourly_symbol", day=day_s, ts=str(ts_local), symbol=sym, has_price=(p_intraday is not None), checks=','.join(sorted(list(checks))))
                        if p_intraday is None:
                            continue
                        hourly_debug["price_points_found"] += 1

                        trail_enabled, trail_start, trail_pct = _trail_cfg(sym)
                        if trail_enabled and trail_pct is not None and trail_pct > 0 and _can_sell_today(sym):
                            peak = _trail_peak(sym, day, float(p_intraday))
                            positions_peak_mark[sym] = peak
                            avg_cost = positions_avg_cost.get(sym, float(p_intraday))
                            armed = (avg_cost is not None and avg_cost > 0 and peak >= (float(avg_cost) * (1 + float(trail_start))))
                            trail_level = peak * (1 - trail_pct)
                            if debug_verbose:
                                _dbg("hourly_trailing_eval", day=day_s, ts=str(ts_local), symbol=sym, price=float(p_intraday), avg_cost=avg_cost, peak=peak, armed=bool(armed), trail_start=float(trail_start), trail_pct=float(trail_pct), trail_level=float(trail_level))
                            if armed and float(p_intraday) <= trail_level:
                                q = positions_qty.get(sym, 0.0)
                                sell_px = float(p_intraday) * (1 - params.slippage_bps / 10000.0)
                                sell_px = _clamp_fill_px(sym, day, sell_px, "sell")
                                cash += q * sell_px
                                avg_cost = positions_avg_cost.get(sym, float(p_intraday))
                                entry_date = positions_entry_date.get(sym)
                                pnl = (sell_px - avg_cost) * q
                                rec = {"symbol": sym, "entry_date": entry_date, "exit_date": day.strftime("%Y-%m-%d"), "qty": q, "entry_price": avg_cost, "exit_price": sell_px, "pnl": pnl, "pnl_pct": (sell_px / avg_cost - 1.0) if avg_cost else None, "reason": ("trailing_stop_crypto" if is_crypto else "trailing_stop_stocks")}
                                _record_trade(rec)
                                _event({"type": "sell", "symbol": sym, "date": day.strftime("%Y-%m-%d"), "qty": float(q), "price": float(sell_px), "notional": float(q * sell_px), "new_qty": 0.0, "reason": rec["reason"], "pnl": float(pnl)})
                                positions_qty.pop(sym, None); positions_avg_cost.pop(sym, None); positions_entry_date.pop(sym, None); positions_peak_mark.pop(sym, None)
                                continue

                        if "strategy_exit" in checks:
                            hourly_debug["strategy_exit_selected"] += 1
                            ex_rule = cr_exit_rule if is_crypto else eq_exit_rule
                            if ex_rule is None:
                                hourly_debug["strategy_exit_no_rule"] += 1
                            if ex_rule is not None:
                                day_key = pd.Timestamp(day).strftime("%Y-%m-%d")
                                cache_key = (sym, day_key)
                                if cache_key in strategy_eval_cache:
                                    should_exit = bool(strategy_eval_cache.get(cache_key, False))
                                    hourly_debug["strategy_exit_cache_hits"] += 1
                                else:
                                    cls = closes_until_day(sym, day)
                                    should_exit = False
                                    ann_factor = 365.0 if is_crypto else 252.0
                                    ctx = _eval_ctx_until_day(sym, day, ann_factor)
                                    if ctx is not None and len(ctx.closes) >= 5:
                                        hourly_debug["strategy_exit_evaluated"] += 1
                                        try:
                                            should_exit = bool(eval_rule(ctx, ex_rule))
                                        except Exception:
                                            should_exit = False
                                    strategy_eval_cache[cache_key] = bool(should_exit)
                                if debug_verbose:
                                    _ctx = _eval_ctx_until_day(sym, day, ann_factor if 'ann_factor' in locals() else (365.0 if is_crypto else 252.0))
                                    snap = _indicator_snapshot(ex_rule, _ctx) if (_ctx is not None and ex_rule is not None and len(_ctx.closes) > 0) else {}
                                    _dbg("hourly_strategy_exit_eval", day=day_s, ts=str(ts_local), symbol=sym, should_exit=bool(should_exit), has_rule=bool(ex_rule), closes_len=(len(_ctx.closes) if _ctx is not None else None), indicators=snap)
                                if should_exit and _can_sell_today(sym):
                                    hourly_debug["strategy_exit_triggered"] += 1
                                    q = positions_qty.get(sym, 0.0)
                                    sell_px = p_intraday * (1 - params.slippage_bps / 10000.0)
                                    sell_px = _clamp_fill_px(sym, day, sell_px, "sell")
                                    cash += q * sell_px
                                    avg_cost = positions_avg_cost.get(sym, p_intraday)
                                    entry_date = positions_entry_date.get(sym)
                                    pnl = (sell_px - avg_cost) * q
                                    rec = {"symbol": sym, "entry_date": entry_date, "exit_date": day.strftime("%Y-%m-%d"), "qty": q, "entry_price": avg_cost, "exit_price": sell_px, "pnl": pnl, "pnl_pct": (sell_px / avg_cost - 1.0) if avg_cost else None, "reason": "strategy_exit_rule"}
                                    _record_trade(rec)
                                    _event({"type": "sell", "symbol": sym, "date": day.strftime("%Y-%m-%d"), "qty": float(q), "price": float(sell_px), "notional": float(q * sell_px), "new_qty": 0.0, "reason": rec["reason"], "pnl": float(pnl)})
                                    positions_qty.pop(sym, None); positions_avg_cost.pop(sym, None); positions_entry_date.pop(sym, None)
                                    continue

                        if "stop_loss" in checks and params.per_asset_stop_loss_pct is not None and params.per_asset_stop_loss_pct > 0:
                            avg_cost = positions_avg_cost.get(sym)
                            if avg_cost is not None and avg_cost > 0 and (p_intraday / avg_cost - 1.0) <= -float(params.per_asset_stop_loss_pct) and _can_sell_today(sym):
                                q = positions_qty.get(sym, 0.0)
                                sell_px = p_intraday * (1 - params.slippage_bps / 10000.0)
                                sell_px = _clamp_fill_px(sym, day, sell_px, "sell")
                                cash += q * sell_px
                                entry_date = positions_entry_date.get(sym)
                                pnl = (sell_px - avg_cost) * q
                                rec = {"symbol": sym, "entry_date": entry_date, "exit_date": day.strftime("%Y-%m-%d"), "qty": q, "entry_price": avg_cost, "exit_price": sell_px, "pnl": pnl, "pnl_pct": (sell_px / avg_cost - 1.0) if avg_cost else None, "reason": "per_asset_stop_loss"}
                                _record_trade(rec)
                                _event({"type": "sell", "symbol": sym, "date": day.strftime("%Y-%m-%d"), "qty": float(q), "price": float(sell_px), "notional": float(q * sell_px), "new_qty": 0.0, "reason": rec["reason"], "pnl": float(pnl)})
                                positions_qty.pop(sym, None); positions_avg_cost.pop(sym, None); positions_entry_date.pop(sym, None)

                    if params.portfolio_dd_stop is not None and peak_equity > 0:
                        run_eq_dd = (eq_risk_freq == "hourly" and "dd_stop" in eq_hourly_checks and ts_local.minute == eq_risk_minute and eq_slot_open)
                        run_cr_dd = (cr_risk_freq == "hourly" and "dd_stop" in cr_hourly_checks and ts_local.minute == cr_risk_minute)
                        if run_eq_dd or run_cr_dd:
                            eq_now = cash
                            for s2, q2 in positions_qty.items():
                                p2 = risk_px_at_ts(s2, ts_local)
                                if p2 is None:
                                    p2 = px(s2, day) or float(positions_avg_cost.get(s2, 0.0) or 0.0)
                                if p2 > 0:
                                    eq_now += q2 * p2
                            peak_equity = max(peak_equity, float(eq_now))
                            dd = (peak_equity - float(eq_now)) / peak_equity if peak_equity > 0 else 0.0
                            max_observed_dd = max(max_observed_dd, float(dd))
                            if debug_verbose:
                                _dbg("hourly_dd_eval", day=day_s, ts=str(ts_local), equity_now=float(eq_now), peak_equity=float(peak_equity), dd=float(dd), threshold=float(params.portfolio_dd_stop))
                            if (not stopped_until_next_rebalance) and dd >= params.portfolio_dd_stop:
                                dd_stop_events += 1
                                for sym in list(positions_qty.keys()):
                                    p0 = risk_px_at_ts(sym, ts_local) or px(sym, day)
                                    if p0 is None:
                                        continue
                                    sell_px = p0 * (1 - params.slippage_bps / 10000.0)
                                    sell_px = _clamp_fill_px(sym, day, sell_px, "sell")
                                    q = positions_qty.get(sym, 0.0)
                                    if not _can_sell_today(sym):
                                        continue
                                    cash += q * sell_px
                                    avg_cost = positions_avg_cost.get(sym, p0)
                                    entry_date = positions_entry_date.get(sym)
                                    pnl = (sell_px - avg_cost) * q
                                    rec = {"symbol": sym, "entry_date": entry_date, "exit_date": day.strftime("%Y-%m-%d"), "qty": q, "entry_price": avg_cost, "exit_price": sell_px, "pnl": pnl, "pnl_pct": (sell_px / avg_cost - 1.0) if avg_cost else None, "reason": "portfolio_dd_stop"}
                                    _record_trade(rec)
                                    _event({"type": "sell", "symbol": sym, "date": day.strftime("%Y-%m-%d"), "qty": float(q), "price": float(sell_px), "notional": float(q * sell_px), "new_qty": 0.0, "reason": rec["reason"], "pnl": float(pnl)})
                                positions_qty.clear(); positions_avg_cost.clear(); positions_entry_date.clear(); positions_peak_mark.clear()
                                stopped_until_next_rebalance = True
                                dd_stop_trigger_day = day
                                break

                    if progress_cb and use_hourly_progress:
                        progress_steps_done += 1
                        progress_cb(progress_steps_done, progress_total_steps, float(cash))

        # Portfolio DD stop: liquidate to cash until next rebalance
        if params.portfolio_dd_stop is not None and peak_equity > 0 and (eq_risk_freq != "hourly" or cr_risk_freq != "hourly"):
            dd = (peak_equity - equity) / peak_equity
            max_observed_dd = max(max_observed_dd, float(dd))
            if debug_verbose:
                _dbg("daily_dd_eval", day=day_s, equity=float(equity), peak_equity=float(peak_equity), dd=float(dd), threshold=float(params.portfolio_dd_stop))
            if (not stopped_until_next_rebalance) and dd >= params.portfolio_dd_stop:
                dd_stop_events += 1
                # liquidate everything at risk-check time - slippage
                for sym in list(positions_qty.keys()):
                    p0 = px(sym, day)
                    if p0 is None:
                        continue
                    base_px = risk_px(sym, day) or p0
                    sell_px = base_px * (1 - params.slippage_bps / 10000.0)
                    sell_px = _clamp_fill_px(sym, day, sell_px, "sell")
                    q = positions_qty.get(sym, 0.0)
                    if not _can_sell_today(sym):
                        continue
                    cash += q * sell_px
                    avg_cost = positions_avg_cost.get(sym, p0)
                    entry_date = positions_entry_date.get(sym)
                    pnl = (sell_px - avg_cost) * q
                    rec = {
                        "symbol": sym,
                        "entry_date": entry_date,
                        "exit_date": day.strftime("%Y-%m-%d"),
                        "qty": q,
                        "entry_price": avg_cost,
                        "exit_price": sell_px,
                        "pnl": pnl,
                        "pnl_pct": (sell_px / avg_cost - 1.0) if avg_cost else None,
                        "reason": "portfolio_dd_stop",
                    }
                    _record_trade(rec)
                    _event({
                        "type": "sell",
                        "symbol": sym,
                        "date": day.strftime("%Y-%m-%d"),
                        "qty": float(q),
                        "price": float(sell_px),
                        "notional": float(q * sell_px),
                        "new_qty": 0.0,
                        "reason": rec["reason"],
                        "pnl": float(pnl),
                    })
                positions_qty.clear()
                positions_avg_cost.clear()
                positions_entry_date.clear()
                stopped_until_next_rebalance = True
                dd_stop_trigger_day = day

        # Optional per-asset user exit rules (from strategy builder), evaluated on risk-check schedule.
        if eq_exit_rule is not None or cr_exit_rule is not None:
            for sym in list(positions_qty.keys()):
                is_crypto = "/" in sym
                if is_crypto and (day not in cr_risk_days or cr_risk_freq == "hourly"):
                    continue
                if (not is_crypto) and (day not in eq_risk_days or eq_risk_freq == "hourly"):
                    continue
                ex_rule = cr_exit_rule if is_crypto else eq_exit_rule
                if ex_rule is None:
                    continue
                ann_factor = 365.0 if is_crypto else 252.0
                ctx = _eval_ctx_until_day(sym, day, ann_factor)
                if ctx is None or len(ctx.closes) < 5:
                    continue
                should_exit = False
                try:
                    should_exit = bool(eval_rule(ctx, ex_rule))
                except Exception:
                    should_exit = False
                if debug_verbose:
                    snap = _indicator_snapshot(ex_rule, ctx)
                    _dbg("daily_strategy_exit_eval", day=day_s, symbol=sym, should_exit=bool(should_exit), closes_len=len(ctx.closes), ann_factor=ann_factor, indicators=snap)
                if not should_exit:
                    continue

                p0 = px(sym, day)
                if p0 is None:
                    continue
                base_px = risk_px(sym, day) or p0
                sell_px = base_px * (1 - params.slippage_bps / 10000.0)
                sell_px = _clamp_fill_px(sym, day, sell_px, "sell")
                q = positions_qty.get(sym, 0.0)
                if not _can_sell_today(sym):
                    continue
                cash += q * sell_px
                avg_cost = positions_avg_cost.get(sym, p0)
                entry_date = positions_entry_date.get(sym)
                pnl = (sell_px - avg_cost) * q
                rec = {
                    "symbol": sym,
                    "entry_date": entry_date,
                    "exit_date": day.strftime("%Y-%m-%d"),
                    "qty": q,
                    "entry_price": avg_cost,
                    "exit_price": sell_px,
                    "pnl": pnl,
                    "pnl_pct": (sell_px / avg_cost - 1.0) if avg_cost else None,
                    "reason": "strategy_exit_rule",
                }
                _record_trade(rec)
                _event({
                    "type": "sell",
                    "symbol": sym,
                    "date": day.strftime("%Y-%m-%d"),
                    "qty": float(q),
                    "price": float(sell_px),
                    "notional": float(q * sell_px),
                    "new_qty": 0.0,
                    "reason": rec["reason"],
                    "pnl": float(pnl),
                })
                positions_qty.pop(sym, None)
                positions_avg_cost.pop(sym, None)
                positions_entry_date.pop(sym, None)

        # Per-asset risk exits (checked on risk schedule): trailing + fixed stop loss
        if ((params.per_asset_stop_loss_pct is not None and params.per_asset_stop_loss_pct > 0)
            or bool(params.trailing_stop_stocks_enabled) or bool(params.trailing_stop_crypto_enabled)):
            sl = float(params.per_asset_stop_loss_pct) if (params.per_asset_stop_loss_pct is not None and params.per_asset_stop_loss_pct > 0) else None
            for sym in list(positions_qty.keys()):
                is_crypto = "/" in sym
                if is_crypto and (day not in cr_risk_days or cr_risk_freq == "hourly"):
                    continue
                if (not is_crypto) and (day not in eq_risk_days or eq_risk_freq == "hourly"):
                    continue
                p0 = px(sym, day)
                if p0 is None:
                    continue
                trail_enabled, trail_start, trail_pct = _trail_cfg(sym)
                if trail_enabled and trail_pct is not None and trail_pct > 0:
                    peak = _trail_peak(sym, day, float(p0))
                    positions_peak_mark[sym] = peak
                    avg_cost_for_arm = positions_avg_cost.get(sym, p0)
                    armed = (avg_cost_for_arm is not None and avg_cost_for_arm > 0 and peak >= (float(avg_cost_for_arm) * (1 + float(trail_start))))
                    trail_level = peak * (1 - trail_pct)
                    if debug_verbose:
                        _dbg("daily_trailing_eval", day=day_s, symbol=sym, price=float(p0), avg_cost=avg_cost_for_arm, peak=peak, armed=bool(armed), trail_start=float(trail_start), trail_pct=float(trail_pct), trail_level=float(trail_level))
                    if armed and float(p0) <= trail_level:
                        base_px = risk_px(sym, day) or p0
                        sell_px = base_px * (1 - params.slippage_bps / 10000.0)
                        sell_px = _clamp_fill_px(sym, day, sell_px, "sell")
                        q = positions_qty.get(sym, 0.0)
                        if not _can_sell_today(sym):
                            continue
                        cash += q * sell_px
                        avg_cost = positions_avg_cost.get(sym, p0)
                        entry_date = positions_entry_date.get(sym)
                        pnl = (sell_px - avg_cost) * q
                        rec = {
                            "symbol": sym,
                            "entry_date": entry_date,
                            "exit_date": day.strftime("%Y-%m-%d"),
                            "qty": q,
                            "entry_price": avg_cost,
                            "exit_price": sell_px,
                            "pnl": pnl,
                            "pnl_pct": (sell_px / avg_cost - 1.0) if avg_cost else None,
                            "reason": "trailing_stop_crypto" if is_crypto else "trailing_stop_stocks",
                        }
                        _record_trade(rec)
                        _event({"type": "sell", "symbol": sym, "date": day.strftime("%Y-%m-%d"), "qty": float(q), "price": float(sell_px), "notional": float(q * sell_px), "new_qty": 0.0, "reason": rec["reason"], "pnl": float(pnl)})
                        positions_qty.pop(sym, None)
                        positions_avg_cost.pop(sym, None)
                        positions_entry_date.pop(sym, None)
                        positions_peak_mark.pop(sym, None)
                        continue

                avg_cost = positions_avg_cost.get(sym)
                if avg_cost is None or avg_cost <= 0:
                    continue
                dd_pct = (p0 / avg_cost - 1.0) if avg_cost else None
                if debug_verbose and sl is not None:
                    _dbg("daily_stop_eval", day=day_s, symbol=sym, price=p0, avg_cost=avg_cost, pnl_pct=dd_pct, stop_loss_pct=-sl)
                if (sl is not None) and ((p0 / avg_cost - 1.0) <= -sl):
                    # stop out full position at risk-check time - slippage
                    base_px = risk_px(sym, day) or p0
                    sell_px = base_px * (1 - params.slippage_bps / 10000.0)
                    sell_px = _clamp_fill_px(sym, day, sell_px, "sell")
                    q = positions_qty.get(sym, 0.0)
                    if not _can_sell_today(sym):
                        continue
                    cash += q * sell_px
                    entry_date = positions_entry_date.get(sym)
                    pnl = (sell_px - avg_cost) * q
                    rec = {
                        "symbol": sym,
                        "entry_date": entry_date,
                        "exit_date": day.strftime("%Y-%m-%d"),
                        "qty": q,
                        "entry_price": avg_cost,
                        "exit_price": sell_px,
                        "pnl": pnl,
                        "pnl_pct": (sell_px / avg_cost - 1.0) if avg_cost else None,
                        "reason": "per_asset_stop_loss",
                    }
                    _record_trade(rec)
                    _event({
                        "type": "sell",
                        "symbol": sym,
                        "date": day.strftime("%Y-%m-%d"),
                        "qty": float(q),
                        "price": float(sell_px),
                        "notional": float(q * sell_px),
                        "new_qty": 0.0,
                        "reason": rec["reason"],
                        "pnl": float(pnl),
                    })
                    positions_qty.pop(sym, None)
                    positions_avg_cost.pop(sym, None)
                    positions_entry_date.pop(sym, None)

        # Rebalance
        do_eq_rebalance_raw = day in eq_rebal_days
        do_cr_rebalance_raw = day in cr_rebal_days
        do_eq_rebalance = bool(do_eq_rebalance_raw) and (params.asset_mode in ("both", "equities"))
        do_cr_rebalance = bool(do_cr_rebalance_raw) and (params.asset_mode in ("both", "crypto"))
        if debug_verbose:
            _dbg(
                "day_schedule",
                day=day_s,
                do_eq_rebalance=bool(do_eq_rebalance),
                do_cr_rebalance=bool(do_cr_rebalance),
                raw_eq_rebalance=bool(do_eq_rebalance_raw),
                raw_cr_rebalance=bool(do_cr_rebalance_raw),
                eq_risk_day=bool(day in eq_risk_days),
                cr_risk_day=bool(day in cr_risk_days),
                asset_mode=str(params.asset_mode),
            )
        if (i == 0) or ((i + 1) % 10 == 0):
            _dbg("day_tick", day=day_s, idx=i + 1, total=len(days), positions=len(positions_qty), pending_limits=len(pending_limits), cash=round(float(cash), 2), equity=round(float(equity), 2))
        if do_eq_rebalance or do_cr_rebalance:
            if stopped_until_next_rebalance:
                # behavior A: stay in cash UNTIL the next scheduled rebalance.
                # If we triggered on this same rebalance day, skip this rebalance entirely.
                if dd_stop_trigger_day is not None and day <= dd_stop_trigger_day:
                    # Snapshot holdings for hover/inspection
                    holdings = []
                    total_unreal = 0.0
                    for sym, q in positions_qty.items():
                        p0 = px(sym, day)
                        avg_cost = float(positions_avg_cost.get(sym, 0.0) or 0.0)
                        if p0 is None:
                            p0 = avg_cost if avg_cost > 0 else None
                        if p0 is None:
                            continue
                        mv = float(q * p0)
                        avg_cost = float(positions_avg_cost.get(sym, p0) or p0)
                        unreal = float((p0 - avg_cost) * q)
                        total_unreal += unreal
                        unreal_pct = float(p0 / avg_cost - 1.0) if avg_cost else None
                        holdings.append({"symbol": sym, "mv": mv, "unreal": unreal, "unreal_pct": unreal_pct})
                    holdings.sort(key=lambda x: abs(x.get("mv", 0.0)), reverse=True)
                    holdings = holdings[:15]

                    curve.append({
                        "date": day.strftime("%Y-%m-%d"),
                        "equity": float(equity),
                        "cash": float(cash),
                        "unrealized_pnl": float(total_unreal),
                        "holdings": holdings,
                        "regime": _regime(day),
                    })
                    if progress_cb and (not use_hourly_progress):
                        progress_cb(i + 1, progress_total_steps, float(equity))
                    continue
                stopped_until_next_rebalance = False
                dd_stop_trigger_day = None
            # compute candidates based on history up to day, via selected per-asset entry strategies

            # build bars dict slices up to current day (excluding banned symbols)
            # Keep full OHLCV columns for strategy parity with live selection logic.
            def _slice_df(src: dict[str, pd.DataFrame], sym: str, day_: pd.Timestamp) -> pd.DataFrame | None:
                df0 = src.get(sym)
                if df0 is None or len(df0) == 0:
                    return None
                try:
                    dfx = df0.loc[:day_]
                    if len(dfx) == 0:
                        return None
                    return dfx
                except Exception:
                    return None

            run_eq_select = bool(do_eq_rebalance) and (params.asset_mode in ("both", "equities"))
            run_cr_select = bool(do_cr_rebalance) and (params.asset_mode in ("both", "crypto"))

            eq_bars_day: dict[str, pd.DataFrame] = {}
            if run_eq_select:
                for sym in stock_universe:
                    if sym in excluded:
                        continue
                    dfx = _slice_df(bars_all, sym, day)
                    if dfx is not None:
                        eq_bars_day[sym] = dfx

            cr_bars_day: dict[str, pd.DataFrame] = {}
            if run_cr_select:
                for sym in crypto_universe:
                    if sym in excluded:
                        continue
                    dfx = _slice_df(bars_all, sym, day)
                    if dfx is not None:
                        cr_bars_day[sym] = dfx

            eq_sel: list[str] = []
            cr_sel: list[str] = []
            _eq_details: dict[str, dict] = {}
            _cr_details: dict[str, dict] = {}
            if run_eq_select:
                eq_sel, _eq_details = eq_strat.select_equities(bars=eq_bars_day, cfg=cfg)
            if run_cr_select:
                cr_sel, _cr_details = cr_strat.select_crypto(bars=cr_bars_day, cfg=cfg)
            _dbg("rebalance_selection", day=day_s, run_eq=run_eq_select, run_cr=run_cr_select, eq_candidates=len(eq_bars_day), cr_candidates=len(cr_bars_day), eq_selected=len(eq_sel), cr_selected=len(cr_sel))
            if debug_verbose and (run_eq_select or run_cr_select):
                _dbg("strategy_run", day=day_s, eq_strategy=str(eq_strategy_id), cr_strategy=str(cr_strategy_id), run_eq=run_eq_select, run_cr=run_cr_select)
                if run_eq_select:
                    _dbg("eq_selected_symbols", day=day_s, symbols=",".join([str(s) for s in eq_sel[:50]]))
                    for s in eq_sel[:20]:
                        d = (_eq_details or {}).get(s) or {}
                        _dbg(
                            "eq_indicator_snapshot",
                            day=day_s,
                            symbol=str(s),
                            score=d.get("score"),
                            reason=d.get("reason") or d.get("reject_reason"),
                            last_close=d.get("last_close"),
                            ann_vol=d.get("ann_vol"),
                            ma_long=d.get("ma_long"),
                            ma_short=d.get("ma_short"),
                            rank=d.get("rank"),
                        )
                if run_cr_select:
                    _dbg("cr_selected_symbols", day=day_s, symbols=",".join([str(s) for s in cr_sel[:50]]))
                    for s in cr_sel[:20]:
                        d = (_cr_details or {}).get(s) or {}
                        _dbg(
                            "cr_indicator_snapshot",
                            day=day_s,
                            symbol=str(s),
                            score=d.get("score"),
                            reason=d.get("reason") or d.get("reject_reason"),
                            last_close=d.get("last_close"),
                            ann_vol=d.get("ann_vol"),
                            ma_long=d.get("ma_long"),
                            ma_short=d.get("ma_short"),
                            rank=d.get("rank"),
                        )

            # Optional crypto price floor (per-run param, else config)
            min_cr_px = params.min_crypto_price if params.min_crypto_price is not None else getattr(cfg.limits, "min_crypto_price", None)
            if min_cr_px is not None:
                cr_sel = [s for s in cr_sel if ((px(s, day) or 0.0) >= float(min_cr_px))]

            # targets (notional)
            equity_now = portfolio_value(day)
            eq_budget = equity_now * cfg.allocation.equities
            cr_budget = equity_now * cfg.allocation.crypto

            if params.asset_mode == "equities":
                cr_sel = []
                cr_budget = 0.0
            elif params.asset_mode == "crypto":
                eq_sel = []
                eq_budget = 0.0

            # apply per-asset schedule gates
            if not do_eq_rebalance:
                eq_sel = []
                eq_budget = 0.0
            if not do_cr_rebalance:
                cr_sel = []
                cr_budget = 0.0

            target_notional: dict[str, float] = {}
            if eq_sel:
                w = eq_budget / len(eq_sel)
                for s in eq_sel:
                    target_notional[s] = w
            if cr_sel:
                w = cr_budget / len(cr_sel)
                for s in cr_sel:
                    target_notional[s] = w

            # Liquidate anything not in target set (optional)
            keep = set(target_notional.keys())
            if params.liquidation_mode == "liquidate_non_selected":
                for sym in list(positions_qty.keys()):
                    is_crypto = "/" in sym
                    if is_crypto and not do_cr_rebalance:
                        continue
                    if (not is_crypto) and not do_eq_rebalance:
                        continue
                    if sym in keep:
                        continue
                    p0 = px(sym, day)
                    if p0 is None:
                        continue
                    base_px = exec_px(sym, day) or p0
                    sell_px = base_px * (1 - params.slippage_bps / 10000.0)
                    sell_px = _clamp_fill_px(sym, day, sell_px, "sell")
                    q = positions_qty.get(sym, 0.0)
                    if not _can_sell_today(sym):
                        continue
                    cash += q * sell_px

                    avg_cost = positions_avg_cost.get(sym, p0)
                    entry_date = positions_entry_date.get(sym)
                    pnl = (sell_px - avg_cost) * q
                    rec = {
                        "symbol": sym,
                        "entry_date": entry_date,
                        "exit_date": day.strftime("%Y-%m-%d"),
                        "qty": q,
                        "entry_price": avg_cost,
                        "exit_price": sell_px,
                        "pnl": pnl,
                        "pnl_pct": (sell_px / avg_cost - 1.0) if avg_cost else None,
                        "reason": "rebalance_liquidate",
                    }
                    _record_trade(rec)
                    _event({
                        "type": "sell",
                        "symbol": sym,
                        "date": day.strftime("%Y-%m-%d"),
                        "qty": float(q),
                        "price": float(sell_px),
                        "notional": float(q * sell_px),
                        "new_qty": 0.0,
                        "reason": rec["reason"],
                        "pnl": float(pnl),
                    })

                    positions_qty.pop(sym, None)
                    positions_avg_cost.pop(sym, None)
                    positions_entry_date.pop(sym, None)

            # Rebalance into targets
            for sym, tgtN in target_notional.items():
                p0 = px(sym, day)
                if p0 is None:
                    continue
                p_exec = exec_px(sym, day) or p0
                curQ = positions_qty.get(sym, 0.0)
                # Use execution-time price for sizing so slippage is applied to transaction value (not close).
                curN = curQ * p_exec
                deltaN = tgtN - curN
                if abs(deltaN) < 1e-6:
                    continue

                if deltaN > 0:
                    # optional rule: don't add to losers (skip topping up if below avg cost)
                    if params.rebalance_mode == "no_add_to_losers":
                        prevQ = positions_qty.get(sym, 0.0)
                        if prevQ > 0:
                            avg_cost = positions_avg_cost.get(sym)
                            if avg_cost is not None and p0 < avg_cost:
                                # keep current position; don't add
                                continue

                    # buy at execution time + slippage
                    base_px = p_exec
                    if _use_limit_for(sym):
                        limit_px = base_px * (1 + _limit_off_bps_for(sym) / 10000.0)
                        desired_notional = max(0.0, float(deltaN))
                        if desired_notional <= 0:
                            continue
                        po = {
                            "symbol": sym,
                            "side": "buy",
                            "placed_day": day.strftime("%Y-%m-%d"),
                            "limit_px": float(limit_px),
                            "notional": float(desired_notional),
                            "fallback": _fallback_for(sym),
                        }
                        pending_limits.append(po)
                        _event({"type":"order", "symbol":sym, "date":day.strftime("%Y-%m-%d"), "side":"buy", "limit_px":float(limit_px), "notional":float(desired_notional), "reason":"limit_placed"})
                    else:
                        buy_px = base_px * (1 + params.slippage_bps / 10000.0)
                        buy_px = _clamp_fill_px(sym, day, buy_px, "buy")

                        # desired add in notional terms at base_px
                        desired_q = (deltaN / base_px) if base_px else 0.0
                        if desired_q <= 0:
                            continue
                        # spend at buy_px (includes slippage)
                        q_add = min(desired_q, cash / buy_px) if buy_px else 0.0
                        cost = q_add * buy_px
                        if q_add <= 0:
                            continue
                        cash -= cost
                        prevQ = positions_qty.get(sym, 0.0)
                        newQ = prevQ + q_add

                        # avg cost update
                        prevCost = positions_avg_cost.get(sym, buy_px)
                        if prevQ <= 0:
                            positions_entry_date[sym] = day.strftime("%Y-%m-%d")
                            positions_avg_cost[sym] = buy_px
                            positions_peak_mark[sym] = float(buy_px)
                        else:
                            positions_avg_cost[sym] = (prevQ * prevCost + q_add * buy_px) / (prevQ + q_add)
                        positions_qty[sym] = newQ

                        # record buy event
                        _event({
                            "type": "buy",
                            "symbol": sym,
                            "date": day.strftime("%Y-%m-%d"),
                            "qty": float(q_add),
                            "price": float(buy_px),
                            "notional": float(cost),
                            "new_qty": float(newQ),
                            "reason": "rebalance_buy",
                        })

                else:
                    if not _can_sell_today(sym):
                        continue
                    sellN = min(curN, abs(deltaN))
                    q_sub = (sellN / p_exec) if p_exec else 0.0
                    q_sub = min(q_sub, positions_qty.get(sym, 0.0))
                    if q_sub <= 0:
                        continue

                    if _use_limit_for(sym):
                        limit_px = p_exec * (1 - _limit_off_bps_for(sym) / 10000.0)
                        po = {
                            "symbol": sym,
                            "side": "sell",
                            "placed_day": day.strftime("%Y-%m-%d"),
                            "limit_px": float(limit_px),
                            "qty": float(q_sub),
                            "fallback": _fallback_for(sym),
                        }
                        pending_limits.append(po)
                        _event({"type":"order", "symbol":sym, "date":day.strftime("%Y-%m-%d"), "side":"sell", "limit_px":float(limit_px), "qty":float(q_sub), "reason":"limit_placed"})
                    else:
                        sell_px = p_exec * (1 - params.slippage_bps / 10000.0)
                        sell_px = _clamp_fill_px(sym, day, sell_px, "sell")
                        proceeds = q_sub * sell_px
                        cash += proceeds

                        avg_cost = positions_avg_cost.get(sym, p0)
                        entry_date = positions_entry_date.get(sym)
                        pnl = (sell_px - avg_cost) * q_sub
                        rec = {
                            "symbol": sym,
                            "entry_date": entry_date,
                            "exit_date": day.strftime("%Y-%m-%d"),
                            "qty": q_sub,
                            "entry_price": avg_cost,
                            "exit_price": sell_px,
                            "pnl": pnl,
                            "pnl_pct": (sell_px / avg_cost - 1.0) if avg_cost else None,
                            "reason": "rebalance_trim" if sym in keep else "rebalance_sell",
                        }
                        _record_trade(rec)
                        _event({
                            "type": "sell",
                            "symbol": sym,
                            "date": day.strftime("%Y-%m-%d"),
                            "qty": float(q_sub),
                            "price": float(sell_px),
                            "notional": float(proceeds),
                            "new_qty": float(max(0.0, positions_qty.get(sym, 0.0) - q_sub)),
                            "reason": rec["reason"],
                            "pnl": float(pnl),
                        })

                        newQ = max(0.0, positions_qty.get(sym, 0.0) - q_sub)
                        positions_qty[sym] = newQ
                        if newQ <= 1e-12:
                            positions_qty.pop(sym, None)
                            positions_avg_cost.pop(sym, None)
                            positions_entry_date.pop(sym, None)

            # If any symbol hit the realized P/L floor during this rebalance, optionally liquidate it immediately.
            if params.symbol_pnl_floor_liquidate:
                _liquidate_excluded(day, reason="symbol_pnl_floor_exclude")

        # Snapshot holdings for hover/inspection
        holdings = []
        total_unreal = 0.0
        for sym, q in positions_qty.items():
            p0 = px(sym, day)
            avg_cost = float(positions_avg_cost.get(sym, 0.0) or 0.0)
            if p0 is None:
                p0 = avg_cost if avg_cost > 0 else None
            if p0 is None:
                continue
            mv = float(q * p0)
            avg_cost = float(positions_avg_cost.get(sym, p0) or p0)
            unreal = float((p0 - avg_cost) * q)
            total_unreal += unreal
            unreal_pct = float(p0 / avg_cost - 1.0) if avg_cost else None
            holdings.append({"symbol": sym, "mv": mv, "unreal": unreal, "unreal_pct": unreal_pct})
        holdings.sort(key=lambda x: abs(x.get("mv", 0.0)), reverse=True)
        holdings = holdings[:15]

        curve.append({
            "date": day.strftime("%Y-%m-%d"),
            "equity": float(equity),
            "cash": float(cash),
            "unrealized_pnl": float(total_unreal),
            "holdings": holdings,
            "regime": _regime(day),
        })

        if progress_cb and (not use_hourly_progress):
            progress_cb(i + 1, progress_total_steps, float(equity))
        if debug_verbose:
            _dbg("day_end", day=day_s, idx=i + 1, total=len(days), positions=len(positions_qty), cash=round(float(cash), 2), equity=round(float(equity), 2))
        elif (i == len(days) - 1) or ((i + 1) % 10 == 0):
            _dbg("day_done", day=day_s, idx=i + 1, total=len(days), positions=len(positions_qty), cash=round(float(cash), 2), equity=round(float(equity), 2))

    # metrics
    eq0 = curve[0]["equity"] if curve else params.initial_equity
    eq1 = curve[-1]["equity"] if curve else params.initial_equity
    rets = np.array([curve[j]["equity"] for j in range(len(curve))], dtype=float)
    peak = np.maximum.accumulate(rets) if len(rets) else np.array([eq0])
    dd = (peak - rets) / np.where(peak == 0, 1, peak)
    max_dd = float(np.max(dd)) if len(dd) else 0.0

    days_n = max(1, len(rets) - 1)
    years = days_n / 365.0
    cagr = (eq1 / eq0) ** (1 / years) - 1 if years > 0 and eq0 > 0 else 0.0

    # daily returns for vol/sharpe
    dr = np.diff(rets) / np.where(rets[:-1] == 0, 1, rets[:-1]) if len(rets) > 1 else np.array([0.0])
    ann_factor = 365.0 if str(params.asset_mode) == "crypto" else 252.0
    vol = float(np.std(dr, ddof=0) * np.sqrt(ann_factor)) if len(dr) > 2 else 0.0
    sharpe = float((np.mean(dr) / (np.std(dr, ddof=0) + 1e-12)) * np.sqrt(ann_factor)) if len(dr) > 2 else 0.0
    sharpe_252 = float((np.mean(dr) / (np.std(dr, ddof=0) + 1e-12)) * np.sqrt(252.0)) if len(dr) > 2 else 0.0
    sharpe_365 = float((np.mean(dr) / (np.std(dr, ddof=0) + 1e-12)) * np.sqrt(365.0)) if len(dr) > 2 else 0.0

    wins = [t for t in trades if (t.get("pnl") or 0.0) > 0]
    losses = [t for t in trades if (t.get("pnl") or 0.0) <= 0]
    dd_stops = [t for t in trades if t.get("reason") == "portfolio_dd_stop"]

    metrics = {
        "start_equity": float(eq0),
        "end_equity": float(eq1),
        "return": float(eq1 / eq0 - 1.0) if eq0 else 0.0,
        "cagr": float(cagr),
        "max_drawdown": max_dd,
        "ann_vol": vol,
        "sharpe": sharpe,
        "sharpe_252": sharpe_252,
        "sharpe_365": sharpe_365,
        "sharpe_annualization_days": int(ann_factor),
        "trade_count": len(trades),
        "win_rate": (len(wins) / len(trades)) if trades else None,
        "avg_win": float(np.mean([t["pnl"] for t in wins])) if wins else None,
        "avg_loss": float(np.mean([t["pnl"] for t in losses])) if losses else None,
        "dd_stop_trade_count": len(dd_stops),
        "dd_stop_event_count": int(dd_stop_events),
        "max_observed_drawdown": float(max_observed_dd),
        "open_position_count": int(len(positions_qty)),
        "per_asset_stop_loss_pct": params.per_asset_stop_loss_pct,
        "ts": datetime.now(timezone.utc).isoformat(),
        "hourly_debug": hourly_debug,
    }

    # Open positions (unrealized) at end
    end_day = days[-1] if len(days) else pd.to_datetime(params.end)
    open_pos = []
    for sym, q in positions_qty.items():
        p0 = px(sym, end_day)
        if p0 is None:
            continue
        avg_cost = positions_avg_cost.get(sym, p0)
        mv = q * p0
        pnl = (p0 - avg_cost) * q
        open_pos.append(
            {
                "symbol": sym,
                "qty": q,
                "avg_cost": avg_cost,
                "last_close": p0,
                "market_value": mv,
                "unrealized_pnl": pnl,
                "unrealized_pnl_pct": (p0 / avg_cost - 1.0) if avg_cost else None,
            }
        )
    open_pos.sort(key=lambda x: abs(x.get("market_value", 0.0)), reverse=True)

    return BacktestResult(
        params=params.__dict__,
        equity_curve=curve,
        metrics=metrics,
        trades=trades,
        events=events,
        open_positions=open_pos,
        realized_pnl_by_symbol=realized_pnl_by_symbol,
        excluded_symbols=sorted(excluded),
    )
