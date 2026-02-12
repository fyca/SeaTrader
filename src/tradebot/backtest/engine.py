from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Literal

import numpy as np
import pandas as pd

from tradebot.risk.exits import trend_break_exit
from tradebot.strategies.registry import get_strategy


@dataclass(frozen=True)
class BacktestParams:
    start: str  # YYYY-MM-DD
    end: str    # YYYY-MM-DD
    initial_equity: float = 100000.0
    slippage_bps: float = 10.0
    use_limit_orders: bool = False
    limit_offset_bps: float = 10.0
    # Per-asset execution options (fallback to global use_limit_orders/offset when unset)
    order_type_equities: Literal["market", "limit"] | None = None
    order_type_crypto: Literal["market", "limit"] | None = None
    limit_offset_bps_equities: float | None = None
    limit_offset_bps_crypto: float | None = None
    # Optional parity with live: if limit not fillable by open, convert to market-at-open.
    limit_fallback_to_market_open: bool = False
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
    execution_time_mode: Literal["daily", "intraday"] = "daily"
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
    risk_check_frequency_equities: Literal["weekly", "daily"] | None = None
    risk_check_day_equities: Literal["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"] | None = None
    risk_check_frequency_crypto: Literal["weekly", "daily"] | None = None
    risk_check_day_crypto: Literal["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"] | None = None

    strategy_id: str = "baseline_trendvol"
    asset_mode: Literal["both", "equities", "crypto"] = "both"
    rebalance_mode: Literal["target_notional", "no_add_to_losers"] = "target_notional"
    liquidation_mode: Literal["liquidate_non_selected", "hold_until_exit"] = "liquidate_non_selected"
    per_asset_stop_loss_pct: float | None = None

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
    if mode == "daily":
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
    eq_risk_days = _rebalance_days(days, params.risk_check_frequency_equities or "daily", params.risk_check_day_equities or params.rebalance_day)
    cr_risk_days = _rebalance_days(days, params.risk_check_frequency_crypto or "daily", params.risk_check_day_crypto or params.rebalance_day)

    equity = float(params.initial_equity)
    cash = equity
    positions_qty: dict[str, float] = {}
    positions_avg_cost: dict[str, float] = {}
    positions_entry_date: dict[str, str] = {}
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

    # Precompute close/open series
    closes: dict[str, pd.Series] = {}
    opens: dict[str, pd.Series] = {}
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

    for sym, df in {**stock_bars, **crypto_bars}.items():
        if df is not None and len(df) and "close" in df.columns:
            dfx = df.copy()
            dfx.index = _naive_utc_index(dfx.index)
            dfx = dfx.sort_index()
            closes[sym] = dfx["close"].astype(float)
            if "open" in dfx.columns:
                opens[sym] = dfx["open"].astype(float)
            else:
                opens[sym] = pd.Series(dtype=float)
        else:
            closes[sym] = pd.Series(dtype=float)
            opens[sym] = pd.Series(dtype=float)

    def px(sym: str, day: pd.Timestamp) -> float | None:
        s = closes.get(sym)
        if s is None or len(s) == 0:
            return None
        # use last available close on/before day
        sub = s.loc[:day]
        if len(sub) == 0:
            return None
        v = float(sub.iloc[-1])
        return v if np.isfinite(v) and v > 0 else None

    def px_open(sym: str, day: pd.Timestamp) -> float | None:
        s = opens.get(sym)
        if s is None or len(s) == 0:
            return None
        sub = s.loc[:day]
        if len(sub) == 0:
            return None
        v = float(sub.iloc[-1])
        return v if np.isfinite(v) and v > 0 else None

    def px_col(sym: str, day: pd.Timestamp, col: str) -> float | None:
        df = stock_bars.get(sym) if sym in stock_bars else crypto_bars.get(sym)
        if df is None or len(df) == 0 or col not in df.columns:
            return None
        try:
            dfx = df.copy()
            dfx.index = _naive_utc_index(dfx.index)
            dfx = dfx.sort_index()
            sub = dfx[col].astype(float).loc[:day]
            if len(sub) == 0:
                return None
            v = float(sub.iloc[-1])
            return v if np.isfinite(v) and v > 0 else None
        except Exception:
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

    def _use_limit_for(sym: str) -> bool:
        is_crypto = "/" in sym
        ot = (params.order_type_crypto if is_crypto else params.order_type_equities)
        if ot in ("market", "limit"):
            return ot == "limit"
        return bool(params.use_limit_orders)

    def _limit_off_bps_for(sym: str) -> float:
        is_crypto = "/" in sym
        v = params.limit_offset_bps_crypto if is_crypto else params.limit_offset_bps_equities
        return float(v if v is not None else params.limit_offset_bps)

    def portfolio_value(day: pd.Timestamp) -> float:
        total = cash
        for sym, q in positions_qty.items():
            p = px(sym, day)
            if p is None:
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

    def _event(e: dict) -> None:
        events.append(e)

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
            q = positions_qty.get(sym, 0.0)
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

    for i, day in enumerate(days):
        # Process pending simulated limit orders first
        if params.use_limit_orders or params.order_type_equities == "limit" or params.order_type_crypto == "limit":
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

        # Portfolio DD stop: liquidate to cash until next rebalance
        if params.portfolio_dd_stop is not None and peak_equity > 0:
            dd = (peak_equity - equity) / peak_equity
            max_observed_dd = max(max_observed_dd, float(dd))
            if (not stopped_until_next_rebalance) and dd >= params.portfolio_dd_stop:
                dd_stop_events += 1
                # liquidate everything at risk-check time - slippage
                for sym in list(positions_qty.keys()):
                    p0 = px(sym, day)
                    if p0 is None:
                        continue
                    base_px = risk_px(sym, day) or p0
                    sell_px = base_px * (1 - params.slippage_bps / 10000.0)
                    q = positions_qty.get(sym, 0.0)
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

        # Per-asset stop loss (checked daily at close)
        if params.per_asset_stop_loss_pct is not None and params.per_asset_stop_loss_pct > 0:
            sl = float(params.per_asset_stop_loss_pct)
            for sym in list(positions_qty.keys()):
                is_crypto = "/" in sym
                if is_crypto and day not in cr_risk_days:
                    continue
                if (not is_crypto) and day not in eq_risk_days:
                    continue
                p0 = px(sym, day)
                if p0 is None:
                    continue
                avg_cost = positions_avg_cost.get(sym)
                if avg_cost is None or avg_cost <= 0:
                    continue
                if (p0 / avg_cost - 1.0) <= -sl:
                    # stop out full position at risk-check time - slippage
                    base_px = risk_px(sym, day) or p0
                    sell_px = base_px * (1 - params.slippage_bps / 10000.0)
                    q = positions_qty.get(sym, 0.0)
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
        do_eq_rebalance = day in eq_rebal_days
        do_cr_rebalance = day in cr_rebal_days
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
                    if progress_cb and (i % 10 == 0 or i == len(days) - 1):
                        progress_cb(i + 1, len(days))
                    continue
                stopped_until_next_rebalance = False
                dd_stop_trigger_day = None
            # compute candidates based on history up to day, via selected strategy
            strat = get_strategy(params.strategy_id)

            # build bars dict slices up to current day (excluding banned symbols)
            eq_bars_day: dict[str, pd.DataFrame] = {}
            for sym in stock_universe:
                if sym in excluded:
                    continue
                s = closes.get(sym)
                if s is None or len(s) == 0:
                    continue
                s2 = s.loc[:day]
                if len(s2) == 0:
                    continue
                eq_bars_day[sym] = pd.DataFrame({"close": s2.values}, index=s2.index)

            cr_bars_day: dict[str, pd.DataFrame] = {}
            for sym in crypto_universe:
                if sym in excluded:
                    continue
                s = closes.get(sym)
                if s is None or len(s) == 0:
                    continue
                s2 = s.loc[:day]
                if len(s2) == 0:
                    continue
                cr_bars_day[sym] = pd.DataFrame({"close": s2.values}, index=s2.index)

            eq_sel, _eq_details = strat.select_equities(bars=eq_bars_day, cfg=cfg)
            cr_sel, _cr_details = strat.select_crypto(bars=cr_bars_day, cfg=cfg)

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
                    q = positions_qty.get(sym, 0.0)
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
                            "fallback": bool(params.limit_fallback_to_market_open),
                        }
                        pending_limits.append(po)
                        _event({"type":"order", "symbol":sym, "date":day.strftime("%Y-%m-%d"), "side":"buy", "limit_px":float(limit_px), "notional":float(desired_notional), "reason":"limit_placed"})
                    else:
                        buy_px = base_px * (1 + params.slippage_bps / 10000.0)

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
                            "fallback": bool(params.limit_fallback_to_market_open),
                        }
                        pending_limits.append(po)
                        _event({"type":"order", "symbol":sym, "date":day.strftime("%Y-%m-%d"), "side":"sell", "limit_px":float(limit_px), "qty":float(q_sub), "reason":"limit_placed"})
                    else:
                        sell_px = p_exec * (1 - params.slippage_bps / 10000.0)
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

        if progress_cb and (i % 10 == 0 or i == len(days) - 1):
            progress_cb(i + 1, len(days))

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
    vol = float(np.std(dr, ddof=0) * np.sqrt(365.0)) if len(dr) > 2 else 0.0
    sharpe = float((np.mean(dr) / (np.std(dr, ddof=0) + 1e-12)) * np.sqrt(365.0)) if len(dr) > 2 else 0.0

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
