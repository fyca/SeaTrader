from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import numpy as np
import pandas as pd

from tradebot.signals.trend_vol import compute_trend_vol_signal
from tradebot.risk.exits import trend_break_exit


@dataclass(frozen=True)
class BacktestParams:
    start: str  # YYYY-MM-DD
    end: str    # YYYY-MM-DD
    initial_equity: float = 100000.0
    slippage_bps: float = 10.0
    rebalance: Literal["weekly", "daily"] = "weekly"
    asset_mode: Literal["both", "equities", "crypto"] = "both"
    rebalance_mode: Literal["target_notional", "no_add_to_losers"] = "target_notional"
    liquidation_mode: Literal["liquidate_non_selected", "hold_until_exit"] = "liquidate_non_selected"
    per_asset_stop_loss_pct: float | None = None

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
    trades: list[dict]
    open_positions: list[dict]


def _date_range(start: str, end: str) -> pd.DatetimeIndex:
    # Use tz-naive dates consistently
    s = pd.to_datetime(start).tz_localize(None)
    e = pd.to_datetime(end).tz_localize(None)
    idx = pd.date_range(s, e, freq="D")
    return idx


def _rebalance_days(days: pd.DatetimeIndex, mode: str) -> set[pd.Timestamp]:
    if mode == "daily":
        return set(days)
    # weekly: Mondays
    return set([d for d in days if d.weekday() == 0])


def run_backtest(
    *,
    stock_bars: dict[str, pd.DataFrame],
    crypto_bars: dict[str, pd.DataFrame],
    stock_universe: list[str],
    crypto_universe: list[str],
    cfg,
    params: BacktestParams,
    progress_cb=None,
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
    rebal_days = _rebalance_days(days, params.rebalance)

    equity = float(params.initial_equity)
    cash = equity
    positions_qty: dict[str, float] = {}
    positions_avg_cost: dict[str, float] = {}
    positions_entry_date: dict[str, str] = {}
    trades: list[dict] = []

    peak_equity = equity
    stopped_until_next_rebalance = False
    dd_stop_events = 0
    max_observed_dd = 0.0
    dd_stop_trigger_day: pd.Timestamp | None = None

    # Precompute close series
    closes: dict[str, pd.Series] = {}
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
        else:
            closes[sym] = pd.Series(dtype=float)

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

    def portfolio_value(day: pd.Timestamp) -> float:
        total = cash
        for sym, q in positions_qty.items():
            p = px(sym, day)
            if p is None:
                continue
            total += q * p
        return float(total)

    curve: list[dict] = []

    for i, day in enumerate(days):
        # Mark-to-market
        equity = portfolio_value(day)
        peak_equity = max(peak_equity, equity)

        # Portfolio DD stop: liquidate to cash until next rebalance
        if params.portfolio_dd_stop is not None and peak_equity > 0:
            dd = (peak_equity - equity) / peak_equity
            max_observed_dd = max(max_observed_dd, float(dd))
            if (not stopped_until_next_rebalance) and dd >= params.portfolio_dd_stop:
                dd_stop_events += 1
                # liquidate everything at close - slippage
                for sym in list(positions_qty.keys()):
                    p0 = px(sym, day)
                    if p0 is None:
                        continue
                    sell_px = p0 * (1 - params.slippage_bps / 10000.0)
                    q = positions_qty.get(sym, 0.0)
                    cash += q * sell_px
                    avg_cost = positions_avg_cost.get(sym, p0)
                    entry_date = positions_entry_date.get(sym)
                    pnl = (sell_px - avg_cost) * q
                    trades.append(
                        {
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
                    )
                positions_qty.clear()
                positions_avg_cost.clear()
                positions_entry_date.clear()
                stopped_until_next_rebalance = True
                dd_stop_trigger_day = day

        # Rebalance
        if day in rebal_days:
            if stopped_until_next_rebalance:
                # behavior A: stay in cash UNTIL the next scheduled rebalance.
                # If we triggered on this same rebalance day, skip this rebalance entirely.
                if dd_stop_trigger_day is not None and day <= dd_stop_trigger_day:
                    curve.append({"date": day.strftime("%Y-%m-%d"), "equity": float(equity), "cash": float(cash)})
                    if progress_cb and (i % 10 == 0 or i == len(days) - 1):
                        progress_cb(i + 1, len(days))
                    continue
                stopped_until_next_rebalance = False
                dd_stop_trigger_day = None
            # compute candidates based on history up to day
            eq_ok: list[tuple[str, float]] = []
            for sym in stock_universe:
                s = closes.get(sym)
                if s is None or len(s) == 0:
                    continue
                s2 = s.loc[:day]
                if len(s2) == 0:
                    continue
                last = float(s2.iloc[-1])
                if last < cfg.limits.min_stock_price:
                    continue
                sig = compute_trend_vol_signal(
                    s2,
                    ma_long=cfg.signals.equity.ma_long,
                    ma_short=cfg.signals.equity.ma_short,
                    vol_lookback=cfg.signals.equity.vol_lookback,
                    max_ann_vol=cfg.signals.equity.max_ann_vol,
                    ann_factor=252.0,
                )
                if sig.ok:
                    eq_ok.append((sym, float(sig.score)))

            cr_ok: list[tuple[str, float]] = []
            for sym in crypto_universe:
                s = closes.get(sym)
                if s is None or len(s) == 0:
                    continue
                s2 = s.loc[:day]
                if len(s2) == 0:
                    continue
                sig = compute_trend_vol_signal(
                    s2,
                    ma_long=cfg.signals.crypto.ma_long,
                    ma_short=cfg.signals.crypto.ma_short,
                    vol_lookback=cfg.signals.crypto.vol_lookback,
                    max_ann_vol=cfg.signals.crypto.max_ann_vol,
                    ann_factor=365.0,
                )
                if sig.ok:
                    cr_ok.append((sym, float(sig.score)))

            eq_sel = [s for s, _ in sorted(eq_ok, key=lambda x: x[1], reverse=True)[: cfg.limits.max_equity_positions]]
            cr_sel = [s for s, _ in sorted(cr_ok, key=lambda x: x[1], reverse=True)[: cfg.limits.max_crypto_positions]]

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
                    if sym in keep:
                        continue
                    p0 = px(sym, day)
                    if p0 is None:
                        continue
                    sell_px = p0 * (1 - params.slippage_bps / 10000.0)
                    q = positions_qty.get(sym, 0.0)
                    cash += q * sell_px

                    avg_cost = positions_avg_cost.get(sym, p0)
                    entry_date = positions_entry_date.get(sym)
                    pnl = (sell_px - avg_cost) * q
                    trades.append(
                        {
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
                    )

                    positions_qty.pop(sym, None)
                    positions_avg_cost.pop(sym, None)
                    positions_entry_date.pop(sym, None)

            # Rebalance into targets
            for sym, tgtN in target_notional.items():
                p0 = px(sym, day)
                if p0 is None:
                    continue
                curQ = positions_qty.get(sym, 0.0)
                curN = curQ * p0
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

                    # buy at close + slippage
                    buy_px = p0 * (1 + params.slippage_bps / 10000.0)
                    cost = deltaN * (1 + params.slippage_bps / 10000.0)
                    if cost > cash:
                        cost = cash
                    q_add = cost / buy_px
                    cash -= cost
                    newQ = positions_qty.get(sym, 0.0) + q_add
                    # avg cost update
                    prevQ = positions_qty.get(sym, 0.0)
                    prevCost = positions_avg_cost.get(sym, buy_px)
                    if prevQ <= 0:
                        positions_entry_date[sym] = day.strftime("%Y-%m-%d")
                        positions_avg_cost[sym] = buy_px
                    else:
                        positions_avg_cost[sym] = (prevQ * prevCost + q_add * buy_px) / (prevQ + q_add)
                    positions_qty[sym] = newQ

                else:
                    # sell at close - slippage
                    sell_px = p0 * (1 - params.slippage_bps / 10000.0)
                    sellN = min(curN, abs(deltaN))
                    q_sub = sellN / p0
                    q_sub = min(q_sub, positions_qty.get(sym, 0.0))
                    proceeds = q_sub * sell_px
                    cash += proceeds

                    avg_cost = positions_avg_cost.get(sym, p0)
                    entry_date = positions_entry_date.get(sym)
                    pnl = (sell_px - avg_cost) * q_sub
                    trades.append(
                        {
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
                    )

                    newQ = max(0.0, positions_qty.get(sym, 0.0) - q_sub)
                    positions_qty[sym] = newQ
                    if newQ <= 1e-12:
                        positions_qty.pop(sym, None)
                        positions_avg_cost.pop(sym, None)
                        positions_entry_date.pop(sym, None)

        curve.append({"date": day.strftime("%Y-%m-%d"), "equity": float(equity), "cash": float(cash)})

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

    return BacktestResult(params=params.__dict__, equity_curve=curve, metrics=metrics, trades=trades, open_positions=open_pos)
