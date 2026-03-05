from __future__ import annotations

import argparse
import uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from rich import print
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus, OrderType
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest

from tradebot.adapters.alpaca_client import make_alpaca_clients
from tradebot.adapters.bars import fetch_stock_bars, fetch_crypto_bars
from tradebot.risk.drawdown import update_drawdown_state
from tradebot.risk.exits import trend_break_exit
from tradebot.util.config import load_config
from tradebot.strategies.registry import get_strategy
from tradebot.strategies.resolver import resolve_for_risk_check, resolve_for_rebalance, strategy_snapshot, validate_strategy_refs
from tradebot.util.env import load_env
from tradebot.util.state import load_state, save_state
from tradebot.util.artifacts import write_artifact
from tradebot.util.equity_curve import append_equity_point
from tradebot.util.market_hours import get_market_status
from tradebot.util.live_ledger import append_live_run, append_live_events
from tradebot.universe.equities import list_tradable_equities
from tradebot.universe.crypto import list_tradable_crypto


def _fallback_convert_open_limits(cfg, clients, market_status: dict) -> list[dict]:
    """Convert stale OPEN limit orders to market orders when fallback window is reached.

    This runs during risk-check so fallback still happens even if rebalance process exits
    before/while waiting for fallback time.
    """
    converted: list[dict] = []
    try:
        tz = ZoneInfo(getattr(cfg.scheduling, "timezone", "America/Los_Angeles"))
    except Exception:
        tz = ZoneInfo("America/Los_Angeles")
    now = datetime.now(tz)
    grace = int(getattr(cfg.execution, "fallback_grace_seconds", 20) or 20)

    try:
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500)
        orders = clients.trading.get_orders(filter=req)
    except Exception as ex:
        print(f"[yellow]Fallback scan warning[/yellow]: {ex}")
        return converted

    for o in orders:
        try:
            typ = str(getattr(o, "type", "") or "").upper()
            if "LIMIT" not in typ:
                continue
            symbol = str(getattr(o, "symbol", "") or "").strip()
            if not symbol:
                continue

            is_crypto = "/" in symbol
            ex_cfg = cfg.execution.crypto if is_crypto else cfg.execution.equities
            enabled = bool(getattr(ex_cfg, "fallback_to_market_at_open", False))
            if not enabled:
                continue

            t_local = str(getattr(ex_cfg, "fallback_time_local", "06:30") or "06:30")
            try:
                hh, mm = [int(x) for x in t_local.split(":")]
                target = now.replace(hour=hh, minute=mm, second=0, microsecond=0) + timedelta(seconds=grace)
            except Exception:
                continue
            if now < target:
                continue

            # Respect equity market-hours guard for market orders.
            if (not is_crypto) and (not bool(market_status.get("can_place_equity_orders", False))):
                converted.append({
                    "symbol": symbol,
                    "status": "skipped_market_closed",
                    "reason": "fallback_window_reached_but_market_closed",
                })
                continue

            side = str(getattr(o, "side", "") or "").upper()
            qty = getattr(o, "qty", None)
            notional = getattr(o, "notional", None)
            filled_qty = float(getattr(o, "filled_qty", 0.0) or 0.0)
            if filled_qty > 0:
                converted.append({
                    "symbol": symbol,
                    "status": "skipped_partial_fill",
                    "reason": "partial_fill_present",
                    "filled_qty": filled_qty,
                    "order_id": str(getattr(o, "id", "")),
                })
                continue

            order_id = str(getattr(o, "id", ""))
            if order_id:
                try:
                    clients.trading.cancel_order_by_id(order_id)
                except Exception:
                    pass

            req_kwargs = dict(
                symbol=symbol,
                side=OrderSide.BUY if side.endswith("BUY") else OrderSide.SELL,
                time_in_force=TimeInForce.GTC if is_crypto else TimeInForce.DAY,
            )
            try:
                q = float(qty) if qty is not None else 0.0
            except Exception:
                q = 0.0
            try:
                n = float(notional) if notional is not None else 0.0
            except Exception:
                n = 0.0

            if q > 0:
                req_kwargs["qty"] = q
            elif n > 0:
                req_kwargs["notional"] = round(n, 2)
            else:
                converted.append({
                    "symbol": symbol,
                    "status": "skipped_missing_size",
                    "reason": "open_limit_has_no_qty_or_notional",
                    "order_id": order_id,
                })
                continue

            mo = clients.trading.submit_order(MarketOrderRequest(**req_kwargs))
            converted.append({
                "symbol": symbol,
                "status": "submitted",
                "reason": "fallback_convert_open_limit",
                "from_order_id": order_id,
                "to_order_id": str(getattr(mo, "id", "")),
                "side": "buy" if side.endswith("BUY") else "sell",
                "qty": q if q > 0 else None,
                "notional": n if (q <= 0 and n > 0) else None,
            })
        except Exception as ex:
            converted.append({
                "symbol": str(getattr(o, "symbol", "") or ""),
                "status": "error",
                "reason": "fallback_conversion_error",
                "error": str(ex),
            })
    return converted


def preview_rebalance_selections(cfg, clients) -> dict:
    """Preview what the rebalance strategy would select without running a full rebalance.
    
    Returns:
        {
            "stocks_would_select": [...],
            "crypto_would_select": [...],
            "held_stocks": [...],
            "held_crypto": [...],
            "stocks_to_add": [...],      # selected but not held
            "stocks_to_remove": [...],   # held but not selected
            "crypto_to_add": [...],
            "crypto_to_remove": [...],
        }
    """
    try:
        # Fetch universes
        eq_univ = list_tradable_equities(clients.trading, exclude_leveraged_etfs=cfg.universe.exclude_leveraged_etfs)
        cr_univ = list_tradable_crypto(clients.trading)
        
        # Build candidate symbol lists (same logic as rebalance)
        try:
            from tradebot.universe.sp500 import get_sp500_symbols
            sp500 = set(get_sp500_symbols())
        except Exception:
            sp500 = set()
        
        eq_all = [x.symbol for x in eq_univ]
        eq_symbols = [s for s in eq_all if s in sp500] if sp500 else eq_all
        eq_symbols = eq_symbols[:500]
        
        cr_all = cfg.universe.crypto_symbols_allowlist or [x.symbol for x in cr_univ]
        cr_symbols = [s for s in cr_all if str(s).endswith("USD")]
        cr_symbols = cr_symbols[:200]
        
        # Fetch bars
        eq_bars = fetch_stock_bars(clients.stocks, eq_symbols, lookback_days=cfg.signals.lookback_days)
        cr_bars = fetch_crypto_bars(clients.crypto, cr_symbols, lookback_days=cfg.signals.lookback_days)
        
        # Resolve entry strategies
        resolved = resolve_for_rebalance(cfg)
        eq_strategy_id = resolved["stocks_entry"].strategy_id
        cr_strategy_id = resolved["crypto_entry"].strategy_id
        
        eq_strat = get_strategy(eq_strategy_id)
        cr_strat = get_strategy(cr_strategy_id)
        
        # Run strategy selections
        eq_sel, _ = eq_strat.select_equities(bars=eq_bars, cfg=cfg)
        cr_sel, _ = cr_strat.select_crypto(bars=cr_bars, cfg=cfg)
        
        # Get current positions
        positions = clients.trading.get_all_positions()
        held_stocks = [p.symbol for p in positions if float(p.qty) != 0.0 and "/" not in p.symbol]
        held_crypto = [p.symbol for p in positions if float(p.qty) != 0.0 and "/" in p.symbol]
        
        eq_sel_set = set(eq_sel)
        cr_sel_set = set(cr_sel)
        held_stocks_set = set(held_stocks)
        held_crypto_set = set(held_crypto)
        
        return {
            "stocks_would_select": list(eq_sel),
            "crypto_would_select": list(cr_sel),
            "held_stocks": held_stocks,
            "held_crypto": held_crypto,
            "stocks_to_add": list(eq_sel_set - held_stocks_set),
            "stocks_to_remove": list(held_stocks_set - eq_sel_set),
            "crypto_to_add": list(cr_sel_set - held_crypto_set),
            "crypto_to_remove": list(held_crypto_set - cr_sel_set),
        }
    except Exception as ex:
        return {
            "error": str(ex),
            "stocks_would_select": [],
            "crypto_would_select": [],
            "held_stocks": [],
            "held_crypto": [],
            "stocks_to_add": [],
            "stocks_to_remove": [],
            "crypto_to_add": [],
            "crypto_to_remove": [],
        }


def _symbols_bought_today(clients, tz_name: str) -> set[str]:
    out: set[str] = set()
    try:
        tz = ZoneInfo(tz_name or "America/Los_Angeles")
    except Exception:
        tz = ZoneInfo("America/Los_Angeles")
    today = datetime.now(tz).date()
    try:
        req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=500)
        orders = clients.trading.get_orders(filter=req)
    except Exception:
        return out

    for o in orders:
        try:
            sym = str(getattr(o, "symbol", "") or "").strip()
            if not sym:
                continue
            side = str(getattr(o, "side", "") or "").upper()
            if not side.endswith("BUY"):
                continue
            st = str(getattr(o, "status", "") or "").upper()
            if "FILLED" not in st:
                continue
            fa = getattr(o, "filled_at", None)
            if fa is None:
                continue
            if isinstance(fa, datetime):
                dt = fa
            else:
                dt = datetime.fromisoformat(str(fa).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=tz)
            if dt.astimezone(tz).date() == today:
                out.add(sym)
        except Exception:
            continue
    return out


def cmd_risk_check(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, preset_override=getattr(args, "preset", None))
    ref_errors = validate_strategy_refs(cfg)
    if ref_errors:
        raise RuntimeError("Invalid strategy references: " + "; ".join(ref_errors))
    run_id = str(uuid.uuid4())
    run_asset_mode = str(getattr(args, "asset_mode", None) or "both").lower()
    if run_asset_mode not in ("both", "equities", "crypto"):
        run_asset_mode = "both"
    env = load_env()
    clients = make_alpaca_clients(env)

    acct = clients.trading.get_account()
    equity = float(acct.equity)
    market_status = get_market_status(clients.trading)
    
    # Preview what rebalance strategy would select (to inform exit decisions)
    rebalance_preview = preview_rebalance_selections(cfg, clients)

    state = load_state()
    dd_trigger = cfg.risk.portfolio_dd_stop if cfg.risk.portfolio_dd_stop is not None else cfg.risk.max_drawdown_freeze
    dd_state = update_drawdown_state(prior_peak_equity=state.peak_equity, current_equity=equity, freeze_at=dd_trigger)
    state.peak_equity = dd_state.peak_equity
    save_state(state)

    print(f"Equity: {equity:.2f}  Peak: {dd_state.peak_equity:.2f}  Drawdown: {dd_state.drawdown:.1%}")
    if dd_state.frozen:
        print(f"[red]FROZEN[/red] (>= {cfg.risk.max_drawdown_freeze:.0%}) -> should not open new positions")
    else:
        print("OK")

    # Exits-only logic: if a held position breaks trend OR hits stop-loss, propose a SELL (dry-run only)
    positions = clients.trading.get_all_positions()
    held = [p.symbol for p in positions if float(p.qty) != 0.0]

    eq_syms = [s for s in held if "/" not in s]
    cr_syms = [s for s in held if "/" in s]
    if state.trailing_peaks:
        held_set = set([str(s).upper() for s in held])
        state.trailing_peaks = {k: v for k, v in state.trailing_peaks.items() if str(k).upper() in held_set}
    if run_asset_mode == "equities":
        cr_syms = []
    elif run_asset_mode == "crypto":
        eq_syms = []

    exit_plans = []
    stocks_to_remove = set(rebalance_preview.get("stocks_to_remove", []))
    crypto_to_remove = set(rebalance_preview.get("crypto_to_remove", []))

    stop_pct = cfg.risk.per_asset_stop_loss_pct
    if stop_pct is not None:
        stop_pct = float(stop_pct)

    trailing_anchor = str(getattr(cfg.risk, "trailing_stop_anchor", "highest_since_entry") or "highest_since_entry")
    trail_eq_enabled = bool(getattr(cfg.risk, "trailing_stop_stocks_enabled", False))
    trail_cr_enabled = bool(getattr(cfg.risk, "trailing_stop_crypto_enabled", False))
    trail_eq_start = getattr(cfg.risk, "trailing_stop_stocks_start_gain_pct", 0.05)
    trail_cr_start = getattr(cfg.risk, "trailing_stop_crypto_start_gain_pct", 0.05)
    trail_eq_pct = getattr(cfg.risk, "trailing_stop_stocks_pct", None)
    trail_cr_pct = getattr(cfg.risk, "trailing_stop_crypto_pct", None)
    trail_eq_start = float(trail_eq_start) if trail_eq_start is not None else 0.05
    trail_cr_start = float(trail_cr_start) if trail_cr_start is not None else 0.05
    trail_eq_pct = float(trail_eq_pct) if trail_eq_pct is not None else None
    trail_cr_pct = float(trail_cr_pct) if trail_cr_pct is not None else None
    state.trailing_peaks = dict(state.trailing_peaks or {})

    # Resolve per-asset exit strategies (with legacy fallback to cfg.strategy_id)
    resolved = resolve_for_risk_check(cfg)

    def _exit_rule_for(asset_class: str):
        # Toggle gate: when disabled, do not evaluate user-defined exit strategy for that asset.
        enabled = bool(cfg.strategies.crypto.exit_enabled) if asset_class == "crypto" else bool(cfg.strategies.stocks.exit_enabled)
        if not enabled:
            return None
        strategy_id = resolved["crypto_exit"].strategy_id if asset_class == "crypto" else resolved["stocks_exit"].strategy_id
        try:
            strat = get_strategy(strategy_id)
            if strat is not None and hasattr(strat, "spec"):
                return getattr(strat, "spec", {}).get("exit")
        except Exception:
            return None
        return None

    user_exit_eq = _exit_rule_for("stocks")
    user_exit_cr = _exit_rule_for("crypto")

    if eq_syms:
        eq_bars = fetch_stock_bars(clients.stocks, eq_syms, lookback_days=cfg.signals.lookback_days)
        for sym, df in eq_bars.items():
            if df is None or len(df) == 0 or "close" not in df.columns:
                continue
            closes = df["close"].dropna()
            if len(closes) == 0:
                continue
            last_px = float(closes.iloc[-1])

            # user exit rule (if present/enabled for stocks)
            if user_exit_eq:
                try:
                    from tradebot.strategies.rule_engine import EvalContext, eval_rule
                    highs = df["high"].dropna() if "high" in df.columns else None
                    lows = df["low"].dropna() if "low" in df.columns else None
                    opens = df["open"].dropna() if "open" in df.columns else None
                    volumes = df["volume"].dropna() if "volume" in df.columns else None
                    ctx = EvalContext(closes=closes, ann_factor=252.0, highs=highs, lows=lows, opens=opens, volumes=volumes)
                    if eval_rule(ctx, user_exit_eq):
                        exit_plans.append({
                            "symbol": sym,
                            "asset_class": "equity",
                            "reason": "user_exit_rule",
                            "last_close": last_px,
                            "strategy_removing": sym in stocks_to_remove,
                        })
                        continue
                except Exception:
                    pass

            # stop-loss check from avg entry
            if stop_pct is not None:
                # pull avg entry from Alpaca position
                pos = next((p for p in positions if p.symbol == sym), None)
                if pos is not None:
                    avg_entry = float(pos.avg_entry_price)
                    stop_level = avg_entry * (1 - stop_pct)
                    if last_px <= stop_level:
                        exit_plans.append({
                            "symbol": sym,
                            "asset_class": "equity",
                            "reason": f"stop_loss_{int(stop_pct*100)}%",
                            "last_close": last_px,
                            "stop_level": stop_level,
                            "avg_entry": avg_entry,
                            "strategy_removing": sym in stocks_to_remove,
                        })
                        continue

            if trail_eq_enabled and (trail_eq_pct is not None) and trail_eq_pct > 0:
                pos = next((p for p in positions if p.symbol == sym), None)
                avg_entry = float(getattr(pos, "avg_entry_price", 0.0) or 0.0) if pos is not None else 0.0
                prev_peak = float((state.trailing_peaks or {}).get(sym, 0.0) or 0.0)
                peak = float(closes.max()) if trailing_anchor == "highest_close_since_entry" else max(prev_peak, float(last_px))
                if peak > 0:
                    state.trailing_peaks[sym] = peak
                    armed = (avg_entry > 0) and (peak >= (avg_entry * (1 + trail_eq_start)))
                    if armed:
                        trail_level = peak * (1 - trail_eq_pct)
                        if last_px <= trail_level:
                            exit_plans.append({
                                "symbol": sym,
                                "asset_class": "equity",
                                "reason": "trailing_stop_stocks",
                                "last_close": last_px,
                                "trail_level": trail_level,
                                "trail_peak": peak,
                                "strategy_removing": sym in stocks_to_remove,
                            })
                            continue

            should, reason, last, maL = trend_break_exit(closes, ma_long=cfg.signals.equity.ma_long)
            if should:
                exit_plans.append({
                    "symbol": sym,
                    "asset_class": "equity",
                    "reason": reason,
                    "last_close": last,
                    "ma_long": maL,
                    "strategy_removing": sym in stocks_to_remove,
                })

    if cr_syms:
        cr_bars = fetch_crypto_bars(clients.crypto, cr_syms, lookback_days=cfg.signals.lookback_days)
        for sym, df in cr_bars.items():
            if df is None or len(df) == 0 or "close" not in df.columns:
                continue
            closes = df["close"].dropna()
            if len(closes) == 0:
                continue
            last_px = float(closes.iloc[-1])

            # user exit rule (if present/enabled for crypto)
            if user_exit_cr:
                try:
                    from tradebot.strategies.rule_engine import EvalContext, eval_rule
                    highs = df["high"].dropna() if "high" in df.columns else None
                    lows = df["low"].dropna() if "low" in df.columns else None
                    opens = df["open"].dropna() if "open" in df.columns else None
                    volumes = df["volume"].dropna() if "volume" in df.columns else None
                    ctx = EvalContext(closes=closes, ann_factor=365.0, highs=highs, lows=lows, opens=opens, volumes=volumes)
                    if eval_rule(ctx, user_exit_cr):
                        exit_plans.append({
                            "symbol": sym,
                            "asset_class": "crypto",
                            "reason": "user_exit_rule",
                            "last_close": last_px,
                            "strategy_removing": sym in crypto_to_remove,
                        })
                        continue
                except Exception:
                    pass

            if stop_pct is not None:
                pos = next((p for p in positions if p.symbol == sym), None)
                if pos is not None:
                    avg_entry = float(pos.avg_entry_price)
                    stop_level = avg_entry * (1 - stop_pct)
                    if last_px <= stop_level:
                        exit_plans.append({
                            "symbol": sym,
                            "asset_class": "crypto",
                            "reason": f"stop_loss_{int(stop_pct*100)}%",
                            "last_close": last_px,
                            "stop_level": stop_level,
                            "avg_entry": avg_entry,
                            "strategy_removing": sym in crypto_to_remove,
                        })
                        continue

            if trail_cr_enabled and (trail_cr_pct is not None) and trail_cr_pct > 0:
                pos = next((p for p in positions if p.symbol == sym), None)
                avg_entry = float(getattr(pos, "avg_entry_price", 0.0) or 0.0) if pos is not None else 0.0
                prev_peak = float((state.trailing_peaks or {}).get(sym, 0.0) or 0.0)
                peak = float(closes.max()) if trailing_anchor == "highest_close_since_entry" else max(prev_peak, float(last_px))
                if peak > 0:
                    state.trailing_peaks[sym] = peak
                    armed = (avg_entry > 0) and (peak >= (avg_entry * (1 + trail_cr_start)))
                    if armed:
                        trail_level = peak * (1 - trail_cr_pct)
                        if last_px <= trail_level:
                            exit_plans.append({
                                "symbol": sym,
                                "asset_class": "crypto",
                                "reason": "trailing_stop_crypto",
                                "last_close": last_px,
                                "trail_level": trail_level,
                                "trail_peak": peak,
                                "strategy_removing": sym in crypto_to_remove,
                            })
                            continue

            should, reason, last, maL = trend_break_exit(closes, ma_long=cfg.signals.crypto.ma_long)
            if should:
                exit_plans.append({
                    "symbol": sym,
                    "asset_class": "crypto",
                    "reason": reason,
                    "last_close": last,
                    "ma_long": maL,
                    "strategy_removing": sym in crypto_to_remove,
                })

    blocked_same_day: list[dict] = []
    if bool(getattr(cfg.risk, "block_same_day_roundtrip", True)) and exit_plans:
        bought_today = _symbols_bought_today(clients, getattr(cfg.scheduling, "timezone", "America/Los_Angeles"))
        if bought_today:
            kept = []
            for e in exit_plans:
                sym = str(e.get("symbol") or "")
                if sym in bought_today:
                    blocked_same_day.append({
                        "symbol": sym,
                        "asset_class": e.get("asset_class"),
                        "reason": e.get("reason"),
                        "status": "skipped_same_day_roundtrip",
                    })
                else:
                    kept.append(e)
            exit_plans = kept

    if exit_plans:
        print("\nExit signals:")
        for e in exit_plans:
            safe_note = " [strategy removing]" if e.get("strategy_removing") else ""
            print(f"- SELL {e['symbol']:12s} ({e['asset_class']}) reason={e['reason']}{safe_note}")
    if blocked_same_day:
        print("\nSame-day roundtrip guard blocked exits:")
        for e in blocked_same_day:
            print(f"- SKIP {e['symbol']:12s} reason={e['reason']}")

    fallback_conversions = _fallback_convert_open_limits(cfg, clients, market_status)
    if fallback_conversions:
        print(f"Fallback conversions processed: {len(fallback_conversions)}")

    executed_liquidations = []
    if exit_plans and bool(getattr(cfg.risk, "execute_exit_liquidations", False)):
        # de-dup by symbol to avoid double-ordering the same asset
        pos_by_symbol = {p.symbol: p for p in positions}
        seen: set[str] = set()
        for e in exit_plans:
            sym = str(e.get("symbol") or "").strip()
            if not sym or sym in seen:
                continue
            seen.add(sym)
            pos = pos_by_symbol.get(sym)
            if pos is None:
                continue
            try:
                qty = abs(float(pos.qty))
            except Exception:
                qty = 0.0
            if qty <= 0:
                continue
            if "/" not in sym and not bool(market_status.get("can_place_equity_orders", False)):
                executed_liquidations.append({
                    "symbol": sym,
                    "qty": qty,
                    "side": "sell",
                    "status": "skipped_market_closed",
                    "reason": e.get("reason"),
                })
                continue
            if bool(getattr(cfg, "dry_run", False)):
                executed_liquidations.append({
                    "symbol": sym,
                    "qty": qty,
                    "side": "sell",
                    "status": "skipped_dry_run",
                    "reason": e.get("reason"),
                })
                continue
            tif = TimeInForce.GTC if "/" in sym else TimeInForce.DAY
            try:
                req = MarketOrderRequest(symbol=sym, qty=qty, side=OrderSide.SELL, time_in_force=tif)
                o = clients.trading.submit_order(req)
                executed_liquidations.append({
                    "symbol": sym,
                    "qty": qty,
                    "side": "sell",
                    "status": "submitted",
                    "order_id": str(getattr(o, "id", "")),
                    "reason": e.get("reason"),
                })
                print(f"[green]Submitted SELL[/green] {sym} qty={qty}")
            except Exception as ex:
                executed_liquidations.append({
                    "symbol": sym,
                    "qty": qty,
                    "side": "sell",
                    "status": "error",
                    "error": str(ex),
                    "reason": e.get("reason"),
                })
                print(f"[red]Failed SELL[/red] {sym}: {ex}")

    risk_payload = {
        "run_id": run_id,
        "strategy_selection": {
            "stocks_exit": resolved["stocks_exit"].strategy_id,
            "crypto_exit": resolved["crypto_exit"].strategy_id,
            "stocks_exit_enabled": bool(cfg.strategies.stocks.exit_enabled),
            "crypto_exit_enabled": bool(cfg.strategies.crypto.exit_enabled),
        },
        "strategy_snapshot": strategy_snapshot(cfg),
        "rebalance_preview": rebalance_preview,  # What the rebalance strategy would select
        "equity": equity,
        "peak_equity": dd_state.peak_equity,
        "drawdown": dd_state.drawdown,
        "frozen": dd_state.frozen,
        "exit_signals": exit_plans,
        "execute_exit_liquidations": bool(getattr(cfg.risk, "execute_exit_liquidations", False)),
        "executed_liquidations": executed_liquidations,
        "blocked_same_day_roundtrip": blocked_same_day,
        "fallback_conversions": fallback_conversions,
        "market_status": market_status,
    }
    write_artifact("last_risk_check.json", risk_payload)
    append_live_run(
        run_id=run_id,
        kind="risk_check",
        payload={
            "paper": bool(env.paper),
            "asset_mode": run_asset_mode,
            "equity": equity,
            "drawdown": dd_state.drawdown,
            "peak_equity": dd_state.peak_equity,
            "frozen": dd_state.frozen,
            "exit_signal_count": len(exit_plans),
            "blocked_same_day_roundtrip_count": len(blocked_same_day),
            "liquidation_count": len(executed_liquidations),
            "fallback_conversion_count": len([x for x in fallback_conversions if str(x.get("status")) == "submitted"]),
            "market_status": market_status,
        },
    )
    append_live_events(
        run_id=run_id,
        kind="risk_check",
        events=[
            {
                "event_type": "exit_signal",
                **e,
            }
            for e in exit_plans
        ]
        + [
            {
                "event_type": "liquidation",
                **e,
            }
            for e in executed_liquidations
        ]
        + [
            {
                "event_type": "exit_blocked_same_day_roundtrip",
                **e,
            }
            for e in blocked_same_day
        ]
        + [
            {
                "event_type": "fallback_conversion",
                **e,
            }
            for e in fallback_conversions
        ],
    )
    append_equity_point(equity=equity, cash=float(getattr(acct, "cash", 0.0) or 0.0))
    save_state(state)

    return 0
