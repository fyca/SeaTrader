from __future__ import annotations

import argparse
import json
import subprocess
import threading
import uuid
from datetime import datetime, timedelta, timezone
import io
import contextlib
from pathlib import Path
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from tradebot.dashboard.auth import check_token
from tradebot.dashboard.actions import require_token
from tradebot.dashboard.config_api import load_config_file, save_config_file, validate_config_payload
from tradebot.backtest.job import start_backtest, get_status as bt_status, get_result as bt_result, list_jobs as bt_list_jobs, get_latest_job_id
from tradebot.util.config import load_config
from tradebot.util.env import load_env
from tradebot.adapters.alpaca_client import make_alpaca_clients
from tradebot.util.state import load_state, save_state
from tradebot.adapters.bars import fetch_stock_bars, fetch_crypto_bars
from tradebot.util.market_hours import get_market_status
from tradebot.util.live_ledger import get_runs as live_ledger_runs, get_events as live_ledger_events


def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def create_app(*, config_path: str) -> FastAPI:
    app = FastAPI(title="tradebot dashboard")
    action_jobs: dict[str, dict] = {}
    schedule_state: dict[str, object] = {
        "rebalance_weekly_enabled": False,
        "rebalance_started_at": None,
        "rebalance_next_run": None,
        "rebalance_last_run": None,
        "rebalance_last_state": None,
        "rebalance_last_error": None,

        "risk_daily_enabled": False,
        "risk_daily_started_at": None,
        "risk_next_run": None,
        "risk_last_run": None,
        "risk_last_state": None,
        "risk_last_error": None,
    }

    # Static assets (themes, icons, etc.)
    static_dir = Path(__file__).with_name("static")
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        html_path = Path(__file__).with_name("index.html")
        return HTMLResponse(html_path.read_text())

    @app.get("/builder", response_class=HTMLResponse)
    def builder():
        html_path = Path(__file__).with_name("builder.html")
        return HTMLResponse(html_path.read_text())

    @app.get("/api/health")
    def health():
        return {"ok": True, "ts": datetime.now(timezone.utc).isoformat()}

    @app.get("/api/config")
    def config():
        cfg = load_config_file(config_path)
        return JSONResponse(cfg.model_dump())

    @app.get("/api/strategies")
    def strategies():
        from tradebot.strategies.registry import list_strategies

        return list_strategies()

    # Strategy Builder API (user strategies)
    @app.get("/api/strategy/{strategy_id}")
    def strategy_get(strategy_id: str):
        from tradebot.strategies.user_store import load_user_strategy

        return load_user_strategy(strategy_id)

    @app.post("/api/strategy/{strategy_id}")
    async def strategy_save(strategy_id: str, req: Request):
        # token gate is controlled by DASHBOARD_REQUIRE_TOKEN
        require_token(req)
        body = await req.json()
        from tradebot.strategies.user_store import save_user_strategy

        save_user_strategy(strategy_id, body)
        return {"ok": True}

    @app.delete("/api/strategy/{strategy_id}")
    async def strategy_delete(strategy_id: str, req: Request):
        require_token(req)
        from tradebot.strategies.user_store import delete_user_strategy

        delete_user_strategy(strategy_id)
        return {"ok": True}

    @app.post("/api/strategy/{strategy_id}/preview")
    async def strategy_preview(strategy_id: str, req: Request):
        body = await req.json()
        symbol = str(body.get("symbol") or "").strip()
        if not symbol:
            raise HTTPException(status_code=400, detail="Missing symbol")

        from tradebot.strategies.user_store import load_user_strategy
        from tradebot.strategies.rule_engine import EvalContext, eval_rule, eval_score_with_breakdown, eval_indicator

        spec = load_user_strategy(strategy_id)
        entry = spec.get("entry") or {"all": []}
        exit_rule = spec.get("exit") or {"any": []}
        factors = spec.get("score_factors") or []

        env = load_env()
        clients = make_alpaca_clients(env)

        # fetch recent bars for just this symbol
        is_crypto = "/" in symbol
        if is_crypto:
            bars = fetch_crypto_bars(clients.crypto, [symbol], lookback_days=420)
            df = bars.get(symbol)
            ann_factor = 365.0
        else:
            bars = fetch_stock_bars(clients.stocks, [symbol], lookback_days=420)
            df = bars.get(symbol)
            ann_factor = 252.0

        if df is None or len(df) == 0 or "close" not in df.columns:
            raise HTTPException(status_code=400, detail="No bars for symbol")

        closes = df["close"].dropna()
        ctx = EvalContext(closes=closes, ann_factor=ann_factor)

        entry_ok = bool(eval_rule(ctx, entry))
        exit_ok = bool(eval_rule(ctx, exit_rule))
        score, breakdown = eval_score_with_breakdown(ctx, factors)

        # Helpful indicator cache for common values
        indicators = {}
        for ind in [
            {"kind": "close"},
            {"kind": "sma", "n": 50},
            {"kind": "sma", "n": 200},
            {"kind": "rsi", "n": 14},
            {"kind": "ann_vol", "n": 20},
            {"kind": "highest", "n": 20},
        ]:
            try:
                indicators[f"{ind['kind']}:{ind.get('n','')}"] = eval_indicator(ctx, ind)
            except Exception:
                pass

        return {
            "symbol": symbol,
            "is_crypto": is_crypto,
            "entry_ok": entry_ok,
            "exit_ok": exit_ok,
            "score": float(score),
            "breakdown": breakdown,
            "indicators": indicators,
            "last_close": float(closes.iloc[-1]) if len(closes) else None,
        }

    @app.post("/api/config")
    async def config_save(req: Request):
        require_token(req)
        body = await req.json()
        cfg = validate_config_payload(body)
        save_config_file(config_path, cfg)
        return {"ok": True}

    @app.get("/api/state")
    def state():
        st = load_state()
        return {"peak_equity": st.peak_equity}

    @app.get("/api/exclusions")
    def exclusions():
        st = load_state()
        excluded = sorted(set([str(s).upper() for s in (st.excluded_symbols or []) if str(s).strip()]))

        env = load_env()
        clients = make_alpaca_clients(env)

        pos_map: dict[str, dict] = {}
        try:
            for p in clients.trading.get_all_positions():
                sym = str(getattr(p, "symbol", "") or "").upper()
                if not sym:
                    continue
                pos_map[sym] = {
                    "qty": float(getattr(p, "qty", 0.0) or 0.0),
                    "qty_available": float(getattr(p, "qty_available", 0.0) or 0.0),
                    "market_value": float(getattr(p, "market_value", 0.0) or 0.0),
                    "avg_entry_price": float(getattr(p, "avg_entry_price", 0.0) or 0.0),
                    "unrealized_pl": float(getattr(p, "unrealized_pl", 0.0) or 0.0),
                    "unrealized_plpc": float(getattr(p, "unrealized_plpc", 0.0) or 0.0),
                    "current_price": float(getattr(p, "current_price", 0.0) or 0.0),
                    "lastday_price": float(getattr(p, "lastday_price", 0.0) or 0.0),
                }
        except Exception:
            pass

        eq_syms = [s for s in excluded if "/" not in s]
        cr_syms = [s for s in excluded if "/" in s]
        close_map: dict[str, float] = {}
        if eq_syms:
            try:
                bars = fetch_stock_bars(clients.stocks, eq_syms, lookback_days=10)
                for s, df in bars.items():
                    if df is not None and len(df) and "close" in df.columns:
                        close_map[str(s).upper()] = float(df["close"].dropna().iloc[-1])
            except Exception:
                pass
        if cr_syms:
            try:
                bars = fetch_crypto_bars(clients.crypto, cr_syms, lookback_days=10)
                for s, df in bars.items():
                    if df is not None and len(df) and "close" in df.columns:
                        close_map[str(s).upper()] = float(df["close"].dropna().iloc[-1])
            except Exception:
                pass

        rows = []
        for sym in excluded:
            p = pos_map.get(sym, {})
            last_trade = p.get("current_price") or close_map.get(sym)
            rows.append(
                {
                    "symbol": sym,
                    "held": sym in pos_map,
                    "qty": p.get("qty"),
                    "qty_available": p.get("qty_available"),
                    "market_value": p.get("market_value"),
                    "avg_entry_price": p.get("avg_entry_price"),
                    "unrealized_pl": p.get("unrealized_pl"),
                    "unrealized_plpc": p.get("unrealized_plpc"),
                    "last_trade_price": last_trade,
                    "last_close": close_map.get(sym),
                    "reason": "symbol_pnl_floor",
                }
            )

        cfg = load_config(config_path)
        return {
            "count": len(rows),
            "floor_pct": getattr(cfg.rebalance, "symbol_pnl_floor_pct", None),
            "symbols": rows,
        }

    @app.post("/api/exclusions/remove")
    async def exclusions_remove(req: Request):
        require_token(req)
        body = await req.json()
        symbol = str((body or {}).get("symbol") or "").strip().upper()
        if not symbol:
            raise HTTPException(status_code=400, detail="Missing symbol")
        st = load_state()
        cur = [str(s).upper() for s in (st.excluded_symbols or [])]
        st.excluded_symbols = [s for s in cur if s != symbol]
        save_state(st)
        return {"ok": True, "removed": symbol, "count": len(st.excluded_symbols or [])}

    @app.get("/api/account")
    def account():
        env = load_env()
        clients = make_alpaca_clients(env)
        acct = clients.trading.get_account()

        def _f(x, d=0.0):
            try:
                return float(x)
            except Exception:
                return float(d)

        eq = _f(getattr(acct, "equity", 0.0))
        cash = _f(getattr(acct, "cash", 0.0))
        bod_eq = _f(getattr(acct, "last_equity", 0.0))
        bod_cash = _f(getattr(acct, "last_cash", 0.0))

        return {
            "account_number": acct.account_number,
            "paper": bool(env.paper),

            # current
            "equity": eq,
            "cash": cash,
            "buying_power": _f(getattr(acct, "buying_power", 0.0)),
            "portfolio_value": _f(getattr(acct, "portfolio_value", eq)),

            # beginning-of-day (previous close snapshot from broker)
            "bod_equity": bod_eq,
            "bod_cash": bod_cash,
            "day_change_equity": (eq - bod_eq) if bod_eq else None,
            "day_change_equity_pct": ((eq / bod_eq - 1.0) if bod_eq else None),
            "day_change_cash": (cash - bod_cash) if bod_cash else None,

            # useful health/margin/account fields
            "status": str(getattr(acct, "status", "")),
            "currency": str(getattr(acct, "currency", "USD")),
            "multiplier": str(getattr(acct, "multiplier", "")),
            "regt_buying_power": _f(getattr(acct, "regt_buying_power", 0.0)),
            "daytrading_buying_power": _f(getattr(acct, "daytrading_buying_power", 0.0)),
            "initial_margin": _f(getattr(acct, "initial_margin", 0.0)),
            "maintenance_margin": _f(getattr(acct, "maintenance_margin", 0.0)),
            "sma": _f(getattr(acct, "sma", 0.0)),
            "pattern_day_trader": bool(getattr(acct, "pattern_day_trader", False)),
            "trading_blocked": bool(getattr(acct, "trading_blocked", False)),
            "transfers_blocked": bool(getattr(acct, "transfers_blocked", False)),
            "account_blocked": bool(getattr(acct, "account_blocked", False)),
            "shorting_enabled": bool(getattr(acct, "shorting_enabled", False)),
            "crypto_status": str(getattr(acct, "crypto_status", "")),
        }

    @app.get("/api/market-status")
    def market_status():
        env = load_env()
        clients = make_alpaca_clients(env)
        return get_market_status(clients.trading)

    @app.get("/api/positions")
    def positions():
        """Held positions only (Alpaca positions)."""
        env = load_env()
        clients = make_alpaca_clients(env)
        pos = clients.trading.get_all_positions()
        out = []
        for p in pos:
            out.append(
                {
                    "symbol": p.symbol,
                    "qty": float(p.qty),
                    "side": getattr(p, "side", ""),
                    "market_value": float(p.market_value),
                    "avg_entry_price": float(p.avg_entry_price),
                    "unrealized_pl": float(p.unrealized_pl),
                    "unrealized_plpc": float(p.unrealized_plpc),
                }
            )
        out.sort(key=lambda x: abs(x["market_value"]), reverse=True)
        return out

    @app.get("/api/exposure")
    def exposure():
        """Combined view: held positions + open orders (pending exposure).

        Adds estimated price/qty for notional market orders using last daily close.
        """
        env = load_env()
        clients = make_alpaca_clients(env)

        # held positions
        held: dict[str, dict] = {}
        for p in clients.trading.get_all_positions():
            held[p.symbol] = {
                "symbol": p.symbol,
                "held_qty": float(p.qty),
                "market_value": float(p.market_value),
                "avg_entry_price": float(p.avg_entry_price),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
                "pending_buy_qty": 0.0,
                "pending_sell_qty": 0.0,
                "pending_buy_notional": 0.0,
                "pending_sell_notional": 0.0,
                "est_price": None,
                "est_pending_buy_qty": None,
                "open_orders": [],
            }

        # open orders
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500)
        orders = clients.trading.get_orders(filter=req)
        for o in orders:
            sym = getattr(o, "symbol", "")
            if not sym:
                continue
            row = held.get(sym)
            if row is None:
                row = {
                    "symbol": sym,
                    "held_qty": 0.0,
                    "market_value": 0.0,
                    "avg_entry_price": 0.0,
                    "unrealized_pl": 0.0,
                    "unrealized_plpc": 0.0,
                    "pending_buy_qty": 0.0,
                    "pending_sell_qty": 0.0,
                    "pending_buy_notional": 0.0,
                    "pending_sell_notional": 0.0,
                    "est_price": None,
                    "est_pending_buy_qty": None,
                    "open_orders": [],
                }
                held[sym] = row

            side = _norm_enum(getattr(o, "side", ""))
            qty = getattr(o, "qty", None)
            notional = getattr(o, "notional", None)
            try:
                qty_f = float(qty) if qty is not None else 0.0
            except Exception:
                qty_f = 0.0
            try:
                notional_f = float(notional) if notional is not None else 0.0
            except Exception:
                notional_f = 0.0

            if side.lower() == "buy":
                row["pending_buy_qty"] += qty_f
                row["pending_buy_notional"] += notional_f
            elif side.lower() == "sell":
                row["pending_sell_qty"] += qty_f
                row["pending_sell_notional"] += notional_f

            row["open_orders"].append(
                {
                    "id": str(getattr(o, "id", "")),
                    "side": side,
                    "status": _norm_enum(getattr(o, "status", "")),
                    "qty": qty,
                    "notional": notional,
                    "limit_price": getattr(o, "limit_price", None),
                    "stop_price": getattr(o, "stop_price", None),
                    "created_at": str(getattr(o, "created_at", "")),
                    "type": _norm_enum(getattr(o, "type", "")),
                }
            )

        # Estimate price/qty for symbols that have pending buy notional but no qty
        syms = list(held.keys())
        eq_syms = [s for s in syms if "/" not in s]
        cr_syms = [s for s in syms if "/" in s]

        price_map: dict[str, float] = {}
        if eq_syms:
            try:
                bars = fetch_stock_bars(clients.stocks, eq_syms, lookback_days=10)
                for s, df in bars.items():
                    if df is not None and len(df) and "close" in df.columns:
                        price_map[s] = float(df["close"].dropna().iloc[-1])
            except Exception:
                pass
        if cr_syms:
            try:
                bars = fetch_crypto_bars(clients.crypto, cr_syms, lookback_days=10)
                for s, df in bars.items():
                    if df is not None and len(df) and "close" in df.columns:
                        price_map[s] = float(df["close"].dropna().iloc[-1])
            except Exception:
                pass

        for s, row in held.items():
            px = price_map.get(s)
            if px and px > 0:
                row["est_price"] = px
                if row.get("pending_buy_notional", 0.0) and (row.get("pending_buy_qty", 0.0) == 0.0):
                    row["est_pending_buy_qty"] = float(row["pending_buy_notional"]) / px

        out = list(held.values())
        out.sort(key=lambda x: abs(x.get("market_value", 0.0)) + abs(x.get("pending_buy_notional", 0.0)) + abs(x.get("pending_sell_notional", 0.0)), reverse=True)
        return out

    def _norm_enum(val) -> str:
        s = str(val or "")
        # alpaca enums often stringify as 'OrderSide.BUY' / 'OrderStatus.ACCEPTED'
        if "." in s:
            s = s.split(".")[-1]
        s = s.strip()
        return s.upper()

    def _orders_by_status(status, limit: int = 500, include_current_price: bool = False):
        env = load_env()
        clients = make_alpaca_clients(env)
        from alpaca.trading.requests import GetOrdersRequest

        req = GetOrdersRequest(status=status, limit=limit)
        orders = clients.trading.get_orders(filter=req)

        out = []
        symbols: list[str] = []
        for o in orders:
            symbol = getattr(o, "symbol", "")
            if symbol:
                symbols.append(symbol)
            out.append(
                {
                    "id": str(getattr(o, "id", "")),
                    "symbol": symbol,
                    "side": _norm_enum(getattr(o, "side", "")),
                    "status": _norm_enum(getattr(o, "status", "")),
                    "type": _norm_enum(getattr(o, "type", "")),
                    "qty": getattr(o, "qty", None),
                    "filled_qty": getattr(o, "filled_qty", None),
                    "notional_usd": getattr(o, "notional", None),
                    "filled_avg_price": getattr(o, "filled_avg_price", None),
                    "limit_price": getattr(o, "limit_price", None),
                    "stop_price": getattr(o, "stop_price", None),
                    "created_at": str(getattr(o, "created_at", "")),
                    "filled_at": str(getattr(o, "filled_at", "")),
                }
            )

        if include_current_price and out:
            uniq_syms = sorted(set(s for s in symbols if s))
            eq_syms = [s for s in uniq_syms if "/" not in s]
            cr_syms = [s for s in uniq_syms if "/" in s]
            price_map: dict[str, float] = {}
            if eq_syms:
                try:
                    bars = fetch_stock_bars(clients.stocks, eq_syms, lookback_days=10)
                    for s, df in bars.items():
                        if df is not None and len(df) and "close" in df.columns:
                            price_map[s] = float(df["close"].dropna().iloc[-1])
                except Exception:
                    pass
            if cr_syms:
                try:
                    bars = fetch_crypto_bars(clients.crypto, cr_syms, lookback_days=10)
                    for s, df in bars.items():
                        if df is not None and len(df) and "close" in df.columns:
                            price_map[s] = float(df["close"].dropna().iloc[-1])
                except Exception:
                    pass
            for row in out:
                px = price_map.get(str(row.get("symbol") or ""))
                row["current_price"] = px if (px is not None and px > 0) else None

        return out

    @app.get("/api/open-orders")
    def open_orders():
        from alpaca.trading.enums import QueryOrderStatus

        return _orders_by_status(QueryOrderStatus.OPEN, limit=500, include_current_price=True)

    @app.get("/api/recent-fills")
    def recent_fills(limit: int = 200):
        from alpaca.trading.enums import QueryOrderStatus

        # Alpaca API: status='closed' captures filled/canceled; we filter to filled
        closed = _orders_by_status(QueryOrderStatus.CLOSED, limit=limit)
        filled = [o for o in closed if str(o.get("filled_at") or "").strip() not in ("", "None")]
        # sort newest first
        filled.sort(key=lambda x: x.get("filled_at") or "", reverse=True)
        return filled

    @app.get("/api/order-summary")
    def order_summary():
        from alpaca.trading.enums import QueryOrderStatus

        open_orders = _orders_by_status(QueryOrderStatus.OPEN, limit=500)
        closed = _orders_by_status(QueryOrderStatus.CLOSED, limit=500)
        fills = [o for o in closed if str(o.get("filled_at") or "").strip() not in ("", "None")]

        def _sum(field: str, arr):
            tot = 0.0
            for o in arr:
                v = o.get(field)
                try:
                    tot += float(v) if v is not None else 0.0
                except Exception:
                    pass
            return tot

        return {
            "open_count": len(open_orders),
            "open_notional": _sum("notional_usd", open_orders),
            "fills_count": len(fills),
            "fills_qty": _sum("filled_qty", fills),
        }

    @app.post("/api/cancel-all-open-orders")
    async def cancel_all_open_orders(req: Request):
        require_token(req)
        env = load_env()
        if not env.paper:
            raise HTTPException(status_code=400, detail="Refusing: APCA_PAPER is not true")
        clients = make_alpaca_clients(env)
        res = clients.trading.cancel_orders()
        return {"ok": True, "result": str(res)}

    def _seconds_until_local_hhmm(hhmm: str, tz_name: str) -> tuple[float, str]:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(tz_name)
        now = datetime.now(tz)
        hh, mm = [int(x) for x in str(hhmm).split(":")]
        target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if target <= now:
            target = target + timedelta(days=1)
        return (target - now).total_seconds(), target.isoformat()

    def _seconds_until_weekly(day_name: str, hhmm: str, tz_name: str) -> tuple[float, str]:
        from zoneinfo import ZoneInfo

        day_map = {"MON":0, "TUE":1, "WED":2, "THU":3, "FRI":4, "SAT":5, "SUN":6}
        wd = day_map.get(str(day_name).upper(), 0)
        tz = ZoneInfo(tz_name)
        now = datetime.now(tz)
        hh, mm = [int(x) for x in str(hhmm).split(":")]
        target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        delta_days = (wd - now.weekday()) % 7
        if delta_days == 0 and target <= now:
            delta_days = 7
        target = target + timedelta(days=delta_days)
        return (target - now).total_seconds(), target.isoformat()

    def _is_equity_trading_day(trading_client, day_local) -> bool:
        try:
            ds = day_local.isoformat()
            cal = trading_client.get_calendar(start=ds, end=ds)
            for r in cal or []:
                d = str(getattr(r, "date", "") or "")
                if d.startswith(ds):
                    return True
            return False
        except Exception:
            # fallback: weekday heuristic if calendar unavailable
            return int(day_local.weekday()) < 5

    def _seconds_until_weekly_equity_market_day(day_name: str, hhmm: str, tz_name: str, trading_client) -> tuple[float, str, str | None]:
        from zoneinfo import ZoneInfo

        day_map = {"MON":0, "TUE":1, "WED":2, "THU":3, "FRI":4, "SAT":5, "SUN":6}
        wd = day_map.get(str(day_name).upper(), 0)
        tz = ZoneInfo(tz_name)
        now = datetime.now(tz)
        hh, mm = [int(x) for x in str(hhmm).split(":")]

        target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        delta_days = (wd - now.weekday()) % 7
        if delta_days == 0 and target <= now:
            delta_days = 7
        target = target + timedelta(days=delta_days)
        base_target = target

        # Roll forward to next equity trading day (holiday/weekend fallback)
        for _ in range(10):
            if _is_equity_trading_day(trading_client, target.date()):
                break
            target = target + timedelta(days=1)

        note = None
        if target.date() != base_target.date():
            note = f"holiday/weekend fallback from {base_target.date().isoformat()}"

        return (target - now).total_seconds(), target.isoformat(), note

    def _ensure_weekly_rebalance_scheduler(preset, place_orders: bool = True):
        if bool(schedule_state.get("rebalance_weekly_enabled")):
            return

        schedule_state["rebalance_weekly_enabled"] = True
        schedule_state["rebalance_started_at"] = datetime.now(timezone.utc).isoformat()

        def _loop():
            from tradebot.cli import cmd_rebalance

            while bool(schedule_state.get("rebalance_weekly_enabled")):
                try:
                    cfg = load_config(config_path, preset_override=preset)
                    tz_name = str(getattr(cfg.scheduling, "timezone", "America/Los_Angeles"))

                    eq_s = getattr(cfg.scheduling, "equities", None)
                    cr_s = getattr(cfg.scheduling, "crypto", None)

                    eq_freq = str(getattr(eq_s, "rebalance_frequency", "weekly"))
                    eq_day = str(getattr(eq_s, "rebalance_day", getattr(cfg.scheduling, "weekly_rebalance_day", "MON")))
                    eq_tm = str(getattr(eq_s, "rebalance_time_local", getattr(cfg.scheduling, "weekly_rebalance_time_local", "06:35")))

                    cr_freq = str(getattr(cr_s, "rebalance_frequency", "weekly"))
                    cr_day = str(getattr(cr_s, "rebalance_day", getattr(cfg.scheduling, "weekly_rebalance_day", "MON")))
                    cr_tm = str(getattr(cr_s, "rebalance_time_local", getattr(cfg.scheduling, "weekly_rebalance_time_local", "06:35")))

                    waits = []
                    schedule_state["rebalance_next_run_reason_equities"] = None
                    if eq_freq == "daily":
                        sec_eq, when_eq = _seconds_until_local_hhmm(eq_tm, tz_name)
                        waits.append(("equities", sec_eq, when_eq))
                    else:
                        env2 = load_env()
                        clients2 = make_alpaca_clients(env2)
                        sec_eq, when_eq, note_eq = _seconds_until_weekly_equity_market_day(eq_day, eq_tm, tz_name, clients2.trading)
                        waits.append(("equities", sec_eq, when_eq))
                        if note_eq:
                            schedule_state["rebalance_next_run_reason_equities"] = note_eq
                    if cr_freq == "daily":
                        waits.append(("crypto", *_seconds_until_local_hhmm(cr_tm, tz_name)))
                    else:
                        waits.append(("crypto", *_seconds_until_weekly(cr_day, cr_tm, tz_name)))

                    asset_mode, sec, when_iso = min(waits, key=lambda x: float(x[1]))
                    schedule_state[f"rebalance_next_run_{asset_mode}"] = when_iso
                    schedule_state["rebalance_next_run"] = when_iso
                    import time as _time
                    _time.sleep(max(1.0, sec))
                    if not bool(schedule_state.get("rebalance_weekly_enabled")):
                        break

                    rc = int(cmd_rebalance(argparse.Namespace(config=config_path, place_orders=place_orders, wait_until=None, preset=preset, asset_mode=asset_mode)))
                    schedule_state["rebalance_last_run"] = datetime.now(timezone.utc).isoformat()
                    schedule_state["rebalance_last_asset_mode"] = asset_mode
                    schedule_state["rebalance_last_state"] = "done" if rc == 0 else f"rc={rc}"
                    schedule_state["rebalance_last_error"] = None
                except Exception as e:
                    schedule_state["rebalance_last_run"] = datetime.now(timezone.utc).isoformat()
                    schedule_state["rebalance_last_state"] = "error"
                    schedule_state["rebalance_last_error"] = str(e)
                    import time as _time
                    _time.sleep(5)

        threading.Thread(target=_loop, daemon=True).start()

    def _ensure_daily_risk_scheduler(preset):
        if bool(schedule_state.get("risk_daily_enabled")):
            return

        schedule_state["risk_daily_enabled"] = True
        schedule_state["risk_daily_started_at"] = datetime.now(timezone.utc).isoformat()

        def _loop():
            from tradebot.commands.risk_check import cmd_risk_check

            while bool(schedule_state.get("risk_daily_enabled")):
                try:
                    cfg = load_config(config_path, preset_override=preset)
                    tz_name = str(getattr(cfg.scheduling, "timezone", "America/Los_Angeles"))
                    eq_s = getattr(cfg.scheduling, "equities", None)
                    cr_s = getattr(cfg.scheduling, "crypto", None)

                    eq_freq = str(getattr(eq_s, "risk_check_frequency", "daily"))
                    eq_day = str(getattr(eq_s, "risk_check_day", getattr(cfg.scheduling, "weekly_rebalance_day", "MON")))
                    eq_tm = str(getattr(eq_s, "risk_check_time_local", getattr(cfg.scheduling, "daily_risk_check_time_local", "18:05")))

                    cr_freq = str(getattr(cr_s, "risk_check_frequency", "daily"))
                    cr_day = str(getattr(cr_s, "risk_check_day", getattr(cfg.scheduling, "weekly_rebalance_day", "MON")))
                    cr_tm = str(getattr(cr_s, "risk_check_time_local", getattr(cfg.scheduling, "daily_risk_check_time_local", "18:05")))

                    waits = []
                    if eq_freq == "weekly":
                        waits.append(("equities", *_seconds_until_weekly(eq_day, eq_tm, tz_name)))
                    else:
                        waits.append(("equities", *_seconds_until_local_hhmm(eq_tm, tz_name)))
                    if cr_freq == "weekly":
                        waits.append(("crypto", *_seconds_until_weekly(cr_day, cr_tm, tz_name)))
                    else:
                        waits.append(("crypto", *_seconds_until_local_hhmm(cr_tm, tz_name)))

                    asset_mode, sec, when_iso = min(waits, key=lambda x: float(x[1]))
                    schedule_state[f"risk_next_run_{asset_mode}"] = when_iso
                    schedule_state["risk_next_run"] = when_iso
                    import time as _time
                    _time.sleep(max(1.0, sec))
                    if not bool(schedule_state.get("risk_daily_enabled")):
                        break
                    rc = int(cmd_risk_check(argparse.Namespace(config=config_path, preset=preset, asset_mode=asset_mode)))
                    schedule_state["risk_last_run"] = datetime.now(timezone.utc).isoformat()
                    schedule_state["risk_last_asset_mode"] = asset_mode
                    schedule_state["risk_last_state"] = "done" if rc == 0 else f"rc={rc}"
                    schedule_state["risk_last_error"] = None
                except Exception as e:
                    schedule_state["risk_last_run"] = datetime.now(timezone.utc).isoformat()
                    schedule_state["risk_last_state"] = "error"
                    schedule_state["risk_last_error"] = str(e)
                    # wait briefly before retrying schedule calc
                    import time as _time
                    _time.sleep(5)

        threading.Thread(target=_loop, daemon=True).start()

    @app.post("/api/actions/run")
    async def run_action(req: Request):
        require_token(req)
        body = await req.json()
        kind = str(body.get("kind") or "").strip()
        preset = body.get("preset")
        place_orders = bool(body.get("place_orders", True))
        wait_until_configured = bool(body.get("wait_until_configured", False))
        queue_daily_risk_check = bool(body.get("queue_daily_risk_check", True))
        asset_mode = str(body.get("asset_mode") or "both")

        if kind not in ("rebalance", "risk-check"):
            raise HTTPException(status_code=400, detail="kind must be rebalance or risk-check")

        job_id = str(uuid.uuid4())
        repo_dir = Path(config_path).resolve().parent.parent
        log_dir = repo_dir / "data" / "action_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{job_id}.log"
        action_jobs[job_id] = {
            "state": "starting",
            "kind": kind,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "wait_until_configured": wait_until_configured,
            "place_orders": place_orders,
            "log_path": str(log_path),
        }

        def _run():
            class _JobWriter(io.TextIOBase):
                def __init__(self, file_path: Path):
                    self._buf = ""
                    self._file_path = file_path

                def _emit(self, line: str):
                    action_jobs[job_id]["message"] = line
                    logs = action_jobs[job_id].setdefault("logs", [])
                    logs.append(line)
                    # Keep a generous in-memory tail for UI while writing full log to disk.
                    if len(logs) > 2000:
                        del logs[:-2000]
                    try:
                        with self._file_path.open("a", encoding="utf-8") as f:
                            f.write(line + "\n")
                    except Exception:
                        pass

                def write(self, s):
                    try:
                        txt = str(s)
                    except Exception:
                        return 0
                    self._buf += txt
                    while "\n" in self._buf:
                        line, self._buf = self._buf.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        self._emit(line)
                    return len(txt)

                def flush(self):
                    if self._buf.strip():
                        self._emit(self._buf.strip())
                    self._buf = ""

            try:
                from tradebot.cli import cmd_rebalance
                from tradebot.commands.risk_check import cmd_risk_check

                action_jobs[job_id]["state"] = "running"
                w = _JobWriter(log_path)
                with contextlib.redirect_stdout(w), contextlib.redirect_stderr(w):
                    if kind == "rebalance":
                        if wait_until_configured:
                            # queue persistent weekly scheduler instead of one-shot sleep/run
                            _ensure_weekly_rebalance_scheduler(preset, place_orders=place_orders)
                            if queue_daily_risk_check:
                                _ensure_daily_risk_scheduler(preset)
                            action_jobs[job_id]["wait_until"] = "configured schedule"
                            rc = 0
                        else:
                            ns = argparse.Namespace(config=config_path, place_orders=place_orders, wait_until=None, preset=preset, asset_mode=asset_mode)
                            rc = int(cmd_rebalance(ns))
                    else:
                        ns = argparse.Namespace(config=config_path, preset=preset, asset_mode=asset_mode)
                        rc = int(cmd_risk_check(ns))
                w.flush()
                action_jobs[job_id]["state"] = "done"
                action_jobs[job_id]["rc"] = rc
                action_jobs[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()
            except Exception as e:
                action_jobs[job_id]["state"] = "error"
                action_jobs[job_id]["error"] = str(e)
                action_jobs[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()

        threading.Thread(target=_run, daemon=True).start()
        return {"ok": True, "job_id": job_id}

    @app.get("/api/actions/status")
    def action_status(job_id: str):
        return action_jobs.get(job_id) or {"state": "missing"}

    @app.get("/api/actions/latest")
    def actions_latest(kind: str = "rebalance"):
        rows = [dict(job_id=k, **v) for k, v in action_jobs.items() if str(v.get("kind")) == str(kind)]
        if not rows:
            return {"state": "missing"}
        rows.sort(key=lambda x: str(x.get("started_at") or ""), reverse=True)
        return rows[0]

    @app.get("/api/actions/schedule-status")
    def actions_schedule_status():
        return dict(schedule_state)

    CRON_TAG_REB = "# SEATRADER_REBALANCE"
    CRON_TAG_RISK = "# SEATRADER_RISK"
    CRON_TAG_REB_EQ = "# SEATRADER_REBALANCE_EQ"
    CRON_TAG_REB_CR = "# SEATRADER_REBALANCE_CR"
    CRON_TAG_RISK_EQ = "# SEATRADER_RISK_EQ"
    CRON_TAG_RISK_CR = "# SEATRADER_RISK_CR"

    def _cron_get_lines() -> list[str]:
        try:
            out = subprocess.check_output(["crontab", "-l"], text=True, stderr=subprocess.STDOUT)
            return out.splitlines()
        except subprocess.CalledProcessError as e:
            txt = str(getattr(e, "output", "") or "")
            if "no crontab for" in txt.lower():
                return []
            raise

    def _cron_write_lines(lines: list[str]) -> None:
        payload = "\n".join(lines).rstrip() + "\n"
        subprocess.run(["crontab", "-"], input=payload, text=True, check=True)

    def _dow_num(day: str) -> str:
        m = {"SUN":"0", "MON":"1", "TUE":"2", "WED":"3", "THU":"4", "FRI":"5", "SAT":"6"}
        return m.get(str(day).upper(), "1")

    @app.get("/api/scheduler/cron-status")
    def cron_status():
        lines = _cron_get_lines()
        reb = [ln for ln in lines if (CRON_TAG_REB in ln or CRON_TAG_REB_EQ in ln or CRON_TAG_REB_CR in ln)]
        risk = [ln for ln in lines if (CRON_TAG_RISK in ln or CRON_TAG_RISK_EQ in ln or CRON_TAG_RISK_CR in ln)]

        def _sched(line: str):
            try:
                p = line.split()
                if len(p) < 5:
                    return None
                mm, hh, _dom, _mon, dow = p[:5]
                dow_map = {"0":"SUN","1":"MON","2":"TUE","3":"WED","4":"THU","5":"FRI","6":"SAT"}
                day = dow_map.get(str(dow), str(dow))
                return {"hhmm": f"{int(hh):02d}:{int(mm):02d}", "day": day}
            except Exception:
                return None

        def _line_for_tag(tag: str):
            for ln in lines:
                if tag in ln:
                    return ln
            return None

        def _read_text(path: Path):
            try:
                if path.exists():
                    return path.read_text().strip() or None
            except Exception:
                return None
            return None

        reb_s = _sched(reb[0]) if reb else None
        risk_s = _sched(risk[0]) if risk else None

        reb_eq_line = _line_for_tag(CRON_TAG_REB_EQ)
        reb_cr_line = _line_for_tag(CRON_TAG_REB_CR)
        risk_eq_line = _line_for_tag(CRON_TAG_RISK_EQ)
        risk_cr_line = _line_for_tag(CRON_TAG_RISK_CR)

        repo = Path(config_path).resolve().parent.parent
        stamp_reb_eq = _read_text(repo / "data" / "cron_last_run_rebalance_equities.txt")
        stamp_reb_cr = _read_text(repo / "data" / "cron_last_run_rebalance_crypto.txt")
        stamp_risk_eq = _read_text(repo / "data" / "cron_last_run_risk_equities.txt")
        stamp_risk_cr = _read_text(repo / "data" / "cron_last_run_risk_crypto.txt")

        return {
            "ok": True,
            "enabled": bool(reb or risk),
            "rebalance": reb,
            "risk": risk,
            "rebalance_enabled": bool(reb),
            "risk_enabled": bool(risk),
            "rebalance_schedule": reb_s,
            "risk_schedule": risk_s,
            "rebalance_equities_enabled": bool(reb_eq_line),
            "rebalance_crypto_enabled": bool(reb_cr_line),
            "risk_equities_enabled": bool(risk_eq_line),
            "risk_crypto_enabled": bool(risk_cr_line),
            "rebalance_schedule_equities": _sched(reb_eq_line) if reb_eq_line else None,
            "rebalance_schedule_crypto": _sched(reb_cr_line) if reb_cr_line else None,
            "risk_schedule_equities": _sched(risk_eq_line) if risk_eq_line else None,
            "risk_schedule_crypto": _sched(risk_cr_line) if risk_cr_line else None,
            "rebalance_last_run_equities": stamp_reb_eq,
            "rebalance_last_run_crypto": stamp_reb_cr,
            "risk_last_run_equities": stamp_risk_eq,
            "risk_last_run_crypto": stamp_risk_cr,
        }

    @app.post("/api/scheduler/cron/setup")
    async def cron_setup(req: Request):
        require_token(req)
        cfg = load_config(config_path)
        tags = (CRON_TAG_REB, CRON_TAG_RISK, CRON_TAG_REB_EQ, CRON_TAG_REB_CR, CRON_TAG_RISK_EQ, CRON_TAG_RISK_CR)
        lines = [ln for ln in _cron_get_lines() if not any(t in ln for t in tags)]

        eqs = getattr(cfg.scheduling, "equities", None)
        crs = getattr(cfg.scheduling, "crypto", None)

        eq_reb_freq = str(getattr(eqs, "rebalance_frequency", "weekly"))
        eq_reb_day = str(getattr(eqs, "rebalance_day", getattr(cfg.scheduling, "weekly_rebalance_day", "MON")))
        eq_reb_time = str(getattr(eqs, "rebalance_time_local", getattr(cfg.scheduling, "weekly_rebalance_time_local", "06:35")))

        cr_reb_freq = str(getattr(crs, "rebalance_frequency", "weekly"))
        cr_reb_day = str(getattr(crs, "rebalance_day", getattr(cfg.scheduling, "weekly_rebalance_day", "MON")))
        cr_reb_time = str(getattr(crs, "rebalance_time_local", getattr(cfg.scheduling, "weekly_rebalance_time_local", "06:35")))

        eq_risk_freq = str(getattr(eqs, "risk_check_frequency", "daily"))
        eq_risk_day = str(getattr(eqs, "risk_check_day", getattr(cfg.scheduling, "weekly_rebalance_day", "MON")))
        eq_risk_time = str(getattr(eqs, "risk_check_time_local", getattr(cfg.scheduling, "daily_risk_check_time_local", "18:05")))

        cr_risk_freq = str(getattr(crs, "risk_check_frequency", "daily"))
        cr_risk_day = str(getattr(crs, "risk_check_day", getattr(cfg.scheduling, "weekly_rebalance_day", "MON")))
        cr_risk_time = str(getattr(crs, "risk_check_time_local", getattr(cfg.scheduling, "daily_risk_check_time_local", "18:05")))

        repo = Path(config_path).resolve().parent.parent
        cmd_base = f"cd {repo} && source .venv/bin/activate"
        reb_eq_stamp = str(repo / "data" / "cron_last_run_rebalance_equities.txt")
        reb_cr_stamp = str(repo / "data" / "cron_last_run_rebalance_crypto.txt")
        risk_eq_stamp = str(repo / "data" / "cron_last_run_risk_equities.txt")
        risk_cr_stamp = str(repo / "data" / "cron_last_run_risk_crypto.txt")

        reb_eq_cmd = f"{cmd_base} && date -Iseconds > {reb_eq_stamp} && tradebot rebalance --config {config_path} --place-orders --asset-mode equities"
        reb_cr_cmd = f"{cmd_base} && date -Iseconds > {reb_cr_stamp} && tradebot rebalance --config {config_path} --place-orders --asset-mode crypto"
        risk_eq_cmd = f"{cmd_base} && date -Iseconds > {risk_eq_stamp} && tradebot risk-check --config {config_path} --asset-mode equities"
        risk_cr_cmd = f"{cmd_base} && date -Iseconds > {risk_cr_stamp} && tradebot risk-check --config {config_path} --asset-mode crypto"

        def _line(hhmm: str, day: str | None, cmd: str, tag: str):
            hh, mm = [int(x) for x in str(hhmm).split(":")]
            if day is None:
                return f"{mm} {hh} * * * /bin/bash -lc '{cmd}' {tag}"
            return f"{mm} {hh} * * {_dow_num(day)} /bin/bash -lc '{cmd}' {tag}"

        lines.append(_line(eq_reb_time, None if eq_reb_freq == "daily" else eq_reb_day, reb_eq_cmd, CRON_TAG_REB_EQ))
        lines.append(_line(cr_reb_time, None if cr_reb_freq == "daily" else cr_reb_day, reb_cr_cmd, CRON_TAG_REB_CR))
        lines.append(_line(eq_risk_time, None if eq_risk_freq == "daily" else eq_risk_day, risk_eq_cmd, CRON_TAG_RISK_EQ))
        lines.append(_line(cr_risk_time, None if cr_risk_freq == "daily" else cr_risk_day, risk_cr_cmd, CRON_TAG_RISK_CR))
        _cron_write_lines(lines)
        return {"ok": True}

    @app.post("/api/scheduler/cron/stop")
    async def cron_stop(req: Request):
        require_token(req)
        tags = (CRON_TAG_REB, CRON_TAG_RISK, CRON_TAG_REB_EQ, CRON_TAG_REB_CR, CRON_TAG_RISK_EQ, CRON_TAG_RISK_CR)
        lines = [ln for ln in _cron_get_lines() if not any(t in ln for t in tags)]
        _cron_write_lines(lines)
        return {"ok": True}

    @app.post("/api/scheduler/cron/restart")
    async def cron_restart(req: Request):
        require_token(req)
        await cron_stop(req)
        return await cron_setup(req)

    # Backtest (token-gated)
    @app.post("/api/backtest/start")
    async def backtest_start(req: Request):
        require_token(req)
        body = await req.json()
        job_id = start_backtest(config_path=config_path, params=body)
        return {"ok": True, "job_id": job_id}

    @app.get("/api/backtest/status")
    def backtest_status(job_id: str | None = None):
        job_id = job_id or get_latest_job_id()
        if not job_id:
            return {"state": "missing"}
        return bt_status(job_id)

    @app.get("/api/backtest/result")
    def backtest_result(job_id: str | None = None):
        job_id = job_id or get_latest_job_id()
        if not job_id:
            return {"state": "missing"}
        res = bt_result(job_id)
        return res or {"state": "missing"}

    @app.get("/api/backtest/jobs")
    def backtest_jobs(limit: int = 20):
        return bt_list_jobs(limit=limit)

    @app.get("/api/backtest/symbol")
    def backtest_symbol(job_id: str, symbol: str):
        # Return price series for symbol + SPY, plus realized trades markers for that symbol.
        res = bt_result(job_id)
        if not res:
            return {"ok": False, "error": "missing job"}
        params = (res.get("params") or {})
        start = str(params.get("start"))
        end = str(params.get("end"))
        symbol = str(symbol).strip().upper()

        from tradebot.dashboard.backtest_symbol import fetch_close_series, to_points

        sym_series = fetch_close_series(symbol, start, end)
        spy_series = fetch_close_series("SPY", start, end)

        trades = [t for t in (res.get("trades") or []) if str(t.get("symbol") or "").upper() == symbol]
        events = [e for e in (res.get("events") or []) if str(e.get("symbol") or "").upper() == symbol]

        # markers by date from events (preferred)
        markers = []
        for e in events:
            if e.get("type") in ("buy", "sell"):
                markers.append({
                    "date": e.get("date"),
                    "type": e.get("type"),
                    "qty": e.get("qty"),
                    "price": e.get("price"),
                    "notional": e.get("notional"),
                    "reason": e.get("reason"),
                    "pnl": e.get("pnl"),
                })

        # Back-compat fallback: if no events recorded, derive minimal markers from realized trades
        if not markers:
            for t in trades:
                if t.get("entry_date"):
                    markers.append({"date": t.get("entry_date"), "type": "buy", "qty": t.get("qty"), "price": t.get("entry_price"), "pnl": None})
                if t.get("exit_date"):
                    markers.append({"date": t.get("exit_date"), "type": "sell", "qty": t.get("qty"), "price": t.get("exit_price"), "pnl": t.get("pnl"), "reason": t.get("reason")})

        return {
            "ok": True,
            "symbol": symbol,
            "start": start,
            "end": end,
            "series": to_points(sym_series),
            "spy": to_points(spy_series),
            "trades": trades,
            "events": events,
            "markers": markers,
        }

    @app.get("/api/backtest/presets")
    def backtest_presets():
        from tradebot.dashboard.presets import load_presets
        return {"presets": load_presets()}

    @app.post("/api/backtest/presets/save")
    async def backtest_presets_save(req: Request):
        require_token(req)
        body = await req.json()
        name = body.get("name")
        params = body.get("params") or {}
        from tradebot.dashboard.presets import save_preset
        save_preset(name, params)
        return {"ok": True}

    @app.post("/api/backtest/presets/apply-live")
    async def backtest_presets_apply_live(req: Request):
        require_token(req)
        body = await req.json()
        name = body.get("name")
        from tradebot.dashboard.presets import get_preset
        p = get_preset(name)
        if not p:
            return {"ok": False, "error": "unknown preset"}
        params = p.get("params") or {}

        # Unified preset application: set config.active_preset.
        # Also write mapped overlap fields directly into config so values are visible immediately
        # even when preset bot patch is stale.
        from pathlib import Path
        import yaml
        from tradebot.dashboard.presets import _bt_to_bot_patch

        def _deep_merge(a: dict, b: dict) -> dict:
            out = dict(a or {})
            for k, v in (b or {}).items():
                if isinstance(v, dict) and isinstance(out.get(k), dict):
                    out[k] = _deep_merge(out.get(k) or {}, v)
                else:
                    out[k] = v
            return out

        cfg_path = Path(config_path)
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        cfg["active_preset"] = name

        # derive bot patch from backtest params on apply
        bt = (p.get("params") or {})
        derived = _bt_to_bot_patch(bt)
        cfg = _deep_merge(cfg, derived)

        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        return {"ok": True, "active_preset": name}

    @app.post("/api/backtest/clear-cache")
    async def backtest_clear_cache(req: Request):
        require_token(req)
        # Delete cached bars (safe to regenerate)
        cache_dir = Path("data/cache/bars")
        if cache_dir.exists():
            # manual recursive delete to avoid shelling out
            for p in sorted(cache_dir.glob("**/*"), reverse=True):
                try:
                    if p.is_file() or p.is_symlink():
                        p.unlink()
                    elif p.is_dir():
                        p.rmdir()
                except Exception:
                    pass
        return {"ok": True}

    @app.post("/api/backtest/notify-telegram")
    async def backtest_notify_telegram(req: Request):
        require_token(req)
        body = await req.json()
        text = str((body or {}).get("text") or "").strip()
        if not text:
            return {"ok": False, "error": "missing text"}

        import os
        import requests

        bot_token = os.getenv("SEATRADER_TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("SEATRADER_TELEGRAM_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID")
        if not bot_token or not chat_id:
            return {
                "ok": False,
                "error": "telegram env not configured; set SEATRADER_TELEGRAM_BOT_TOKEN and SEATRADER_TELEGRAM_CHAT_ID"
            }

        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        r = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=15)
        if r.status_code >= 400:
            return {"ok": False, "error": f"telegram send failed: {r.status_code} {r.text[:300]}"}
        return {"ok": True}

    @app.post("/api/backtest/iterations/save")
    async def backtest_iterations_save(req: Request):
        require_token(req)
        body = await req.json()
        from datetime import datetime, timezone
        import uuid

        base = Path("data/backtests/iterations")
        base.mkdir(parents=True, exist_ok=True)

        created_at = datetime.now(timezone.utc).isoformat()
        axis = str((body or {}).get("axis") or "iteration")
        asset_mode = str((body or {}).get("asset_mode") or "both")
        report_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{asset_mode}_{axis}_{uuid.uuid4().hex[:8]}"
        payload = {
            "id": report_id,
            "created_at": created_at,
            **(body or {}),
        }
        (base / f"{report_id}.json").write_text(json.dumps(payload, indent=2))
        return {"ok": True, "id": report_id}

    @app.get("/api/backtest/iterations/list")
    def backtest_iterations_list(limit: int = 50):
        base = Path("data/backtests/iterations")
        if not base.exists():
            return {"reports": []}
        files = sorted(base.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[: max(1, min(int(limit), 200))]
        out = []
        for p in files:
            try:
                o = json.loads(p.read_text())
                out.append({
                    "id": o.get("id") or p.stem,
                    "created_at": o.get("created_at"),
                    "axis": o.get("axis"),
                    "asset_mode": o.get("asset_mode"),
                    "count": len(o.get("rows") or []),
                    "best": o.get("best"),
                })
            except Exception:
                continue
        return {"reports": out}

    @app.get("/api/backtest/iterations/get")
    def backtest_iterations_get(id: str):
        base = Path("data/backtests/iterations")
        p = base / f"{id}.json"
        if not p.exists():
            return {"ok": False, "error": "missing report"}
        try:
            o = json.loads(p.read_text())
        except Exception:
            return {"ok": False, "error": "failed to read report"}
        return {"ok": True, "report": o}

    @app.get("/api/artifacts")
    def artifacts():
        base = Path("data")
        return {
            "last_account": _read_json(base / "last_account.json"),
            "last_rebalance": _read_json(base / "last_rebalance.json"),
            "last_risk_check": _read_json(base / "last_risk_check.json"),
            "last_placed_orders": _read_json(base / "last_placed_orders.json"),
        }

    @app.get("/api/equity-curve")
    def equity_curve(limit: int = 500):
        path = Path("data/equity_curve.jsonl")
        if not path.exists():
            return []
        lines = path.read_text().strip().splitlines()
        tail = lines[-max(0, min(limit, 5000)) :]
        out = []
        for ln in tail:
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
        return out

    @app.get("/api/live-ledger/runs")
    def live_ledger_runs_api(limit: int = 200):
        return {"runs": live_ledger_runs(limit=limit)}

    @app.get("/api/live-ledger/events")
    def live_ledger_events_api(limit: int = 500, run_id: str | None = None, kind: str | None = None):
        return {"events": live_ledger_events(limit=limit, run_id=run_id, kind=kind)}

    @app.get("/api/live-ledger/metrics")
    def live_ledger_metrics_api(days: int = 90):
        path = Path("data/equity_curve.jsonl")
        curve: list[dict] = []
        if path.exists():
            for ln in path.read_text().splitlines():
                if not ln.strip():
                    continue
                try:
                    curve.append(json.loads(ln))
                except Exception:
                    continue

        curve.sort(key=lambda x: str(x.get("ts") or ""))
        events = live_ledger_events(limit=5000)

        def _equity_at_or_before(ts_iso: str | None):
            if not ts_iso:
                return None
            v = None
            for row in curve:
                rts = str(row.get("ts") or "")
                if rts and rts <= ts_iso:
                    try:
                        v = float(row.get("equity"))
                    except Exception:
                        pass
                else:
                    break
            return v

        latest_eq = float(curve[-1].get("equity")) if curve else None
        latest_cash = float(curve[-1].get("cash")) if curve else None
        start_eq = float(curve[0].get("equity")) if curve else None

        max_dd = None
        cur_dd = None
        peak = None
        if curve:
            peak = float(curve[0].get("equity") or 0.0)
            dds = []
            for r in curve:
                eq = float(r.get("equity") or 0.0)
                if eq > peak:
                    peak = eq
                dd = 0.0 if peak <= 0 else (peak - eq) / peak
                dds.append(dd)
            max_dd = max(dds) if dds else None
            cur_dd = dds[-1] if dds else None

        now_ts = str(curve[-1].get("ts")) if curve else None
        d1 = None
        d7 = None
        d30 = None
        if now_ts:
            try:
                now_dt = datetime.fromisoformat(now_ts.replace("Z", "+00:00"))
                e1 = _equity_at_or_before((now_dt - timedelta(days=1)).isoformat())
                e7 = _equity_at_or_before((now_dt - timedelta(days=7)).isoformat())
                e30 = _equity_at_or_before((now_dt - timedelta(days=30)).isoformat())
                if latest_eq is not None and e1 not in (None, 0):
                    d1 = (latest_eq / e1) - 1.0
                if latest_eq is not None and e7 not in (None, 0):
                    d7 = (latest_eq / e7) - 1.0
                if latest_eq is not None and e30 not in (None, 0):
                    d30 = (latest_eq / e30) - 1.0
            except Exception:
                pass

        total_return = None
        if latest_eq is not None and start_eq not in (None, 0):
            total_return = (latest_eq / start_eq) - 1.0

        event_counts: dict[str, int] = defaultdict(int)
        buy_notional = 0.0
        sell_notional = 0.0
        for e in events:
            et = str(e.get("event_type") or "unknown")
            event_counts[et] += 1
            try:
                n = float(e.get("notional_usd") or 0.0)
            except Exception:
                n = 0.0
            if "buy" in et:
                buy_notional += n
            if "sell" in et:
                sell_notional += n

        turnover = None
        if latest_eq and latest_eq > 0:
            turnover = (buy_notional + sell_notional) / latest_eq

        daily_curve_last: dict[str, dict] = {}
        for r in curve:
            ts = str(r.get("ts") or "")
            day = ts[:10] if len(ts) >= 10 else ""
            if not day:
                continue
            daily_curve_last[day] = r

        daily_ev: dict[str, dict] = defaultdict(lambda: {
            "order_count": 0,
            "buy_notional": 0.0,
            "sell_notional": 0.0,
            "exit_signal_count": 0,
            "liquidation_count": 0,
        })
        for e in events:
            ts = str(e.get("ts") or "")
            day = ts[:10] if len(ts) >= 10 else ""
            if not day:
                continue
            et = str(e.get("event_type") or "")
            row = daily_ev[day]
            if et == "order_submitted":
                row["order_count"] += 1
            if et == "exit_signal":
                row["exit_signal_count"] += 1
            if et == "liquidation":
                row["liquidation_count"] += 1

        # Buy/sell notional in daily summary should reflect actual FILLS only
        # (not planned orders and not merely submitted orders).
        try:
            from alpaca.trading.enums import QueryOrderStatus

            closed = _orders_by_status(QueryOrderStatus.CLOSED, limit=500)
            fills = [o for o in closed if str(o.get("filled_at") or "").strip() not in ("", "None")]
            for o in fills:
                ts = str(o.get("filled_at") or "")
                day = ts[:10] if len(ts) >= 10 else ""
                if not day:
                    continue
                row = daily_ev[day]
                side = str(o.get("side") or "").lower()
                try:
                    n = float(o.get("filled_notional_usd") or 0.0)
                except Exception:
                    n = 0.0
                if n <= 0:
                    try:
                        fq = float(o.get("filled_qty") or 0.0)
                    except Exception:
                        fq = 0.0
                    try:
                        fpx = float(o.get("filled_avg_price") or 0.0)
                    except Exception:
                        fpx = 0.0
                    n = max(0.0, fq * fpx)
                if side == "buy":
                    row["buy_notional"] += n
                elif side == "sell":
                    row["sell_notional"] += n
        except Exception:
            pass

        all_days = sorted(set(daily_curve_last.keys()) | set(daily_ev.keys()))
        if days > 0:
            all_days = all_days[-max(1, min(int(days), 3650)) :]

        daily_rows: list[dict] = []
        prev_eq = None
        for day in all_days:
            c = daily_curve_last.get(day) or {}
            ev = daily_ev.get(day) or {}
            eq = c.get("equity")
            cash = c.get("cash")
            eq_delta = None
            if eq is not None and prev_eq is not None:
                try:
                    eq_delta = float(eq) - float(prev_eq)
                except Exception:
                    eq_delta = None
            if eq is not None:
                prev_eq = eq
            daily_rows.append({
                "day": day,
                "equity": eq,
                "cash": cash,
                "equity_delta": eq_delta,
                "order_count": ev.get("order_count", 0),
                "buy_notional": ev.get("buy_notional", 0.0),
                "sell_notional": ev.get("sell_notional", 0.0),
                "exit_signal_count": ev.get("exit_signal_count", 0),
                "liquidation_count": ev.get("liquidation_count", 0),
            })

        chart = []
        peak2 = None
        for r in curve[-max(1, min(int(days) * 24, 5000)) :] if curve else []:
            try:
                eq = float(r.get("equity") or 0.0)
            except Exception:
                continue
            if peak2 is None or eq > peak2:
                peak2 = eq
            dd = 0.0 if (not peak2 or peak2 <= 0) else (peak2 - eq) / peak2
            chart.append({"ts": r.get("ts"), "equity": eq, "drawdown": dd})

        return {
            "summary": {
                "latest_equity": latest_eq,
                "latest_cash": latest_cash,
                "start_equity": start_eq,
                "total_return": total_return,
                "return_1d": d1,
                "return_7d": d7,
                "return_30d": d30,
                "current_drawdown": cur_dd,
                "max_drawdown": max_dd,
                "order_submitted_count": int(event_counts.get("order_submitted", 0)),
                "exit_signal_count": int(event_counts.get("exit_signal", 0)),
                "liquidation_count": int(event_counts.get("liquidation", 0)),
                "buy_notional_usd": buy_notional,
                "sell_notional_usd": sell_notional,
                "turnover_vs_equity": turnover,
            },
            "daily": daily_rows,
            "chart": chart,
        }

    @app.get("/api/benchmarks")
    def benchmarks(start: str, end: str):
        from tradebot.dashboard.benchmarks import get_sp500_series, get_spy_series, normalize

        spy = normalize(get_spy_series(start, end))
        spx = normalize(get_sp500_series(start, end))
        return {"SPY": spy, "SP500": spx}

    return app
