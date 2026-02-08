from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from tradebot.dashboard.auth import check_token
from tradebot.dashboard.actions import require_token
from tradebot.dashboard.config_api import load_config_file, save_config_file, validate_config_payload
from tradebot.backtest.job import start_backtest, get_status as bt_status, get_result as bt_result, list_jobs as bt_list_jobs, get_latest_job_id
from pathlib import Path
from tradebot.util.config import load_config
from tradebot.util.env import load_env
from tradebot.adapters.alpaca_client import make_alpaca_clients
from tradebot.util.state import load_state
from tradebot.adapters.bars import fetch_stock_bars, fetch_crypto_bars


def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def create_app(*, config_path: str) -> FastAPI:
    app = FastAPI(title="tradebot dashboard")

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

    @app.get("/api/account")
    def account():
        env = load_env()
        clients = make_alpaca_clients(env)
        acct = clients.trading.get_account()
        return {
            "account_number": acct.account_number,
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(getattr(acct, "buying_power", 0.0) or 0.0),
            "paper": bool(env.paper),
        }

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

    def _orders_by_status(status, limit: int = 500):
        env = load_env()
        clients = make_alpaca_clients(env)
        from alpaca.trading.requests import GetOrdersRequest

        req = GetOrdersRequest(status=status, limit=limit)
        orders = clients.trading.get_orders(filter=req)

        out = []
        for o in orders:
            out.append(
                {
                    "id": str(getattr(o, "id", "")),
                    "symbol": getattr(o, "symbol", ""),
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
        return out

    @app.get("/api/open-orders")
    def open_orders():
        from alpaca.trading.enums import QueryOrderStatus

        return _orders_by_status(QueryOrderStatus.OPEN, limit=500)

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
        # markers by date
        markers = []
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

        # Apply overlapping params to live config
        # (Backtest has more knobs than live; we only map safe overlaps.)
        from pathlib import Path
        import yaml

        cfg_path = Path(config_path)
        cfg = yaml.safe_load(cfg_path.read_text()) or {}

        if params.get("strategy_id"):
            cfg["strategy_id"] = params.get("strategy_id")

        if params.get("per_asset_stop_loss_pct") is not None:
            cfg.setdefault("risk", {})
            cfg["risk"]["per_asset_stop_loss_pct"] = params.get("per_asset_stop_loss_pct")

        # Optional: map backtest portfolio_dd_stop to live freeze threshold if user wants
        # (not done automatically; different semantics).

        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        return {"ok": True}

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

    @app.get("/api/benchmarks")
    def benchmarks(start: str, end: str):
        from tradebot.dashboard.benchmarks import get_sp500_series, get_spy_series, normalize

        spy = normalize(get_spy_series(start, end))
        spx = normalize(get_sp500_series(start, end))
        return {"SPY": spy, "SP500": spx}

    return app
