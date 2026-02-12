from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from datetime import datetime, timedelta, timezone

from tradebot.adapters.bars import fetch_crypto_bars_range, fetch_stock_bars_range
from tradebot.adapters.alpaca_client import make_alpaca_clients
from tradebot.util.config import load_config
from tradebot.util.env import load_env
from tradebot.universe.sp500 import get_sp500_symbols
from tradebot.universe.crypto import list_tradable_crypto
from tradebot.backtest.engine import BacktestParams, run_backtest
from tradebot.backtest.cache import load_cached_frames, save_cached_frames


BASE = Path("data/backtests")
LATEST_PATH = BASE / "latest_job_id.txt"


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def _read_json_safe(path: Path, retries: int = 5, delay_s: float = 0.05) -> dict | None:
    """Best-effort JSON reader resilient to transient partial writes."""
    for _ in range(max(1, retries)):
        try:
            txt = path.read_text()
            if not txt.strip():
                raise ValueError("empty json")
            return json.loads(txt)
        except Exception:
            time.sleep(delay_s)
    return None


def start_backtest(*, config_path: str, params: dict) -> str:
    job_id = str(uuid.uuid4())
    job_dir = BASE / job_id
    status_path = job_dir / "status.json"
    result_path = job_dir / "result.json"

    _write(status_path, {"state": "starting", "progress": 0, "total": 1})

    def run():
        try:
            _write(status_path, {"state": "fetching_data", "progress": 0, "total": 1})
            cfg = load_config(config_path)
            env = load_env()
            clients = make_alpaca_clients(env)

            # Universe
            sp500 = set(get_sp500_symbols())
            assets = clients.trading.get_all_assets()
            tradable_eq = sorted({a.symbol for a in assets if getattr(a, "tradable", False) and getattr(a, "status", None) == "active" and a.symbol in sp500})
            # Ensure SPY is available for regime filter strategies
            if "SPY" not in tradable_eq:
                tradable_eq.append("SPY")

            crypto_assets = list_tradable_crypto(clients.trading)
            tradable_cr = sorted({a.symbol for a in crypto_assets if a.symbol.endswith("/USD")})
            # Ensure BTC/USD available for regime filter strategies
            if "BTC/USD" not in tradable_cr:
                tradable_cr.append("BTC/USD")

            # Single-symbol mode
            sym = params.get("symbol")
            if params.get("universe_mode") == "single" and sym:
                sym = str(sym).strip().upper()
                if "/" in sym:
                    tradable_cr = [sym]
                    tradable_eq = []
                else:
                    tradable_eq = [sym]
                    tradable_cr = []

            # Fetch bars: cover evaluation window plus warmup for indicators
            start_dt = datetime.fromisoformat(params["start"]).replace(tzinfo=timezone.utc)
            end_dt = datetime.fromisoformat(params["end"]).replace(tzinfo=timezone.utc) + timedelta(days=1)
            warmup_start = start_dt - timedelta(days=cfg.signals.lookback_days)

            asset_mode = params.get("asset_mode", "both")

            # Fetch bars (lookback includes MA history) with caching
            cache_start = warmup_start.date().isoformat()
            cache_end = end_dt.date().isoformat()

            stock_bars: dict[str, pd.DataFrame] = {}
            crypto_bars: dict[str, pd.DataFrame] = {}

            if asset_mode in ("both", "equities"):
                stock_bars = load_cached_frames("stocks", tradable_eq, cfg.signals.lookback_days, cache_start, cache_end)
                if stock_bars is None:
                    stock_bars = {}
                    chunk = 100
                    for i in range(0, len(tradable_eq), chunk):
                        syms = tradable_eq[i : i + chunk]
                        stock_bars.update(fetch_stock_bars_range(clients.stocks, syms, start=warmup_start, end=end_dt))
                        _write(status_path, {"state": "fetching_data", "progress": min(i + chunk, len(tradable_eq)), "total": len(tradable_eq)})
                    save_cached_frames("stocks", tradable_eq, cfg.signals.lookback_days, cache_start, cache_end, stock_bars)

            if asset_mode in ("both", "crypto"):
                crypto_bars = load_cached_frames("crypto", tradable_cr, cfg.signals.lookback_days, cache_start, cache_end)
                if crypto_bars is None:
                    crypto_bars = {}
                    chunkc = 50
                    for i in range(0, len(tradable_cr), chunkc):
                        syms = tradable_cr[i : i + chunkc]
                        crypto_bars.update(fetch_crypto_bars_range(clients.crypto, syms, start=warmup_start, end=end_dt))
                        _write(status_path, {"state": "fetching_crypto", "progress": min(i + chunkc, len(tradable_cr)), "total": len(tradable_cr)})
                    save_cached_frames("crypto", tradable_cr, cfg.signals.lookback_days, cache_start, cache_end, crypto_bars)

            # Normalize stop-loss input: allow UI to pass 5 meaning 5%
            if params.get("per_asset_stop_loss_pct") is not None:
                try:
                    v = float(params.get("per_asset_stop_loss_pct"))
                    if v > 1.0:
                        v = v / 100.0
                    params["per_asset_stop_loss_pct"] = v
                except Exception:
                    params["per_asset_stop_loss_pct"] = None

            # Run backtest
            p = BacktestParams(**params)

            def prog(done, total):
                _write(status_path, {"state": "running", "progress": done, "total": total})

            intraday_cb = None
            intraday_limit_touch_cb = None
            risk_intraday_cb = None
            if getattr(p, "execution_time_mode", "daily") == "intraday":
                from tradebot.backtest.intraday import IntradayPriceProvider

                prov = IntradayPriceProvider(
                    stocks_client=clients.stocks,
                    crypto_client=clients.crypto,
                    exec_time_local=p.execution_time_local,
                    tz=p.execution_tz,
                )
                risk_prov = IntradayPriceProvider(
                    stocks_client=clients.stocks,
                    crypto_client=clients.crypto,
                    exec_time_local=getattr(p, "risk_check_time_local", "12:30"),
                    tz=p.execution_tz,
                )

                # Only apply intraday pricing on rebalance days.
                # Precompute the rebalance day set once (avoid O(days) work per symbol call).
                start_d = pd.to_datetime(p.start)
                end_d = pd.to_datetime(p.end)
                all_days = pd.date_range(start_d, end_d, freq="D")
                if p.rebalance == "daily":
                    reb_days = set(all_days)
                else:
                    day_map = {"MON":0, "TUE":1, "WED":2, "THU":3, "FRI":4, "SAT":5, "SUN":6}
                    wd = day_map.get(str(getattr(p, "rebalance_day", "MON")).upper(), 0)
                    reb_days = set([d for d in all_days if d.weekday() == wd])

                def intraday_cb(sym, day):
                    if day not in reb_days:
                        return None
                    return prov.price(sym, day)

                def intraday_limit_touch_cb(sym, day, side, limit_px):
                    return prov.limit_touched(sym, day, side, float(limit_px))

                def risk_intraday_cb(sym, day):
                    return risk_prov.price(sym, day)

            res = run_backtest(
                stock_bars=stock_bars,
                crypto_bars=crypto_bars,
                stock_universe=tradable_eq,
                crypto_universe=tradable_cr,
                cfg=cfg,
                params=p,
                progress_cb=prog,
                intraday_price_cb=intraday_cb,
                intraday_limit_touch_cb=intraday_limit_touch_cb,
                risk_intraday_price_cb=risk_intraday_cb,
            )

            _write(result_path, {"job_id": job_id, **asdict(res)})
            _write(status_path, {"state": "done", "progress": 1, "total": 1})
        except Exception as e:
            import traceback

            _write(status_path, {"state": "error", "error": str(e), "traceback": traceback.format_exc()})

    # record latest job id
    BASE.mkdir(parents=True, exist_ok=True)
    LATEST_PATH.write_text(job_id)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return job_id


def get_latest_job_id() -> str | None:
    try:
        if LATEST_PATH.exists():
            return LATEST_PATH.read_text().strip() or None
    except Exception:
        return None
    return None


def get_status(job_id: str) -> dict:
    p = BASE / job_id / "status.json"
    if not p.exists():
        return {"state": "missing"}
    st = _read_json_safe(p)
    return st or {"state": "reading"}


def get_result(job_id: str) -> dict | None:
    p = BASE / job_id / "result.json"
    if not p.exists():
        return None
    return _read_json_safe(p)


def list_jobs(limit: int = 20) -> list[dict]:
    BASE.mkdir(parents=True, exist_ok=True)
    jobs: list[dict] = []
    for d in BASE.iterdir():
        if not d.is_dir():
            continue
        status_p = d / "status.json"
        if not status_p.exists():
            continue
        st = _read_json_safe(status_p)
        if not st:
            continue
        item: dict = {"job_id": d.name, **st}

        # Attach lightweight result summary for easier scanning
        res_p = d / "result.json"
        if res_p.exists():
            try:
                res = _read_json_safe(res_p)
                if not res:
                    raise ValueError("result not readable yet")
                m = (res or {}).get("metrics") or {}
                p = (res or {}).get("params") or {}
                item["result_metrics"] = {
                    "return": m.get("return"),
                    "cagr": m.get("cagr"),
                    "sharpe": m.get("sharpe"),
                    "max_drawdown": m.get("max_drawdown"),
                    "end_equity": m.get("end_equity"),
                }
                item["result_params"] = {
                    "strategy_id": p.get("strategy_id"),
                    "asset_mode": p.get("asset_mode"),
                    "rebalance": p.get("rebalance"),
                }
            except Exception:
                pass

        jobs.append(item)

    # sort by mtime desc
    jobs.sort(key=lambda x: (BASE / x["job_id"] / "status.json").stat().st_mtime, reverse=True)
    return jobs[: max(1, min(limit, 200))]
