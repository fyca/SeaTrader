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
from tradebot.backtest.engine import BacktestParams, BacktestStopped, run_backtest
from tradebot.backtest.cache import load_cached_frames, save_cached_frames


BASE = Path("data/backtests")
LATEST_PATH = BASE / "latest_job_id.txt"
_STOP_EVENTS: dict[str, threading.Event] = {}


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
    debug_log_path = job_dir / "debug.log"
    trace_dir = job_dir / "trace"
    trace_run_path = trace_dir / "run.jsonl"
    trace_symbols_dir = trace_dir / "symbols"

    _write(status_path, {"state": "starting", "progress": 0, "total": 1})
    stop_event = threading.Event()
    _STOP_EVENTS[job_id] = stop_event

    def _trace(event: str, **fields) -> None:
        try:
            ts = datetime.now(timezone.utc).isoformat()
            rec = {"ts": ts, "event": str(event)}
            rec.update(fields or {})
            trace_dir.mkdir(parents=True, exist_ok=True)
            with trace_run_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
            sym = rec.get("symbol")
            if sym is not None:
                s = str(sym).strip().upper().replace("/", "_")
                if s:
                    trace_symbols_dir.mkdir(parents=True, exist_ok=True)
                    with (trace_symbols_dir / f"{s}.jsonl").open("a", encoding="utf-8") as sf:
                        sf.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
        except Exception:
            pass

    def _log(msg: str, **fields) -> None:
        try:
            job_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).isoformat()
            if fields:
                extras = " ".join([f"{k}={fields[k]}" for k in sorted(fields.keys())])
                line = f"[{ts}] {msg} | {extras}"
            else:
                line = f"[{ts}] {msg}"
            with debug_log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            _trace(msg, **fields)
        except Exception:
            pass

    def run():
        try:
            t_job_start = time.perf_counter()
            _log("backtest_started", config_path=config_path)
            t_fetch_start = t_job_start
            _write(status_path, {"state": "fetching_data", "progress": 0, "total": 1})
            _log("phase", state="fetching_data")
            cfg = load_config(config_path)
            # Optional backtest-only liquidity override from dashboard controls.
            try:
                adv_override = params.get("min_avg_crypto_dollar_volume_20d")
                if adv_override is not None:
                    cfg.limits.min_avg_crypto_dollar_volume_20d = float(adv_override)
            except Exception:
                pass
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

            _log("universe_resolved", equities=len(tradable_eq), crypto=len(tradable_cr), mode=params.get("asset_mode", "both"))

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
            stock_cache_hit = False
            crypto_cache_hit = False

            if asset_mode in ("both", "equities"):
                stock_bars = load_cached_frames("stocks", tradable_eq, cfg.signals.lookback_days, cache_start, cache_end)
                if stock_bars is None:
                    _log("stock_cache", hit=False, symbols=len(tradable_eq))
                    stock_bars = {}
                    chunk = 100
                    for i in range(0, len(tradable_eq), chunk):
                        syms = tradable_eq[i : i + chunk]
                        stock_bars.update(fetch_stock_bars_range(clients.stocks, syms, start=warmup_start, end=end_dt))
                        _write(status_path, {"state": "fetching_data", "progress": min(i + chunk, len(tradable_eq)), "total": len(tradable_eq)})
                        _log("fetch_stocks_chunk", done=min(i + chunk, len(tradable_eq)), total=len(tradable_eq))
                    save_cached_frames("stocks", tradable_eq, cfg.signals.lookback_days, cache_start, cache_end, stock_bars)
                    _log("stock_cache_saved", frames=len(stock_bars))
                else:
                    stock_cache_hit = True
                    _log("stock_cache", hit=True, frames=len(stock_bars))

            if asset_mode in ("both", "crypto"):
                crypto_bars = load_cached_frames("crypto", tradable_cr, cfg.signals.lookback_days, cache_start, cache_end)
                if crypto_bars is None:
                    _log("crypto_cache", hit=False, symbols=len(tradable_cr))
                    crypto_bars = {}
                    chunkc = 50
                    for i in range(0, len(tradable_cr), chunkc):
                        syms = tradable_cr[i : i + chunkc]
                        crypto_bars.update(fetch_crypto_bars_range(clients.crypto, syms, start=warmup_start, end=end_dt))
                        _write(status_path, {"state": "fetching_crypto", "progress": min(i + chunkc, len(tradable_cr)), "total": len(tradable_cr)})
                        _log("fetch_crypto_chunk", done=min(i + chunkc, len(tradable_cr)), total=len(tradable_cr))
                    save_cached_frames("crypto", tradable_cr, cfg.signals.lookback_days, cache_start, cache_end, crypto_bars)
                    _log("crypto_cache_saved", frames=len(crypto_bars))
                else:
                    crypto_cache_hit = True
                    _log("crypto_cache", hit=True, frames=len(crypto_bars))

            # Normalize stop-loss input: allow UI to pass 5 meaning 5%
            if params.get("per_asset_stop_loss_pct") is not None:
                try:
                    v = float(params.get("per_asset_stop_loss_pct"))
                    if v > 1.0:
                        v = v / 100.0
                    params["per_asset_stop_loss_pct"] = v
                except Exception:
                    params["per_asset_stop_loss_pct"] = None

            t_fetch_end = time.perf_counter()

            # Run backtest
            debug_verbose = bool(params.get("debug_verbose", False))
            p_params = dict(params)
            p_params.pop("debug_verbose", None)
            p = BacktestParams(**p_params)
            _log("phase", state="running", debug_verbose=debug_verbose)
            _log(
                "effective_schedule",
                rebalance=str(getattr(p, "rebalance", "weekly")),
                rebalance_day=str(getattr(p, "rebalance_day", "MON")),
                eq_rebalance=str(getattr(p, "rebalance_frequency_equities", None) or getattr(p, "rebalance", "weekly")),
                eq_day=str(getattr(p, "rebalance_day_equities", None) or getattr(p, "rebalance_day", "MON")),
                cr_rebalance=str(getattr(p, "rebalance_frequency_crypto", None) or getattr(p, "rebalance", "weekly")),
                cr_day=str(getattr(p, "rebalance_day_crypto", None) or getattr(p, "rebalance_day", "MON")),
            )
            _last_prog_log = {"done": -1}

            def prog(done, total, current_equity=None):
                st = {"state": "running", "progress": done, "total": total}
                if current_equity is not None:
                    try:
                        st["current_equity"] = float(current_equity)
                    except Exception:
                        pass
                _write(status_path, st)
                last_done = int(_last_prog_log.get("done", -1))
                if (done == total) or (done - last_done >= 10) or (last_done < 0):
                    _log("simulate_progress", done=done, total=total, current_equity=st.get("current_equity"))
                    _last_prog_log["done"] = int(done)

            def dbg(msg, **fields):
                _log(f"engine:{msg}", **fields)

            intraday_cb = None
            intraday_limit_touch_cb = None
            risk_intraday_cb = None
            if getattr(p, "execution_time_mode", "daily") == "intraday":
                _log("intraday_mode_enabled", execution_tz=p.execution_tz)
                from tradebot.backtest.intraday import IntradayPriceProvider

                eq_exec_t = getattr(p, "execution_time_local_equities", None) or p.execution_time_local
                cr_exec_t = getattr(p, "execution_time_local_crypto", None) or p.execution_time_local
                eq_risk_t = getattr(p, "risk_check_time_local_equities", None) or getattr(p, "risk_check_time_local", "12:30")
                cr_risk_t = getattr(p, "risk_check_time_local_crypto", None) or getattr(p, "risk_check_time_local", "12:30")

                prov_eq = IntradayPriceProvider(
                    stocks_client=clients.stocks,
                    crypto_client=clients.crypto,
                    exec_time_local=eq_exec_t,
                    tz=p.execution_tz,
                )
                prov_cr = IntradayPriceProvider(
                    stocks_client=clients.stocks,
                    crypto_client=clients.crypto,
                    exec_time_local=cr_exec_t,
                    tz=p.execution_tz,
                )
                risk_prov_eq = IntradayPriceProvider(
                    stocks_client=clients.stocks,
                    crypto_client=clients.crypto,
                    exec_time_local=eq_risk_t,
                    tz=p.execution_tz,
                )
                risk_prov_cr = IntradayPriceProvider(
                    stocks_client=clients.stocks,
                    crypto_client=clients.crypto,
                    exec_time_local=cr_risk_t,
                    tz=p.execution_tz,
                )

                # Only apply intraday pricing on asset-specific rebalance days.
                start_d = pd.to_datetime(p.start)
                end_d = pd.to_datetime(p.end)
                all_days = pd.date_range(start_d, end_d, freq="D")
                day_map = {"MON":0, "TUE":1, "WED":2, "THU":3, "FRI":4, "SAT":5, "SUN":6}

                eq_freq = str(getattr(p, "rebalance_frequency_equities", None) or p.rebalance)
                eq_day = str(getattr(p, "rebalance_day_equities", None) or p.rebalance_day)
                cr_freq = str(getattr(p, "rebalance_frequency_crypto", None) or p.rebalance)
                cr_day = str(getattr(p, "rebalance_day_crypto", None) or p.rebalance_day)

                eq_reb_days = set(all_days) if eq_freq == "daily" else set([d for d in all_days if d.weekday() == day_map.get(eq_day.upper(), 0)])
                cr_reb_days = set(all_days) if cr_freq == "daily" else set([d for d in all_days if d.weekday() == day_map.get(cr_day.upper(), 0)])

                def intraday_cb(sym, day):
                    is_crypto = "/" in str(sym)
                    if is_crypto and day not in cr_reb_days:
                        return None
                    if (not is_crypto) and day not in eq_reb_days:
                        return None
                    return (prov_cr if is_crypto else prov_eq).price(sym, day)

                def intraday_limit_touch_cb(sym, day, side, limit_px):
                    is_crypto = "/" in str(sym)
                    return (prov_cr if is_crypto else prov_eq).limit_touched(sym, day, side, float(limit_px))

                def risk_intraday_cb(sym, day_or_ts):
                    is_crypto = "/" in str(sym)
                    prov = (risk_prov_cr if is_crypto else risk_prov_eq)
                    ts = pd.Timestamp(day_or_ts)
                    if ts.hour == 0 and ts.minute == 0 and ts.second == 0:
                        return prov.price(sym, ts)
                    return prov.price_at_local_ts(sym, ts, exact_minute=True)

            t_prepare_end = time.perf_counter()
            t_sim_start = t_prepare_end

            _hb_stop = threading.Event()

            def _sim_heartbeat():
                while not _hb_stop.wait(5.0):
                    try:
                        st = _read_json_safe(status_path) or {}
                        if str(st.get("state")) != "running":
                            continue
                        _log("simulate_heartbeat", done=st.get("progress", 0), total=st.get("total", 0))
                    except Exception:
                        pass

            hb_t = threading.Thread(target=_sim_heartbeat, daemon=True)
            hb_t.start()
            try:
                res = run_backtest(
                    stock_bars=stock_bars,
                    crypto_bars=crypto_bars,
                    stock_universe=tradable_eq,
                    crypto_universe=tradable_cr,
                    cfg=cfg,
                    params=p,
                    progress_cb=prog,
                    debug_cb=dbg,
                    stop_cb=stop_event.is_set,
                    debug_verbose=debug_verbose,
                    intraday_price_cb=intraday_cb,
                    intraday_limit_touch_cb=intraday_limit_touch_cb,
                    risk_intraday_price_cb=risk_intraday_cb,
                )
            finally:
                _hb_stop.set()

            t_sim_end = time.perf_counter()
            t_write_start = t_sim_end

            payload = {"job_id": job_id, **asdict(res)}
            metrics = payload.get("metrics") or {}
            metrics["timing"] = {
                "total_seconds": round(float(time.perf_counter() - t_job_start), 4),
                "fetch_data_seconds": round(float(t_fetch_end - t_fetch_start), 4),
                "prepare_seconds": round(float(t_prepare_end - t_fetch_end), 4),
                "simulate_seconds": round(float(t_sim_end - t_sim_start), 4),
                "write_seconds": None,
                "stock_cache_hit": bool(stock_cache_hit),
                "crypto_cache_hit": bool(crypto_cache_hit),
                "stock_universe_size": len(tradable_eq),
                "crypto_universe_size": len(tradable_cr),
            }
            payload["metrics"] = metrics
            _write(result_path, payload)

            t_write_end = time.perf_counter()
            payload["metrics"]["timing"]["write_seconds"] = round(float(t_write_end - t_write_start), 4)
            payload["metrics"]["timing"]["total_seconds"] = round(float(t_write_end - t_job_start), 4)
            _write(result_path, payload)
            _write(status_path, {"state": "done", "progress": 1, "total": 1})
            _log("backtest_done", total_seconds=payload["metrics"]["timing"].get("total_seconds"))
        except BacktestStopped:
            _write(status_path, {"state": "stopped", "progress": 0, "total": 1})
            _log("backtest_stopped")
        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            _write(status_path, {"state": "error", "error": str(e), "traceback": tb})
            _log("backtest_error", error=str(e))
        finally:
            _STOP_EVENTS.pop(job_id, None)

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


def stop_backtest(job_id: str) -> dict:
    ev = _STOP_EVENTS.get(job_id)
    if ev is None:
        return {"ok": False, "error": "job_not_running", "job_id": job_id}
    ev.set()
    p = BASE / job_id / "status.json"
    try:
        _write(p, {"state": "stopping", "progress": 0, "total": 1})
    except Exception:
        pass
    return {"ok": True, "job_id": job_id}


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


def get_debug_log(job_id: str, *, offset: int = 0, limit: int = 200) -> dict:
    p = BASE / job_id / "debug.log"
    if not p.exists():
        return {"job_id": job_id, "lines": [], "next_offset": 0, "eof": True}
    try:
        lines_all = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return {"job_id": job_id, "lines": [], "next_offset": 0, "eof": True}
    off = max(0, int(offset))
    lim = max(1, min(int(limit), 2000))
    out = lines_all[off : off + lim]
    nxt = off + len(out)
    return {"job_id": job_id, "lines": out, "next_offset": nxt, "eof": nxt >= len(lines_all)}


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
