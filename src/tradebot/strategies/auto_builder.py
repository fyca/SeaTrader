from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from tradebot.adapters.alpaca_client import make_alpaca_clients
from tradebot.adapters.bars import fetch_crypto_bars_range, fetch_stock_bars_range
from tradebot.backtest.engine import BacktestParams, run_backtest
from tradebot.strategies.user_store import save_user_strategy, delete_user_strategy
from tradebot.util.config import load_config
from tradebot.util.env import load_env


@dataclass
class AutoBuildJob:
    state: str
    progress: int
    total: int
    started_at: str
    updated_at: str
    error: str | None = None
    result: dict | None = None
    phase: str = "starting"
    current_symbol: str | None = None
    detail: str | None = None


_JOBS: dict[str, AutoBuildJob] = {}
_STOPS: dict[str, threading.Event] = {}
_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rsi_series(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).rolling(n).mean()
    loss = (-d.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(100.0)


def _metrics(equity: pd.Series) -> dict:
    if equity is None or len(equity) < 3:
        return {"return": 0.0, "cagr": 0.0, "max_drawdown": 0.0, "sharpe": 0.0}
    rets = equity.pct_change().dropna()
    total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    years = max(1e-9, len(equity) / 252.0)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    max_dd = float(dd.min())
    vol = float(rets.std(ddof=0) * np.sqrt(252.0)) if len(rets) else 0.0
    sharpe = float((rets.mean() * 252.0) / vol) if vol > 1e-12 else 0.0
    return {"return": total_ret, "cagr": cagr, "max_drawdown": max_dd, "sharpe": sharpe}


def _objective(metrics: dict, objective: str) -> float:
    o = str(objective or "balanced").lower()
    if o == "sharpe":
        return float(metrics.get("sharpe") or 0.0)
    if o == "return":
        return float(metrics.get("return") or 0.0) - 1.5 * abs(float(metrics.get("max_drawdown") or 0.0))
    if o == "cagr":
        return float(metrics.get("cagr") or 0.0) - 1.5 * abs(float(metrics.get("max_drawdown") or 0.0))
    return float((metrics.get("sharpe") or 0.0) + 0.5 * (metrics.get("return") or 0.0) + 0.2 * (metrics.get("cagr") or 0.0) - 2.0 * abs(float(metrics.get("max_drawdown") or 0.0)))


def _ma_series(s: pd.Series, kind: str, n: int) -> pd.Series:
    if str(kind).lower() == "ema":
        return s.ewm(span=int(n), adjust=False).mean()
    return s.rolling(int(n)).mean()


def _simulate_symbol(close: pd.Series, cfg: dict) -> dict:
    s = close.dropna().astype(float)
    ma_long = int(cfg.get("ma_long", 200))
    ma_short = int(cfg.get("ma_short", 50))
    exit_n = int(cfg.get("exit_n", 100))
    if len(s) < max(ma_long, ma_short, exit_n, 60) + 5:
        return {"ok": False}

    ma_kind = str(cfg.get("ma_kind", "sma"))
    exit_kind = str(cfg.get("exit_kind", "sma"))
    maL = _ma_series(s, ma_kind, ma_long)
    maS = _ma_series(s, ma_kind, ma_short)
    maE = _ma_series(s, exit_kind, exit_n)
    rsi = _rsi_series(s, 14)
    rets = s.pct_change()

    pos = 0
    eq = [1.0]
    idx = s.index
    trades = 0
    for i in range(1, len(s)):
        px_prev = float(s.iloc[i - 1]); px_now = float(s.iloc[i])
        eq.append(eq[-1] * (px_now / px_prev) if pos else eq[-1])

        base_enter = bool(np.isfinite(maL.iloc[i]) and np.isfinite(maS.iloc[i]) and (px_now > float(maL.iloc[i])) and (float(maS.iloc[i]) > float(maL.iloc[i])))
        if not base_enter:
            enter = False
        else:
            ft = str(cfg.get("filter_type", "rsi"))
            if ft == "none":
                enter = True
            elif ft == "rsi":
                enter = bool(np.isfinite(rsi.iloc[i]) and float(rsi.iloc[i]) < float(cfg.get("rsi_max", 70)))
            elif ft == "roc":
                n = int(cfg.get("roc_n", 20))
                enter = bool(i >= n and np.isfinite(s.iloc[i - n]) and float(s.iloc[i - n]) > 0 and ((px_now / float(s.iloc[i - n]) - 1.0) > float(cfg.get("roc_min", 0.0))))
            elif ft == "ann_vol":
                n = int(cfg.get("vol_n", 20))
                vv = rets.iloc[max(0, i - n + 1): i + 1].std(ddof=0) * np.sqrt(252.0)
                enter = bool(np.isfinite(vv) and float(vv) < float(cfg.get("vol_max", 0.8)))
            elif ft == "breakout":
                n = int(cfg.get("breakout_n", 20))
                hi = s.iloc[max(0, i - n + 1): i + 1].max()
                enter = bool(np.isfinite(hi) and px_now >= float(hi))
            elif ft == "dist_sma":
                n = int(cfg.get("dist_n", 200))
                sm = _ma_series(s, "sma", n).iloc[i]
                enter = bool(np.isfinite(sm) and sm > 0 and ((px_now / float(sm) - 1.0) <= float(cfg.get("dist_max", 0.25))))
            elif ft == "ret_1d":
                enter = bool(np.isfinite(rets.iloc[i]) and float(rets.iloc[i]) >= float(cfg.get("ret1d_min", -0.03)))
            elif ft == "lowest":
                n = int(cfg.get("lowest_n", 20))
                lo = s.iloc[max(0, i - n + 1): i + 1].min()
                enter = bool(np.isfinite(lo) and lo > 0 and (px_now / float(lo) - 1.0) >= float(cfg.get("rebound_min", 0.03)))
            else:
                enter = True

        et = str(cfg.get("exit_type", "ma"))
        if et == "ma":
            exit_ = bool(np.isfinite(maE.iloc[i]) and px_now < float(maE.iloc[i]))
        elif et == "rsi":
            exit_ = bool(np.isfinite(rsi.iloc[i]) and float(rsi.iloc[i]) > float(cfg.get("exit_rsi", 75)))
        elif et == "roc":
            n = int(cfg.get("exit_roc_n", 20))
            exit_ = bool(i >= n and np.isfinite(s.iloc[i - n]) and float(s.iloc[i - n]) > 0 and ((px_now / float(s.iloc[i - n]) - 1.0) < float(cfg.get("exit_roc_min", -0.06))))
        elif et == "breakdown":
            n = int(cfg.get("exit_break_n", 20))
            lo = s.iloc[max(0, i - n + 1): i + 1].min()
            exit_ = bool(np.isfinite(lo) and px_now <= float(lo))
        else:
            exit_ = False

        if pos == 0 and enter:
            pos = 1; trades += 1
        elif pos == 1 and exit_:
            pos = 0; trades += 1

    equity = pd.Series(eq, index=idx)
    m = _metrics(equity)
    return {"ok": True, "metrics": m, "trades": trades}


def _build_spec(params: dict, asset_class: str = "stocks") -> dict:
    ml = int(params["ma_long"])
    ms = int(params["ma_short"])
    mk = str(params.get("ma_kind", "sma"))
    exn = int(params.get("exit_n", 100))
    exk = str(params.get("exit_kind", "sma"))
    sid = f"auto_{asset_class}_{mk}_ml{ml}_ms{ms}_x{exn}"

    entry_all = [
        {"left": {"kind": "close"}, "op": ">", "right": {"kind": mk, "n": ml}},
        {"left": {"kind": mk, "n": ms}, "op": ">", "right": {"kind": mk, "n": ml}},
    ]
    ft = str(params.get("filter_type", "none"))
    if ft == "rsi":
        entry_all.append({"left": {"kind": "rsi", "n": 14}, "op": "<", "right": float(params.get("rsi_max", 70))})
    elif ft == "roc":
        entry_all.append({"left": {"kind": "roc", "n": int(params.get("roc_n", 20))}, "op": ">", "right": float(params.get("roc_min", 0.0))})
    elif ft == "ann_vol":
        entry_all.append({"left": {"kind": "ann_vol", "n": int(params.get("vol_n", 20))}, "op": "<", "right": float(params.get("vol_max", 0.8))})
    elif ft == "breakout":
        entry_all.append({"left": {"kind": "close"}, "op": ">=", "right": {"kind": "highest", "n": int(params.get("breakout_n", 20))}})
    elif ft == "dist_sma":
        entry_all.append({"left": {"kind": "dist_sma", "n": int(params.get("dist_n", 200))}, "op": "<=", "right": float(params.get("dist_max", 0.25))})
    elif ft == "ret_1d":
        entry_all.append({"left": {"kind": "ret_1d"}, "op": ">=", "right": float(params.get("ret1d_min", -0.03))})
    elif ft == "lowest":
        entry_all.append({"left": {"kind": "close"}, "op": ">", "right": {"kind": "lowest", "n": int(params.get("lowest_n", 20))}})

    et = str(params.get("exit_type", "ma"))
    if et == "ma":
        exit_any = [{"left": {"kind": "close"}, "op": "<", "right": {"kind": exk, "n": exn}}]
    elif et == "rsi":
        exit_any = [{"left": {"kind": "rsi", "n": 14}, "op": ">", "right": float(params.get("exit_rsi", 75))}]
    elif et == "roc":
        exit_any = [{"left": {"kind": "roc", "n": int(params.get("exit_roc_n", 20))}, "op": "<", "right": float(params.get("exit_roc_min", -0.06))}]
    else:
        exit_any = [{"left": {"kind": "close"}, "op": "<=", "right": {"kind": "lowest", "n": int(params.get("exit_break_n", 20))}}]

    return {
        "id": sid,
        "name": f"Auto {asset_class} {mk} ml={ml} ms={ms} {ft} exit={et}",
        "version": 1,
        "type": "entry",
        "asset_class": asset_class,
        "entry": {"all": entry_all},
        "exit": {"any": exit_any},
        "score_factors": [
            {"weight": 1.0, "value": {"kind": "dist_sma", "n": ml}},
            {"weight": -0.25, "value": {"kind": "ann_vol", "n": 20}},
        ],
    }


def _candidate_configs(search_mode: str) -> list[dict]:
    mode = str(search_mode or "standard").lower()
    ma_longs = [100, 150, 200] if mode != "exhaustive" else [80, 100, 120, 150, 180, 200, 250]
    ma_shorts = [20, 50] if mode != "exhaustive" else [10, 20, 30, 50, 80]
    exit_ns = [50, 100, 150] if mode != "exhaustive" else [20, 50, 80, 100, 150, 200]
    ma_kinds = ["sma"] if mode != "exhaustive" else ["sma", "ema"]
    filter_types = ["rsi"] if mode != "exhaustive" else ["none", "rsi", "roc", "ann_vol", "breakout", "dist_sma", "ret_1d", "lowest"]
    exit_types = ["ma"] if mode != "exhaustive" else ["ma", "rsi", "roc", "breakdown"]

    out: list[dict] = []
    for mk in ma_kinds:
        for ml in ma_longs:
            for ms in ma_shorts:
                if ms >= ml:
                    continue
                for exn in exit_ns:
                    for et in exit_types:
                        for ft in filter_types:
                            cfg = {
                                "ma_kind": mk,
                                "ma_long": int(ml),
                                "ma_short": int(ms),
                                "filter_type": ft,
                                "exit_type": et,
                                "exit_n": int(exn),
                                "exit_kind": mk,
                            }
                            if ft == "rsi":
                                for v in ([60, 65, 70] if mode == "exhaustive" else [65]):
                                    c = dict(cfg); c["rsi_max"] = v; out.append(c)
                                continue
                            if ft == "roc":
                                for n in ([10, 20, 40] if mode == "exhaustive" else [20]):
                                    for mn in ([0.0, 0.03, 0.06] if mode == "exhaustive" else [0.03]):
                                        c = dict(cfg); c["roc_n"] = n; c["roc_min"] = mn; out.append(c)
                                continue
                            if ft == "ann_vol":
                                for n in ([10, 20, 40] if mode == "exhaustive" else [20]):
                                    for vm in ([0.5, 0.7, 0.9, 1.2] if mode == "exhaustive" else [0.8]):
                                        c = dict(cfg); c["vol_n"] = n; c["vol_max"] = vm; out.append(c)
                                continue
                            if ft == "breakout":
                                for n in ([10, 20, 55, 100] if mode == "exhaustive" else [20]):
                                    c = dict(cfg); c["breakout_n"] = n; out.append(c)
                                continue
                            if ft == "dist_sma":
                                for n in ([100, 150, 200] if mode == "exhaustive" else [200]):
                                    for dm in ([0.15, 0.25, 0.35] if mode == "exhaustive" else [0.25]):
                                        c = dict(cfg); c["dist_n"] = n; c["dist_max"] = dm; out.append(c)
                                continue
                            if ft == "ret_1d":
                                for rv in ([-0.05, -0.03, -0.01, 0.0] if mode == "exhaustive" else [-0.03]):
                                    c = dict(cfg); c["ret1d_min"] = rv; out.append(c)
                                continue
                            if ft == "lowest":
                                for n in ([10, 20, 40] if mode == "exhaustive" else [20]):
                                    for rb in ([0.01, 0.03, 0.05] if mode == "exhaustive" else [0.03]):
                                        c = dict(cfg); c["lowest_n"] = n; c["rebound_min"] = rb; out.append(c)
                                continue
                            out.append(cfg)
    return out


def _evaluate_candidate_parity(*, cfg_obj, bars_by_symbol: dict[str, pd.DataFrame], symbol: str, asset_class: str, cand_cfg: dict, params_overrides: dict, start: str, end: str) -> dict:
    temp_id = f"__ab_{str(uuid.uuid4())[:8]}_{symbol.replace('/','_')}"
    spec = _build_spec({**cand_cfg, "asset_class": asset_class}, asset_class=asset_class)
    spec["id"] = temp_id
    spec["name"] = f"AUTO TMP {temp_id}"

    try:
        save_user_strategy(temp_id, spec)

        p = dict(params_overrides or {})
        p.setdefault("initial_equity", 100000)
        p.setdefault("slippage_bps", 10)
        p["start"] = start
        p["end"] = end
        p["universe_mode"] = "single"
        p["symbol"] = symbol
        p["strategy_id"] = temp_id
        if asset_class == "stocks":
            p["asset_mode"] = "equities"
            p["strategy_id_equities"] = temp_id
            p["strategy_id_crypto"] = p.get("strategy_id_crypto") or temp_id
        else:
            p["asset_mode"] = "crypto"
            p["strategy_id_crypto"] = temp_id
            p["strategy_id_equities"] = p.get("strategy_id_equities") or temp_id

        bt_params = BacktestParams(**{k: v for k, v in p.items() if k in BacktestParams.__dataclass_fields__})
        stock_bars = {symbol: bars_by_symbol[symbol]} if asset_class == "stocks" else {}
        crypto_bars = {symbol: bars_by_symbol[symbol]} if asset_class == "crypto" else {}

        res = run_backtest(
            stock_bars=stock_bars,
            crypto_bars=crypto_bars,
            stock_universe=([symbol] if asset_class == "stocks" else []),
            crypto_universe=([symbol] if asset_class == "crypto" else []),
            cfg=cfg_obj,
            params=bt_params,
        )
        m = res.metrics or {}
        trades = int(m.get("trade_count") or 0)
        return {"ok": True, "metrics": m, "trades": trades}
    except Exception:
        return {"ok": False}
    finally:
        try:
            delete_user_strategy(temp_id)
        except Exception:
            pass


def _fmt_eta(seconds: float) -> str:
    s = max(0, int(seconds))
    d, r = divmod(s, 86400)
    h, r = divmod(r, 3600)
    m, sec = divmod(r, 60)
    return f"{d}d {h}h {m}m {sec}s"


def _fetch_with_timeout(fetch_fn, timeout_s: float = 25.0):
    box = {"res": None, "err": None}

    def _run():
        try:
            box["res"] = fetch_fn()
        except Exception as e:
            box["err"] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        raise TimeoutError(f"fetch_timeout_{timeout_s}s")
    if box["err"] is not None:
        raise box["err"]
    return box["res"]


def start_auto_build(*, symbols: list[str], years: int = 5, asset_class: str = "stocks", objective: str = "balanced", min_trades: int = 8, train_ratio: float = 0.7, folds: int = 3, search_mode: str = "standard", parity_mode: bool = True, base_params: dict | None = None, config_path: str | None = None) -> str:
    job_id = str(uuid.uuid4())
    now = _now()
    with _LOCK:
        _JOBS[job_id] = AutoBuildJob(state="starting", progress=0, total=max(1, len(symbols)), started_at=now, updated_at=now)
        _STOPS[job_id] = threading.Event()

    def upd(**kw):
        with _LOCK:
            j = _JOBS.get(job_id)
            if not j:
                return
            for k, v in kw.items():
                setattr(j, k, v)
            j.updated_at = _now()

    def run():
        try:
            upd(state="running", phase="fetching", detail="initializing")
            env = load_env()
            clients = make_alpaca_clients(env)
            cfg_obj = load_config(config_path) if config_path else None
            params_overrides = dict(base_params or {})
            end_dt = datetime.now(timezone.utc)
            start_dt = end_dt - timedelta(days=int(years) * 365)
            candidates = _candidate_configs(search_mode)

            per_symbol_best: list[dict] = []
            done = 0
            total_evals = 0
            for sym in symbols:
                stop_ev = _STOPS.get(job_id)
                if stop_ev is not None and stop_ev.is_set():
                    upd(state="stopped", phase="stopped", detail="stopped_by_user")
                    return

                sym = str(sym).strip().upper()
                if not sym:
                    continue
                upd(state="running", phase="fetching", current_symbol=sym, detail=f"fetching {done+1}/{len(symbols)}")

                bars = None
                last_fetch_err = None
                for attempt in range(1, 4):
                    try:
                        if "/" in sym:
                            bars = _fetch_with_timeout(lambda: fetch_crypto_bars_range(clients.crypto, [sym], start=start_dt, end=end_dt), timeout_s=25.0)
                        else:
                            bars = _fetch_with_timeout(lambda: fetch_stock_bars_range(clients.stocks, [sym], start=start_dt, end=end_dt), timeout_s=25.0)
                        last_fetch_err = None
                        break
                    except Exception as e:
                        last_fetch_err = e
                        upd(state="running", phase="fetching", current_symbol=sym, detail=f"retry {attempt}/3 after {type(e).__name__}")
                        time.sleep(0.6 * attempt)

                if bars is None:
                    done += 1
                    upd(progress=done, state="running", phase="fetching", current_symbol=sym, detail=f"skip fetch error: {last_fetch_err}")
                    continue

                df = bars.get(sym)
                if df is None or len(df) < 250 or "close" not in df.columns:
                    done += 1
                    upd(progress=done, state="running", phase="running", current_symbol=sym, detail="insufficient bars")
                    continue

                close = df["close"].dropna().astype(float)
                n = len(close)
                fcnt = max(1, int(folds))
                fold_windows: list[tuple[pd.Series, pd.Series]] = []
                for fi in range(fcnt):
                    start_i = int((n * 0.05) * fi)
                    sub = close.iloc[start_i:]
                    if len(sub) < 260:
                        continue
                    split_idx = max(200, int(len(sub) * float(train_ratio)))
                    if split_idx >= len(sub) - 30:
                        split_idx = int(len(sub) * 0.7)
                    train_close = sub.iloc[:split_idx]
                    valid_close = sub.iloc[split_idx:]
                    if len(train_close) < 200 or len(valid_close) < 30:
                        continue
                    fold_windows.append((train_close, valid_close))

                if not fold_windows:
                    done += 1
                    upd(progress=done, state="running", phase="running", current_symbol=sym, detail="no valid fold windows")
                    continue

                best = None
                evals_symbol = 0
                eval_start = time.perf_counter()
                upd(state="running", phase="optimizing", current_symbol=sym, detail=f"optimizing 0/{len(candidates)} candidates • ETA unknown")
                for cfg in candidates:
                    evals_symbol += 1
                    fold_scores: list[float] = []
                    fold_train_scores: list[float] = []
                    fold_valid_scores: list[float] = []
                    fold_trades: list[int] = []
                    for train_close, valid_close in fold_windows:
                        if parity_mode and cfg_obj is not None:
                            r_tr = _evaluate_candidate_parity(
                                cfg_obj=cfg_obj,
                                bars_by_symbol={sym: df},
                                symbol=sym,
                                asset_class=asset_class,
                                cand_cfg=cfg,
                                params_overrides=params_overrides,
                                start=pd.Timestamp(train_close.index[0]).strftime("%Y-%m-%d"),
                                end=pd.Timestamp(train_close.index[-1]).strftime("%Y-%m-%d"),
                            )
                            r_va = _evaluate_candidate_parity(
                                cfg_obj=cfg_obj,
                                bars_by_symbol={sym: df},
                                symbol=sym,
                                asset_class=asset_class,
                                cand_cfg=cfg,
                                params_overrides=params_overrides,
                                start=pd.Timestamp(valid_close.index[0]).strftime("%Y-%m-%d"),
                                end=pd.Timestamp(valid_close.index[-1]).strftime("%Y-%m-%d"),
                            )
                        else:
                            r_tr = _simulate_symbol(train_close, cfg)
                            r_va = _simulate_symbol(valid_close, cfg)
                        if not r_tr.get("ok") or not r_va.get("ok"):
                            continue
                        tr_count = int(r_tr.get("trades") or 0)
                        if tr_count < int(min_trades):
                            continue
                        obj_tr = _objective(r_tr["metrics"], objective)
                        obj_va = _objective(r_va["metrics"], objective)
                        obj = 0.6 * float(obj_tr) + 0.4 * float(obj_va)
                        fold_scores.append(float(obj))
                        fold_train_scores.append(float(obj_tr))
                        fold_valid_scores.append(float(obj_va))
                        fold_trades.append(tr_count)

                    if not fold_scores:
                        # still count completion for ETA progression
                        elapsed = max(1e-9, time.perf_counter() - eval_start)
                        rate = evals_symbol / elapsed
                        rem = max(0, len(candidates) - evals_symbol)
                        eta = _fmt_eta(rem / rate) if rate > 1e-9 else "unknown"
                        upd(state="running", phase="optimizing", current_symbol=sym, detail=f"optimizing {evals_symbol}/{len(candidates)} candidates • ETA {eta}")
                        continue

                    row = {
                        "symbol": sym,
                        **cfg,
                        "objective": float(np.mean(fold_scores)),
                        "train_objective": float(np.mean(fold_train_scores)),
                        "valid_objective": float(np.mean(fold_valid_scores)),
                        "trades": int(np.mean(fold_trades)) if fold_trades else 0,
                        "folds_used": int(len(fold_scores)),
                        "folds_requested": int(fcnt),
                    }
                    if best is None or float(row["objective"]) > float(best["objective"]):
                        best = row

                    elapsed = max(1e-9, time.perf_counter() - eval_start)
                    rate = evals_symbol / elapsed
                    rem = max(0, len(candidates) - evals_symbol)
                    eta = _fmt_eta(rem / rate) if rate > 1e-9 else "unknown"
                    upd(state="running", phase="optimizing", current_symbol=sym, detail=f"optimizing {evals_symbol}/{len(candidates)} candidates • ETA {eta}")
                total_evals += int(evals_symbol)
                if best is not None:
                    best["evaluations"] = int(evals_symbol)
                    per_symbol_best.append(best)

                done += 1
                upd(progress=done, state="running", phase="running", current_symbol=sym, detail=f"done {done}/{len(symbols)}")

            if not per_symbol_best:
                upd(state="error", phase="error", error="no viable symbol results")
                return

            top = sorted(per_symbol_best, key=lambda x: float(x.get("objective") or -1e9), reverse=True)
            topk = top[: max(3, min(len(top), 10))]
            agg = {
                "ma_long": int(np.median([x.get("ma_long", 200) for x in topk])),
                "ma_short": int(np.median([x.get("ma_short", 50) for x in topk])),
                "exit_n": int(np.median([x.get("exit_n", 100) for x in topk])),
                "ma_kind": max({str(x.get("ma_kind", "sma")) for x in topk}, key=lambda k: sum(1 for z in topk if str(z.get("ma_kind", "sma")) == k)),
                "exit_kind": max({str(x.get("exit_kind", "sma")) for x in topk}, key=lambda k: sum(1 for z in topk if str(z.get("exit_kind", "sma")) == k)),
                "filter_type": max({str(x.get("filter_type", "none")) for x in topk}, key=lambda k: sum(1 for z in topk if str(z.get("filter_type", "none")) == k)),
                "exit_type": max({str(x.get("exit_type", "ma")) for x in topk}, key=lambda k: sum(1 for z in topk if str(z.get("exit_type", "ma")) == k)),
            }
            best0 = topk[0]
            for k in ["rsi_max", "roc_n", "roc_min", "vol_n", "vol_max", "breakout_n", "dist_n", "dist_max", "ret1d_min", "lowest_n", "rebound_min", "exit_rsi", "exit_roc_n", "exit_roc_min", "exit_break_n"]:
                if k in best0:
                    agg[k] = best0[k]

            spec = _build_spec(agg, asset_class=asset_class)
            out = {
                "symbols": symbols,
                "years": years,
                "asset_class": asset_class,
                "objective": objective,
                "min_trades": int(min_trades),
                "train_ratio": float(train_ratio),
                "folds": int(folds),
                "search_mode": str(search_mode),
                "parity_mode": bool(parity_mode),
                "candidate_count": int(len(candidates)),
                "total_evaluations": int(total_evals),
                "aggregate_params": agg,
                "strategy": spec,
                "per_symbol_best": per_symbol_best,
            }
            upd(state="done", phase="done", progress=max(1, len(symbols)), total=max(1, len(symbols)), result=out, current_symbol=None, detail="complete")
        except Exception as e:
            upd(state="error", phase="error", error=str(e), detail=type(e).__name__)
        finally:
            with _LOCK:
                _STOPS.pop(job_id, None)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return job_id


def stop_auto_build(job_id: str) -> dict:
    with _LOCK:
        ev = _STOPS.get(job_id)
    if ev is None:
        return {"ok": False, "error": "job_not_running", "job_id": job_id}
    ev.set()
    return {"ok": True, "job_id": job_id}


def get_auto_build_status(job_id: str) -> dict:
    with _LOCK:
        j = _JOBS.get(job_id)
        if not j:
            return {"state": "missing"}
        return {
            "state": j.state,
            "phase": j.phase,
            "progress": j.progress,
            "total": j.total,
            "current_symbol": j.current_symbol,
            "detail": j.detail,
            "error": j.error,
            "started_at": j.started_at,
            "updated_at": j.updated_at,
        }


def get_auto_build_result(job_id: str) -> dict | None:
    with _LOCK:
        j = _JOBS.get(job_id)
        if not j:
            return None
        return j.result
