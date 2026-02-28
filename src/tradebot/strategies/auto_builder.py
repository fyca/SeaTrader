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


_JOBS: dict[str, AutoBuildJob] = {}
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


def _simulate_symbol(close: pd.Series, ma_long: int, ma_short: int, rsi_max: int, exit_ma: int) -> dict:
    s = close.dropna().astype(float)
    if len(s) < max(ma_long, ma_short, exit_ma, 50) + 5:
        return {"ok": False}
    maL = s.rolling(ma_long).mean()
    maS = s.rolling(ma_short).mean()
    maE = s.rolling(exit_ma).mean()
    rsi = _rsi_series(s, 14)

    pos = 0
    eq = [1.0]
    idx = s.index
    trades = 0
    for i in range(1, len(s)):
        px_prev = float(s.iloc[i - 1])
        px_now = float(s.iloc[i])
        if pos:
            eq.append(eq[-1] * (px_now / px_prev))
        else:
            eq.append(eq[-1])

        enter = bool((px_now > float(maL.iloc[i])) and (float(maS.iloc[i]) > float(maL.iloc[i])) and (float(rsi.iloc[i]) < float(rsi_max))) if np.isfinite(maL.iloc[i]) and np.isfinite(maS.iloc[i]) and np.isfinite(rsi.iloc[i]) else False
        exit_ = bool(px_now < float(maE.iloc[i])) if np.isfinite(maE.iloc[i]) else False

        if pos == 0 and enter:
            pos = 1
            trades += 1
        elif pos == 1 and exit_:
            pos = 0
            trades += 1

    equity = pd.Series(eq, index=idx)
    m = _metrics(equity)
    obj = float(m["sharpe"] + 0.5 * m["return"] + 0.2 * m["cagr"] - 2.0 * abs(m["max_drawdown"]))
    return {"ok": True, "metrics": m, "objective": obj, "trades": trades}


def _build_spec(params: dict, asset_class: str = "stocks") -> dict:
    ml = int(params["ma_long"])
    ms = int(params["ma_short"])
    rm = int(params["rsi_max"])
    ex = int(params["exit_ma"])
    sid = f"auto_{asset_class}_ml{ml}_ms{ms}_r{rm}_x{ex}"
    return {
        "id": sid,
        "name": f"Auto {asset_class} trend ml={ml} ms={ms} rsi<{rm} exit={ex}",
        "version": 1,
        "type": "entry",
        "asset_class": asset_class,
        "entry": {
            "all": [
                {"left": {"kind": "close"}, "op": ">", "right": {"kind": "sma", "n": ml}},
                {"left": {"kind": "sma", "n": ms}, "op": ">", "right": {"kind": "sma", "n": ml}},
                {"left": {"kind": "rsi", "n": 14}, "op": "<", "right": rm},
            ]
        },
        "exit": {
            "any": [
                {"left": {"kind": "close"}, "op": "<", "right": {"kind": "sma", "n": ex}}
            ]
        },
        "score_factors": [
            {"weight": 1.0, "value": {"kind": "dist_sma", "n": ml}},
            {"weight": -0.25, "value": {"kind": "ann_vol", "n": 20}},
        ],
    }


def start_auto_build(*, symbols: list[str], years: int = 5, asset_class: str = "stocks") -> str:
    job_id = str(uuid.uuid4())
    now = _now()
    with _LOCK:
        _JOBS[job_id] = AutoBuildJob(state="starting", progress=0, total=max(1, len(symbols)), started_at=now, updated_at=now)

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
            upd(state="fetching")
            env = load_env()
            clients = make_alpaca_clients(env)
            end_dt = datetime.now(timezone.utc)
            start_dt = end_dt - timedelta(days=int(years) * 365)

            grid = {
                "ma_long": [100, 150, 200],
                "ma_short": [20, 50],
                "rsi_max": [60, 65, 70],
                "exit_ma": [50, 100, 150],
            }

            per_symbol_best: list[dict] = []
            done = 0
            for sym in symbols:
                sym = str(sym).strip().upper()
                if not sym:
                    continue
                if "/" in sym:
                    bars = fetch_crypto_bars_range(clients.crypto, [sym], start=start_dt, end=end_dt)
                else:
                    bars = fetch_stock_bars_range(clients.stocks, [sym], start=start_dt, end=end_dt)
                df = bars.get(sym)
                if df is None or len(df) < 250 or "close" not in df.columns:
                    done += 1
                    upd(progress=done, state="running")
                    continue
                close = df["close"].dropna().astype(float)

                best = None
                for ml in grid["ma_long"]:
                    for ms in grid["ma_short"]:
                        if ms >= ml:
                            continue
                        for rm in grid["rsi_max"]:
                            for ex in grid["exit_ma"]:
                                r = _simulate_symbol(close, ml, ms, rm, ex)
                                if not r.get("ok"):
                                    continue
                                row = {"symbol": sym, "ma_long": ml, "ma_short": ms, "rsi_max": rm, "exit_ma": ex, **r}
                                if best is None or float(row["objective"]) > float(best["objective"]):
                                    best = row
                if best is not None:
                    per_symbol_best.append(best)

                done += 1
                upd(progress=done, state="running")

            if not per_symbol_best:
                upd(state="error", error="no viable symbol results")
                return

            agg = {
                "ma_long": int(np.median([x["ma_long"] for x in per_symbol_best])),
                "ma_short": int(np.median([x["ma_short"] for x in per_symbol_best])),
                "rsi_max": int(np.median([x["rsi_max"] for x in per_symbol_best])),
                "exit_ma": int(np.median([x["exit_ma"] for x in per_symbol_best])),
            }
            spec = _build_spec(agg, asset_class=asset_class)
            out = {
                "symbols": symbols,
                "years": years,
                "asset_class": asset_class,
                "aggregate_params": agg,
                "strategy": spec,
                "per_symbol_best": per_symbol_best,
            }
            upd(state="done", progress=max(1, len(symbols)), total=max(1, len(symbols)), result=out)
        except Exception as e:
            upd(state="error", error=str(e))

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return job_id


def get_auto_build_status(job_id: str) -> dict:
    with _LOCK:
        j = _JOBS.get(job_id)
        if not j:
            return {"state": "missing"}
        return {
            "state": j.state,
            "progress": j.progress,
            "total": j.total,
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
