"""Strategy + parameter sweep vs SPY benchmark (cached bars, no network).

Outputs CSV under data/analysis/.

Run:
  cd tradebot
  .venv/bin/python scripts/sweep_vs_spy.py

Notes:
- Uses a small cached universe (largest parquet files) for runtime.
- Uses daily closes; results are approximate.
- No live trading.
"""

from __future__ import annotations

import itertools
import json
from copy import deepcopy
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tradebot.backtest.engine import BacktestParams, run_backtest
from tradebot.strategies.registry import list_strategies
from tradebot.util.config import load_config

BASE = Path(__file__).resolve().parents[1]
CACHE_DIR = BASE / "data" / "cache" / "bars"
OUT_DIR = BASE / "data" / "analysis"


def load_cache_frames(prefix: str, *, max_symbols: int | None = None) -> tuple[dict[str, pd.DataFrame], list[str]]:
    metas = sorted(CACHE_DIR.glob(f"{prefix}-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not metas:
        raise FileNotFoundError(f"No cache meta JSON found for prefix={prefix} in {CACHE_DIR}")

    last_err: Exception | None = None
    for meta in metas:
        try:
            obj = json.loads(meta.read_text())
            files: dict[str, str] = obj.get("files") or {}

            items = []
            for sym, rel in files.items():
                p = CACHE_DIR / rel
                if p.exists():
                    items.append((sym, rel, p.stat().st_size))

            items.sort(key=lambda x: x[2], reverse=True)
            if max_symbols is not None:
                # ensure SPY is included if present
                syms = [x[0] for x in items]
                spy_item = None
                for it in items:
                    if it[0] == "SPY":
                        spy_item = it
                        break
                items = items[:max_symbols]
                if spy_item and "SPY" not in [x[0] for x in items]:
                    items = [spy_item] + items[:-1]

            frames: dict[str, pd.DataFrame] = {}
            nonempty = 0
            for sym, rel, _sz in items:
                df = pd.read_parquet(CACHE_DIR / rel)
                frames[sym] = df
                if len(df) > 0:
                    nonempty += 1

            if nonempty == 0:
                continue
            universe = sorted(frames.keys())
            return frames, universe
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Failed to load any non-empty cache for prefix={prefix}. last_err={last_err}")


def _series_close(df: pd.DataFrame) -> pd.Series:
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)
    if "close" not in df.columns:
        return pd.Series(dtype=float)
    s = df["close"].astype(float).dropna()
    idx = pd.to_datetime(s.index)
    # drop tz if present
    try:
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert(None)
    except Exception:
        pass
    s.index = idx
    s = s.sort_index()
    return s


def _benchmark_metrics(spy_close: pd.Series, *, start: str, end: str) -> dict[str, float]:
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    sub = spy_close.loc[:e]
    sub = sub[sub.index >= s]
    if len(sub) < 2:
        return {"spy_return": float("nan"), "spy_cagr": float("nan"), "spy_max_drawdown": float("nan"), "spy_sharpe": float("nan"), "spy_ann_vol": float("nan")}

    rets = sub.to_numpy(dtype=float)
    eq0, eq1 = float(rets[0]), float(rets[-1])
    curve = rets / (eq0 if eq0 else 1.0)
    peak = np.maximum.accumulate(curve)
    dd = (peak - curve) / np.where(peak == 0, 1, peak)
    max_dd = float(np.max(dd))

    days_n = max(1, len(curve) - 1)
    years = days_n / 365.0
    cagr = (curve[-1]) ** (1 / years) - 1 if years > 0 else 0.0

    dr = np.diff(curve) / np.where(curve[:-1] == 0, 1, curve[:-1])
    vol = float(np.std(dr, ddof=0) * np.sqrt(365.0)) if len(dr) > 2 else 0.0
    sharpe = float((np.mean(dr) / (np.std(dr, ddof=0) + 1e-12)) * np.sqrt(365.0)) if len(dr) > 2 else 0.0

    return {
        "spy_return": float(eq1 / eq0 - 1.0) if eq0 else 0.0,
        "spy_cagr": float(cagr),
        "spy_max_drawdown": float(max_dd),
        "spy_sharpe": float(sharpe),
        "spy_ann_vol": float(vol),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg0 = load_config(str(BASE / "config" / "config.yaml"))

    # Universe caps for runtime
    stock_bars, stock_universe = load_cache_frames("stocks", max_symbols=15)
    crypto_bars, crypto_universe = load_cache_frames("crypto", max_symbols=8)

    any_df = next(df for df in stock_bars.values() if len(df))
    last_ts = pd.to_datetime(any_df.index.max()).tz_convert(None).date()
    end = last_ts.isoformat()
    # Use 3-year window for sweep runtime; you can widen later once narrowed.
    start = (last_ts - timedelta(days=365 * 3)).isoformat()

    spy_close = _series_close(stock_bars.get("SPY"))
    spy_m = _benchmark_metrics(spy_close, start=start, end=end)

    base_params = BacktestParams(start=start, end=end, slippage_bps=10.0, rebalance="weekly")

    # Drawdown constraint used for ranking report
    dd_constraint = 0.25

    # Strategy sweep list (builtin + user)
    strats = list_strategies()

    # Parameter grid (kept modest; deterministic sample below)
    ma_long = [150, 200, 250]
    ma_short = [30, 50, 70]
    eq_max_vol = [0.6, 0.8, 1.0]
    cr_max_vol = [2.0, 2.5, 3.0]
    max_eq_pos = [7, 10, 15]
    max_cr_pos = [4, 5, 8]

    # Keep the sweep tight for runtime: weekly-only, fixed slippage.
    rebalance = ["weekly"]
    slippage_bps = [10.0]
    dd_stop = [None, 0.20]
    per_asset_stop = [None]
    rebalance_mode = ["target_notional", "no_add_to_losers"]
    liquidation_mode = ["liquidate_non_selected", "hold_until_exit"]

    grid = list(
        itertools.product(
            ma_long,
            ma_short,
            eq_max_vol,
            cr_max_vol,
            max_eq_pos,
            max_cr_pos,
            rebalance,
            slippage_bps,
            dd_stop,
            per_asset_stop,
            rebalance_mode,
            liquidation_mode,
        )
    )

    # Deterministic downsample for runtime
    # Keep a small number of configs per strategy (runtime-friendly)
    target_per_strategy = 30

    def keep_cfg(tup: tuple) -> bool:
        h = hash(tup)
        return (h % 3) == 0

    sampled = [g for g in grid if keep_cfg(g)]
    # Ensure some baseline-style configs are always present
    pinned = [
        (200, 50, 0.8, 2.5, 10, 5, "weekly", 10.0, None, None, "target_notional", "liquidate_non_selected"),
        (200, 50, 0.8, 2.5, 10, 5, "weekly", 10.0, 0.20, None, "target_notional", "liquidate_non_selected"),
        (200, 50, 0.8, 2.5, 10, 5, "weekly", 10.0, 0.20, None, "no_add_to_losers", "hold_until_exit"),
    ]
    for p in pinned:
        if p not in sampled:
            sampled.append(p)

    sampled = sampled[:target_per_strategy]

    rows: list[dict[str, Any]] = []

    for s_i, s in enumerate(strats, 1):
        sid = s["id"]
        sname = s["name"]
        source = s["source"]
        print(f"[{s_i}/{len(strats)}] strategy={sid} ({source}) sweep_n={len(sampled)}", flush=True)

        for j, g in enumerate(sampled, 1):
            (
                el,
                es,
                eqv,
                crv,
                me,
                mc,
                rb,
                sl,
                dds,
                pas,
                rbm,
                lm,
            ) = g

            cfg = deepcopy(cfg0)
            cfg.signals.equity.ma_long = int(el)
            cfg.signals.equity.ma_short = int(es)
            cfg.signals.crypto.ma_long = int(min(150, el))  # keep crypto MAs a bit shorter
            cfg.signals.crypto.ma_short = int(min(40, es))
            cfg.signals.equity.max_ann_vol = float(eqv)
            cfg.signals.crypto.max_ann_vol = float(crv)
            cfg.limits.max_equity_positions = int(me)
            cfg.limits.max_crypto_positions = int(mc)

            p = BacktestParams(
                **{
                    **asdict(base_params),
                    "strategy_id": sid,
                    "rebalance": rb,
                    "slippage_bps": float(sl),
                    "portfolio_dd_stop": dds,
                    "per_asset_stop_loss_pct": pas,
                    "rebalance_mode": rbm,
                    "liquidation_mode": lm,
                }
            )

            res = run_backtest(
                stock_bars=stock_bars,
                crypto_bars=crypto_bars,
                stock_universe=stock_universe,
                crypto_universe=crypto_universe,
                cfg=cfg,
                params=p,
            )

            m = dict(res.metrics)
            m.update(
                {
                    "strategy_id": sid,
                    "strategy_name": sname,
                    "strategy_source": source,
                    "start": start,
                    "end": end,
                    "dd_constraint": dd_constraint,
                    "eq_ma_long": int(el),
                    "eq_ma_short": int(es),
                    "eq_max_ann_vol": float(eqv),
                    "cr_max_ann_vol": float(crv),
                    "max_equity_positions": int(me),
                    "max_crypto_positions": int(mc),
                    "rebalance": rb,
                    "slippage_bps": float(sl),
                    "portfolio_dd_stop": dds,
                    "per_asset_stop_loss_pct": pas,
                    "rebalance_mode": rbm,
                    "liquidation_mode": lm,
                    **spy_m,
                    "excess_return_vs_spy": float(m.get("return", 0.0) - spy_m.get("spy_return", 0.0)),
                    "excess_sharpe_vs_spy": float(m.get("sharpe", 0.0) - spy_m.get("spy_sharpe", 0.0)),
                }
            )
            rows.append(m)

            if j % 20 == 0:
                print(f"  {j}/{len(sampled)}", flush=True)

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "sweep_vs_spy_3y.csv"
    df.to_csv(out_csv, index=False)

    # Produce a small JSON summary with top configs under drawdown constraint
    ok = df[df["max_drawdown"] <= dd_constraint].copy()
    ok = ok.sort_values(["excess_return_vs_spy", "sharpe", "return"], ascending=False)
    top = ok.head(10).to_dict(orient="records")
    summary = {
        "window": {"start": start, "end": end},
        "universe": {"stocks": len(stock_universe), "crypto": len(crypto_universe)},
        "spy": spy_m,
        "dd_constraint": dd_constraint,
        "top10": top,
    }
    out_json = OUT_DIR / "sweep_vs_spy_3y_top10.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str))

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
