"""5-year backtest + small parameter sweep.

Reads cached Alpaca bars from tradebot/data/cache/bars (no network calls).
Outputs a CSV + JSON summary under tradebot/data/analysis/.

Run:
  cd tradebot
  .venv/bin/python scripts/optimize_5y.py
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from tradebot.backtest.engine import BacktestParams, BacktestResult, run_backtest
from tradebot.util.config import load_config

BASE = Path(__file__).resolve().parents[1]
CACHE_DIR = BASE / "data" / "cache" / "bars"
OUT_DIR = BASE / "data" / "analysis"


def _load_cache_frames(prefix: str, *, max_symbols: int | None = None) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Load the most-recent non-empty cache set for given prefix.

    NOTE: Loading *all* cached symbols can be slow (hundreds of parquets).
    For experiments we optionally load only the largest parquet files as a proxy
    for the most-liquid/complete symbols.
    """

    metas = sorted(CACHE_DIR.glob(f"{prefix}-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not metas:
        raise FileNotFoundError(f"No cache meta JSON found for prefix={prefix} in {CACHE_DIR}")

    last_err = None
    for meta in metas:
        try:
            obj = json.loads(meta.read_text())
            files: dict[str, str] = obj.get("files") or {}

            items = []
            for sym, rel in files.items():
                p = CACHE_DIR / rel
                if not p.exists():
                    continue
                items.append((sym, rel, p.stat().st_size))

            # Prefer bigger files first (usually more history)
            items.sort(key=lambda x: x[2], reverse=True)
            if max_symbols is not None:
                items = items[: max_symbols]

            frames: dict[str, pd.DataFrame] = {}
            nonempty = 0
            for sym, rel, _sz in items:
                p = CACHE_DIR / rel
                df = pd.read_parquet(p)
                frames[sym] = df
                if len(df) > 0:
                    nonempty += 1

            if nonempty == 0:
                continue
            universe = sorted(list(frames.keys()))
            return frames, universe
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Failed to load any non-empty cache for prefix={prefix}. last_err={last_err}")


def _metrics_row(res: BacktestResult, tag: str, cfg_overrides: dict[str, Any]) -> dict[str, Any]:
    m = dict(res.metrics)
    m.update({
        "tag": tag,
        "start": res.params["start"],
        "end": res.params["end"],
        "slippage_bps": res.params["slippage_bps"],
    })
    for k, v in cfg_overrides.items():
        m[k] = v
    return m


def _calc_max_dd(curve: list[dict]) -> float:
    eq = [float(x["equity"]) for x in curve]
    peak = -1e18
    maxdd = 0.0
    for v in eq:
        peak = max(peak, v)
        if peak > 0:
            maxdd = max(maxdd, (peak - v) / peak)
    return float(maxdd)


# DD-stop is now implemented in-engine via BacktestParams.portfolio_dd_stop


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_config(str(BASE / "config" / "config.yaml"))

    stock_bars, stock_universe = _load_cache_frames("stocks", max_symbols=15)
    crypto_bars, crypto_universe = _load_cache_frames("crypto", max_symbols=8)

    print(f"Loaded {len(stock_universe)} stock symbols, {len(crypto_universe)} crypto symbols from cache", flush=True)

    # 5-year window ending last available day (from stocks cache)
    # pick max timestamp across a couple representative symbols
    any_df = next(df for df in stock_bars.values() if len(df))
    last_ts = pd.to_datetime(any_df.index.max()).tz_convert(None).date()
    end = last_ts.isoformat()
    start = (last_ts - timedelta(days=365 * 5)).isoformat()
    print(f"Backtest window: {start} -> {end}", flush=True)

    baseline_params = BacktestParams(start=start, end=end, slippage_bps=10.0, rebalance="weekly")

    # Small sweep (intentionally limited for runtime)
    # Build a grid then take a deterministic sample.
    grid: list[dict[str, Any]] = []
    # (eq_long, eq_short, cr_long, cr_short)
    ma_sets = [
        (200, 50, 120, 30),  # baseline
        (150, 50, 90, 30),
        (250, 50, 150, 30),
        (200, 30, 120, 20),
        (200, 70, 120, 40),
    ]
    vol_sets = [
        (0.80, 2.50),  # baseline
        (0.60, 2.00),
        (1.00, 3.00),
    ]
    maxpos_sets = [
        (10, 5),   # baseline
        (7, 4),
        (15, 8),
    ]
    slippages = [5.0, 10.0, 20.0]

    for (el, es, cl, cs) in ma_sets:
        for (eqv, crv) in vol_sets:
            for (me, mc) in maxpos_sets:
                for sl in slippages:
                    grid.append({
                        "eq_ma_long": el,
                        "eq_ma_short": es,
                        "cr_ma_long": cl,
                        "cr_ma_short": cs,
                        "eq_max_ann_vol": eqv,
                        "cr_max_ann_vol": crv,
                        "max_equity_positions": me,
                        "max_crypto_positions": mc,
                        "slippage_bps": sl,
                    })

    # Deterministic sample: always include baseline + a spread of others
    baseline_overrides = {
        "eq_ma_long": 200,
        "eq_ma_short": 50,
        "cr_ma_long": 120,
        "cr_ma_short": 30,
        "eq_max_ann_vol": 0.80,
        "cr_max_ann_vol": 2.50,
        "max_equity_positions": 10,
        "max_crypto_positions": 5,
        "slippage_bps": 10.0,
    }
    sample_n = 6
    sweep = [baseline_overrides]
    for x in grid:
        if x == baseline_overrides:
            continue
        # simple hash-based pseudo-random filter
        h = hash(tuple(sorted(x.items())))
        if (h % 7) == 0 and len(sweep) < sample_n:
            sweep.append(x)
        if len(sweep) >= sample_n:
            break

    rows: list[dict[str, Any]] = []
    base_rows: list[dict[str, Any]] = []

    def run_one(i: int, overrides: dict[str, Any]) -> BacktestResult:
        # mutate cfg in-place (restore after)
        cfg2 = deepcopy(cfg)
        cfg2.signals.equity.ma_long = overrides["eq_ma_long"]
        cfg2.signals.equity.ma_short = overrides["eq_ma_short"]
        cfg2.signals.crypto.ma_long = overrides["cr_ma_long"]
        cfg2.signals.crypto.ma_short = overrides["cr_ma_short"]
        cfg2.signals.equity.max_ann_vol = overrides["eq_max_ann_vol"]
        cfg2.signals.crypto.max_ann_vol = overrides["cr_max_ann_vol"]
        cfg2.limits.max_equity_positions = overrides["max_equity_positions"]
        cfg2.limits.max_crypto_positions = overrides["max_crypto_positions"]

        p = BacktestParams(**{**asdict(baseline_params), "slippage_bps": overrides["slippage_bps"]})
        return run_backtest(
            stock_bars=stock_bars,
            crypto_bars=crypto_bars,
            stock_universe=stock_universe,
            crypto_universe=crypto_universe,
            cfg=cfg2,
            params=p,
        )

    # 1) Run base backtests for the sampled sweep
    for i, overrides in enumerate(sweep, 1):
        res = run_one(i, overrides)
        row = _metrics_row(res, tag="base", cfg_overrides=overrides)
        rows.append(row)
        base_rows.append(row)

        print(f"base {i}/{len(sweep)} done", flush=True)

    # 2) Apply DD-stop variant only to baseline + top configs (by Sharpe)
    base_df = pd.DataFrame(base_rows).sort_values(["sharpe", "return"], ascending=False)
    top_k = 5
    dd_targets = [baseline_overrides] + [
        {
            "eq_ma_long": int(r.eq_ma_long),
            "eq_ma_short": int(r.eq_ma_short),
            "cr_ma_long": int(r.cr_ma_long),
            "cr_ma_short": int(r.cr_ma_short),
            "eq_max_ann_vol": float(r.eq_max_ann_vol),
            "cr_max_ann_vol": float(r.cr_max_ann_vol),
            "max_equity_positions": int(r.max_equity_positions),
            "max_crypto_positions": int(r.max_crypto_positions),
            "slippage_bps": float(r.slippage_bps),
        }
        for r in base_df.head(top_k).itertuples(index=False)
    ]

    seen = set()
    dd_targets_u = []
    for x in dd_targets:
        key = tuple(sorted(x.items()))
        if key in seen:
            continue
        seen.add(key)
        dd_targets_u.append(x)

    for j, overrides in enumerate(dd_targets_u, 1):
        # Run DD-stop variant (behavior A) in-engine
        cfg2 = deepcopy(cfg)
        cfg2.signals.equity.ma_long = overrides["eq_ma_long"]
        cfg2.signals.equity.ma_short = overrides["eq_ma_short"]
        cfg2.signals.crypto.ma_long = overrides["cr_ma_long"]
        cfg2.signals.crypto.ma_short = overrides["cr_ma_short"]
        cfg2.signals.equity.max_ann_vol = overrides["eq_max_ann_vol"]
        cfg2.signals.crypto.max_ann_vol = overrides["cr_max_ann_vol"]
        cfg2.limits.max_equity_positions = overrides["max_equity_positions"]
        cfg2.limits.max_crypto_positions = overrides["max_crypto_positions"]

        p_stop = BacktestParams(**{**asdict(baseline_params), "slippage_bps": overrides["slippage_bps"], "portfolio_dd_stop": 0.20})
        stopped = run_backtest(
            stock_bars=stock_bars,
            crypto_bars=crypto_bars,
            stock_universe=stock_universe,
            crypto_universe=crypto_universe,
            cfg=cfg2,
            params=p_stop,
        )
        row2 = _metrics_row(stopped, tag="ddstop", cfg_overrides=overrides)
        rows.append(row2)
        if j % 5 == 0 or j == len(dd_targets_u):
            print(f"ddstop {j}/{len(dd_targets_u)} done", flush=True)

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "sweep_5y_metrics.csv"
    df.to_csv(out_csv, index=False)

    # Top 5 by Sharpe (base only)
    base_only = df[df["tag"] == "base"].copy()
    top5 = base_only.sort_values(["sharpe", "return"], ascending=False).head(5)
    out_json = OUT_DIR / "top5_5y.json"
    out_json.write_text(json.dumps({"top5": top5.to_dict(orient="records")}, indent=2, default=str))

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
