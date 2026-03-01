#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class Candidate:
    entry: tuple
    exit: str
    risk: dict


def load_sp500_symbols() -> list[str]:
    urls = [
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
        "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv",
    ]
    for u in urls:
        try:
            t = pd.read_csv(u)
            col = "Symbol" if "Symbol" in t.columns else "symbol"
            syms = t[col].astype(str).str.replace(".", "-", regex=False).tolist()
            if len(syms) > 300:
                return syms
        except Exception:
            pass
    raise RuntimeError("failed to load sp500 symbol list")


def candidates() -> list[Candidate]:
    entries = [(mk, s, l, f) for mk in ["sma", "ema"] for l in [100, 150, 200] for s in [20, 50, 80] if s < l for f in ["macd", "rsi", "adx"]]
    exits = ["ma100", "ma150", "ma200", "macd_flip", "vwap_loss", "roc_break"]
    risks = [{"sl": None, "dd": None}, {"sl": 0.10, "dd": None}, {"sl": 0.12, "dd": 0.15}, {"sl": 0.15, "dd": 0.20}]
    return [Candidate(e, x, r) for e in entries for x in exits for r in risks]


def fetch_df(sym: str, start: datetime, end: datetime) -> pd.DataFrame | None:
    d = yf.download(sym, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval="1d", auto_adjust=True, progress=False)
    if d is None or d.empty:
        return None
    d.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in d.columns]
    for c in ["high", "low", "close", "volume"]:
        if c not in d.columns:
            d[c] = d["close"] if c != "volume" else 1.0
    d = d.dropna(subset=["close"]).copy()
    return d if len(d) >= 260 else None


def prep(df: pd.DataFrame) -> dict[str, pd.Series]:
    c, h, l, v = df["close"].astype(float), df["high"].astype(float), df["low"].astype(float), df["volume"].astype(float)
    out = {}
    for n in [20, 50, 80, 100, 150, 200]:
        out[f"sma{n}"] = c.rolling(n).mean()
        out[f"ema{n}"] = c.ewm(span=n, adjust=False).mean()
    ef = c.ewm(span=12, adjust=False).mean()
    es = c.ewm(span=26, adjust=False).mean()
    out["macd_hist"] = (ef - es) - (ef - es).ewm(span=9, adjust=False).mean()
    d = c.diff()
    g = d.clip(lower=0).rolling(14).mean()
    ls = (-d.clip(upper=0)).rolling(14).mean()
    rs = g / ls.replace(0, np.nan)
    out["rsi14"] = (100 - 100 / (1 + rs)).fillna(100)
    up = h.diff()
    dn = -l.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    mdm = dn.where((dn > up) & (dn > 0), 0.0)
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    pdi = 100 * (pdm.rolling(14).mean() / atr.replace(0, np.nan))
    mdi = 100 * (mdm.rolling(14).mean() / atr.replace(0, np.nan))
    dx = 100 * ((pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan))
    out["adx14"] = dx.rolling(14).mean()
    tp = (h + l + c) / 3
    out["vwap"] = (tp * v).cumsum() / v.cumsum().replace(0, np.nan)
    out["roc20"] = c / c.shift(20) - 1
    return out


def run_symbol(df: pd.DataFrame, ind: dict[str, pd.Series], cand: Candidate) -> float:
    c = df["close"].astype(float)
    i0 = 220
    mk, s, l, f = cand.entry
    kS, kL = f"{mk}{s}", f"{mk}{l}"

    def ent(i: int) -> bool:
        base = (c.iloc[i] > ind[kL].iloc[i]) and (ind[kS].iloc[i] > ind[kL].iloc[i])
        if not base:
            return False
        if f == "macd":
            return bool(ind["macd_hist"].iloc[i] > 0)
        if f == "rsi":
            return bool(ind["rsi14"].iloc[i] < 70)
        return bool(ind["adx14"].iloc[i] > 18)

    if cand.exit.startswith("ma") and cand.exit[2:].isdigit():
        n = int(cand.exit[2:])
        ma = c.rolling(n).mean()
        ex = lambda i: c.iloc[i] < ma.iloc[i]
    elif cand.exit == "vwap_loss":
        ex = lambda i: c.iloc[i] < ind["vwap"].iloc[i]
    elif cand.exit == "roc_break":
        ex = lambda i: ind["roc20"].iloc[i] < -0.06
    else:
        ex = lambda i: ind["macd_hist"].iloc[i] < 0

    cash, qty, entry, peak = 1.0, 0.0, None, 1.0
    for i in range(i0, len(df)):
        px = float(c.iloc[i])
        eq = cash + qty * px
        peak = max(peak, eq)
        if cand.risk["dd"] is not None and qty > 0 and peak > 0 and (peak - eq) / peak >= cand.risk["dd"]:
            cash += qty * px
            qty = 0.0
            entry = None
        if cand.risk["sl"] is not None and qty > 0 and entry is not None and (px / entry - 1) <= -cand.risk["sl"]:
            cash += qty * px
            qty = 0.0
            entry = None
        if qty > 0 and ex(i):
            cash += qty * px
            qty = 0.0
            entry = None
        if qty == 0 and ent(i):
            qty = cash / px
            cash = 0.0
            entry = px
    if qty > 0:
        cash += qty * float(c.iloc[-1])
    return cash - 1


def geom_mean(returns: list[float]) -> float:
    vals = [1 + r for r in returns]
    return float(np.prod(vals) ** (1 / len(vals)) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="data/optimizer_sp500")
    ap.add_argument("--chunks", type=int, default=5)
    ap.add_argument("--chunk-index", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-symbols", type=int, default=0)
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    wd = Path(args.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    status_p = wd / f"chunk_{args.chunk_index:02d}_status.json"
    result_p = wd / f"chunk_{args.chunk_index:02d}_result.json"

    syms = load_sp500_symbols()
    random.Random(args.seed).shuffle(syms)
    if args.max_symbols and args.max_symbols > 0:
        syms = syms[: args.max_symbols]
    n = len(syms)
    chunk_size = math.ceil(n / args.chunks)
    i0 = (args.chunk_index - 1) * chunk_size
    i1 = min(n, i0 + chunk_size)
    chunk = syms[i0:i1]

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * 5 + 20)

    cache = {}
    for k, s in enumerate(chunk, start=1):
        st = {"state": "fetching", "symbol": s, "symbol_progress": f"{k}/{len(chunk)}"}
        status_p.write_text(json.dumps(st, indent=2))
        d = fetch_df(s, start, end)
        if d is not None:
            cache[s] = (d, prep(d))

    symbols = list(cache.keys())
    if len(symbols) < 10:
        status_p.write_text(json.dumps({"state": "error", "error": "insufficient symbols with data"}, indent=2))
        return

    split = max(8, int(len(symbols) * 0.65))
    train, valid = symbols[:split], symbols[split:]

    cands = candidates()
    scores = []
    for idx, cand in enumerate(cands, start=1):
        if idx % 10 == 0 or idx == 1 or idx == len(cands):
            status_p.write_text(json.dumps({"state": "optimizing", "candidate_progress": f"{idx}/{len(cands)}"}, indent=2))
        try:
            tr = []
            va = []
            for s in train:
                tr.append(run_symbol(*cache[s], cand))
            for s in valid:
                va.append(run_symbol(*cache[s], cand))
            if len(tr) < 5 or len(va) < 3:
                continue
            tr_g = geom_mean(tr)
            va_g = geom_mean(va)
            joint = 0.6 * tr_g + 0.4 * va_g
            scores.append({
                "candidate": {"entry": cand.entry, "exit": cand.exit, "risk": cand.risk},
                "train": tr_g,
                "valid": va_g,
                "joint": joint,
            })
        except Exception:
            continue

    scores = sorted(scores, key=lambda x: x["joint"], reverse=True)
    top = scores[: args.top_n]
    out = {
        "chunk_index": args.chunk_index,
        "chunks": args.chunks,
        "symbols_in_chunk": len(chunk),
        "symbols_with_data": len(symbols),
        "top": top,
    }
    result_p.write_text(json.dumps(out, indent=2))
    status_p.write_text(json.dumps({"state": "done", "result_file": str(result_p)}, indent=2))
    print(json.dumps({"ok": True, "result_file": str(result_p), "top_count": len(top)}, indent=2))


if __name__ == "__main__":
    main()
