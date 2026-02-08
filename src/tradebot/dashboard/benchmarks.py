from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from tradebot.adapters.bars import fetch_stock_bars_range
from tradebot.util.env import load_env
from tradebot.adapters.alpaca_client import make_alpaca_clients


CACHE_DIR = Path("data/cache/benchmarks")


def _cache_path(name: str, start: str, end: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = name.replace("/", "_")
    return CACHE_DIR / f"{safe}-{start}-{end}.parquet"


def get_spy_series(start: str, end: str) -> pd.Series:
    p = _cache_path("SPY", start, end)
    if p.exists():
        df = pd.read_parquet(p)
        return df["close"].astype(float)

    env = load_env()
    clients = make_alpaca_clients(env)
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)

    bars = fetch_stock_bars_range(clients.stocks, ["SPY"], start=start_dt, end=end_dt)
    df = bars.get("SPY")
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)

    df = df[["close"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.to_parquet(p)
    return df["close"].astype(float)


def get_sp500_series(start: str, end: str) -> pd.Series:
    """S&P 500 index from Stooq (free). Symbol: ^SPX.

    NOTE: This is EOD and may differ slightly vs SPY.
    """
    p = _cache_path("SP500", start, end)
    if p.exists():
        df = pd.read_parquet(p)
        return df["close"].astype(float)

    # Stooq daily CSV
    url = "https://stooq.com/q/d/l/?s=^spx&i=d"
    df = pd.read_csv(url)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.set_index("Date").sort_index()
    df = df.loc[pd.to_datetime(start) : pd.to_datetime(end)]
    if "Close" not in df.columns:
        return pd.Series(dtype=float)
    out = df["Close"].rename("close").astype(float)
    out.to_frame().to_parquet(p)
    return out


def normalize(series: pd.Series) -> list[dict]:
    s = series.dropna().astype(float)
    if len(s) == 0:
        return []
    base = float(s.iloc[0])
    if base == 0:
        base = 1.0
    res = []
    for idx, v in s.items():
        res.append({"date": pd.to_datetime(idx).strftime("%Y-%m-%d"), "value": float(v), "norm": float(v / base * 100.0)})
    return res
