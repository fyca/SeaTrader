from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from tradebot.adapters.alpaca_client import make_alpaca_clients
from tradebot.adapters.bars import fetch_crypto_bars_range, fetch_stock_bars_range
from tradebot.util.env import load_env


def fetch_close_series(symbol: str, start: str, end: str) -> pd.Series:
    """Fetch daily close series for equity or crypto.

    - Equities: via Alpaca stocks (IEX)
    - Crypto: via Alpaca crypto
    """
    env = load_env()
    clients = make_alpaca_clients(env)
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    # include end day
    end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc) + timedelta(days=1)

    is_crypto = "/" in symbol
    if is_crypto:
        bars = fetch_crypto_bars_range(clients.crypto, [symbol], start=start_dt, end=end_dt)
    else:
        bars = fetch_stock_bars_range(clients.stocks, [symbol], start=start_dt, end=end_dt)

    df = bars.get(symbol)
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)
    df = df[["close"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()
    return df["close"].astype(float)


def to_points(series: pd.Series) -> list[dict]:
    s = series.dropna().astype(float)
    out: list[dict] = []
    for idx, v in s.items():
        out.append({"date": pd.to_datetime(idx).strftime("%Y-%m-%d"), "close": float(v)})
    return out
