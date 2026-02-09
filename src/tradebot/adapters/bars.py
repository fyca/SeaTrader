from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed


def _to_frame(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    # alpaca returns multi-index dataframe with symbol as top level sometimes
    if df is None or len(df) == 0:
        return pd.DataFrame()

    if isinstance(df.index, pd.MultiIndex):
        try:
            sub = df.xs(symbol)
        except Exception:
            sub = df[df.index.get_level_values(0) == symbol]
            if isinstance(sub.index, pd.MultiIndex):
                sub = sub.droplevel(0)
    else:
        sub = df

    sub = sub.copy()
    sub.index = pd.to_datetime(sub.index)
    sub = sub.sort_index()
    return sub


def fetch_stock_bars_range(
    stocks_client,
    symbols: list[str],
    *,
    start: datetime,
    end: datetime,
) -> dict[str, pd.DataFrame]:
    # Use IEX feed for free/retail accounts (SIP often requires subscription)
    req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Day, start=start, end=end, feed=DataFeed.IEX)
    resp = stocks_client.get_stock_bars(req)
    df = resp.df if hasattr(resp, "df") else resp

    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        out[sym] = _to_frame(df, sym)
    return out


def fetch_stock_bars_range_1m(
    stocks_client,
    symbols: list[str],
    *,
    start: datetime,
    end: datetime,
) -> dict[str, pd.DataFrame]:
    # IEX feed for equities
    req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Minute, start=start, end=end, feed=DataFeed.IEX)
    resp = stocks_client.get_stock_bars(req)
    df = resp.df if hasattr(resp, "df") else resp
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        out[sym] = _to_frame(df, sym)
    return out


def fetch_stock_bars(stocks_client, symbols: list[str], *, lookback_days: int) -> dict[str, pd.DataFrame]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    return fetch_stock_bars_range(stocks_client, symbols, start=start, end=end)


def fetch_stock_closes(stocks_client, symbols: list[str], *, lookback_days: int) -> dict[str, pd.Series]:
    bars = fetch_stock_bars(stocks_client, symbols, lookback_days=lookback_days)
    out: dict[str, pd.Series] = {}
    for sym, df in bars.items():
        if df is None or len(df) == 0 or "close" not in df.columns:
            out[sym] = pd.Series(dtype=float)
        else:
            out[sym] = df["close"].copy()
    return out


def fetch_crypto_bars_range(
    crypto_client,
    symbols: list[str],
    *,
    start: datetime,
    end: datetime,
) -> dict[str, pd.DataFrame]:
    req = CryptoBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Day, start=start, end=end)
    resp = crypto_client.get_crypto_bars(req)
    df = resp.df if hasattr(resp, "df") else resp

    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        out[sym] = _to_frame(df, sym)
    return out


def fetch_crypto_bars_range_1m(
    crypto_client,
    symbols: list[str],
    *,
    start: datetime,
    end: datetime,
) -> dict[str, pd.DataFrame]:
    req = CryptoBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Minute, start=start, end=end)
    resp = crypto_client.get_crypto_bars(req)
    df = resp.df if hasattr(resp, "df") else resp
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        out[sym] = _to_frame(df, sym)
    return out


def fetch_crypto_bars(crypto_client, symbols: list[str], *, lookback_days: int) -> dict[str, pd.DataFrame]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    return fetch_crypto_bars_range(crypto_client, symbols, start=start, end=end)


def fetch_crypto_closes(crypto_client, symbols: list[str], *, lookback_days: int) -> dict[str, pd.Series]:
    bars = fetch_crypto_bars(crypto_client, symbols, lookback_days=lookback_days)
    out: dict[str, pd.Series] = {}
    for sym, df in bars.items():
        if df is None or len(df) == 0 or "close" not in df.columns:
            out[sym] = pd.Series(dtype=float)
        else:
            out[sym] = df["close"].copy()
    return out
