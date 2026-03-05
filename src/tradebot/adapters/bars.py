from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

import pandas as pd

from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

from tradebot.adapters.rate_limit import retry_on_rate_limit


def _log_fetch_summary(kind: str, timeframe: str, symbols: list[str], out: dict[str, pd.DataFrame]) -> None:
    """Emit and persist a compact fetch summary for debugging/forensics."""
    requested = len(symbols)
    with_data = 0
    total_rows = 0
    for _s, df in out.items():
        try:
            n = int(len(df)) if df is not None else 0
        except Exception:
            n = 0
        if n > 0:
            with_data += 1
            total_rows += n

    evt = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "timeframe": timeframe,
        "requested_symbols": requested,
        "symbols_with_data": with_data,
        "total_rows": total_rows,
    }
    print(f"[bars] {kind} {timeframe}: requested_symbols={requested} symbols_with_data={with_data} total_rows={total_rows}")

    # Persist latest summary for dashboard/UI visibility
    try:
        p = Path("data/last_bar_fetch_stats.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        existing = {"events": []}
        if p.exists():
            try:
                existing = json.loads(p.read_text()) or {"events": []}
            except Exception:
                existing = {"events": []}
        events = list(existing.get("events") or [])
        events.append(evt)
        events = events[-200:]
        p.write_text(json.dumps({"events": events, "latest": evt}, indent=2))
    except Exception:
        pass


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


@retry_on_rate_limit(max_retries=5, initial_backoff=1.0)
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
    _log_fetch_summary("stocks", "1d", symbols, out)
    return out


@retry_on_rate_limit(max_retries=5, initial_backoff=1.0)
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
    _log_fetch_summary("stocks", "1m", symbols, out)
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


@retry_on_rate_limit(max_retries=5, initial_backoff=1.0)
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
    _log_fetch_summary("crypto", "1d", symbols, out)
    return out


@retry_on_rate_limit(max_retries=5, initial_backoff=1.0)
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
    _log_fetch_summary("crypto", "1m", symbols, out)
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
