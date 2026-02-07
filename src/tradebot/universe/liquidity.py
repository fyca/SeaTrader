from __future__ import annotations

import pandas as pd


def avg_dollar_volume(df: pd.DataFrame, *, lookback: int = 20) -> float:
    """Average (close * volume) over last N bars."""
    if df is None or len(df) == 0:
        return 0.0
    if "close" not in df.columns or "volume" not in df.columns:
        return 0.0
    tail = df.tail(lookback)
    if len(tail) == 0:
        return 0.0
    dv = (tail["close"] * tail["volume"]).dropna()
    if len(dv) == 0:
        return 0.0
    return float(dv.mean())
