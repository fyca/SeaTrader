from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ExitDecision:
    symbol: str
    should_exit: bool
    reason: str
    last_close: float | None = None
    ma_long: float | None = None


def trend_break_exit(closes: pd.Series, *, ma_long: int) -> tuple[bool, str, float | None, float | None]:
    closes = closes.dropna()
    if len(closes) < ma_long + 5:
        return False, "insufficient_history", None, None

    last = float(closes.iloc[-1])
    maL = float(closes.rolling(ma_long).mean().iloc[-1])

    if last < maL:
        return True, "close_below_long_ma", last, maL
    return False, "ok", last, maL
