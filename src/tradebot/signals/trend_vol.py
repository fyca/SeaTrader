from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalResult:
    ok: bool
    score: float
    reason: str
    last_close: float
    ann_vol: float
    ma_long: float
    ma_short: float


def compute_trend_vol_signal(
    closes: pd.Series,
    *,
    ma_long: int,
    ma_short: int,
    vol_lookback: int,
    max_ann_vol: float,
    ann_factor: float,
) -> SignalResult:
    """Simple, robust swing signal.

    - Trend: last close > MA_long and MA_short slope positive
    - Vol filter: annualized realized vol below cap

    score: distance above MA_long minus vol penalty.
    """
    closes = closes.dropna()
    if len(closes) < max(ma_long, ma_short, vol_lookback) + 5:
        return SignalResult(False, 0.0, "insufficient_history", float("nan"), float("nan"), float("nan"), float("nan"))

    last = float(closes.iloc[-1])
    maL = float(closes.rolling(ma_long).mean().iloc[-1])
    maS = closes.rolling(ma_short).mean()
    maS_last = float(maS.iloc[-1])
    maS_prev = float(maS.iloc[-6])  # ~1 week slope on daily data

    rets = closes.pct_change().dropna()
    vol = float(rets.tail(vol_lookback).std(ddof=0) * np.sqrt(ann_factor))

    if not np.isfinite(last) or not np.isfinite(maL) or not np.isfinite(vol):
        return SignalResult(False, 0.0, "nan_values", last, vol, maL, maS_last)

    if vol > max_ann_vol:
        return SignalResult(False, 0.0, f"vol_too_high:{vol:.2f}", last, vol, maL, maS_last)

    if last <= maL:
        return SignalResult(False, 0.0, "below_long_ma", last, vol, maL, maS_last)

    if maS_last <= maS_prev:
        return SignalResult(False, 0.0, "short_ma_not_rising", last, vol, maL, maS_last)

    dist = (last / maL) - 1.0
    score = dist - 0.25 * vol
    return SignalResult(True, float(score), "ok", last, vol, maL, maS_last)
