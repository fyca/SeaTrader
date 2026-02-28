from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
import pandas as pd


@dataclass
class _SeriesCache:
    data: dict[tuple, pd.Series]


class IndicatorService:
    """Central indicator compute/cache service.

    All new indicators should be added here so strategy/risk callers get
    caching automatically.
    """

    def __init__(self):
        self._series = _SeriesCache(data={})

    def _skey(self, closes: pd.Series, kind: str, *params: object) -> tuple:
        s = closes.dropna().astype(float)
        end = s.index[-1] if len(s) else None
        return (kind, tuple(params), len(s), end, tuple(s.index[-3:]))

    def _get(self, closes: pd.Series, kind: str, *params: object) -> pd.Series:
        s = closes.dropna().astype(float)
        key = self._skey(s, kind, *params)
        out = self._series.data.get(key)
        if out is not None:
            return out

        if kind == "sma":
            n = int(params[0])
            out = s.rolling(n).mean()
        elif kind == "ema":
            n = int(params[0])
            out = s.ewm(span=n, adjust=False).mean()
        elif kind == "ret":
            out = s.pct_change()
        elif kind == "ann_vol":
            n = int(params[0])
            ann_factor = float(params[1])
            out = s.pct_change().rolling(n).std(ddof=0) * np.sqrt(ann_factor)
        elif kind == "rsi":
            n = int(params[0])
            d = s.diff()
            gain = d.clip(lower=0).rolling(n).mean()
            loss = (-d.clip(upper=0)).rolling(n).mean()
            rs = gain / loss.replace(0, np.nan)
            out = 100.0 - (100.0 / (1.0 + rs))
            out = out.fillna(100.0)
        elif kind == "highest":
            n = int(params[0])
            out = s.rolling(n).max()
        elif kind == "lowest":
            n = int(params[0])
            out = s.rolling(n).min()
        else:
            raise ValueError(f"unknown indicator kind: {kind}")

        self._series.data[key] = out
        return out

    @staticmethod
    def _last_valid(series: pd.Series) -> float | None:
        if series is None or len(series) == 0:
            return None
        v = series.iloc[-1]
        if v is None or (isinstance(v, float) and (not isfinite(v))):
            return None
        try:
            f = float(v)
        except Exception:
            return None
        return f if isfinite(f) else None

    def close(self, closes: pd.Series) -> float | None:
        s = closes.dropna().astype(float)
        if len(s) == 0:
            return None
        v = float(s.iloc[-1])
        return v if isfinite(v) else None

    def sma_series(self, closes: pd.Series, n: int) -> pd.Series:
        return self._get(closes, "sma", int(n))

    def sma(self, closes: pd.Series, n: int) -> float | None:
        if n <= 0 or len(closes.dropna()) < n:
            return None
        return self._last_valid(self.sma_series(closes, n))

    def ema(self, closes: pd.Series, n: int) -> float | None:
        if n <= 0 or len(closes.dropna()) < n:
            return None
        return self._last_valid(self._get(closes, "ema", int(n)))

    def highest(self, closes: pd.Series, n: int) -> float | None:
        if n <= 0 or len(closes.dropna()) < n:
            return None
        return self._last_valid(self._get(closes, "highest", int(n)))

    def lowest(self, closes: pd.Series, n: int) -> float | None:
        if n <= 0 or len(closes.dropna()) < n:
            return None
        return self._last_valid(self._get(closes, "lowest", int(n)))

    def roc(self, closes: pd.Series, n: int) -> float | None:
        s = closes.dropna().astype(float)
        if n <= 0 or len(s) < n + 1:
            return None
        prev = float(s.iloc[-(n + 1)])
        now = float(s.iloc[-1])
        if prev == 0:
            return None
        return float(now / prev - 1.0)

    def ret_1d(self, closes: pd.Series) -> float | None:
        s = closes.dropna().astype(float)
        if len(s) < 2:
            return None
        prev = float(s.iloc[-2])
        now = float(s.iloc[-1])
        if prev == 0:
            return None
        return float(now / prev - 1.0)

    def ann_vol(self, closes: pd.Series, n: int, ann_factor: float) -> float | None:
        s = closes.dropna().astype(float)
        if len(s) < n + 2:
            return None
        return self._last_valid(self._get(s, "ann_vol", int(n), float(ann_factor)))

    def rsi(self, closes: pd.Series, n: int = 14) -> float | None:
        s = closes.dropna().astype(float)
        if len(s) < n + 2:
            return None
        return self._last_valid(self._get(s, "rsi", int(n)))


indicator_service = IndicatorService()
