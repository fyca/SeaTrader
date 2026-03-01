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
        elif kind == "mom":
            n = int(params[0])
            out = s - s.shift(n)
        elif kind == "bb_mid":
            n = int(params[0])
            out = s.rolling(n).mean()
        elif kind == "bb_upper":
            n = int(params[0]); k = float(params[1])
            m = s.rolling(n).mean(); sd = s.rolling(n).std(ddof=0)
            out = m + k * sd
        elif kind == "bb_lower":
            n = int(params[0]); k = float(params[1])
            m = s.rolling(n).mean(); sd = s.rolling(n).std(ddof=0)
            out = m - k * sd
        elif kind == "bb_width":
            n = int(params[0]); k = float(params[1])
            m = s.rolling(n).mean(); sd = s.rolling(n).std(ddof=0)
            up = m + k * sd
            lo = m - k * sd
            out = (up - lo) / m.replace(0, np.nan)
        elif kind == "macd_line":
            fast = int(params[0]); slow = int(params[1])
            ef = s.ewm(span=fast, adjust=False).mean()
            es = s.ewm(span=slow, adjust=False).mean()
            out = ef - es
        elif kind == "macd_signal":
            fast = int(params[0]); slow = int(params[1]); sig = int(params[2])
            ef = s.ewm(span=fast, adjust=False).mean()
            es = s.ewm(span=slow, adjust=False).mean()
            ml = ef - es
            out = ml.ewm(span=sig, adjust=False).mean()
        elif kind == "macd_hist":
            fast = int(params[0]); slow = int(params[1]); sig = int(params[2])
            ef = s.ewm(span=fast, adjust=False).mean()
            es = s.ewm(span=slow, adjust=False).mean()
            ml = ef - es
            ms = ml.ewm(span=sig, adjust=False).mean()
            out = ml - ms
        elif kind == "atr":
            n = int(params[0])
            tr = s.diff().abs()
            out = tr.rolling(n).mean()
        elif kind == "adx":
            n = int(params[0])
            tr = s.diff().abs().rolling(n).mean()
            dm = s.diff().clip(lower=0).rolling(n).mean()
            out = 100.0 * (dm / tr.replace(0, np.nan)).fillna(0.0)
        elif kind == "stoch_k":
            n = int(params[0])
            lo = s.rolling(n).min(); hi = s.rolling(n).max()
            out = 100.0 * ((s - lo) / (hi - lo).replace(0, np.nan))
        elif kind == "stoch_d":
            n = int(params[0]); d = int(params[1])
            lo = s.rolling(n).min(); hi = s.rolling(n).max()
            k = 100.0 * ((s - lo) / (hi - lo).replace(0, np.nan))
            out = k.rolling(d).mean()
        elif kind == "cci":
            n = int(params[0])
            tp = s
            ma = tp.rolling(n).mean()
            md = (tp - ma).abs().rolling(n).mean()
            out = (tp - ma) / (0.015 * md.replace(0, np.nan))
        elif kind == "vwap":
            out = s.expanding().mean()
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

    def mom(self, closes: pd.Series, n: int = 10) -> float | None:
        return self._last_valid(self._get(closes, "mom", int(n)))

    def bb_upper(self, closes: pd.Series, n: int = 20, k: float = 2.0) -> float | None:
        return self._last_valid(self._get(closes, "bb_upper", int(n), float(k)))

    def bb_lower(self, closes: pd.Series, n: int = 20, k: float = 2.0) -> float | None:
        return self._last_valid(self._get(closes, "bb_lower", int(n), float(k)))

    def bb_width(self, closes: pd.Series, n: int = 20, k: float = 2.0) -> float | None:
        return self._last_valid(self._get(closes, "bb_width", int(n), float(k)))

    def macd_line(self, closes: pd.Series, fast: int = 12, slow: int = 26) -> float | None:
        return self._last_valid(self._get(closes, "macd_line", int(fast), int(slow)))

    def macd_signal(self, closes: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> float | None:
        return self._last_valid(self._get(closes, "macd_signal", int(fast), int(slow), int(signal)))

    def macd_hist(self, closes: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> float | None:
        return self._last_valid(self._get(closes, "macd_hist", int(fast), int(slow), int(signal)))

    def atr(self, closes: pd.Series, n: int = 14, *, highs: pd.Series | None = None, lows: pd.Series | None = None) -> float | None:
        c = closes.dropna().astype(float)
        h = highs.reindex(c.index).astype(float) if highs is not None and len(highs) else c
        l = lows.reindex(c.index).astype(float) if lows is not None and len(lows) else c
        prev_c = c.shift(1)
        tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        return self._last_valid(tr.rolling(int(n)).mean())

    def adx(self, closes: pd.Series, n: int = 14, *, highs: pd.Series | None = None, lows: pd.Series | None = None) -> float | None:
        c = closes.dropna().astype(float)
        h = highs.reindex(c.index).astype(float) if highs is not None and len(highs) else c
        l = lows.reindex(c.index).astype(float) if lows is not None and len(lows) else c
        up_move = h.diff()
        down_move = -l.diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        prev_c = c.shift(1)
        tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        atr = tr.rolling(int(n)).mean()
        plus_di = 100.0 * (plus_dm.rolling(int(n)).mean() / atr.replace(0, np.nan))
        minus_di = 100.0 * (minus_dm.rolling(int(n)).mean() / atr.replace(0, np.nan))
        dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        adx = dx.rolling(int(n)).mean()
        return self._last_valid(adx)

    def stoch_k(self, closes: pd.Series, n: int = 14, *, highs: pd.Series | None = None, lows: pd.Series | None = None) -> float | None:
        c = closes.dropna().astype(float)
        h = highs.reindex(c.index).astype(float) if highs is not None and len(highs) else c
        l = lows.reindex(c.index).astype(float) if lows is not None and len(lows) else c
        lo = l.rolling(int(n)).min(); hi = h.rolling(int(n)).max()
        k = 100.0 * ((c - lo) / (hi - lo).replace(0, np.nan))
        return self._last_valid(k)

    def stoch_d(self, closes: pd.Series, n: int = 14, d: int = 3, *, highs: pd.Series | None = None, lows: pd.Series | None = None) -> float | None:
        c = closes.dropna().astype(float)
        h = highs.reindex(c.index).astype(float) if highs is not None and len(highs) else c
        l = lows.reindex(c.index).astype(float) if lows is not None and len(lows) else c
        lo = l.rolling(int(n)).min(); hi = h.rolling(int(n)).max()
        k = 100.0 * ((c - lo) / (hi - lo).replace(0, np.nan))
        return self._last_valid(k.rolling(int(d)).mean())

    def cci(self, closes: pd.Series, n: int = 20, *, highs: pd.Series | None = None, lows: pd.Series | None = None) -> float | None:
        c = closes.dropna().astype(float)
        h = highs.reindex(c.index).astype(float) if highs is not None and len(highs) else c
        l = lows.reindex(c.index).astype(float) if lows is not None and len(lows) else c
        tp = (h + l + c) / 3.0
        ma = tp.rolling(int(n)).mean()
        md = (tp - ma).abs().rolling(int(n)).mean()
        cci = (tp - ma) / (0.015 * md.replace(0, np.nan))
        return self._last_valid(cci)

    def vwap(self, closes: pd.Series, *, highs: pd.Series | None = None, lows: pd.Series | None = None, volumes: pd.Series | None = None) -> float | None:
        c = closes.dropna().astype(float)
        h = highs.reindex(c.index).astype(float) if highs is not None and len(highs) else c
        l = lows.reindex(c.index).astype(float) if lows is not None and len(lows) else c
        v = volumes.reindex(c.index).astype(float) if volumes is not None and len(volumes) else pd.Series(1.0, index=c.index)
        tp = (h + l + c) / 3.0
        num = (tp * v).cumsum()
        den = v.cumsum().replace(0, np.nan)
        return self._last_valid(num / den)


indicator_service = IndicatorService()
