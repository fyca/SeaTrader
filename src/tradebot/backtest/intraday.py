from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from tradebot.adapters.bars import fetch_crypto_bars_range_1m, fetch_stock_bars_range_1m


@dataclass
class IntradayPriceProvider:
    stocks_client: object
    crypto_client: object
    exec_time_local: str
    tz: str

    def __post_init__(self):
        self._tz = ZoneInfo(self.tz)
        hh, mm = [int(x) for x in self.exec_time_local.split(":")]
        self._t_local = time(hh, mm)
        self._cache: dict[tuple[str, str], float | None] = {}
        self._day_bars_cache: dict[tuple[str, str], pd.DataFrame | None] = {}

    def _target_utc(self, day: pd.Timestamp) -> datetime:
        # day is tz-naive date; interpret as local date
        d = pd.to_datetime(day).date()
        dt_local = datetime(d.year, d.month, d.day, self._t_local.hour, self._t_local.minute, tzinfo=self._tz)
        return dt_local.astimezone(ZoneInfo("UTC"))

    def _day_bars(self, sym: str, day: pd.Timestamp) -> pd.DataFrame | None:
        key = (sym, day.strftime("%Y-%m-%d"))
        if key in self._day_bars_cache:
            return self._day_bars_cache[key]
        target = self._target_utc(day)
        start = target - timedelta(hours=12)
        end = target + timedelta(hours=12)
        try:
            if "/" in sym:
                bars = fetch_crypto_bars_range_1m(self.crypto_client, [sym], start=start, end=end)
            else:
                bars = fetch_stock_bars_range_1m(self.stocks_client, [sym], start=start, end=end)
            df = bars.get(sym)
            if df is None or len(df) == 0:
                self._day_bars_cache[key] = None
                return None
            df = df.sort_index()
            self._day_bars_cache[key] = df
            return df
        except Exception:
            self._day_bars_cache[key] = None
            return None

    def price(self, sym: str, day: pd.Timestamp) -> float | None:
        key = (sym, day.strftime("%Y-%m-%d"))
        if key in self._cache:
            return self._cache[key]

        target = self._target_utc(day)
        start = target - timedelta(hours=6)
        end = target + timedelta(minutes=2)

        try:
            df = self._day_bars(sym, day)
            if df is None or len(df) == 0:
                self._cache[key] = None
                return None
            # pick last bar at/before target
            df2 = df.loc[: pd.to_datetime(target)]
            if len(df2) == 0:
                self._cache[key] = None
                return None
            # use close of minute bar
            v = float(df2["close"].iloc[-1])
            self._cache[key] = v
            return v
        except Exception:
            self._cache[key] = None
            return None

    def limit_touched(self, sym: str, day: pd.Timestamp, side: str, limit_px: float) -> bool:
        df = self._day_bars(sym, day)
        if df is None or len(df) == 0:
            return False
        try:
            if str(side).lower() == "buy":
                if "low" not in df.columns:
                    return False
                return bool((df["low"].astype(float) <= float(limit_px)).any())
            else:
                if "high" not in df.columns:
                    return False
                return bool((df["high"].astype(float) >= float(limit_px)).any())
        except Exception:
            return False
