from __future__ import annotations

import pandas as pd

from tradebot.risk.exits import trend_break_exit
from tradebot.strategies.baseline_trendvol import BaselineTrendVolStrategy


class RegimeFilteredTrendVolStrategy(BaselineTrendVolStrategy):
    id = "regime_filtered_trendvol"
    name = "Trend/vol + regime filter (SPY/BTC)"

    def select_equities(self, *, bars: dict[str, pd.DataFrame], cfg):
        # Require SPY in uptrend (close >= MA_long) to allocate equities
        spy = bars.get("SPY")
        if spy is not None and len(spy) and "close" in spy.columns:
            should_exit, reason, last, maL = trend_break_exit(spy["close"], ma_long=cfg.signals.equity.ma_long)
            if should_exit:
                return [], {}
        return super().select_equities(bars=bars, cfg=cfg)

    def select_crypto(self, *, bars: dict[str, pd.DataFrame], cfg):
        # Require BTC/USD in uptrend to allocate crypto
        btc = bars.get("BTC/USD")
        if btc is not None and len(btc) and "close" in btc.columns:
            should_exit, reason, last, maL = trend_break_exit(btc["close"], ma_long=cfg.signals.crypto.ma_long)
            if should_exit:
                return [], {}
        return super().select_crypto(bars=bars, cfg=cfg)
