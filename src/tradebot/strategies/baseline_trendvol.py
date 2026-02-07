from __future__ import annotations

import pandas as pd

from tradebot.signals.trend_vol import compute_trend_vol_signal


class BaselineTrendVolStrategy:
    id = "baseline_trendvol"
    name = "Baseline trend/vol"

    def select_equities(self, *, bars: dict[str, pd.DataFrame], cfg):
        ok: list[tuple[str, float, float]] = []
        details: dict[str, dict] = {}
        for sym, df in bars.items():
            if df is None or len(df) == 0 or "close" not in df.columns:
                continue
            closes = df["close"].dropna()
            if len(closes) == 0:
                continue
            last = float(closes.iloc[-1])
            if last < cfg.limits.min_stock_price:
                continue
            if "volume" in df.columns:
                adv = float((df.tail(20)["close"] * df.tail(20)["volume"]).dropna().mean())
                if adv < cfg.limits.min_avg_dollar_volume_20d:
                    continue
            sig = compute_trend_vol_signal(
                closes,
                ma_long=cfg.signals.equity.ma_long,
                ma_short=cfg.signals.equity.ma_short,
                vol_lookback=cfg.signals.equity.vol_lookback,
                max_ann_vol=cfg.signals.equity.max_ann_vol,
                ann_factor=252.0,
            )
            if sig.ok:
                ok.append((sym, float(sig.score), 0.0))
                details[sym] = {
                    "score": float(sig.score),
                    "reason": sig.reason,
                    "last_close": sig.last_close,
                    "ann_vol": sig.ann_vol,
                    "ma_long": sig.ma_long,
                    "ma_short": sig.ma_short,
                }
        sel = [s for s, _, _ in sorted(ok, key=lambda x: x[1], reverse=True)[: cfg.limits.max_equity_positions]]
        return sel, {s: details.get(s) for s in sel}

    def select_crypto(self, *, bars: dict[str, pd.DataFrame], cfg):
        ok: list[tuple[str, float]] = []
        details: dict[str, dict] = {}
        for sym, df in bars.items():
            if df is None or len(df) == 0 or "close" not in df.columns:
                continue
            closes = df["close"].dropna()
            if len(closes) == 0:
                continue
            if "volume" in df.columns:
                adv = float((df.tail(20)["close"] * df.tail(20)["volume"]).dropna().mean())
                if adv < cfg.limits.min_avg_crypto_dollar_volume_20d:
                    continue
            sig = compute_trend_vol_signal(
                closes,
                ma_long=cfg.signals.crypto.ma_long,
                ma_short=cfg.signals.crypto.ma_short,
                vol_lookback=cfg.signals.crypto.vol_lookback,
                max_ann_vol=cfg.signals.crypto.max_ann_vol,
                ann_factor=365.0,
            )
            if sig.ok:
                ok.append((sym, float(sig.score)))
                details[sym] = {
                    "score": float(sig.score),
                    "reason": sig.reason,
                    "last_close": sig.last_close,
                    "ann_vol": sig.ann_vol,
                    "ma_long": sig.ma_long,
                    "ma_short": sig.ma_short,
                }
        sel = [s for s, _ in sorted(ok, key=lambda x: x[1], reverse=True)[: cfg.limits.max_crypto_positions]]
        return sel, {s: details.get(s) for s in sel}
