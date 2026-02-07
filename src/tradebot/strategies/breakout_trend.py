from __future__ import annotations

import pandas as pd

from tradebot.signals.trend_vol import compute_trend_vol_signal


class BreakoutTrendStrategy:
    id = "breakout_trend"
    name = "Breakout trend (20D high + trend/vol filter)"

    def _select(self, *, bars: dict[str, pd.DataFrame], cfg, is_crypto: bool):
        ok: list[tuple[str, float]] = []
        details: dict[str, dict] = {}

        sp = cfg.signals.crypto if is_crypto else cfg.signals.equity
        ann_factor = 365.0 if is_crypto else 252.0
        max_pos = cfg.limits.max_crypto_positions if is_crypto else cfg.limits.max_equity_positions

        for sym, df in bars.items():
            if df is None or len(df) == 0 or "close" not in df.columns:
                continue
            closes = df["close"].dropna()
            if len(closes) < max(sp.ma_long, 50):
                continue

            # Optional liquidity filters
            if "volume" in df.columns:
                adv = float((df.tail(20)["close"] * df.tail(20)["volume"]).dropna().mean())
                if is_crypto:
                    if adv < cfg.limits.min_avg_crypto_dollar_volume_20d:
                        continue
                else:
                    last = float(closes.iloc[-1])
                    if last < cfg.limits.min_stock_price:
                        continue
                    if adv < cfg.limits.min_avg_dollar_volume_20d:
                        continue

            sig = compute_trend_vol_signal(
                closes,
                ma_long=sp.ma_long,
                ma_short=sp.ma_short,
                vol_lookback=sp.vol_lookback,
                max_ann_vol=sp.max_ann_vol,
                ann_factor=ann_factor,
            )
            if not sig.ok:
                continue

            # Breakout condition: close >= max(close over last 20 trading days)
            lookback = 20
            if len(closes) < lookback + 5:
                continue
            highest = float(closes.tail(lookback).max())
            last_close = float(closes.iloc[-1])
            if last_close < highest:
                continue

            # Score: base trend score + breakout bonus
            breakout_bonus = 0.05
            score = float(sig.score) + breakout_bonus
            ok.append((sym, score))
            details[sym] = {
                "score": score,
                "reason": "breakout_20d",
                "last_close": sig.last_close,
                "ann_vol": sig.ann_vol,
                "ma_long": sig.ma_long,
                "ma_short": sig.ma_short,
                "highest_20d": highest,
            }

        sel = [s for s, _ in sorted(ok, key=lambda x: x[1], reverse=True)[:max_pos]]
        return sel, {s: details.get(s) for s in sel}

    def select_equities(self, *, bars: dict[str, pd.DataFrame], cfg):
        return self._select(bars=bars, cfg=cfg, is_crypto=False)

    def select_crypto(self, *, bars: dict[str, pd.DataFrame], cfg):
        return self._select(bars=bars, cfg=cfg, is_crypto=True)
