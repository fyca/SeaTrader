from __future__ import annotations

import pandas as pd

from tradebot.signals.trend_vol import compute_trend_vol_signal


class PullbackInTrendStrategy:
    id = "pullback_in_trend"
    name = "Pullback-in-trend (above MA_long, near MA_short)"

    def _select(self, *, bars: dict[str, pd.DataFrame], cfg, is_crypto: bool):
        sp = cfg.signals.crypto if is_crypto else cfg.signals.equity
        ann_factor = 365.0 if is_crypto else 252.0
        max_pos = cfg.limits.max_crypto_positions if is_crypto else cfg.limits.max_equity_positions

        ok: list[tuple[str, float]] = []
        details: dict[str, dict] = {}

        for sym, df in bars.items():
            if df is None or len(df) == 0 or "close" not in df.columns:
                continue
            closes = df["close"].dropna()
            if len(closes) < max(sp.ma_long, sp.ma_short) + 10:
                continue

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

            last = float(closes.iloc[-1])
            maS = float(closes.rolling(sp.ma_short).mean().iloc[-1])
            if maS <= 0:
                continue

            # Pullback condition: price within +2% of MA_short (i.e., not too extended)
            ratio = last / maS
            if ratio > 1.02:
                continue

            # Score: prefer closer to MA_short (lower ratio) while still trending
            score = float(sig.score) + (1.02 - ratio)
            ok.append((sym, score))
            details[sym] = {
                "score": score,
                "reason": "pullback_near_ma_short",
                "last_close": sig.last_close,
                "ann_vol": sig.ann_vol,
                "ma_long": sig.ma_long,
                "ma_short": sig.ma_short,
                "ma_short_value": maS,
                "close_to_ma_short_ratio": ratio,
            }

        sel = [s for s, _ in sorted(ok, key=lambda x: x[1], reverse=True)[:max_pos]]
        return sel, {s: details.get(s) for s in sel}

    def select_equities(self, *, bars: dict[str, pd.DataFrame], cfg):
        return self._select(bars=bars, cfg=cfg, is_crypto=False)

    def select_crypto(self, *, bars: dict[str, pd.DataFrame], cfg):
        return self._select(bars=bars, cfg=cfg, is_crypto=True)
