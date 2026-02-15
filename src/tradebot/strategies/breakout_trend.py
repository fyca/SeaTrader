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
                details[sym] = {"eligible": False, "reject_reason": "missing_close_data"}
                continue
            closes = df["close"].dropna()
            if len(closes) < max(sp.ma_long, 50):
                details[sym] = {"eligible": False, "reject_reason": "insufficient_history", "bars": int(len(closes))}
                continue

            # Optional liquidity filters
            adv = None
            if "volume" in df.columns:
                adv = float((df.tail(20)["close"] * df.tail(20)["volume"]).dropna().mean())
                if is_crypto:
                    if adv < cfg.limits.min_avg_crypto_dollar_volume_20d:
                        details[sym] = {
                            "eligible": False,
                            "reject_reason": "below_min_crypto_adv",
                            "adv20": adv,
                            "min_adv20": float(cfg.limits.min_avg_crypto_dollar_volume_20d),
                        }
                        continue
                else:
                    last = float(closes.iloc[-1])
                    if last < cfg.limits.min_stock_price:
                        details[sym] = {
                            "eligible": False,
                            "reject_reason": "below_min_stock_price",
                            "last_close": last,
                            "min_stock_price": float(cfg.limits.min_stock_price),
                        }
                        continue
                    if adv < cfg.limits.min_avg_dollar_volume_20d:
                        details[sym] = {
                            "eligible": False,
                            "reject_reason": "below_min_equity_adv",
                            "adv20": adv,
                            "min_adv20": float(cfg.limits.min_avg_dollar_volume_20d),
                        }
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
                details[sym] = {
                    "eligible": False,
                    "reject_reason": f"trend_vol_{sig.reason}",
                    "last_close": sig.last_close,
                    "ann_vol": sig.ann_vol,
                    "ma_long": sig.ma_long,
                    "ma_short": sig.ma_short,
                }
                continue

            # Breakout condition: close >= max(close over last 20 trading days)
            lookback = 20
            if len(closes) < lookback + 5:
                details[sym] = {"eligible": False, "reject_reason": "insufficient_breakout_window", "bars": int(len(closes))}
                continue
            highest = float(closes.tail(lookback).max())
            last_close = float(closes.iloc[-1])
            if last_close < highest:
                details[sym] = {
                    "eligible": False,
                    "reject_reason": "no_20d_breakout",
                    "last_close": last_close,
                    "highest_20d": highest,
                    "breakout_gap_pct": ((last_close / highest) - 1.0) if highest > 0 else None,
                    "ann_vol": sig.ann_vol,
                    "ma_long": sig.ma_long,
                    "ma_short": sig.ma_short,
                }
                continue

            # Score: base trend score + breakout bonus
            breakout_bonus = 0.05
            score = float(sig.score) + breakout_bonus
            ok.append((sym, score))
            details[sym] = {
                "eligible": True,
                "score": score,
                "reason": "breakout_20d",
                "last_close": sig.last_close,
                "ann_vol": sig.ann_vol,
                "ma_long": sig.ma_long,
                "ma_short": sig.ma_short,
                "highest_20d": highest,
                "adv20": adv,
            }

        ranked = sorted(ok, key=lambda x: x[1], reverse=True)
        sel = [s for s, _ in ranked[:max_pos]]
        rank_map = {s: i + 1 for i, (s, _v) in enumerate(ranked)}
        for s, _v in ranked:
            d = details.setdefault(s, {})
            d["rank"] = int(rank_map.get(s, 0))
            d["selected"] = bool(s in sel)
            if not d.get("selected"):
                d["reject_reason"] = "rank_below_cutoff"
        return sel, details

    def select_equities(self, *, bars: dict[str, pd.DataFrame], cfg):
        return self._select(bars=bars, cfg=cfg, is_crypto=False)

    def select_crypto(self, *, bars: dict[str, pd.DataFrame], cfg):
        return self._select(bars=bars, cfg=cfg, is_crypto=True)
