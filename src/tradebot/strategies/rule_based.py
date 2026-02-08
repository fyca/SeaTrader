from __future__ import annotations

import pandas as pd

from tradebot.strategies.rule_engine import EvalContext, eval_rule, eval_score


class RuleBasedStrategy:
    def __init__(self, spec: dict):
        self.spec = spec
        self.id = spec.get("id")
        self.name = spec.get("name") or self.id

    def _select(self, *, bars: dict[str, pd.DataFrame], cfg, is_crypto: bool):
        entry_rule = self.spec.get("entry") or {"all": []}
        factors = self.spec.get("score_factors") or []
        max_pos = cfg.limits.max_crypto_positions if is_crypto else cfg.limits.max_equity_positions
        ann_factor = 365.0 if is_crypto else 252.0

        ok: list[tuple[str, float]] = []
        details: dict[str, dict] = {}

        for sym, df in bars.items():
            if df is None or len(df) == 0 or "close" not in df.columns:
                continue
            closes = df["close"].dropna()
            if len(closes) < 60:
                continue
            ctx = EvalContext(closes=closes, ann_factor=ann_factor)
            if not eval_rule(ctx, entry_rule):
                continue
            score = eval_score(ctx, factors)
            ok.append((sym, float(score)))
            details[sym] = {"score": float(score), "reason": "rule_based"}

        sel = [s for s, _ in sorted(ok, key=lambda x: x[1], reverse=True)[:max_pos]]
        return sel, {s: details.get(s) for s in sel}

    def select_equities(self, *, bars: dict[str, pd.DataFrame], cfg):
        return self._select(bars=bars, cfg=cfg, is_crypto=False)

    def select_crypto(self, *, bars: dict[str, pd.DataFrame], cfg):
        return self._select(bars=bars, cfg=cfg, is_crypto=True)
