from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def sma(closes: pd.Series, n: int) -> float | None:
    if n <= 0:
        return None
    s = closes.dropna()
    if len(s) < n:
        return None
    return float(s.rolling(n).mean().iloc[-1])


def highest(closes: pd.Series, n: int) -> float | None:
    if n <= 0:
        return None
    s = closes.dropna()
    if len(s) < n:
        return None
    return float(s.tail(n).max())


def lowest(closes: pd.Series, n: int) -> float | None:
    if n <= 0:
        return None
    s = closes.dropna()
    if len(s) < n:
        return None
    return float(s.tail(n).min())


def ema(closes: pd.Series, n: int) -> float | None:
    if n <= 0:
        return None
    s = closes.dropna().astype(float)
    if len(s) < n:
        return None
    v = float(s.ewm(span=n, adjust=False).mean().iloc[-1])
    return v if np.isfinite(v) else None


def roc(closes: pd.Series, n: int) -> float | None:
    """Rate of change over n days as fractional return (close/close[-n]-1)."""
    if n <= 0:
        return None
    s = closes.dropna().astype(float)
    if len(s) < n + 1:
        return None
    prev = float(s.iloc[-(n + 1)])
    now = float(s.iloc[-1])
    if prev == 0:
        return None
    return float(now / prev - 1.0)


def ret_1d(closes: pd.Series) -> float | None:
    s = closes.dropna().astype(float)
    if len(s) < 2:
        return None
    prev = float(s.iloc[-2])
    now = float(s.iloc[-1])
    if prev == 0:
        return None
    return float(now / prev - 1.0)


def ann_vol(closes: pd.Series, n: int, ann_factor: float) -> float | None:
    s = closes.dropna()
    if len(s) < n + 2:
        return None
    rets = s.pct_change().dropna()
    v = float(rets.tail(n).std(ddof=0) * np.sqrt(ann_factor))
    return v if np.isfinite(v) else None


def rsi(closes: pd.Series, n: int = 14) -> float | None:
    s = closes.dropna().astype(float)
    if len(s) < n + 2:
        return None
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(n).mean().iloc[-1]
    avg_loss = loss.rolling(n).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def close(closes: pd.Series) -> float | None:
    s = closes.dropna()
    if len(s) == 0:
        return None
    return float(s.iloc[-1])


@dataclass(frozen=True)
class EvalContext:
    closes: pd.Series
    ann_factor: float


def eval_indicator(ctx: EvalContext, spec: dict) -> float | None:
    kind = (spec.get("kind") or "").lower()
    if kind == "close":
        return close(ctx.closes)
    if kind == "sma":
        return sma(ctx.closes, int(spec.get("n", 0)))
    if kind == "rsi":
        return rsi(ctx.closes, int(spec.get("n", 14)))
    if kind == "highest":
        return highest(ctx.closes, int(spec.get("n", 0)))
    if kind == "lowest":
        return lowest(ctx.closes, int(spec.get("n", 0)))
    if kind == "ema":
        return ema(ctx.closes, int(spec.get("n", 0)))
    if kind == "roc":
        return roc(ctx.closes, int(spec.get("n", 20)))
    if kind == "ret_1d":
        return ret_1d(ctx.closes)
    if kind == "ann_vol":
        return ann_vol(ctx.closes, int(spec.get("n", 20)), ctx.ann_factor)

    # Derived / preset-style indicators
    if kind == "dist_sma":
        n = int(spec.get("n", 0))
        c = close(ctx.closes)
        m = sma(ctx.closes, n)
        if c is None or m is None or m == 0:
            return None
        return float(c / m - 1.0)

    if kind == "breakout":
        n = int(spec.get("n", 20))
        c = close(ctx.closes)
        h = highest(ctx.closes, n)
        if c is None or h is None:
            return None
        return 1.0 if c >= h else 0.0

    if kind == "cross_above":
        # returns 1 if SMA(n_fast) crossed above SMA(n_slow) today
        nf = int(spec.get("fast", 0))
        ns = int(spec.get("slow", 0))
        s = ctx.closes.dropna().astype(float)
        if len(s) < max(nf, ns) + 2:
            return None
        f = s.rolling(nf).mean()
        g = s.rolling(ns).mean()
        if pd.isna(f.iloc[-2]) or pd.isna(g.iloc[-2]) or pd.isna(f.iloc[-1]) or pd.isna(g.iloc[-1]):
            return None
        prev = f.iloc[-2] <= g.iloc[-2]
        now = f.iloc[-1] > g.iloc[-1]
        return 1.0 if (prev and now) else 0.0

    if kind == "sma_slope":
        # slope of SMA(n) over lookback days, as fractional change
        n = int(spec.get("n", 0))
        lb = int(spec.get("lookback", 5))
        s = ctx.closes.dropna().astype(float)
        if len(s) < n + lb + 2:
            return None
        ma = s.rolling(n).mean()
        ma_now = ma.iloc[-1]
        ma_prev = ma.iloc[-(lb + 1)]
        if pd.isna(ma_now) or pd.isna(ma_prev) or ma_prev == 0:
            return None
        return float((ma_now / ma_prev) - 1.0)

    raise ValueError(f"unknown indicator kind: {kind}")


def eval_condition(ctx: EvalContext, cond: dict) -> bool:
    # cond: {left: <indicator>, op: '>'|'>='|'<'|'<='|'==' , right: <indicator|number>}
    left = eval_indicator(ctx, cond.get("left") or {})
    if left is None:
        return False

    op = cond.get("op")
    right_spec = cond.get("right")
    if isinstance(right_spec, (int, float)):
        right = float(right_spec)
    elif isinstance(right_spec, dict):
        right = eval_indicator(ctx, right_spec)
    else:
        right = None

    if right is None:
        return False

    if op == ">":
        return left > right
    if op == ">=":
        return left >= right
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    if op == "==":
        return left == right
    return False


def eval_rule(ctx: EvalContext, rule: dict) -> bool:
    # rule can be:
    # {"all": [rule|condition, ...]}  AND
    # {"any": [rule|condition, ...]}  OR
    # or a condition object (has 'left'/'op')
    if "all" in rule:
        items = rule.get("all") or []
        return all(eval_rule(ctx, it) for it in items)
    if "any" in rule:
        items = rule.get("any") or []
        return any(eval_rule(ctx, it) for it in items)
    if "left" in rule and "op" in rule:
        return eval_condition(ctx, rule)
    return False


def eval_score(ctx: EvalContext, factors: list[dict]) -> float:
    return eval_score_with_breakdown(ctx, factors)[0]


def eval_score_with_breakdown(ctx: EvalContext, factors: list[dict]) -> tuple[float, list[dict]]:
    # factors: [{weight: float, value: indicator|number|condition_as_bool}, ...]
    total = 0.0
    rows: list[dict] = []
    for f in factors or []:
        w = float(f.get("weight", 0.0))
        val_spec = f.get("value")
        v: float | None
        kind = None

        if isinstance(val_spec, (int, float)):
            v = float(val_spec)
            kind = "number"
        elif isinstance(val_spec, dict) and ("left" in val_spec and "op" in val_spec):
            v = 1.0 if eval_condition(ctx, val_spec) else 0.0
            kind = "condition"
        elif isinstance(val_spec, dict):
            v = eval_indicator(ctx, val_spec)
            kind = val_spec.get("kind")
        else:
            v = None

        if v is None:
            rows.append({"weight": w, "value": None, "contrib": None, "spec": val_spec, "kind": kind})
            continue
        contrib = w * float(v)
        total += contrib
        rows.append({"weight": w, "value": float(v), "contrib": float(contrib), "spec": val_spec, "kind": kind})

    return float(total), rows
