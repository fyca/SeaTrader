from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GuardrailResult:
    ok: bool
    reason: str
    total_notional: float
    count: int


def check_order_guardrails(
    plans,
    *,
    max_orders: int | None,
    max_single_notional: float | None,
    max_total_notional: float | None,
) -> GuardrailResult:
    if max_orders is not None and len(plans) > max_orders:
        return GuardrailResult(False, f"too_many_orders:{len(plans)}>{max_orders}", 0.0, len(plans))

    total = 0.0
    for p in plans:
        n = float(p.notional_usd)
        if max_single_notional is not None and n > max_single_notional:
            return GuardrailResult(False, f"single_order_too_large:{p.symbol}:{n:.2f}>{max_single_notional:.2f}", 0.0, len(plans))
        total += n

    if max_total_notional is not None and total > max_total_notional:
        return GuardrailResult(False, f"total_notional_too_large:{total:.2f}>{max_total_notional:.2f}", total, len(plans))

    return GuardrailResult(True, "ok", total, len(plans))
