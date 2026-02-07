from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DrawdownState:
    peak_equity: float
    current_equity: float
    drawdown: float  # 0..1
    frozen: bool


def update_drawdown_state(
    *,
    prior_peak_equity: float | None,
    current_equity: float,
    freeze_at: float,
) -> DrawdownState:
    peak = max(prior_peak_equity or current_equity, current_equity)
    dd = 0.0 if peak <= 0 else (peak - current_equity) / peak
    frozen = dd >= freeze_at
    return DrawdownState(peak_equity=peak, current_equity=current_equity, drawdown=dd, frozen=frozen)
