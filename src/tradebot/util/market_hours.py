from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from typing import Any


def _to_iso(x: Any) -> str | None:
    if x is None:
        return None
    try:
        if isinstance(x, datetime):
            return x.astimezone(timezone.utc).isoformat()
    except Exception:
        pass
    s = str(x).strip()
    return s or None


def _to_dt(x: Any) -> datetime | None:
    if x is None:
        return None
    if isinstance(x, datetime):
        return x if x.tzinfo else x.replace(tzinfo=timezone.utc)
    s = str(x).strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def get_market_status(trading_client) -> dict[str, Any]:
    """Return equity market/holiday status for guardrails + dashboard.

    Alpaca clock is regular-hours only. We add an approximate extended-hours window
    (04:00-20:00 America/New_York) on trading days.
    """
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    now_utc = datetime.now(timezone.utc)
    now_et = now_utc.astimezone(et)

    clock = None
    try:
        clock = trading_client.get_clock()
    except Exception:
        clock = None

    clock_is_open = bool(getattr(clock, "is_open", False)) if clock is not None else False
    next_open = getattr(clock, "next_open", None) if clock is not None else None
    next_close = getattr(clock, "next_close", None) if clock is not None else None

    is_trading_day = None
    cal_open = None
    cal_close = None
    try:
        start = now_et.date().isoformat()
        end = (now_et.date() + timedelta(days=7)).isoformat()
        cal_rows = trading_client.get_calendar(start=start, end=end)
        today_row = None
        for r in cal_rows or []:
            d = str(getattr(r, "date", "") or "")
            if d.startswith(start):
                today_row = r
                break
        if today_row is not None:
            is_trading_day = True
            cal_open = getattr(today_row, "open", None)
            cal_close = getattr(today_row, "close", None)
        else:
            is_trading_day = False
    except Exception:
        is_trading_day = None

    if is_trading_day is None:
        is_trading_day = now_et.weekday() < 5

    t = now_et.timetz().replace(tzinfo=None)
    in_extended_window = (time(4, 0) <= t < time(20, 0)) if is_trading_day else False
    can_place_equity_orders = bool(clock_is_open or in_extended_window)

    if clock_is_open:
        phase = "regular"
    elif in_extended_window and t < time(9, 30):
        phase = "pre"
    elif in_extended_window and t >= time(16, 0):
        phase = "post"
    else:
        phase = "closed"

    return {
        "now_utc": now_utc.isoformat(),
        "now_et": now_et.isoformat(),
        "clock_is_open": bool(clock_is_open),
        "is_trading_day": bool(is_trading_day),
        "market_phase": phase,
        "in_extended_window": bool(in_extended_window),
        "can_place_equity_orders": bool(can_place_equity_orders),
        "next_open": _to_iso(next_open),
        "next_close": _to_iso(next_close),
        "calendar_open": _to_iso(cal_open),
        "calendar_close": _to_iso(cal_close),
    }
