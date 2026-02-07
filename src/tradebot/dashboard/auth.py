from __future__ import annotations

import os


def require_token_enabled() -> bool:
    # Default: require token (safer). Set DASHBOARD_REQUIRE_TOKEN=false to disable.
    v = os.getenv("DASHBOARD_REQUIRE_TOKEN", "true").strip().lower()
    return v not in ("0", "false", "no", "n")


def dashboard_token() -> str | None:
    t = os.getenv("DASHBOARD_TOKEN")
    if not t:
        return None
    t = t.strip()
    return t or None


def check_token(provided: str | None) -> bool:
    if not require_token_enabled():
        return True

    expected = dashboard_token()
    if expected is None:
        # Token required but not set
        return False
    return (provided or "") == expected
