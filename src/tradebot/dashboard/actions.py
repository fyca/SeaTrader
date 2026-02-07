from __future__ import annotations

from fastapi import HTTPException, Request

from tradebot.dashboard.auth import check_token, require_token_enabled


def require_token(req: Request) -> None:
    if not require_token_enabled():
        return
    token = req.query_params.get("token") or req.headers.get("x-dashboard-token")
    if not check_token(token):
        raise HTTPException(status_code=401, detail="Unauthorized")
