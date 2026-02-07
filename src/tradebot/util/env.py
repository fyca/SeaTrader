from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Env:
    key_id: str
    secret_key: str
    paper: bool
    dry_run: bool


def load_env() -> Env:
    # Loads from .env in cwd if present
    load_dotenv(override=False)

    key_id = os.getenv("APCA_API_KEY_ID", "").strip()
    secret = os.getenv("APCA_API_SECRET_KEY", "").strip()
    paper = os.getenv("APCA_PAPER", "true").lower() in ("1", "true", "yes", "y")
    dry_run = os.getenv("DRY_RUN", "true").lower() in ("1", "true", "yes", "y")

    if not key_id or not secret:
        raise RuntimeError("Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY in environment")

    return Env(key_id=key_id, secret_key=secret, paper=paper, dry_run=dry_run)
