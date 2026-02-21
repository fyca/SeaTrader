from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE_DIR = Path("data/live_ledger")
RUNS_PATH = BASE_DIR / "runs.jsonl"
EVENTS_PATH = BASE_DIR / "events.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True, default=str) + "\n")


def append_live_run(*, run_id: str | None, kind: str, payload: dict[str, Any], ts: str | None = None) -> None:
    _append_jsonl(
        RUNS_PATH,
        {
            "ts": ts or _now_iso(),
            "run_id": run_id,
            "kind": kind,
            "payload": payload,
        },
    )


def append_live_events(*, run_id: str | None, kind: str, events: list[dict[str, Any]], ts: str | None = None) -> None:
    base_ts = ts or _now_iso()
    for e in events or []:
        row = {
            "ts": e.get("ts") or base_ts,
            "run_id": run_id,
            "kind": kind,
            **e,
        }
        _append_jsonl(EVENTS_PATH, row)


def read_jsonl(path: Path, *, limit: int = 200) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            rows.append(json.loads(ln))
        except Exception:
            continue
    rows = rows[-max(1, min(int(limit), 5000)) :]
    rows.reverse()
    return rows


def get_runs(*, limit: int = 200) -> list[dict[str, Any]]:
    return read_jsonl(RUNS_PATH, limit=limit)


def get_events(*, limit: int = 500, run_id: str | None = None, kind: str | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(EVENTS_PATH, limit=limit * 5)
    out: list[dict[str, Any]] = []
    for r in rows:
        if run_id and str(r.get("run_id") or "") != str(run_id):
            continue
        if kind and str(r.get("kind") or "") != str(kind):
            continue
        out.append(r)
        if len(out) >= max(1, min(int(limit), 5000)):
            break
    return out
