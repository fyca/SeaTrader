#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    base = Path("data/optimizer_sp500")
    chunks = [base / f"chunk_{i:02d}_result.json" for i in range(1, 6)]
    missing = [str(p) for p in chunks if not p.exists()]
    if missing:
        print(json.dumps({"ok": False, "missing": missing}, indent=2))
        return

    all_rows = []
    for p in chunks:
        obj = json.loads(p.read_text())
        for row in (obj.get("top") or []):
            row = dict(row)
            row["chunk"] = p.name
            all_rows.append(row)

    all_rows.sort(key=lambda x: float(x.get("joint") or -1e9), reverse=True)
    out = {
        "ok": True,
        "candidate_count": len(all_rows),
        "top20": all_rows[:20],
        "winner": all_rows[0] if all_rows else None,
    }
    out_path = base / "aggregate_result.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps({"ok": True, "out": str(out_path), "winner": out.get("winner")}, indent=2))


if __name__ == "__main__":
    main()
