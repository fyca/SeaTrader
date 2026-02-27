#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "data" / "config-snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    src = ROOT / "config" / "config.yaml"
    dst = out_dir / f"config-{ts}.yaml"
    shutil.copy2(src, dst)
    print(f"Saved config snapshot: {dst.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
