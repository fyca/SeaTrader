#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def venv_python() -> str:
    if os.name == "nt":
        p = ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        p = ROOT / ".venv" / "bin" / "python"
    if p.exists():
        return str(p)
    return sys.executable


def ensure_venv() -> None:
    venv_dir = ROOT / ".venv"
    if not venv_dir.exists():
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)], cwd=str(ROOT))


def main() -> int:
    ensure_venv()
    py = venv_python()
    subprocess.check_call([py, "-m", "pip", "install", "-e", "."], cwd=str(ROOT))

    host = os.getenv("HOST", "127.0.0.1")
    port = os.getenv("PORT", "8008")
    config = os.getenv("CONFIG", "config/config.yaml")

    cmd = [py, "-m", "tradebot.cli", "dashboard", "--config", config, "--host", host, "--port", str(port)]
    os.execv(cmd[0], cmd)


if __name__ == "__main__":
    raise SystemExit(main())
