#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # tradebot/multibot
REPO = ROOT.parent
PID_DIR = ROOT / "pids"
LOG_DIR = ROOT / "logs"

BOTS = {
    "alpha": 8008,
    "beta": 8009,
    "gamma": 8010,
    "delta": 8011,
    "epsilon": 8012,
    "zeta": 8013,
    "eta": 8014,
    "theta": 8015,
    "iota": 8016,
}


def _load_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for ln in path.read_text().splitlines():
        s = ln.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _venv_python() -> str:
    if os.name == "nt":
        p = REPO / ".venv" / "Scripts" / "python.exe"
    else:
        p = REPO / ".venv" / "bin" / "python"
    if p.exists():
        return str(p)
    return sys.executable


def _pid_path(bot: str) -> Path:
    return PID_DIR / f"{bot}-dashboard.pid"


def _is_running(pid: int) -> bool:
    try:
        if pid <= 0:
            return False
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _spawn_dashboard(bot: str, port: int) -> tuple[bool, str]:
    bot_dir = ROOT / "bots" / bot
    cfg = bot_dir / "config" / "config.yaml"
    if not bot_dir.exists() or not cfg.exists():
        return False, f"Missing bot/config for {bot}: {cfg}"

    PID_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    pid_file = _pid_path(bot)
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            if _is_running(pid):
                return True, f"{bot} dashboard already running (pid {pid})"
        except Exception:
            pass

    log_path = LOG_DIR / f"{bot}-dashboard.log"
    log_f = open(log_path, "ab")
    py = _venv_python()
    cmd = [
        py,
        "-m",
        "tradebot.cli",
        "dashboard",
        "--config",
        str(cfg),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]

    child_env = os.environ.copy()
    child_env.update(_load_env_file(bot_dir / ".env"))

    kwargs = {
        "cwd": str(bot_dir),
        "stdout": log_f,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
        "close_fds": True,
        "env": child_env,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True

    p = subprocess.Popen(cmd, **kwargs)
    pid_file.write_text(str(p.pid))
    return True, f"Started {bot} dashboard on :{port} (pid {p.pid})"


def _stop_dashboard(bot: str) -> tuple[bool, str]:
    pid_file = _pid_path(bot)
    if not pid_file.exists():
        return True, f"{bot} dashboard pid file not found"
    try:
        pid = int(pid_file.read_text().strip())
    except Exception:
        pid_file.unlink(missing_ok=True)
        return True, f"{bot} invalid pid file removed"

    if not _is_running(pid):
        pid_file.unlink(missing_ok=True)
        return True, f"{bot} dashboard not running"

    try:
        if os.name == "nt":
            os.kill(pid, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
        msg = f"Stopped {bot} dashboard (pid {pid})"
    except Exception as e:
        msg = f"Failed stopping {bot} (pid {pid}): {e}"
        return False, msg
    finally:
        pid_file.unlink(missing_ok=True)
    return True, msg


def cmd_start_all() -> int:
    ok_all = True
    for b, p in BOTS.items():
        ok, msg = _spawn_dashboard(b, p)
        print(msg)
        ok_all = ok_all and ok
    return 0 if ok_all else 1


def cmd_stop_all() -> int:
    ok_all = True
    for b in BOTS:
        ok, msg = _stop_dashboard(b)
        print(msg)
        ok_all = ok_all and ok
    return 0 if ok_all else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Cross-platform dashboard control")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("start-all")
    sub.add_parser("stop-all")

    p1 = sub.add_parser("start")
    p1.add_argument("bot", choices=sorted(BOTS.keys()))

    p2 = sub.add_parser("stop")
    p2.add_argument("bot", choices=sorted(BOTS.keys()))

    args = ap.parse_args()

    if args.cmd == "start-all":
        return cmd_start_all()
    if args.cmd == "stop-all":
        return cmd_stop_all()
    if args.cmd == "start":
        ok, msg = _spawn_dashboard(args.bot, BOTS[args.bot])
        print(msg)
        return 0 if ok else 1
    if args.cmd == "stop":
        ok, msg = _stop_dashboard(args.bot)
        print(msg)
        return 0 if ok else 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
