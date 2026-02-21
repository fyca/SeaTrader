#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import statistics
import urllib.request
import yaml

# Repo-local paths (unified under tradebot/multibot)
REPO = Path(__file__).resolve().parents[1]          # .../tradebot
ROOT = REPO / "multibot"                           # .../tradebot/multibot
BOTS_DIR = ROOT / "bots"
HUB_HTML = ROOT / "dashboard-hub.html"
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


def run_cmd(cmd: list[str], timeout: int = 60) -> tuple[bool, str]:
    try:
        p = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, timeout=timeout)
        out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
        return (p.returncode == 0, out.strip())
    except Exception as e:
        return (False, str(e))


def _fetch_json(url: str, timeout: float = 1.8):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        raw = r.read().decode("utf-8")
        return json.loads(raw)


def is_up(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1.5) as _r:
            return True
    except Exception:
        return False


def _bot_name_by_port(port: int) -> str | None:
    for k, p in BOTS.items():
        if int(p) == int(port):
            return k
    return None


def _load_bot_selection_context(bot: str) -> tuple[str | None, dict, dict, dict]:
    strategy_id = None
    entry_signals = {}
    held_diag = {}
    selection_logic = {}
    try:
        cfg_p = BOTS_DIR / bot / "config" / "config.yaml"
        if cfg_p.exists():
            cfg = yaml.safe_load(cfg_p.read_text()) or {}
            strategy_id = cfg.get("strategy_id")
            limits = cfg.get("limits") or {}
            signals = cfg.get("signals") or {}
            eq_sig = signals.get("equity") or {}
            cr_sig = signals.get("crypto") or {}
            sched = cfg.get("scheduling") or {}
            eq_sc = sched.get("equities") or {}
            cr_sc = sched.get("crypto") or {}
            exe = cfg.get("execution") or {}
            selection_logic = {
                "max_equity_positions": limits.get("max_equity_positions"),
                "max_crypto_positions": limits.get("max_crypto_positions"),
                "min_avg_dollar_volume_20d": limits.get("min_avg_dollar_volume_20d"),
                "min_avg_crypto_dollar_volume_20d": limits.get("min_avg_crypto_dollar_volume_20d"),
                "equity_max_ann_vol": eq_sig.get("max_ann_vol"),
                "crypto_max_ann_vol": cr_sig.get("max_ann_vol"),
                "equity_rebalance_time_local": eq_sc.get("rebalance_time_local"),
                "crypto_rebalance_time_local": cr_sc.get("rebalance_time_local"),
                "crypto_risk_check_time_local": cr_sc.get("risk_check_time_local"),
                "extended_hours_start_time_local": exe.get("extended_hours_start_time_local"),
            }
    except Exception:
        pass
    try:
        rb_p = BOTS_DIR / bot / "data" / "last_rebalance.json"
        if rb_p.exists():
            rb = json.loads(rb_p.read_text()).get("payload", {})
            es = rb.get("entry_signals") or {}
            entry_signals = {}
            entry_signals.update(es.get("equities") or {})
            entry_signals.update(es.get("crypto") or {})
            held_diag = rb.get("held_not_selected_diagnostics") or {}
    except Exception:
        pass
    return strategy_id, entry_signals, held_diag, selection_logic


def _parse_iso_ts(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _compute_live_sharpe(bot: str | None) -> float | None:
    if not bot:
        return None
    p = BOTS_DIR / bot / "data" / "equity_curve.jsonl"
    if not p.exists():
        return None

    rows: list[tuple[datetime, float]] = []
    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    ts = _parse_iso_ts(str(rec.get("ts") or ""))
                    eq = rec.get("equity")
                    if ts is not None and eq is not None:
                        rows.append((ts, float(eq)))
                except Exception:
                    continue
    except Exception:
        return None

    if len(rows) < 3:
        return None

    rows.sort(key=lambda x: x[0])
    returns: list[float] = []
    intervals_days: list[float] = []
    prev_t, prev_e = rows[0]
    for t, e in rows[1:]:
        if prev_e > 0:
            returns.append((e / prev_e) - 1.0)
            dt_days = (t - prev_t).total_seconds() / 86400.0
            if dt_days > 0:
                intervals_days.append(dt_days)
        prev_t, prev_e = t, e

    if len(returns) < 2:
        return None

    stdev = statistics.pstdev(returns)
    if stdev <= 0:
        return None

    mean_r = statistics.fmean(returns)
    median_days = statistics.median(intervals_days) if intervals_days else 1.0
    periods_per_year = 252.0 if median_days <= 0 else max(1.0, 365.0 / median_days)
    return (mean_r / stdev) * (periods_per_year ** 0.5)


def bot_overview(port: int) -> dict:
    up = is_up(port)
    bot = _bot_name_by_port(port)
    strategy_id, sig_map, held_diag, selection_logic = _load_bot_selection_context(bot) if bot else (None, {}, {}, {})
    out = {
        "up": up,
        "selection_strategy": strategy_id,
        "selection_logic": selection_logic,
        "equity": None,
        "day_pl": None,
        "day_pl_pct": None,
        "open_positions": 0,
        "gross_exposure": 0.0,
        "best_asset": None,
        "worst_asset": None,
        "sharpe": _compute_live_sharpe(bot),
        "assets": [],
    }
    if not up:
        return out

    try:
        acct = _fetch_json(f"http://127.0.0.1:{port}/api/account")
        pos = _fetch_json(f"http://127.0.0.1:{port}/api/positions")

        out["equity"] = acct.get("equity")
        out["day_pl"] = acct.get("day_change_equity")
        out["day_pl_pct"] = acct.get("day_change_equity_pct")
        out["open_positions"] = len(pos) if isinstance(pos, list) else 0

        if isinstance(pos, list) and pos:
            # gross exposure by absolute market value
            out["gross_exposure"] = float(sum(abs(float(p.get("market_value") or 0.0)) for p in pos))

            best = max(pos, key=lambda p: float(p.get("unrealized_pl") or 0.0))
            worst = min(pos, key=lambda p: float(p.get("unrealized_pl") or 0.0))

            out["best_asset"] = {
                "symbol": best.get("symbol"),
                "unrealized_pl": float(best.get("unrealized_pl") or 0.0),
                "unrealized_plpc": float(best.get("unrealized_plpc") or 0.0),
            }
            out["worst_asset"] = {
                "symbol": worst.get("symbol"),
                "unrealized_pl": float(worst.get("unrealized_pl") or 0.0),
                "unrealized_plpc": float(worst.get("unrealized_plpc") or 0.0),
            }

            assets = []
            for p in sorted(pos, key=lambda x: abs(float(x.get("market_value") or 0.0)), reverse=True):
                sym = str(p.get("symbol") or "")
                sig = (sig_map or {}).get(sym) or {}
                hd = (held_diag or {}).get(sym) or {}
                src = sig if sig else hd
                reason = sig.get("reason") or hd.get("reject_reason") or "n/a"
                assets.append({
                    "symbol": sym,
                    "market_value": float(p.get("market_value") or 0.0),
                    "unrealized_pl": float(p.get("unrealized_pl") or 0.0),
                    "unrealized_plpc": float(p.get("unrealized_plpc") or 0.0),
                    "selection_strategy": strategy_id,
                    "selection_reason": reason,
                    "score": src.get("score"),
                    "ann_vol": src.get("ann_vol"),
                    "adv20": src.get("adv20"),
                    "last_close": src.get("last_close"),
                    "ma_long": src.get("ma_long"),
                    "ma_short": src.get("ma_short"),
                    "highest_20d": src.get("highest_20d"),
                })
            out["assets"] = assets
    except Exception as e:
        out["error"] = str(e)

    return out


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, obj, code=200):
        b = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _send_text(self, txt: str, code=200, ctype="text/plain; charset=utf-8"):
        b = txt.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            if HUB_HTML.exists():
                self._send_text(HUB_HTML.read_text(encoding="utf-8"), ctype="text/html; charset=utf-8")
                return
            self._send_text("Hub HTML not found", 404)
            return

        if self.path == "/api/status":
            out = {k: {"port": p, "up": is_up(p)} for k, p in BOTS.items()}
            self._send_json({"ok": True, "bots": out})
            return

        if self.path == "/api/overview":
            out = {k: {"port": p, **bot_overview(p)} for k, p in BOTS.items()}
            self._send_json({"ok": True, "bots": out})
            return

        self._send_text("Not found", 404)

    def do_POST(self):
        if self.path == "/api/dashboards/start-all":
            ok, out = run_cmd([str(ROOT / "scripts" / "start_dashboards.sh")])
            self._send_json({"ok": ok, "message": out}, 200 if ok else 500)
            return

        if self.path == "/api/dashboards/stop-all":
            ok, out = run_cmd([str(ROOT / "scripts" / "stop_dashboards.sh")])
            self._send_json({"ok": ok, "message": out}, 200 if ok else 500)
            return

        if self.path.startswith("/api/dashboards/start/"):
            bot = self.path.split("/api/dashboards/start/", 1)[1].strip().lower()
            if bot not in BOTS:
                self._send_json({"ok": False, "error": "unknown bot"}, 400)
                return
            ok, out = run_cmd([str(ROOT / "scripts" / "start_one.sh"), bot])
            self._send_json({"ok": ok, "bot": bot, "message": out}, 200 if ok else 500)
            return

        if self.path.startswith("/api/dashboards/stop/"):
            bot = self.path.split("/api/dashboards/stop/", 1)[1].strip().lower()
            if bot not in BOTS:
                self._send_json({"ok": False, "error": "unknown bot"}, 400)
                return
            ok, out = run_cmd([str(ROOT / "scripts" / "stop_one.sh"), bot])
            self._send_json({"ok": ok, "bot": bot, "message": out}, 200 if ok else 500)
            return

        if self.path.startswith("/api/wipe/"):
            bot = self.path.split("/api/wipe/", 1)[1].strip().lower()
            if bot not in BOTS:
                self._send_json({"ok": False, "error": "unknown bot"}, 400)
                return
            script = ROOT / "scripts" / "wipe_bot.sh"
            if not script.exists():
                self._send_json({"ok": False, "error": "wipe script missing"}, 500)
                return
            ok, out = run_cmd([str(script), bot], timeout=90)
            self._send_json({"ok": ok, "bot": bot, "message": out if ok else "", "error": "" if ok else out}, 200 if ok else 500)
            return

        self._send_json({"ok": False, "error": "not found"}, 404)


if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8099
    server = HTTPServer((host, port), Handler)
    print(f"Hub server running: http://{host}:{port}")
    server.serve_forever()
