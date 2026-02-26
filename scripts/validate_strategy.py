#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

ID_RE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
REF_RE = re.compile(r"^ind\.([a-z][a-z0-9_]{2,63})\.[a-z_]+$")


def collect_refs(node, out=None):
    if out is None:
        out = []
    if isinstance(node, dict):
        if isinstance(node.get("ref"), str):
            out.append(node["ref"])
        args = node.get("args")
        if isinstance(args, list):
            for a in args:
                collect_refs(a, out)
    elif isinstance(node, list):
        for a in node:
            collect_refs(a, out)
    return out


def err(errors, code, path, message):
    errors.append({"code": code, "path": path, "message": message})


def validate_strategy(doc):
    errors = []

    for k in ["schema_version", "strategy_id", "asset_policies", "execution"]:
        if k not in doc:
            err(errors, "MISSING_FIELD", f"/{k}", "required field missing")

    sid = doc.get("strategy_id")
    if not isinstance(sid, str) or not ID_RE.match(sid):
        err(errors, "INVALID_ID", "/strategy_id", "must match ^[a-z][a-z0-9_]{2,63}$")

    aps = doc.get("asset_policies", {})
    for cls in ["stocks", "crypto"]:
        if cls not in aps:
            err(errors, "MISSING_POLICY", f"/asset_policies/{cls}", "asset policy required")
            continue
        for side in ["entry_policy", "exit_policy"]:
            p = aps[cls].get(side)
            if not isinstance(p, dict):
                err(errors, "MISSING_POLICY", f"/asset_policies/{cls}/{side}", "policy required")
                continue
            for req in ["enabled", "timeframe", "warmup_bars", "indicators", "rules"]:
                if req not in p:
                    err(errors, "MISSING_FIELD", f"/asset_policies/{cls}/{side}/{req}", "required field missing")

            indicators = p.get("indicators", [])
            rules = p.get("rules", [])

            keys = set()
            for i, ind in enumerate(indicators):
                key = ind.get("key")
                if not isinstance(key, str) or not ID_RE.match(key):
                    err(errors, "INVALID_KEY", f"/asset_policies/{cls}/{side}/indicators/{i}/key", "invalid indicator key")
                elif key in keys:
                    err(errors, "DUPLICATE_KEY", f"/asset_policies/{cls}/{side}/indicators/{i}/key", f"duplicate key {key}")
                else:
                    keys.add(key)

                if ind.get("indicator") == "macd":
                    params = ind.get("params", {})
                    fast, slow = params.get("fast"), params.get("slow")
                    if not (isinstance(fast, int) and isinstance(slow, int) and fast < slow):
                        err(errors, "INVALID_PARAM", f"/asset_policies/{cls}/{side}/indicators/{i}/params", "macd.fast must be < macd.slow")

            priorities = set()
            for i, r in enumerate(rules):
                rid = r.get("id")
                if not isinstance(rid, str) or not ID_RE.match(rid):
                    err(errors, "INVALID_RULE_ID", f"/asset_policies/{cls}/{side}/rules/{i}/id", "invalid rule id")

                pri = r.get("priority")
                if not isinstance(pri, int):
                    err(errors, "INVALID_PRIORITY", f"/asset_policies/{cls}/{side}/rules/{i}/priority", "priority must be integer")
                elif pri in priorities:
                    err(errors, "DUPLICATE_PRIORITY", f"/asset_policies/{cls}/{side}/rules/{i}/priority", f"duplicate priority {pri}")
                else:
                    priorities.add(pri)

                if r.get("action") == "exit_partial":
                    sp = r.get("size_pct")
                    if not (isinstance(sp, (int, float)) and 0 < float(sp) <= 100):
                        err(errors, "INVALID_ACTION_PAYLOAD", f"/asset_policies/{cls}/{side}/rules/{i}/size_pct", "exit_partial requires size_pct in (0,100]")

                refs = collect_refs(r.get("when"))
                for ref in refs:
                    m = REF_RE.match(ref)
                    if m and m.group(1) not in keys:
                        err(errors, "UNRESOLVED_REF", f"/asset_policies/{cls}/{side}/rules/{i}/when", f"unresolved indicator ref: {ref}")

    return errors


def main():
    ap = argparse.ArgumentParser(description="Validate SeaTrader strategy JSON")
    ap.add_argument("strategy", help="Path to strategy JSON")
    args = ap.parse_args()

    p = Path(args.strategy)
    if not p.exists():
        print(json.dumps({"ok": False, "errors": [{"code": "NOT_FOUND", "path": str(p), "message": "file not found"}]}))
        raise SystemExit(2)

    try:
        doc = json.loads(p.read_text())
    except Exception as e:
        print(json.dumps({"ok": False, "errors": [{"code": "BAD_JSON", "path": str(p), "message": str(e)}]}))
        raise SystemExit(2)

    errors = validate_strategy(doc)
    out = {"ok": len(errors) == 0, "errors": errors}
    print(json.dumps(out, indent=2))
    raise SystemExit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
