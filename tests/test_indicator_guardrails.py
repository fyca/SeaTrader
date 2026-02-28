from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "tradebot"

# Indicator math should be centralized in tradebot/indicators/service.py
ALLOWED = {
    (SRC / "indicators" / "service.py").resolve(),
}

WATCH_DIRS = [
    SRC / "strategies",
    SRC / "signals",
]

FORBIDDEN_METHODS = {"rolling", "ewm", "pct_change"}


class _CallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.hits: list[tuple[int, str]] = []

    def visit_Call(self, node: ast.Call):
        fn = node.func
        if isinstance(fn, ast.Attribute) and fn.attr in FORBIDDEN_METHODS:
            self.hits.append((getattr(node, "lineno", 0), fn.attr))
        self.generic_visit(node)


class IndicatorGuardrailTests(unittest.TestCase):
    def test_indicator_math_is_centralized(self):
        violations: list[str] = []
        for base in WATCH_DIRS:
            for py in base.rglob("*.py"):
                rp = py.resolve()
                if rp in ALLOWED:
                    continue
                tree = ast.parse(py.read_text(), filename=str(py))
                v = _CallVisitor()
                v.visit(tree)
                for line, attr in v.hits:
                    violations.append(f"{py.relative_to(ROOT)}:{line} uses .{attr}()")

        if violations:
            joined = "\n".join(violations)
            self.fail(
                "Indicator math must live in tradebot/indicators/service.py. "
                "Add new indicators there, then call via indicator_service.\n"
                f"Violations:\n{joined}"
            )


if __name__ == "__main__":
    unittest.main()
