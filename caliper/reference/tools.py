"""Tools for the built-in reference agent.

Deliberately tiny and fully deterministic. The reference agent exists so that
Caliper is demonstrable with no API key and no network, and so that the
v1-vs-v2 comparison on the website is a real measurement rather than a mock.
That requires the tools themselves to be boring and exact.

Each tool raises ``ToolError`` on bad input. Errors are load-bearing: the
suite includes tasks where the naive agent calls a tool wrongly, so that
``recovered_from_error`` has something real to grade.
"""

from __future__ import annotations

import ast
import operator
import re
from typing import Any, Callable

COST_PER_CALL_USD = 0.0004  # notional per-call price, so cost metrics are non-zero


class ToolError(RuntimeError):
    """Raised by a tool given input it cannot handle."""


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolError(f"unsupported literal: {node.value!r}")
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        right = _eval_node(node.right)
        if type(node.op) in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise ToolError("division by zero")
        return _BIN_OPS[type(node.op)](_eval_node(node.left), right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ToolError(f"unsupported expression element: {type(node).__name__}")


def calculator(expression: str = "") -> float:
    """Evaluate an arithmetic expression. AST-walked, never ``eval``."""
    expr = str(expression).strip()
    if not expr:
        raise ToolError("empty expression")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"cannot parse expression {expr!r}: {exc.msg}") from exc
    value = _eval_node(tree)
    return round(value, 10)


# A small fixed knowledge base. Keys are lowercase.
KV_STORE: dict[str, str] = {
    "capital of france": "Paris",
    "capital of japan": "Tokyo",
    "capital of brazil": "Brasilia",
    "capital of kenya": "Nairobi",
    "speed of light": "299792458 m/s",
    "boiling point of water": "100 C",
    "freezing point of water": "0 C",
    "earth radius": "6371 km",
    "seconds in a day": "86400",
    "days in a leap year": "366",
    "founder of caliper": "Manav Arya Singh",
    "atomic number of carbon": "6",
    "largest ocean": "Pacific Ocean",
    "longest river": "Nile",
    "currency of japan": "yen",
}


def lookup(key: str = "") -> str:
    """Fetch a value from the fixed key-value store. Exact keys only."""
    k = str(key).strip().lower()
    if not k:
        raise ToolError("empty key")
    if k not in KV_STORE:
        raise ToolError(
            f"no entry for {key!r}. Known keys include: "
            + ", ".join(sorted(KV_STORE)[:5])
            + ", ..."
        )
    return KV_STORE[k]


_STRING_OPS: dict[str, Callable[[str], Any]] = {
    "upper": str.upper,
    "lower": str.lower,
    "reverse": lambda s: s[::-1],
    "length": len,
    "title": str.title,
    "word_count": lambda s: len(s.split()),
    "strip": str.strip,
}


def string_op(op: str = "", text: str = "") -> Any:
    """Apply a named string operation."""
    name = str(op).strip().lower()
    if name not in _STRING_OPS:
        raise ToolError(
            f"unknown string op {op!r}. Supported: " + ", ".join(sorted(_STRING_OPS))
        )
    return _STRING_OPS[name](str(text))


TOOLS: dict[str, Callable[..., Any]] = {
    "calculator": calculator,
    "lookup": lookup,
    "string_op": string_op,
}


def call_tool(name: str, **kwargs: Any) -> Any:
    fn = TOOLS.get(name)
    if fn is None:
        raise ToolError(f"unknown tool {name!r}. Available: {', '.join(sorted(TOOLS))}")
    return fn(**kwargs)


def normalise_key(text: str) -> str:
    """Best-effort extraction of a KV key from a natural-language question."""
    t = re.sub(r"[^\w\s]", " ", str(text).lower())
    t = " ".join(t.split())
    for prefix in ("what is the ", "what is ", "whats the ", "tell me the ", "the "):
        if t.startswith(prefix):
            t = t[len(prefix):]
            break
    return t.strip()
