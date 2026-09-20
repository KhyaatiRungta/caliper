"""The built-in reference agent.

Purpose
-------

Caliper must be demonstrable by someone who has just cloned the repo, with no
API key, no network and no sibling project checked out. That requires an agent
that is fully deterministic, and it requires two versions of it that genuinely
differ so the comparison report shows a real measured delta rather than a mock.

Honesty note
------------

This agent is hand-written and rule-based; it makes no LLM calls. ``v1`` and
``v2`` differ in their ROUTING POLICY -- how they decide which tool to reach
for -- which is the same class of change as a prompt edit, and produces the
same class of trajectory difference: wasted calls, avoidable errors, recovery
behaviour. It is a stand-in for a prompt change, not a prompt change. The
numbers Caliper reports about it are real measurements of real executions; what
is simulated is the agent, not the measurement.

v1 -- the naive policy
    Reaches for ``lookup`` first on essentially everything, because "look it up
    before you compute" reads like good advice and is terrible policy. Passes
    the raw question through as the key, fails, retries with a normalised key,
    and only then falls back to the right tool. On string tasks it tries the
    knowledge base first for the same reason. It also re-verifies arithmetic by
    calling the calculator a second time with the same expression.

v2 -- the decompose-then-act policy
    Classifies the question into arithmetic / lookup / string before acting,
    then makes exactly the calls that class needs. It normalises the lookup key
    before the first attempt rather than after the first failure. Crucially it
    also DECOMPOSES: a question like "reverse the capital of Japan" is two
    calls, a lookup feeding a string op, and v2 recognises that while v1 sees
    one unanswerable question and produces a confident wrong answer.

Both share the same tools and the same answer-formatting code, so any measured
difference is attributable to the routing policy alone.
"""

from __future__ import annotations

import re
from typing import Any

from caliper.adapter import TrajectoryRecorder
from caliper.reference import tools as T
from caliper.types import Task, Trajectory

# Notional per-step token accounting so that cost and token metrics are
# non-zero and proportional to work done. These are not billed tokens; they are
# a deterministic model of them, and the site says so.
TOKENS_IN_PER_STEP = 180
TOKENS_OUT_PER_STEP = 40
PRICE_IN_PER_M = 3.00
PRICE_OUT_PER_M = 15.00

_ARITH_RE = re.compile(
    r"(\d[\d,]*(?:\.\d+)?\s*[-+*/^%]\s*\d)|(\bplus\b|\bminus\b|\btimes\b|"
    r"\bdivided by\b|\bmultiplied by\b|\bsquared\b|\bpercent of\b)",
    re.IGNORECASE,
)
_STRING_RE = re.compile(
    r"\b(reverse|uppercase|upper case|lowercase|lower case|title case|"
    r"how many words|word count|length of|how many characters)\b",
    re.IGNORECASE,
)

_WORD_OPS = [
    (r"\bplus\b|\badd\b|\bsum of\b", "+"),
    (r"\bminus\b|\bsubtract\b", "-"),
    (r"\btimes\b|\bmultiplied by\b|\bproduct of\b", "*"),
    (r"\bdivided by\b", "/"),
]


def _step_cost() -> float:
    return (
        TOKENS_IN_PER_STEP * PRICE_IN_PER_M + TOKENS_OUT_PER_STEP * PRICE_OUT_PER_M
    ) / 1_000_000.0


def extract_expression(question: str) -> str:
    """Turn a natural-language arithmetic question into an expression."""
    text = str(question).lower()
    text = text.replace("what is", " ").replace("calculate", " ").replace("compute", " ")
    text = re.sub(r"[?,]", " ", text)

    pct = re.search(r"(\d+(?:\.\d+)?)\s*(?:percent|%)\s*of\s*(\d+(?:\.\d+)?)", text)
    if pct:
        return f"({pct.group(1)}/100)*{pct.group(2)}"
    sq = re.search(r"(\d+(?:\.\d+)?)\s*squared", text)
    if sq:
        return f"{sq.group(1)}**2"

    for pattern, symbol in _WORD_OPS:
        text = re.sub(pattern, symbol, text)
    text = text.replace("^", "**")
    kept = re.findall(r"[\d.()+\-*/%]+|\*\*", text)
    expr = " ".join(kept).replace(" ", "")
    return expr


def extract_string_op(question: str) -> tuple[str, str]:
    """Return ``(op, text)`` for a string-manipulation question."""
    q = str(question)
    low = q.lower()
    if "reverse" in low:
        op = "reverse"
    elif "uppercase" in low or "upper case" in low:
        op = "upper"
    elif "lowercase" in low or "lower case" in low:
        op = "lower"
    elif "title case" in low:
        op = "title"
    elif "word" in low:
        op = "word_count"
    else:
        op = "length"
    quoted = re.search(r"['\"]([^'\"]+)['\"]", q)
    text = quoted.group(1) if quoted else q.split(":")[-1].strip().rstrip("?")
    return op, text


def find_kv_key(question: str) -> str | None:
    """Longest knowledge-base key contained in the question, if any.

    Longest wins so that "capital of japan" is preferred over a shorter key
    that happens to be a substring of it.
    """
    low = " ".join(re.sub(r"[^\w\s]", " ", str(question).lower()).split())
    hits = [k for k in T.KV_STORE if k in low]
    return max(hits, key=len) if hits else None


def has_literal(question: str) -> bool:
    """Does the question carry its own quoted operand?"""
    return bool(re.search(r"['\"][^'\"]+['\"]", str(question)))


def classify(question: str) -> str:
    """arithmetic | string | lookup"""
    q = str(question)
    if _STRING_RE.search(q):
        return "string"
    if _ARITH_RE.search(q):
        return "arithmetic"
    return "lookup"


def _format(value: Any) -> str:
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else f"{value:g}"
    return str(value)


class ReferenceAgent:
    """Adapter over the reference agent. ``version`` selects the routing policy."""

    name = "reference-agent"
    model = "deterministic/reference"

    def __init__(self, version: str = "v2"):
        if version not in ("v1", "v2"):
            raise ValueError(f"reference agent version must be v1 or v2, got {version!r}")
        self.version = version

    # -- step helper -------------------------------------------------------

    def _tool(self, rec: TrajectoryRecorder, name: str, **kwargs: Any):
        """Call a tool, recording the step either way. Returns (ok, value)."""
        step = rec.add(
            "tool",
            name,
            input=kwargs,
            tokens_in=TOKENS_IN_PER_STEP,
            tokens_out=TOKENS_OUT_PER_STEP,
            cost_usd=_step_cost(),
        )
        try:
            value = T.call_tool(name, **kwargs)
        except T.ToolError as exc:
            step.error = str(exc)
            return False, None
        step.output = value
        return True, value

    def _think(self, rec: TrajectoryRecorder, label: str, detail: Any = None) -> None:
        rec.add(
            "internal",
            label,
            input=detail,
            tokens_in=TOKENS_IN_PER_STEP,
            tokens_out=TOKENS_OUT_PER_STEP,
            cost_usd=_step_cost(),
        )

    # -- the loop ----------------------------------------------------------

    def run(self, task: Task) -> Trajectory:
        rec = TrajectoryRecorder(task.id)
        question = str(task.input or "")
        answer = self._run_v2(rec, question) if self.version == "v2" else self._run_v1(
            rec, question
        )
        reason = "completed" if answer is not None else "error"
        return rec.finish(final_output=answer, terminated_reason=reason)

    # -- v2: classify, then act -------------------------------------------

    def _run_v2(self, rec: TrajectoryRecorder, question: str) -> str | None:
        kind = classify(question)
        kv = find_kv_key(question)
        composite = kv is not None and kind != "lookup" and not has_literal(question)
        self._think(rec, "classify", {"kind": kind, "composite": composite, "fact": kv})

        if composite:
            return self._run_composite(rec, question, kind, kv)

        if kind == "arithmetic":
            ok, value = self._tool(rec, "calculator", expression=extract_expression(question))
            return _format(value) if ok else None

        if kind == "string":
            op, text = extract_string_op(question)
            ok, value = self._tool(rec, "string_op", op=op, text=text)
            return _format(value) if ok else None

        ok, value = self._tool(rec, "lookup", key=T.normalise_key(question))
        return _format(value) if ok else None

    def _run_composite(
        self, rec: TrajectoryRecorder, question: str, kind: str, kv: str
    ) -> str | None:
        """Two-call plan: resolve the fact, then operate on it."""
        ok, fact = self._tool(rec, "lookup", key=kv)
        if not ok:
            return None
        if kind == "string":
            op, _ = extract_string_op(question)
            ok, value = self._tool(rec, "string_op", op=op, text=_format(fact))
            return _format(value) if ok else None
        # Arithmetic over a looked-up number: substitute the fact back in.
        low = " ".join(re.sub(r"[^\w\s]", " ", question.lower()).split())
        substituted = low.replace(kv, f" {_format(fact)} ")
        ok, value = self._tool(rec, "calculator", expression=extract_expression(substituted))
        return _format(value) if ok else None

    # -- v1: look it up first, ask questions later -------------------------

    def _run_v1(self, rec: TrajectoryRecorder, question: str) -> str | None:
        # Policy: "always consult the knowledge base first."
        ok, value = self._tool(rec, "lookup", key=question)
        if ok:
            return _format(value)

        ok, value = self._tool(rec, "lookup", key=T.normalise_key(question))
        if ok:
            return _format(value)

        self._think(rec, "reconsider", {"note": "lookup failed twice, trying computation"})

        if _STRING_RE.search(question):
            op, text = extract_string_op(question)
            ok, value = self._tool(rec, "string_op", op=op, text=text)
            return _format(value) if ok else None

        ok, value = self._tool(rec, "calculator", expression=extract_expression(question))
        if not ok:
            return None
        # Policy: "double-check arithmetic." Identical call, no new information.
        self._tool(rec, "calculator", expression=extract_expression(question))
        return _format(value)


def build(version: str = "v2") -> ReferenceAgent:
    return ReferenceAgent(version=version)
