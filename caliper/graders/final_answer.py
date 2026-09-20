"""Graders that look only at ``trajectory.final_output``.

These are the familiar ones. They are cheap, deterministic and necessary, but
they are not sufficient -- see ``caliper/graders/trajectory.py`` for the half
of the picture that final-answer grading structurally cannot see.
"""

from __future__ import annotations

import json
import re
from typing import Any

from caliper.graders.registry import grader
from caliper.types import Score, Task, Trajectory


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def _normalise(text: str, case_sensitive: bool, strip_punct: bool) -> str:
    out = " ".join(text.split())
    if not case_sensitive:
        out = out.lower()
    if strip_punct:
        out = re.sub(r"[^\w\s.\-]", "", out)
    return out.strip()


@grader("exact_match")
def exact_match(
    task: Task,
    trajectory: Trajectory,
    case_sensitive: bool = False,
    strip_punctuation: bool = True,
) -> Score:
    """Whitespace-normalised string equality against ``task.expected``."""
    got = _normalise(_as_text(trajectory.final_output), case_sensitive, strip_punctuation)
    want = _normalise(_as_text(task.expected), case_sensitive, strip_punctuation)
    ok = got == want
    return Score(
        name="exact_match",
        value=1.0 if ok else 0.0,
        unit="ratio",
        passed=ok,
        detail="match" if ok else f"expected {want!r}, got {got!r}",
    )


_NUM_RE = re.compile(r"-?\d+(?:[\d,]*\d)?(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _first_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    match = _NUM_RE.search(_as_text(value))
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


@grader("numeric_tolerance")
def numeric_tolerance(
    task: Task,
    trajectory: Trajectory,
    rel: float | None = 0.01,
    abs: float | None = None,
) -> Score:
    """Numeric comparison with relative and/or absolute tolerance.

    Passes if EITHER tolerance is satisfied. Relative tolerance against an
    expected value of zero is meaningless, so it falls back to the absolute
    check (defaulting to 1e-9) rather than dividing by zero.
    """
    got = _first_number(trajectory.final_output)
    want = _first_number(task.expected)
    if got is None or want is None:
        return Score(
            name="numeric_tolerance",
            value=0.0,
            passed=False,
            detail=f"could not parse a number (expected={want}, got={got})",
        )

    delta = builtins_abs(got - want)
    abs_tol = abs
    ok = False
    reasons = []
    if abs_tol is not None:
        ok = ok or delta <= abs_tol
        reasons.append(f"abs delta {delta:.6g} vs tol {abs_tol:.6g}")
    if rel is not None:
        if want == 0.0:
            fallback = abs_tol if abs_tol is not None else 1e-9
            ok = ok or delta <= fallback
            reasons.append(f"expected is 0, used abs tol {fallback:.6g}")
        else:
            rel_delta = delta / builtins_abs(want)
            ok = ok or rel_delta <= rel
            reasons.append(f"rel delta {rel_delta:.6g} vs tol {rel:.6g}")
    if abs_tol is None and rel is None:
        ok = got == want
        reasons.append("no tolerance given, exact comparison")

    return Score(
        name="numeric_tolerance",
        value=1.0 if ok else 0.0,
        unit="ratio",
        passed=ok,
        detail=f"expected {want:g}, got {got:g}; " + "; ".join(reasons),
    )


builtins_abs = abs  # captured before the parameter name shadows it


@grader("contains_all")
def contains_all(
    task: Task,
    trajectory: Trajectory,
    terms: list[str] | None = None,
    case_sensitive: bool = False,
) -> Score:
    """Fraction of required substrings present. Passes only at 1.0.

    Terms come from ``terms=`` or, failing that, from ``task.expected`` when it
    is a list.
    """
    if terms is None:
        terms = task.expected if isinstance(task.expected, list) else []
    terms = [str(t) for t in terms]
    if not terms:
        return Score(
            name="contains_all",
            value=0.0,
            passed=False,
            detail="no terms configured; give terms= or a list in expected",
        )
    haystack = _as_text(trajectory.final_output)
    if not case_sensitive:
        haystack = haystack.lower()
    missing = [t for t in terms if (t if case_sensitive else t.lower()) not in haystack]
    found = len(terms) - len(missing)
    value = found / len(terms)
    return Score(
        name="contains_all",
        value=value,
        unit="ratio",
        passed=not missing,
        detail=f"{found}/{len(terms)} terms present"
        + (f"; missing: {', '.join(missing)}" if missing else ""),
    )


@grader("regex_match")
def regex_match(
    task: Task,
    trajectory: Trajectory,
    pattern: str | None = None,
    flags: str = "i",
    fullmatch: bool = False,
) -> Score:
    """Regex search (or fullmatch) over the final output.

    ``flags`` is a short string: any of ``i`` (ignorecase), ``m`` (multiline),
    ``s`` (dotall).
    """
    pattern = pattern if pattern is not None else _as_text(task.expected)
    if not pattern:
        return Score(name="regex_match", value=0.0, passed=False, detail="no pattern configured")
    f = 0
    for ch in flags:
        f |= {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL}.get(ch, 0)
    text = _as_text(trajectory.final_output)
    try:
        compiled = re.compile(pattern, f)
    except re.error as exc:
        return Score(name="regex_match", value=0.0, passed=False, detail=f"bad pattern: {exc}")
    m = compiled.fullmatch(text) if fullmatch else compiled.search(text)
    ok = m is not None
    return Score(
        name="regex_match",
        value=1.0 if ok else 0.0,
        unit="ratio",
        passed=ok,
        detail=f"pattern {pattern!r} {'matched' if ok else 'did not match'}",
    )


def _type_ok(value: Any, expected: str) -> bool:
    mapping = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }
    py = mapping.get(expected)
    if py is None:
        return True
    if expected in ("number", "integer") and isinstance(value, bool):
        return False
    return isinstance(value, py)


def _check_schema(value: Any, schema: dict, path: str = "$") -> list[str]:
    """Deliberately small subset of JSON Schema: type, required, properties, items.

    A full validator is a dependency and a distraction. This covers what task
    suites actually assert about agent output.
    """
    errors: list[str] = []
    expected_type = schema.get("type")
    if expected_type and not _type_ok(value, expected_type):
        errors.append(f"{path}: expected type {expected_type}, got {type(value).__name__}")
        return errors
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}.{key}: required key missing")
        for key, sub in (schema.get("properties") or {}).items():
            if key in value:
                errors.extend(_check_schema(value[key], sub, f"{path}.{key}"))
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, item in enumerate(value):
            errors.extend(_check_schema(item, schema["items"], f"{path}[{i}]"))
    return errors


@grader("json_schema_match")
def json_schema_match(
    task: Task,
    trajectory: Trajectory,
    schema: dict | None = None,
) -> Score:
    """Parse the final output as JSON and validate against a small schema subset."""
    schema = schema if schema is not None else (
        task.expected if isinstance(task.expected, dict) else None
    )
    if not schema:
        return Score(
            name="json_schema_match", value=0.0, passed=False, detail="no schema configured"
        )
    raw = trajectory.final_output
    if isinstance(raw, (dict, list)):
        parsed: Any = raw
    else:
        text = _as_text(raw).strip()
        fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            return Score(
                name="json_schema_match",
                value=0.0,
                passed=False,
                detail=f"output is not valid JSON: {exc}",
            )
    errors = _check_schema(parsed, schema)
    ok = not errors
    return Score(
        name="json_schema_match",
        value=1.0 if ok else 0.0,
        unit="ratio",
        passed=ok,
        detail="schema satisfied" if ok else "; ".join(errors[:5]),
    )
