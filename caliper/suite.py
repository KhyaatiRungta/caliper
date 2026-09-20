"""Suite loading.

A suite is a YAML file:

    suite: reference
    defaults:
      timeout_s: 30
      grader: exact_match
    tasks:
      - id: add_two_numbers
        input: "What is 17 plus 25?"
        expected: "42"
        grader:
          - numeric_tolerance: {abs: 0.001}
          - step_efficiency
        expected_tools: [calculator]
        reference_steps: 2
        tags: [arithmetic]

``defaults`` are shallow-merged into every task. Task ids must be unique; a
duplicate is an error rather than a silent overwrite, because a silently
dropped task is a silently missing regression.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from caliper.types import Task


class SuiteError(ValueError):
    """Raised for a malformed suite file."""


def _require_yaml():
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - env dependent
        raise SuiteError(
            "PyYAML is required to load suite files. Install with:\n"
            "  pip install -r requirements.txt"
        ) from exc
    return yaml


def load_suite(path: Path | str) -> tuple[str, list[Task]]:
    """Return ``(suite_name, tasks)``."""
    path = Path(path)
    if not path.exists():
        raise SuiteError(f"suite file not found: {path}")
    yaml = _require_yaml()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SuiteError(f"{path}: top level must be a mapping")
    return parse_suite(data, fallback_name=path.stem)


def parse_suite(data: dict, fallback_name: str = "suite") -> tuple[str, list[Task]]:
    name = str(data.get("suite") or fallback_name)
    defaults = data.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise SuiteError("'defaults' must be a mapping")
    raw_tasks = data.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise SuiteError("'tasks' must be a non-empty list")

    tasks: list[Task] = []
    seen: set[str] = set()
    for i, raw in enumerate(raw_tasks):
        if not isinstance(raw, dict):
            raise SuiteError(f"task #{i} must be a mapping, got {type(raw).__name__}")
        merged: dict[str, Any] = {**defaults, **raw}
        tid = merged.get("id")
        if not tid:
            raise SuiteError(f"task #{i} has no 'id'")
        tid = str(tid)
        if tid in seen:
            raise SuiteError(f"duplicate task id {tid!r}")
        seen.add(tid)
        merged["id"] = tid
        merged["suite"] = name
        known = set(Task.__dataclass_fields__)
        extra = {k: v for k, v in merged.items() if k not in known}
        fields = {k: v for k, v in merged.items() if k in known}
        if extra:
            fields.setdefault("metadata", {})
            fields["metadata"] = {**extra, **(fields.get("metadata") or {})}
        tasks.append(Task(**fields))
    return name, tasks


def filter_tasks(
    tasks: list[Task], tags: list[str] | None = None, limit: int | None = None
) -> list[Task]:
    out = tasks
    if tags:
        wanted = set(tags)
        out = [t for t in out if wanted & set(t.tags)]
    if limit is not None and limit >= 0:
        out = out[:limit]
    return out
