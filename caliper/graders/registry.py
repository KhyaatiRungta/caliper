"""Grader registry.

Every grader is a pure function with the same signature::

    @grader("my_grader")
    def my_grader(task: Task, trajectory: Trajectory, **kwargs) -> Score:
        ...

Pure means: no I/O except the LLM judge, no mutation of the task or the
trajectory, and the same inputs always yield the same Score. That constraint is
what makes a run file re-gradable later without re-running the agent.

Adding a grader
---------------

1. Write the function in a module under ``caliper/graders/``.
2. Decorate it with ``@grader("name")``.
3. Import the module from ``caliper/graders/__init__.py``.
4. Reference it by name from a suite YAML.

There is no step 5. A grader that needs configuration receives it as keyword
arguments from the task's grader spec, e.g.::

    grader:
      - numeric_tolerance: {rel: 0.01}
      - cost_budget
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from caliper.types import Score, Task, Trajectory

GraderFn = Callable[..., Score]

_REGISTRY: dict[str, GraderFn] = {}


class GraderError(RuntimeError):
    """Raised for an unknown or misconfigured grader."""


def grader(name: str) -> Callable[[GraderFn], GraderFn]:
    """Register a grader under ``name``."""

    def deco(fn: GraderFn) -> GraderFn:
        if name in _REGISTRY:
            raise GraderError(f"grader {name!r} is already registered")
        fn.grader_name = name  # type: ignore[attr-defined]
        _REGISTRY[name] = fn
        return fn

    return deco


def get_grader(name: str) -> GraderFn:
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise GraderError(f"unknown grader {name!r}. Registered graders: {known}") from None


def registered() -> list[str]:
    return sorted(_REGISTRY)


def normalise_spec(spec: Any) -> list[tuple[str, dict]]:
    """Turn a task's ``grader`` field into ``[(name, kwargs), ...]``.

    Accepted shapes::

        "exact_match"
        ["exact_match", "cost_budget"]
        {"numeric_tolerance": {"rel": 0.01}}
        ["exact_match", {"llm_judge": {"n_samples": 3}}]
    """
    out: list[tuple[str, dict]] = []
    if spec is None:
        return out
    items: Iterable[Any] = spec if isinstance(spec, list) else [spec]
    for item in items:
        if isinstance(item, str):
            out.append((item, {}))
        elif isinstance(item, dict):
            if len(item) == 1 and not {"name"} & set(item):
                (name, kwargs), = item.items()
                out.append((str(name), dict(kwargs or {})))
            elif "name" in item:
                kwargs = {k: v for k, v in item.items() if k != "name"}
                out.append((str(item["name"]), kwargs))
            else:
                raise GraderError(f"cannot interpret grader spec: {item!r}")
        else:
            raise GraderError(f"cannot interpret grader spec: {item!r}")
    return out


def grade(task: Task, trajectory: Trajectory, **overrides: Any) -> list[Score]:
    """Run every grader attached to ``task`` over ``trajectory``.

    A grader that raises does not abort the others: it yields a failing Score
    carrying the exception text. A broken grader must not look like a broken
    agent, and it must not lose the results of the graders that did work.
    """
    scores: list[Score] = []
    for name, kwargs in normalise_spec(task.grader):
        merged = {**kwargs, **overrides.get(name, {})}
        try:
            fn = get_grader(name)
            score = fn(task, trajectory, **merged)
        except Exception as exc:  # noqa: BLE001
            score = Score(
                name=name,
                value=0.0,
                unit="ratio",
                passed=False,
                detail=f"grader error: {type(exc).__name__}: {exc}",
            )
        scores.append(score)
    return scores
