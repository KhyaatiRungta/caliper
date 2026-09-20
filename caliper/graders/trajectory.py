"""Graders that read the whole trajectory, not just the final answer.

This module is the argument the project exists to make. Two agents can return
the identical final string while one of them called a tool nine times, retried
twice after crashing, and spent seven times the money. Final-answer grading
scores those two identically. That is not a subtle limitation; it is most of
what you need to know before shipping a prompt change.

Every grader here is pure: it reads ``trajectory`` and ``task`` and returns a
Score. None of them re-run the agent.
"""

from __future__ import annotations

import json
from typing import Any

from caliper.graders.registry import grader
from caliper.types import Score, Step, Task, Trajectory


def _counted_steps(trajectory: Trajectory, kinds: tuple[str, ...] | None) -> list[Step]:
    if not kinds:
        return list(trajectory.steps)
    return [s for s in trajectory.steps if s.kind in kinds]


@grader("step_efficiency")
def step_efficiency(
    task: Task,
    trajectory: Trajectory,
    reference_steps: int | None = None,
    kinds: list[str] | None = None,
    slack: float = 1.5,
) -> Score:
    """Steps taken against a reference optimum.

    ``value`` is ``reference / max(actual, reference)``, so a trajectory that
    hits the optimum scores 1.0, one that takes twice as many steps scores 0.5,
    and taking FEWER steps than the reference is not rewarded above 1.0 -- an
    agent that skips necessary work should be caught by the correctness grader,
    not flattered by this one.

    Passes when ``actual <= reference * slack``. Default slack of 1.5 admits
    one extra step on a three-step reference, which is the amount of wandering
    most people are willing to pay for.
    """
    ref = reference_steps if reference_steps is not None else task.reference_steps
    steps = _counted_steps(trajectory, tuple(kinds) if kinds else None)
    actual = len(steps)

    if ref is None or ref <= 0:
        return Score(
            name="step_efficiency",
            value=0.0,
            unit="ratio",
            passed=False,
            detail=(
                "no reference_steps configured for this task; set reference_steps "
                "on the task or pass reference_steps= to the grader"
            ),
        )
    if actual == 0:
        return Score(
            name="step_efficiency",
            value=0.0,
            unit="ratio",
            passed=False,
            detail="empty trajectory: the agent recorded no steps",
        )

    value = ref / max(actual, ref)
    ok = actual <= ref * slack
    return Score(
        name="step_efficiency",
        value=value,
        unit="ratio",
        passed=ok,
        detail=f"{actual} steps vs reference {ref} (slack {slack:g}x)",
    )


@grader("tool_choice_precision")
def tool_choice_precision(
    task: Task,
    trajectory: Trajectory,
    expected_tools: list[str] | None = None,
    require_recall: bool = True,
) -> Score:
    """Did the agent call the tools the task actually needed, and no others?

    Precision is ``|called AND expected| / |called|`` over the SET of distinct
    tool names -- calling ``lookup`` three times is one choice, not three.

    With ``require_recall`` (the default) the grader passes only when every
    expected tool was also used, so an agent that answers from memory and
    happens to be right does not score a clean 1.0 for calling nothing.
    """
    expected = list(expected_tools if expected_tools is not None else task.expected_tools)
    if not expected:
        return Score(
            name="tool_choice_precision",
            value=0.0,
            unit="ratio",
            passed=False,
            detail="no expected_tools configured for this task",
        )
    called = set(trajectory.tool_names)
    want = set(expected)

    if not called:
        return Score(
            name="tool_choice_precision",
            value=0.0,
            unit="ratio",
            passed=False,
            detail=f"no tools called; expected {', '.join(sorted(want))}",
        )

    hits = called & want
    precision = len(hits) / len(called)
    missing = sorted(want - called)
    extra = sorted(called - want)
    ok = precision == 1.0 and (not require_recall or not missing)

    bits = [f"precision {precision:.2f} ({len(hits)}/{len(called)} calls useful)"]
    if extra:
        bits.append(f"unnecessary: {', '.join(extra)}")
    if missing:
        bits.append(f"missing: {', '.join(missing)}")
    return Score(
        name="tool_choice_precision",
        value=precision,
        unit="ratio",
        passed=ok,
        detail="; ".join(bits),
    )


def _call_signature(step: Step) -> str:
    try:
        payload = json.dumps(step.input, sort_keys=True, default=str)
    except (TypeError, ValueError):
        payload = repr(step.input)
    return f"{step.name}::{payload}"


@grader("no_redundant_calls")
def no_redundant_calls(
    task: Task,
    trajectory: Trajectory,
    kinds: list[str] | None = None,
    allow: int = 0,
) -> Score:
    """Detects repeated identical calls -- same tool, same arguments.

    A repeat is pure waste: deterministic tools return the same answer, so the
    second call bought nothing but latency and money. ``allow`` permits a
    budget of repeats, which is occasionally legitimate (polling).

    ``value`` is the fraction of calls that were not redundant, so an agent
    that called four distinct tools and repeated one scores 0.8.
    """
    kind_filter = tuple(kinds) if kinds else ("tool", "retrieval")
    calls = [s for s in trajectory.steps if s.kind in kind_filter]
    if not calls:
        return Score(
            name="no_redundant_calls",
            value=1.0,
            unit="ratio",
            passed=True,
            detail="no tool or retrieval calls to check",
        )

    seen: set[str] = set()
    repeats: list[str] = []
    for step in calls:
        sig = _call_signature(step)
        if sig in seen:
            repeats.append(step.name)
        seen.add(sig)

    value = (len(calls) - len(repeats)) / len(calls)
    ok = len(repeats) <= allow
    detail = (
        "no repeated calls"
        if not repeats
        else f"{len(repeats)} repeated call(s): {', '.join(sorted(set(repeats)))}"
    )
    return Score(
        name="no_redundant_calls",
        value=value,
        unit="ratio",
        passed=ok,
        detail=detail,
    )


@grader("recovered_from_error")
def recovered_from_error(
    task: Task,
    trajectory: Trajectory,
    same_tool_only: bool = False,
) -> Score:
    """Did each errored step get followed by a successful step?

    Recovery is the behaviour that separates an agent from a script, and it is
    invisible to final-answer grading in both directions: an agent that never
    errors looks the same as one that errored and papered over it.

    A trajectory with no errors passes with value 1.0 and says so -- having
    nothing to recover from is the best case, not a missing measurement.

    ``same_tool_only`` requires the retry to be of the same tool, which is the
    stricter reading of "recovered" and the right one when the tool is the
    thing under test.
    """
    errored = trajectory.errored_steps
    if not errored:
        return Score(
            name="recovered_from_error",
            value=1.0,
            unit="ratio",
            passed=True,
            detail="no errored steps in trajectory",
        )

    steps = trajectory.steps
    recovered = 0
    unrecovered: list[str] = []
    for step in errored:
        later = [s for s in steps if s.index > step.index and s.error is None]
        if same_tool_only:
            later = [s for s in later if s.name == step.name]
        if later:
            recovered += 1
        else:
            unrecovered.append(step.name)

    value = recovered / len(errored)
    ok = value == 1.0
    return Score(
        name="recovered_from_error",
        value=value,
        unit="ratio",
        passed=ok,
        detail=f"recovered from {recovered}/{len(errored)} errored step(s)"
        + (f"; never recovered: {', '.join(unrecovered)}" if unrecovered else ""),
    )


def _budget_score(
    name: str, actual: float, budget: float | None, unit: str, fmt: str, what: str
) -> Score:
    if budget is None or budget <= 0:
        return Score(
            name=name,
            value=0.0,
            unit=unit,
            passed=False,
            detail=f"no {what} budget configured for this task",
        )
    ok = actual <= budget
    # 1.0 at or under budget, decaying towards 0 as the overrun grows.
    value = 1.0 if ok else max(0.0, budget / actual) if actual > 0 else 1.0
    return Score(
        name=name,
        value=value,
        unit=unit,
        passed=ok,
        detail=f"{fmt.format(actual)} against budget {fmt.format(budget)}",
    )


@grader("cost_budget")
def cost_budget(
    task: Task, trajectory: Trajectory, budget_usd: float | None = None
) -> Score:
    """Pass/fail against a task-level dollar budget."""
    budget = budget_usd if budget_usd is not None else task.cost_budget_usd
    return _budget_score(
        "cost_budget", trajectory.total_cost_usd, budget, "usd", "${:.4f}", "cost"
    )


@grader("latency_budget")
def latency_budget(
    task: Task, trajectory: Trajectory, budget_s: float | None = None
) -> Score:
    """Pass/fail against a task-level wall-clock budget."""
    budget = budget_s if budget_s is not None else task.latency_budget_s
    return _budget_score(
        "latency_budget", trajectory.wall_time_s, budget, "seconds", "{:.2f}s", "latency"
    )


@grader("completed")
def completed(task: Task, trajectory: Trajectory) -> Score:
    """Did the agent finish at all? Useful as a cheap smoke grader."""
    ok = trajectory.terminated_reason == "completed"
    return Score(
        name="completed",
        value=1.0 if ok else 0.0,
        unit="ratio",
        passed=ok,
        detail=f"terminated_reason={trajectory.terminated_reason}",
    )
