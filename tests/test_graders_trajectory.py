"""Trajectory graders. These are the differentiating ones, so they get the
most edge-case coverage: empty trajectories, zero denominators, and the
distinction between "nothing to measure" and "measured zero"."""

import pytest
from conftest import make_step, make_task, make_trajectory

from caliper.graders import get_grader


def run(name, task, traj, **kw):
    return get_grader(name)(task, traj, **kw)


def tools(*names, **kw):
    return make_trajectory("t", [make_step(i, "tool", n) for i, n in enumerate(names)], **kw)


# --- step_efficiency ------------------------------------------------------


def test_step_efficiency_at_the_optimum():
    score = run("step_efficiency", make_task(reference_steps=2), tools("a", "b"))
    assert score.value == 1.0 and score.passed


def test_step_efficiency_penalises_extra_steps():
    score = run("step_efficiency", make_task(reference_steps=2), tools(*"abcd"))
    assert score.value == 0.5
    assert not score.passed


def test_step_efficiency_allows_slack():
    """3 steps against a reference of 2 is within the default 1.5x slack."""
    score = run("step_efficiency", make_task(reference_steps=2), tools("a", "b", "c"))
    assert score.passed


def test_step_efficiency_does_not_reward_skipping_work():
    """Fewer steps than the reference caps at 1.0, never above."""
    score = run("step_efficiency", make_task(reference_steps=5), tools("a"))
    assert score.value == 1.0


def test_step_efficiency_empty_trajectory():
    score = run("step_efficiency", make_task(reference_steps=3), make_trajectory())
    assert score.value == 0.0 and not score.passed
    assert "empty trajectory" in score.detail


def test_step_efficiency_without_a_reference_says_so():
    score = run("step_efficiency", make_task(), tools("a"))
    assert not score.passed
    assert "no reference_steps" in score.detail


def test_step_efficiency_zero_reference_does_not_divide_by_zero():
    score = run("step_efficiency", make_task(reference_steps=0), tools("a"))
    assert score.value == 0.0


def test_step_efficiency_can_count_only_selected_kinds():
    traj = make_trajectory(
        "t", [make_step(0, "internal", "think"), make_step(1, "tool", "calc")]
    )
    assert run("step_efficiency", make_task(reference_steps=1), traj, kinds=["tool"]).passed


# --- tool_choice_precision ------------------------------------------------


def test_tool_choice_precision_perfect():
    score = run("tool_choice_precision", make_task(expected_tools=["lookup"]), tools("lookup"))
    assert score.value == 1.0 and score.passed


def test_tool_choice_precision_penalises_unnecessary_tools():
    score = run("tool_choice_precision", make_task(expected_tools=["lookup"]),
                tools("lookup", "calculator"))
    assert score.value == 0.5
    assert not score.passed
    assert "unnecessary: calculator" in score.detail


def test_tool_choice_precision_counts_distinct_names_not_calls():
    """Calling one tool three times is one choice."""
    score = run("tool_choice_precision", make_task(expected_tools=["lookup"]),
                tools("lookup", "lookup", "lookup"))
    assert score.value == 1.0


def test_tool_choice_precision_requires_recall_by_default():
    score = run("tool_choice_precision", make_task(expected_tools=["lookup", "calculator"]),
                tools("lookup"))
    assert score.value == 1.0
    assert not score.passed
    assert "missing: calculator" in score.detail


def test_tool_choice_precision_recall_can_be_disabled():
    score = run("tool_choice_precision", make_task(expected_tools=["lookup", "calculator"]),
                tools("lookup"), require_recall=False)
    assert score.passed


def test_tool_choice_precision_no_calls_at_all():
    score = run("tool_choice_precision", make_task(expected_tools=["lookup"]), make_trajectory())
    assert score.value == 0.0 and not score.passed
    assert "no tools called" in score.detail


def test_tool_choice_precision_without_expected_tools_says_so():
    score = run("tool_choice_precision", make_task(), tools("lookup"))
    assert not score.passed
    assert "no expected_tools" in score.detail


# --- no_redundant_calls ---------------------------------------------------


def test_no_redundant_calls_clean_trajectory():
    traj = make_trajectory("t", [
        make_step(0, "tool", "calc", input={"expression": "1+1"}),
        make_step(1, "tool", "calc", input={"expression": "2+2"}),
    ])
    score = run("no_redundant_calls", make_task(), traj)
    assert score.value == 1.0 and score.passed


def test_no_redundant_calls_detects_identical_repeat():
    traj = make_trajectory("t", [
        make_step(0, "tool", "calc", input={"expression": "1+1"}),
        make_step(1, "tool", "calc", input={"expression": "1+1"}),
    ])
    score = run("no_redundant_calls", make_task(), traj)
    assert score.value == 0.5
    assert not score.passed
    assert "calc" in score.detail


def test_no_redundant_calls_ignores_argument_ordering():
    traj = make_trajectory("t", [
        make_step(0, "tool", "op", input={"a": 1, "b": 2}),
        make_step(1, "tool", "op", input={"b": 2, "a": 1}),
    ])
    assert not run("no_redundant_calls", make_task(), traj).passed


def test_no_redundant_calls_with_no_calls_passes():
    """Zero calls means zero redundancy, not a division by zero."""
    score = run("no_redundant_calls", make_task(), make_trajectory())
    assert score.value == 1.0 and score.passed


def test_no_redundant_calls_allowance():
    traj = make_trajectory("t", [make_step(i, "tool", "c", input={"x": 1}) for i in range(2)])
    assert run("no_redundant_calls", make_task(), traj, allow=1).passed


def test_no_redundant_calls_handles_unserialisable_input():
    traj = make_trajectory("t", [make_step(0, "tool", "c", input=object())])
    assert run("no_redundant_calls", make_task(), traj).passed


# --- recovered_from_error -------------------------------------------------


def test_recovered_from_error_with_no_errors():
    score = run("recovered_from_error", make_task(), tools("a"))
    assert score.value == 1.0 and score.passed
    assert "no errored steps" in score.detail


def test_recovered_from_error_successful_retry():
    traj = make_trajectory("t", [
        make_step(0, "tool", "lookup", error="not found"),
        make_step(1, "tool", "lookup", output="Paris"),
    ])
    score = run("recovered_from_error", make_task(), traj)
    assert score.value == 1.0 and score.passed


def test_recovered_from_error_never_recovered():
    traj = make_trajectory("t", [make_step(0, "tool", "lookup", error="not found")])
    score = run("recovered_from_error", make_task(), traj)
    assert score.value == 0.0 and not score.passed
    assert "never recovered" in score.detail


def test_recovered_from_error_partial_credit():
    traj = make_trajectory("t", [
        make_step(0, "tool", "a", error="x"),
        make_step(1, "tool", "a", output="ok"),
        make_step(2, "tool", "b", error="y"),
    ])
    score = run("recovered_from_error", make_task(), traj)
    assert score.value == 0.5 and not score.passed


def test_recovered_from_error_same_tool_only_is_stricter():
    traj = make_trajectory("t", [
        make_step(0, "tool", "a", error="x"),
        make_step(1, "tool", "b", output="ok"),
    ])
    assert run("recovered_from_error", make_task(), traj).passed
    assert not run("recovered_from_error", make_task(), traj, same_tool_only=True).passed


def test_recovery_must_come_after_the_error():
    """A success BEFORE the failure is not a recovery."""
    traj = make_trajectory("t", [
        make_step(0, "tool", "a", output="ok"),
        make_step(1, "tool", "a", error="x"),
    ])
    assert not run("recovered_from_error", make_task(), traj).passed


# --- budgets --------------------------------------------------------------


def test_cost_budget_within():
    traj = make_trajectory("t", [make_step(0, cost_usd=0.01)])
    score = run("cost_budget", make_task(cost_budget_usd=0.05), traj)
    assert score.passed and score.value == 1.0


def test_cost_budget_exceeded_decays():
    traj = make_trajectory("t", [make_step(0, cost_usd=0.10)])
    score = run("cost_budget", make_task(cost_budget_usd=0.05), traj)
    assert not score.passed
    assert score.value == pytest.approx(0.5)


def test_cost_budget_unset_says_so():
    score = run("cost_budget", make_task(), make_trajectory())
    assert not score.passed
    assert "no cost budget" in score.detail


def test_cost_budget_zero_spend_passes():
    score = run("cost_budget", make_task(cost_budget_usd=0.01), make_trajectory())
    assert score.passed


def test_latency_budget():
    traj = make_trajectory("t", wall_time_s=3.0)
    assert run("latency_budget", make_task(latency_budget_s=5.0), traj).passed
    assert not run("latency_budget", make_task(latency_budget_s=1.0), traj).passed


def test_latency_budget_explicit_argument_overrides_task():
    traj = make_trajectory("t", wall_time_s=3.0)
    assert not run("latency_budget", make_task(latency_budget_s=99.0), traj, budget_s=1.0).passed


# --- completed ------------------------------------------------------------


@pytest.mark.parametrize("reason,ok", [("completed", True), ("timeout", False),
                                       ("error", False), ("max_steps", False)])
def test_completed(reason, ok):
    assert run("completed", make_task(), make_trajectory(terminated_reason=reason)).passed is ok
