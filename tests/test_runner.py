"""Runner guarantees: isolation, timeouts, ordering, retries, summary safety."""

import io
import time

import pytest
from conftest import make_task

from caliper.runner import (
    percentile,
    run_suite,
    summarise,
)
from caliper.types import Score, SuiteRun, Trajectory


class ScriptedAgent:
    """Behaviour per task id, so one run can contain every failure mode."""

    name = "scripted"
    version = "v1"
    model = "none"

    def __init__(self, behaviour=None, delay=0.0):
        self.behaviour = behaviour or {}
        self.delay = delay
        self.seen = []
        self.setup_calls = 0
        self.teardown_calls = 0

    def setup(self):
        self.setup_calls += 1

    def teardown(self):
        self.teardown_calls += 1

    def run(self, task):
        self.seen.append(task.id)
        action = self.behaviour.get(task.id, "ok")
        if callable(action):
            return action(task)
        if action == "raise":
            raise ValueError("task exploded")
        if action == "hang":
            time.sleep(10)
        if action == "not_a_trajectory":
            return {"final_output": "wrong type"}
        if self.delay:
            time.sleep(self.delay)
        return Trajectory(task_id=task.id, final_output="ok")


def tasks(n=4, **kw):
    kw.setdefault("expected", "ok")
    kw.setdefault("grader", "exact_match")
    kw.setdefault("timeout_s", 5.0)
    return [make_task(f"t{i}", **kw) for i in range(n)]


def test_happy_path_all_pass():
    run = run_suite(ScriptedAgent(), tasks(3), progress=False, save=False)
    assert run.summary["task_success"] == 1.0
    assert run.summary["passed"] == 3


def test_one_exploding_task_does_not_kill_the_run():
    agent = ScriptedAgent({"t1": "raise"})
    run = run_suite(agent, tasks(4), progress=False, save=False)
    assert len(run.results) == 4
    assert run.summary["passed"] == 3
    failed = [r for r in run.results if r.task.id == "t1"][0]
    assert not failed.passed
    assert failed.trajectory.terminated_reason == "error"
    assert "task exploded" in failed.scores[0].detail


def test_adapter_returning_the_wrong_type_is_caught():
    run = run_suite(ScriptedAgent({"t0": "not_a_trajectory"}), tasks(2),
                    progress=False, save=False)
    assert not run.results[0].passed
    assert "expected Trajectory" in run.results[0].scores[0].detail


def test_timeout_is_enforced_by_the_runner():
    ts = tasks(3)
    ts[1].timeout_s = 0.2
    t0 = time.time()
    run = run_suite(ScriptedAgent({"t1": "hang"}), ts, workers=3, progress=False, save=False)
    assert time.time() - t0 < 5
    assert run.results[1].trajectory.terminated_reason == "timeout"
    assert run.summary["timeouts"] == 1
    assert run.summary["passed"] == 2


def test_timeout_excludes_time_spent_queued():
    """With one worker, a later task must not time out for waiting its turn."""
    ts = tasks(4, timeout_s=0.5)
    run = run_suite(ScriptedAgent(delay=0.15), ts, workers=1, progress=False, save=False)
    assert all(r.trajectory.terminated_reason == "completed" for r in run.results)


@pytest.mark.parametrize("workers", [1, 2, 8])
def test_output_ordering_is_deterministic(workers):
    ts = tasks(8)
    agent = ScriptedAgent(delay=0.0)
    run = run_suite(agent, ts, workers=workers, progress=False, save=False)
    assert [r.task.id for r in run.results] == [t.id for t in ts]


def test_ordering_holds_when_completion_order_is_reversed():
    ts = tasks(4)
    delays = {"t0": 0.20, "t1": 0.15, "t2": 0.10, "t3": 0.0}

    def slow(task):
        time.sleep(delays[task.id])
        return Trajectory(task_id=task.id, final_output="ok")

    agent = ScriptedAgent({tid: slow for tid in delays})
    run = run_suite(agent, ts, workers=4, progress=False, save=False)
    assert [r.task.id for r in run.results] == ["t0", "t1", "t2", "t3"]


def test_transient_failures_are_retried_and_marked_as_infra():
    class Flaky(ScriptedAgent):
        def __init__(self):
            super().__init__()
            self.attempts = 0

        def run(self, task):
            self.attempts += 1
            raise ConnectionError("network blip")

    agent = Flaky()
    run = run_suite(agent, tasks(1), retries=2, progress=False, save=False)
    assert agent.attempts == 3
    assert run.summary["infra_errors"] == 1
    assert run.summary["graded"] == 0
    assert run.results[0].infra_error is not None


def test_transient_failure_that_resolves_is_not_an_error():
    class Flaky(ScriptedAgent):
        def __init__(self):
            super().__init__()
            self.attempts = 0

        def run(self, task):
            self.attempts += 1
            if self.attempts == 1:
                raise ConnectionError("blip")
            return Trajectory(task_id=task.id, final_output="ok")

    run = run_suite(Flaky(), tasks(1), retries=2, progress=False, save=False)
    assert run.summary["passed"] == 1
    assert run.summary["infra_errors"] == 0


def test_genuine_task_failures_are_not_retried():
    agent = ScriptedAgent({"t0": "raise"})
    run = run_suite(agent, tasks(1), retries=3, progress=False, save=False)
    assert agent.seen == ["t0"]
    assert run.results[0].infra_error is None


def test_infra_errors_excluded_from_task_success():
    class Mixed(ScriptedAgent):
        def run(self, task):
            if task.id == "t0":
                raise ConnectionError("blip")
            return Trajectory(task_id=task.id, final_output="ok")

    run = run_suite(Mixed(), tasks(3), retries=0, progress=False, save=False)
    assert run.summary["task_success"] == 1.0
    assert run.summary["graded"] == 2
    assert run.summary["infra_errors"] == 1


def test_setup_and_teardown_run_once():
    agent = ScriptedAgent()
    run_suite(agent, tasks(4), progress=False, save=False)
    assert agent.setup_calls == 1 and agent.teardown_calls == 1


def test_teardown_runs_even_when_tasks_explode():
    agent = ScriptedAgent({f"t{i}": "raise" for i in range(3)})
    run_suite(agent, tasks(3), progress=False, save=False)
    assert agent.teardown_calls == 1


def test_progress_output_is_plain_ascii():
    stream = io.StringIO()
    run_suite(ScriptedAgent(), tasks(2), progress=True, save=False, progress_stream=stream)
    text = stream.getvalue()
    assert text.isascii()
    assert "[  1/2]" in text and "pass" in text


def test_empty_task_list_is_a_usage_error():
    with pytest.raises(ValueError):
        run_suite(ScriptedAgent(), [], progress=False, save=False)


def test_run_is_persisted_and_reloadable(store):
    run = run_suite(ScriptedAgent(), tasks(2), progress=False, store=store)
    assert store.path_for(run.run_id).exists()
    assert store.load(run.run_id).summary == run.summary


# --- summary safety -------------------------------------------------------


def test_summarise_empty_run_has_no_division_by_zero():
    s = summarise(SuiteRun(run_id="r", suite="s", agent_name="a", agent_version="v"))
    assert s["task_success"] == 0.0
    assert s["tool_error_rate"] == 0.0
    assert s["cost_per_task"] == 0.0
    assert s["steps_median"] == 0.0
    assert s["latency_p95"] == 0.0


def test_summarise_all_failed_run():
    agent = ScriptedAgent()
    run = run_suite(agent, tasks(3, expected="something else"), progress=False, save=False)
    assert run.summary["task_success"] == 0.0
    assert run.summary["failed"] == 3


def test_summarise_all_infra_errors():
    class Down(ScriptedAgent):
        def run(self, task):
            raise ConnectionError("down")

    run = run_suite(Down(), tasks(3), retries=0, progress=False, save=False)
    assert run.summary["graded"] == 0
    assert run.summary["task_success"] == 0.0
    assert run.summary["tool_error_rate"] == 0.0


def test_summarise_counts_tool_errors():
    from caliper.types import Step

    def with_tools(task):
        return Trajectory(
            task_id=task.id,
            final_output="ok",
            steps=[
                Step(0, "tool", "a"),
                Step(1, "tool", "b", error="boom"),
                Step(2, "llm", "c", error="ignored by tool rate"),
            ],
        ).recompute()

    run = run_suite(ScriptedAgent({"t0": with_tools}), tasks(1), progress=False, save=False)
    assert run.summary["tool_calls"] == 2
    assert run.summary["tool_error_rate"] == 0.5


def test_per_grader_breakdown():
    ts = [make_task("a", expected="ok", grader=["exact_match", "completed"]),
          make_task("b", expected="no", grader=["exact_match", "completed"])]
    run = run_suite(ScriptedAgent(), ts, progress=False, save=False)
    assert run.summary["graders"]["exact_match"]["pass_rate"] == 0.5
    assert run.summary["graders"]["completed"]["pass_rate"] == 1.0


@pytest.mark.parametrize(
    "values,pct,expected",
    [([], 95, 0.0), ([5.0], 95, 5.0), ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95, 10),
     ([1, 2, 3, 4], 50, 2)],
)
def test_percentile(values, pct, expected):
    assert percentile([float(v) for v in values], pct) == expected
