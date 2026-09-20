"""The adapter boundary and TrajectoryRecorder."""

import pytest
from conftest import make_task

from caliper.adapter import AgentAdapter, TrajectoryRecorder
from caliper.types import Trajectory


def test_recorder_assigns_sequential_indices():
    rec = TrajectoryRecorder("t")
    rec.add("llm", "plan")
    rec.add("tool", "calc")
    assert [s.index for s in rec.steps] == [0, 1]


def test_recorder_finish_computes_totals():
    rec = TrajectoryRecorder("t")
    rec.add("llm", "plan", tokens_in=10, tokens_out=5, cost_usd=0.01)
    rec.add("tool", "calc", tokens_in=1, tokens_out=1, cost_usd=0.02)
    traj = rec.finish(final_output="42")
    assert traj.total_tokens == 17
    assert traj.total_cost_usd == pytest.approx(0.03)
    assert traj.final_output == "42"
    assert traj.terminated_reason == "completed"


def test_recorder_context_manager_times_the_block():
    rec = TrajectoryRecorder("t")
    with rec.step("tool", "calc") as s:
        s.output = 42
    assert rec.steps[0].output == 42
    assert rec.steps[0].latency_s >= 0.0


def test_recorder_captures_exceptions_then_reraises():
    """An agent that dies mid-tool still leaves an accurate record."""
    rec = TrajectoryRecorder("t")
    with pytest.raises(ValueError):
        with rec.step("tool", "calc"):
            raise ValueError("boom")
    assert rec.steps[0].error == "ValueError: boom"
    traj = rec.finish(terminated_reason="error")
    assert traj.errored_steps


def test_recorder_wall_time_is_measured():
    rec = TrajectoryRecorder("t")
    traj = rec.finish()
    assert traj.wall_time_s >= 0.0


def test_minimal_adapter_satisfies_the_protocol():
    class Minimal:
        name = "minimal"
        version = "v1"

        def run(self, task):
            return Trajectory(task_id=task.id, final_output="ok")

    agent = Minimal()
    assert isinstance(agent, AgentAdapter)
    assert agent.run(make_task()).final_output == "ok"


def test_an_object_missing_run_is_not_an_adapter():
    class NotAnAgent:
        name = "x"
        version = "v1"

    assert not isinstance(NotAnAgent(), AgentAdapter)
