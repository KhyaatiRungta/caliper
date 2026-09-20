"""Shared fixtures. Nothing here touches the network or needs an API key."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from caliper.types import Score, Step, SuiteRun, Task, Trajectory  # noqa: E402
from caliper.types import RunResult  # noqa: E402


def make_task(task_id: str = "t1", **kwargs) -> Task:
    kwargs.setdefault("suite", "test")
    return Task(id=task_id, **kwargs)


def make_step(index: int = 0, kind: str = "tool", name: str = "calculator", **kwargs) -> Step:
    return Step(index=index, kind=kind, name=name, **kwargs)


def make_trajectory(task_id: str = "t1", steps: list[Step] | None = None, **kwargs) -> Trajectory:
    traj = Trajectory(task_id=task_id, steps=steps or [], **kwargs)
    return traj.recompute()


def make_result(
    task_id: str,
    passed: bool,
    steps: int = 2,
    cost: float = 0.01,
    latency: float = 1.0,
    infra_error: str | None = None,
) -> RunResult:
    traj = make_trajectory(
        task_id,
        [make_step(i, cost_usd=cost / max(1, steps), latency_s=latency / max(1, steps))
         for i in range(steps)],
        wall_time_s=latency,
    )
    return RunResult(
        task=make_task(task_id),
        trajectory=traj,
        scores=[Score(name="exact_match", value=1.0 if passed else 0.0, passed=passed)],
        passed=passed and infra_error is None,
        infra_error=infra_error,
    )


def make_run(
    run_id: str,
    version: str,
    statuses: dict[str, bool],
    steps: int = 2,
    cost: float = 0.01,
    latency: float = 1.0,
) -> SuiteRun:
    from caliper.runner import summarise

    run = SuiteRun(
        run_id=run_id,
        suite="test",
        agent_name="test-agent",
        agent_version=version,
        started_at=1000.0,
        finished_at=1010.0,
        results=[
            make_result(tid, ok, steps=steps, cost=cost, latency=latency)
            for tid, ok in statuses.items()
        ],
    )
    run.summary = summarise(run)
    return run


@pytest.fixture
def store(tmp_path):
    from caliper.store import RunStore

    return RunStore(tmp_path / ".caliper")
