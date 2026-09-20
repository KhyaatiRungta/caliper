"""The adapter boundary.

This is the only surface an agent must implement to be measurable by Caliper.

    class AgentAdapter(Protocol):
        name: str
        version: str
        def run(self, task: Task) -> Trajectory: ...

That is the whole contract. Caliper does not wrap your agent, does not patch
your LLM client, does not require a framework, and does not care how your loop
is written. You hand back a :class:`Trajectory`; Caliper scores it.

Contract, precisely
-------------------

``name``
    Stable identifier of the agent, e.g. ``"analyst-agent"``. Used for
    grouping runs. Two runs are only meaningfully comparable when the name and
    the suite match -- ``caliper compare`` warns otherwise.

``version``
    Free-form version marker, e.g. ``"v2"`` or a git sha. This is the axis the
    comparison report is built around. Bump it when you change the prompt, the
    tools, the model, or anything else you want attributed.

``run(task) -> Trajectory``
    Must return a Trajectory even on failure. If your agent cannot answer,
    return a trajectory with ``terminated_reason="error"`` and whatever steps
    were recorded before the failure -- a partial trajectory is far more useful
    than an exception, because the trajectory graders can still say where it
    died. If you raise instead, the runner isolates it and records the
    exception, but you lose the step detail.

    ``run`` may be called concurrently from several threads with different
    tasks. Keep per-task state on the stack, not on ``self``.

    Respecting ``task.timeout_s`` is polite but not required: the runner
    enforces it independently.

Optional hooks (duck-typed, all safely absent)
----------------------------------------------

``model: str``
    Reported in the run header. Purely informational.

``setup() -> None`` / ``teardown() -> None``
    Called once before the first task and once after the last.

Recording steps
---------------

Use :class:`TrajectoryRecorder`. It is 60 lines, has no dependency on the rest
of Caliper beyond the types, and is copy-pasteable into an agent repo that does
not want to depend on Caliper at all. The point is that instrumentation should
cost an agent author about ten minutes.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator, Protocol, runtime_checkable

from caliper.types import Step, StepKind, Task, Trajectory


@runtime_checkable
class AgentAdapter(Protocol):
    """Structural protocol. Inherit from it or just match the shape."""

    name: str
    version: str

    def run(self, task: Task) -> Trajectory:  # pragma: no cover - protocol
        ...


class TrajectoryRecorder:
    """Accumulates steps and produces a :class:`Trajectory`.

    Typical use inside an agent loop::

        rec = TrajectoryRecorder(task.id)
        with rec.step("llm", "plan") as s:
            resp = client.complete(messages)
            s.output = resp.text
            s.tokens_in = resp.usage.prompt_tokens
            s.tokens_out = resp.usage.completion_tokens
            s.cost_usd = resp.usage.cost_usd
        return rec.finish(final_output=answer)

    The context manager times the block, assigns the index, and captures any
    exception into ``step.error`` before re-raising -- so an agent that dies
    mid-tool still leaves an accurate record of the tool that killed it.
    """

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.steps: list[Step] = []
        self._started = time.time()

    # -- explicit recording ------------------------------------------------

    def add(
        self,
        kind: StepKind,
        name: str,
        *,
        input: Any = None,
        output: Any = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_usd: float = 0.0,
        latency_s: float = 0.0,
        error: str | None = None,
    ) -> Step:
        step = Step(
            index=len(self.steps),
            kind=kind,
            name=name,
            input=input,
            output=output,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
            latency_s=latency_s,
            error=error,
        )
        self.steps.append(step)
        return step

    # -- timed block recording --------------------------------------------

    @contextmanager
    def step(self, kind: StepKind, name: str, input: Any = None) -> Iterator[Step]:
        step = self.add(kind, name, input=input)
        started = time.time()
        try:
            yield step
        except Exception as exc:  # noqa: BLE001 - record then re-raise
            step.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            step.latency_s = time.time() - started

    # -- finishing ---------------------------------------------------------

    def finish(
        self, final_output: Any = None, terminated_reason: str = "completed"
    ) -> Trajectory:
        traj = Trajectory(
            task_id=self.task_id,
            steps=list(self.steps),
            final_output=final_output,
            wall_time_s=time.time() - self._started,
            terminated_reason=terminated_reason,
        )
        return traj.recompute()
