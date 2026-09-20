"""Core data model for Caliper.

These dataclasses are the spine of the project. Everything else -- adapters,
graders, the runner, the store, the comparison report -- is written against
these types and nothing else.

The shapes are deliberately flat and JSON-round-trippable. A run is a file on
disk, and a file on disk is only useful if it can be read back by a tool that
was not compiled against this exact version of the code. Every type here
therefore exposes ``to_dict`` / ``from_dict`` with no magic: unknown keys are
ignored on read, and no object identity is relied upon.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Literal

StepKind = Literal["llm", "tool", "retrieval", "internal"]

VALID_STEP_KINDS: tuple[str, ...] = ("llm", "tool", "retrieval", "internal")


def _pick(cls, d: dict) -> dict:
    """Keep only the keys that are fields of ``cls``. Forward compatibility."""
    names = {f for f in cls.__dataclass_fields__}
    return {k: v for k, v in d.items() if k in names}


@dataclass
class Task:
    """One unit of evaluation.

    Attributes:
        id: Unique within the suite. Used as the join key across runs, so it
            must be stable -- renaming a task makes it look like a deletion
            plus an addition in a comparison.
        suite: Name of the suite the task belongs to.
        input: Whatever the agent needs to start. Free-form; the adapter owns
            its interpretation. For the reference agent it is a question string.
        expected: Reference answer or structure. Graders interpret this; a
            grader that does not need it ignores it.
        grader: Grader spec. Either a bare name (``"exact_match"``) or a list of
            specs. Each spec may be a name, or a mapping of
            ``{name: {**grader_args}}``.
        tags: Free-form labels, used for ``--tag`` filtering.
        difficulty: Author's own rating. Never used for scoring, only reporting.
        timeout_s: Hard per-task wall clock budget enforced by the runner.
        expected_tools: Tool names a correct solution needs. Consumed by
            ``tool_choice_precision``.
        reference_steps: Step count of a known-good trajectory. Consumed by
            ``step_efficiency``.
        cost_budget_usd / latency_budget_s: Task-level budgets, consumed by the
            budget graders.
        metadata: Escape hatch for suite-specific data. Graders may read it.
    """

    id: str
    suite: str = ""
    input: Any = None
    expected: Any = None
    grader: Any = "exact_match"
    tags: list[str] = field(default_factory=list)
    difficulty: str = "medium"
    timeout_s: float = 60.0
    expected_tools: list[str] = field(default_factory=list)
    reference_steps: int | None = None
    cost_budget_usd: float | None = None
    latency_budget_s: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        return cls(**_pick(cls, d))


@dataclass
class Step:
    """One observable action inside a trajectory.

    A step is whatever the agent author decided is worth recording. Caliper
    does not instrument the agent; the agent emits steps. That is the whole
    contract, and it is why Caliper works on agents it has never seen.

    ``kind`` is one of ``llm``, ``tool``, ``retrieval``, ``internal``.
    ``error`` is None on success and a short string on failure -- the string is
    read by ``recovered_from_error`` and by the tool-error-rate summary metric.
    """

    index: int
    kind: StepKind
    name: str
    input: Any = None
    output: Any = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Step":
        return cls(**_pick(cls, d))


@dataclass
class Trajectory:
    """The full record of one agent attempt at one task.

    ``terminated_reason`` is one of ``completed``, ``timeout``, ``error``,
    ``max_steps``. The runner sets it for the failure cases; the agent sets it
    for the rest.

    The totals are derived from steps via :meth:`recompute`, so an adapter that
    records steps honestly gets accurate cost accounting for free.
    """

    task_id: str
    steps: list[Step] = field(default_factory=list)
    final_output: Any = None
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    wall_time_s: float = 0.0
    terminated_reason: str = "completed"

    def recompute(self) -> "Trajectory":
        """Recalculate totals from steps. Idempotent."""
        self.total_cost_usd = sum(s.cost_usd for s in self.steps)
        self.total_tokens = sum(s.tokens_in + s.tokens_out for s in self.steps)
        return self

    # -- convenience accessors used heavily by trajectory graders --

    def steps_of(self, kind: StepKind) -> list[Step]:
        return [s for s in self.steps if s.kind == kind]

    @property
    def tool_steps(self) -> list[Step]:
        return self.steps_of("tool")

    @property
    def tool_names(self) -> list[str]:
        return [s.name for s in self.tool_steps]

    @property
    def errored_steps(self) -> list[Step]:
        return [s for s in self.steps if s.error is not None]

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Trajectory":
        d = dict(d)
        d["steps"] = [Step.from_dict(s) for s in d.get("steps", [])]
        return cls(**_pick(cls, d))


@dataclass
class Score:
    """The output of one grader on one trajectory.

    ``value`` is always a float in [0, 1] for quality graders so that scores
    aggregate meaningfully. Budget and rate graders also normalise to [0, 1]
    (1.0 = within budget) and carry the raw figure in ``detail``.

    ``passed`` is the binary the CI gate reads. ``unit`` is a display hint.
    """

    name: str
    value: float
    unit: str = "ratio"
    passed: bool = False
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Score":
        return cls(**_pick(cls, d))


@dataclass
class RunResult:
    """One task, its trajectory, and every score computed over it.

    ``passed`` is the AND of all scores: a task passes only if every grader
    attached to it passes. Partial credit lives in the individual scores.

    ``infra_error`` distinguishes "the API fell over" from "the agent was
    wrong". Runs with infra errors are reported separately in the summary and
    never silently counted as task failures.
    """

    task: Task
    trajectory: Trajectory
    scores: list[Score] = field(default_factory=list)
    passed: bool = False
    infra_error: str | None = None

    def score(self, name: str) -> Score | None:
        for s in self.scores:
            if s.name == name:
                return s
        return None

    def recompute_passed(self) -> "RunResult":
        if self.infra_error is not None:
            self.passed = False
        else:
            self.passed = bool(self.scores) and all(s.passed for s in self.scores)
        return self

    def to_dict(self) -> dict:
        return {
            "task": self.task.to_dict(),
            "trajectory": self.trajectory.to_dict(),
            "scores": [s.to_dict() for s in self.scores],
            "passed": self.passed,
            "infra_error": self.infra_error,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RunResult":
        return cls(
            task=Task.from_dict(d["task"]),
            trajectory=Trajectory.from_dict(d["trajectory"]),
            scores=[Score.from_dict(s) for s in d.get("scores", [])],
            passed=bool(d.get("passed", False)),
            infra_error=d.get("infra_error"),
        )


@dataclass
class SuiteRun:
    """Everything produced by one execution of one suite against one agent.

    This object is what gets written to ``.caliper/runs/<run_id>.json`` and is
    the only input to ``caliper compare``. It is self-describing on purpose:
    the agent name, version and model are embedded so a run file read a year
    later still says what it measured.
    """

    run_id: str
    suite: str
    agent_name: str
    agent_version: str
    model: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    results: list[RunResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "suite": self.suite,
            "agent_name": self.agent_name,
            "agent_version": self.agent_version,
            "model": self.model,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SuiteRun":
        return cls(
            run_id=d["run_id"],
            suite=d.get("suite", ""),
            agent_name=d.get("agent_name", ""),
            agent_version=d.get("agent_version", ""),
            model=d.get("model", ""),
            started_at=d.get("started_at", 0.0),
            finished_at=d.get("finished_at", 0.0),
            results=[RunResult.from_dict(r) for r in d.get("results", [])],
            summary=d.get("summary", {}),
        )
