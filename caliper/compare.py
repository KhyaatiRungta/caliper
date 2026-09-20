"""Comparison and regression detection.

This is the headline artifact. Given two runs of the same suite, it answers the
only question a team actually has about a prompt change: did it get better or
worse, and what did it cost?

The output is deliberately a monospace table and not a chart. A table diffs
cleanly in a pull request comment, renders in any terminal, and cannot
exaggerate a two-point move by choosing an axis.

Regression semantics
--------------------

A regression is a task that passed in the baseline run and fails in the
candidate run. Not a metric that moved the wrong way -- a specific, named task
that used to work and now does not. That is the only signal precise enough to
block a merge on, and ``caliper compare --fail-on-regression`` exits 1 when any
exist, which is what makes this a CI gate rather than a report.

Tasks that error at the infrastructure level are excluded from the transition
analysis in both directions. A rate limit is not a regression, and letting one
block a merge is how a team learns to pass ``--no-verify``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from caliper.types import RunResult, SuiteRun

Transition = Literal["pass->fail", "fail->pass", "pass->pass", "fail->fail"]


@dataclass
class TaskDelta:
    task_id: str
    before: str  # pass | fail | absent | infra
    after: str
    transition: str
    steps_before: int = 0
    steps_after: int = 0
    cost_before: float = 0.0
    cost_after: float = 0.0
    detail: str = ""

    @property
    def is_regression(self) -> bool:
        return self.transition == "pass->fail"

    @property
    def is_fix(self) -> bool:
        return self.transition == "fail->pass"


@dataclass
class MetricDelta:
    """One row of the comparison table."""

    label: str
    after: float
    before: float
    fmt: str = "num"          # num | pct | usd | secs
    lower_is_better: bool = False
    delta_style: str = "abs"  # abs | pct | points

    @property
    def delta(self) -> float:
        return self.after - self.before

    @property
    def relative(self) -> float | None:
        if self.before == 0:
            return None
        return (self.after - self.before) / abs(self.before)

    @property
    def improved(self) -> bool | None:
        if self.delta == 0:
            return None
        return (self.delta < 0) if self.lower_is_better else (self.delta > 0)


@dataclass
class Comparison:
    candidate: SuiteRun
    baseline: SuiteRun
    metrics: list[MetricDelta] = field(default_factory=list)
    task_deltas: list[TaskDelta] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def regressions(self) -> list[TaskDelta]:
        return [d for d in self.task_deltas if d.is_regression]

    @property
    def fixes(self) -> list[TaskDelta]:
        return [d for d in self.task_deltas if d.is_fix]

    @property
    def unchanged(self) -> list[TaskDelta]:
        return [d for d in self.task_deltas if d.transition in ("pass->pass", "fail->fail")]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.run_id,
            "baseline": self.baseline.run_id,
            "metrics": [
                {"label": m.label, "after": m.after, "before": m.before, "delta": m.delta}
                for m in self.metrics
            ],
            "task_deltas": [d.__dict__ for d in self.task_deltas],
            "regressions": [d.task_id for d in self.regressions],
            "fixes": [d.task_id for d in self.fixes],
            "warnings": self.warnings,
        }


def _status(result: RunResult | None) -> str:
    if result is None:
        return "absent"
    if result.infra_error:
        return "infra"
    return "pass" if result.passed else "fail"


def _index(run: SuiteRun) -> dict[str, RunResult]:
    return {r.task.id: r for r in run.results}


def compare_runs(candidate: SuiteRun, baseline: SuiteRun) -> Comparison:
    """Diff ``candidate`` against ``baseline``. Candidate is the new run."""
    cmp = Comparison(candidate=candidate, baseline=baseline)

    if candidate.suite != baseline.suite:
        cmp.warnings.append(
            f"suites differ ({candidate.suite!r} vs {baseline.suite!r}); "
            "the comparison is unlikely to mean anything"
        )
    if candidate.agent_name != baseline.agent_name:
        cmp.warnings.append(
            f"agents differ ({candidate.agent_name!r} vs {baseline.agent_name!r}); "
            "this compares two agents, not two versions of one"
        )

    a, b = _index(candidate), _index(baseline)

    # Deterministic order: candidate suite order first, then baseline-only ids.
    ordered_ids = [r.task.id for r in candidate.results]
    ordered_ids += [tid for tid in (r.task.id for r in baseline.results) if tid not in a]

    for tid in ordered_ids:
        after_r, before_r = a.get(tid), b.get(tid)
        after, before = _status(after_r), _status(before_r)
        if before in ("absent", "infra") or after in ("absent", "infra"):
            transition = f"{before}->{after}"
        else:
            transition = f"{before}->{after}"
        detail = ""
        if after_r is not None and not after_r.passed:
            failing = [s for s in after_r.scores if not s.passed]
            if failing:
                detail = f"{failing[0].name}: {failing[0].detail}"[:120]
        cmp.task_deltas.append(
            TaskDelta(
                task_id=tid,
                before=before,
                after=after,
                transition=transition,
                steps_before=len(before_r.trajectory.steps) if before_r else 0,
                steps_after=len(after_r.trajectory.steps) if after_r else 0,
                cost_before=before_r.trajectory.total_cost_usd if before_r else 0.0,
                cost_after=after_r.trajectory.total_cost_usd if after_r else 0.0,
                detail=detail,
            )
        )

    sa, sb = candidate.summary or {}, baseline.summary or {}
    cmp.metrics = [
        MetricDelta(
            "task success",
            sa.get("task_success", 0.0),
            sb.get("task_success", 0.0),
            fmt="pct",
            delta_style="points",
        ),
        MetricDelta(
            "steps (median)",
            float(sa.get("steps_median", 0.0)),
            float(sb.get("steps_median", 0.0)),
            fmt="num",
            lower_is_better=True,
            delta_style="abs",
        ),
        MetricDelta(
            "cost / task",
            sa.get("cost_per_task", 0.0),
            sb.get("cost_per_task", 0.0),
            fmt="usd",
            lower_is_better=True,
            delta_style="pct",
        ),
        MetricDelta(
            "p95 latency",
            sa.get("latency_p95", 0.0),
            sb.get("latency_p95", 0.0),
            fmt="secs",
            lower_is_better=True,
            delta_style="pct",
        ),
        MetricDelta(
            "tool-error rate",
            sa.get("tool_error_rate", 0.0),
            sb.get("tool_error_rate", 0.0),
            fmt="pct",
            lower_is_better=True,
            delta_style="points",
        ),
    ]
    return cmp


def has_regressions(cmp: Comparison) -> bool:
    return bool(cmp.regressions)
