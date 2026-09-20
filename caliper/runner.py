"""Suite runner.

Guarantees this module makes, and which the tests pin down:

1. **Isolation.** A task that raises, hangs, or returns garbage produces a
   failing :class:`RunResult` and nothing else. One exploding task never kills
   the run.
2. **Deterministic output ordering.** Results come back in suite order
   regardless of which worker finished first. A diff of two run files must not
   be full of reordering noise.
3. **Infra errors are not task failures.** A 429 from a provider is not the
   agent being wrong. Transient failures are retried, and if they persist the
   result is marked ``infra_error`` and counted separately in the summary. An
   eval harness that reports a rate-limit as a regression trains you to ignore
   it.
4. **Timeouts are enforced by the runner**, not trusted to the adapter.

A caveat stated plainly: Python cannot safely kill a thread. A timed-out task
is abandoned -- its result is recorded as a timeout immediately and the run
proceeds -- but the worker thread may still be running underneath. The process
therefore shuts its executor down without waiting. This is the correct tradeoff
for an eval harness and the wrong one for a scheduler.
"""

from __future__ import annotations

import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Callable

from caliper.adapter import AgentAdapter
from caliper.graders import grade
from caliper.store import RunStore, new_run_id
from caliper.types import RunResult, Score, SuiteRun, Task, Trajectory

DEFAULT_WORKERS = 4
DEFAULT_RETRIES = 2

# Exception type names treated as transient infrastructure faults rather than
# as the agent being wrong. Matched by name so no provider SDK is imported.
TRANSIENT_NAMES = {
    "RateLimitError",
    "APIConnectionError",
    "APITimeoutError",
    "InternalServerError",
    "APIStatusError",
    "ServiceUnavailableError",
    "LLMCallError",
    "ConnectionError",
    "TimeoutError",
}


def is_transient(exc: BaseException) -> bool:
    if type(exc).__name__ in TRANSIENT_NAMES:
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and (status == 429 or 500 <= status < 600)


class Progress:
    """Plain-text progress to stderr. No spinners, no unicode, no emoji."""

    def __init__(self, total: int, enabled: bool = True, stream=None):
        self.total = total
        self.enabled = enabled
        self.stream = stream or sys.stderr
        self.done = 0
        self._lock = threading.Lock()

    def start(self, run_id: str, suite: str, agent: str, workers: int) -> None:
        if not self.enabled:
            return
        print(
            f"caliper: run {run_id} suite={suite} agent={agent} "
            f"tasks={self.total} workers={workers}",
            file=self.stream,
            flush=True,
        )

    def tick(self, result: RunResult) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.done += 1
            if result.infra_error:
                status = "INFRA"
            elif result.passed:
                status = "pass"
            else:
                status = "FAIL"
            print(
                f"  [{self.done:>3}/{self.total}] {status:<5} {result.task.id} "
                f"({result.trajectory.wall_time_s:.2f}s, "
                f"{len(result.trajectory.steps)} steps)",
                file=self.stream,
                flush=True,
            )

    def finish(self, summary: dict) -> None:
        if not self.enabled:
            return
        print(
            f"caliper: done. {summary['passed']}/{summary['graded']} passed "
            f"({summary['task_success']:.1%}), {summary['infra_errors']} infra error(s), "
            f"${summary['total_cost_usd']:.4f} total",
            file=self.stream,
            flush=True,
        )


def _failed_result(task: Task, reason: str, detail: str, infra: bool = False) -> RunResult:
    traj = Trajectory(task_id=task.id, final_output=None, terminated_reason=reason)
    result = RunResult(
        task=task,
        trajectory=traj,
        scores=[Score(name="run", value=0.0, passed=False, detail=detail)],
        passed=False,
        infra_error=detail if infra else None,
    )
    return result


def _execute_one(
    adapter: AgentAdapter,
    task: Task,
    retries: int,
    grader_overrides: dict[str, dict],
    started_at: dict[str, float] | None = None,
) -> RunResult:
    """Run and grade one task inside a worker thread. Never raises.

    ``started_at`` is written the moment the worker picks the task up, so the
    main thread can enforce a timeout over EXECUTION time and not over time
    spent queued behind other tasks.
    """
    if started_at is not None:
        started_at[task.id] = time.time()
    attempt = 0
    while True:
        attempt += 1
        started = time.time()
        try:
            traj = adapter.run(task)
        except BaseException as exc:  # noqa: BLE001 - isolation is the point
            if is_transient(exc) and attempt <= retries:
                print(
                    f"caliper: transient failure on {task.id} "
                    f"(attempt {attempt}/{retries + 1}): {type(exc).__name__}",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(min(2.0 ** (attempt - 1), 8.0))
                continue
            detail = f"{type(exc).__name__}: {exc}"
            return _failed_result(task, "error", detail, infra=is_transient(exc))

        if not isinstance(traj, Trajectory):
            return _failed_result(
                task,
                "error",
                f"adapter {adapter.name!r} returned {type(traj).__name__}, expected Trajectory",
            )

        traj.task_id = task.id
        if not traj.wall_time_s:
            traj.wall_time_s = time.time() - started
        traj.recompute()
        scores = grade(task, traj, **grader_overrides)
        return RunResult(task=task, trajectory=traj, scores=scores).recompute_passed()


def run_suite(
    adapter: AgentAdapter,
    tasks: list[Task],
    suite_name: str = "",
    workers: int = DEFAULT_WORKERS,
    retries: int = DEFAULT_RETRIES,
    progress: bool = True,
    store: RunStore | None = None,
    save: bool = True,
    run_id: str | None = None,
    grader_overrides: dict[str, dict] | None = None,
    progress_stream=None,
) -> SuiteRun:
    """Execute ``tasks`` against ``adapter`` and return a persisted SuiteRun."""
    if not tasks:
        raise ValueError("no tasks to run (check --tag and --limit filters)")

    suite_name = suite_name or tasks[0].suite or "suite"
    agent_name = getattr(adapter, "name", type(adapter).__name__)
    agent_version = getattr(adapter, "version", "unknown")
    rid = run_id or new_run_id(suite_name, f"{agent_name}-{agent_version}")
    overrides = grader_overrides or {}

    setup = getattr(adapter, "setup", None)
    if callable(setup):
        setup()

    prog = Progress(len(tasks), enabled=progress, stream=progress_stream)
    prog.start(rid, suite_name, f"{agent_name} {agent_version}", workers)

    started_at = time.time()
    results: dict[str, RunResult] = {}
    executor = ThreadPoolExecutor(max_workers=max(1, workers))
    try:
        exec_started: dict[str, float] = {}
        futures = {
            executor.submit(_execute_one, adapter, task, retries, overrides, exec_started): task
            for task in tasks
        }
        for future, task in futures.items():
            budget = max(0.05, task.timeout_s)
            try:
                result = _await_with_timeout(future, task, exec_started, budget)
            except FutureTimeout:
                future.cancel()
                result = _failed_result(
                    task,
                    "timeout",
                    f"exceeded timeout_s={task.timeout_s:g}",
                )
                result.trajectory.wall_time_s = task.timeout_s
            except BaseException as exc:  # noqa: BLE001 - belt and braces
                result = _failed_result(task, "error", f"{type(exc).__name__}: {exc}")
            results[task.id] = result
            prog.tick(result)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        teardown = getattr(adapter, "teardown", None)
        if callable(teardown):
            teardown()

    # Deterministic ordering: suite order, always.
    ordered = [results[t.id] for t in tasks]
    run = SuiteRun(
        run_id=rid,
        suite=suite_name,
        agent_name=agent_name,
        agent_version=agent_version,
        model=str(getattr(adapter, "model", "")),
        started_at=started_at,
        finished_at=time.time(),
        results=ordered,
    )
    run.summary = summarise(run)
    prog.finish(run.summary)

    if save:
        (store or RunStore()).save(run)
    return run


_POLL_S = 0.02


def _await_with_timeout(
    future, task: Task, exec_started: dict[str, float], budget: float
) -> RunResult:
    """Wait for ``future``, counting only time after the worker began the task.

    Raises :class:`FutureTimeout` when the task has been executing for longer
    than its budget. Time spent waiting in the executor queue does not count.
    """
    while True:
        try:
            return future.result(timeout=_POLL_S)
        except FutureTimeout:
            begun = exec_started.get(task.id)
            if begun is not None and (time.time() - begun) > budget:
                raise


def _safe_div(numer: float, denom: float, default: float = 0.0) -> float:
    return numer / denom if denom else default


def summarise(run: SuiteRun) -> dict[str, Any]:
    """Aggregate metrics for a run.

    Every rate here is division-by-zero safe. An empty run, an all-infra-error
    run and an all-failed run all produce a valid summary; a metric with no
    denominator reports 0.0 rather than NaN or an exception.

    ``graded`` excludes infra errors: task_success is the pass rate over tasks
    that actually got a fair attempt.
    """
    results = run.results
    total = len(results)
    infra = [r for r in results if r.infra_error]
    graded = [r for r in results if not r.infra_error]
    passed = [r for r in graded if r.passed]

    step_counts = [len(r.trajectory.steps) for r in graded]
    costs = [r.trajectory.total_cost_usd for r in graded]
    latencies = [r.trajectory.wall_time_s for r in graded]
    tool_steps = [s for r in graded for s in r.trajectory.tool_steps]
    tool_errors = [s for s in tool_steps if s.error is not None]
    timeouts = [r for r in graded if r.trajectory.terminated_reason == "timeout"]

    per_grader: dict[str, dict[str, float]] = {}
    for r in graded:
        for s in r.scores:
            bucket = per_grader.setdefault(s.name, {"n": 0.0, "passed": 0.0, "value_sum": 0.0})
            bucket["n"] += 1
            bucket["passed"] += 1.0 if s.passed else 0.0
            bucket["value_sum"] += s.value
    for name, bucket in per_grader.items():
        n = bucket["n"]
        bucket["pass_rate"] = _safe_div(bucket["passed"], n)
        bucket["mean_value"] = _safe_div(bucket["value_sum"], n)

    return {
        "total_tasks": total,
        "graded": len(graded),
        "passed": len(passed),
        "failed": len(graded) - len(passed),
        "infra_errors": len(infra),
        "timeouts": len(timeouts),
        "task_success": _safe_div(len(passed), len(graded)),
        "steps_median": statistics.median(step_counts) if step_counts else 0.0,
        "steps_mean": _safe_div(sum(step_counts), len(step_counts)),
        "total_steps": sum(step_counts),
        "total_cost_usd": sum(costs),
        "cost_per_task": _safe_div(sum(costs), len(graded)),
        "total_tokens": sum(r.trajectory.total_tokens for r in graded),
        "latency_mean": _safe_div(sum(latencies), len(latencies)),
        "latency_p95": percentile(latencies, 95),
        "wall_time_s": max(0.0, run.finished_at - run.started_at),
        "tool_calls": len(tool_steps),
        "tool_error_rate": _safe_div(len(tool_errors), len(tool_steps)),
        "graders": per_grader,
    }


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile. Empty input is 0.0, not an error."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = max(1, min(len(ordered), int(round(pct / 100.0 * len(ordered) + 0.5))))
    return ordered[rank - 1]
