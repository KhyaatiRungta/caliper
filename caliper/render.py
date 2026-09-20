"""Plain-text renderers.

All Caliper output is monospace text with hyphen rules. No unicode box drawing,
no colour, no emoji, no spinners. Output that survives a pipe, a CI log, a
pull-request comment and a paste into an email is worth more than output that
looks good in exactly one terminal.

The comparison table geometry is fixed and tested against a golden string,
because the table appears on the project website and in CI logs and must not
drift when a metric is added.
"""

from __future__ import annotations

from datetime import datetime, timezone

from caliper.compare import Comparison, MetricDelta
from caliper.types import RunResult, SuiteRun

# --- comparison table geometry (fixed, golden-tested) ---------------------
LABEL_W = 15   # left column, left-justified
AFTER_W = 8    # candidate value, right-justified
BEFORE_W = 11  # baseline value, right-justified
DELTA_W = 8    # delta, right-justified
RULE_W = 43


def _fmt_value(value: float, fmt: str) -> str:
    if fmt == "pct":
        return f"{value * 100:.1f}%"
    if fmt == "usd":
        return f"${value:.3f}"
    if fmt == "secs":
        return f"{value:.1f}s"
    return f"{value:.1f}"


def _fmt_delta(m: MetricDelta) -> str:
    if m.delta_style == "points":
        return f"{m.delta * 100:+.1f}"
    if m.delta_style == "pct":
        rel = m.relative
        return "n/a" if rel is None else f"{rel * 100:+.0f}%"
    return f"{m.delta:+.1f}"


def metric_row(m: MetricDelta) -> str:
    return (
        m.label.ljust(LABEL_W)
        + _fmt_value(m.after, m.fmt).rjust(AFTER_W)
        + _fmt_value(m.before, m.fmt).rjust(BEFORE_W)
        + _fmt_delta(m).rjust(DELTA_W)
    )


def render_comparison(cmp: Comparison, show_fixes: bool = True) -> str:
    """The headline table.

    Shape::

        SUITE: analyst-agent v3   vs   v2
        -------------------------------------------
        task success      78.0%      64.0%   +14.0
        steps (median)      4.0        6.0    -2.0
        cost / task      $0.031     $0.048    -35%
        p95 latency       12.4s      19.1s    -35%
        tool-error rate    9.2%      22.5%   -13.3

        REGRESSIONS (2)
          join_three_tables      pass -> fail
          null_handling_edge     pass -> fail
    """
    agent = cmp.candidate.agent_name or cmp.candidate.suite
    lines: list[str] = [
        f"SUITE: {agent} {cmp.candidate.agent_version}   vs   {cmp.baseline.agent_version}",
        "-" * RULE_W,
    ]
    lines.extend(metric_row(m) for m in cmp.metrics)

    regressions = cmp.regressions
    lines.append("")
    if regressions:
        width = max(len(d.task_id) for d in regressions) + 5
        lines.append(f"REGRESSIONS ({len(regressions)})")
        for d in regressions:
            lines.append("  " + d.task_id.ljust(width) + "pass -> fail")
    else:
        lines.append("REGRESSIONS (0)")
        lines.append("  none")

    if show_fixes and cmp.fixes:
        width = max(len(d.task_id) for d in cmp.fixes) + 5
        lines.append("")
        lines.append(f"FIXES ({len(cmp.fixes)})")
        for d in cmp.fixes:
            lines.append("  " + d.task_id.ljust(width) + "fail -> pass")

    if cmp.warnings:
        lines.append("")
        for w in cmp.warnings:
            lines.append(f"WARNING: {w}")

    return "\n".join(lines)


# --- run listing and summary ---------------------------------------------


def _ts(value: float) -> str:
    if not value:
        return "-"
    return datetime.fromtimestamp(value, timezone.utc).strftime("%Y-%m-%d %H:%M")


def render_run_list(runs: list[SuiteRun]) -> str:
    if not runs:
        return "no runs recorded. Try: caliper run suites/reference.yaml --agent reference-v2"
    header = (
        "RUN ID".ljust(32)
        + "SUITE".ljust(14)
        + "AGENT".ljust(22)
        + "PASS".rjust(9)
        + "COST".rjust(10)
        + "  STARTED"
    )
    lines = [header, "-" * (len(header) + 8)]
    for r in runs:
        s = r.summary or {}
        pass_str = f"{s.get('passed', 0)}/{s.get('graded', 0)}"
        lines.append(
            r.run_id.ljust(32)
            + r.suite[:13].ljust(14)
            + f"{r.agent_name} {r.agent_version}"[:21].ljust(22)
            + pass_str.rjust(9)
            + f"${s.get('total_cost_usd', 0.0):.3f}".rjust(10)
            + "  "
            + _ts(r.started_at)
        )
    return "\n".join(lines)


def render_run_summary(run: SuiteRun, verbose: bool = True) -> str:
    s = run.summary or {}
    lines = [
        f"RUN:    {run.run_id}",
        f"SUITE:  {run.suite}",
        f"AGENT:  {run.agent_name} {run.agent_version}"
        + (f"   MODEL: {run.model}" if run.model else ""),
        f"WHEN:   {_ts(run.started_at)} UTC",
        "-" * 58,
        f"task success     {s.get('task_success', 0.0) * 100:.1f}%  "
        f"({s.get('passed', 0)}/{s.get('graded', 0)} graded)",
        f"infra errors     {s.get('infra_errors', 0)}"
        f"   timeouts {s.get('timeouts', 0)}",
        f"steps (median)   {s.get('steps_median', 0.0):.1f}"
        f"   total {s.get('total_steps', 0)}",
        f"cost / task      ${s.get('cost_per_task', 0.0):.4f}"
        f"   total ${s.get('total_cost_usd', 0.0):.4f}",
        f"p95 latency      {s.get('latency_p95', 0.0):.2f}s"
        f"   wall {s.get('wall_time_s', 0.0):.2f}s",
        f"tool-error rate  {s.get('tool_error_rate', 0.0) * 100:.1f}%"
        f"   over {s.get('tool_calls', 0)} tool calls",
    ]

    graders = s.get("graders") or {}
    if graders:
        lines += ["", "GRADERS", "-" * 58]
        name_w = max(len(n) for n in graders) + 2
        for name in sorted(graders):
            g = graders[name]
            lines.append(
                name.ljust(name_w)
                + f"{g.get('pass_rate', 0.0) * 100:5.1f}% pass"
                + f"   mean {g.get('mean_value', 0.0):.2f}"
                + f"   n={int(g.get('n', 0))}"
            )

    if verbose and run.results:
        lines += ["", "TASKS", "-" * 58]
        id_w = max(len(r.task.id) for r in run.results) + 2
        for r in run.results:
            status = "INFRA" if r.infra_error else ("pass" if r.passed else "FAIL")
            lines.append(
                status.ljust(6)
                + r.task.id.ljust(id_w)
                + f"{len(r.trajectory.steps):>3} steps"
                + f"  ${r.trajectory.total_cost_usd:.4f}"
                + f"  {r.trajectory.wall_time_s:5.2f}s"
            )
            if not r.passed:
                for sc in r.scores:
                    if not sc.passed:
                        lines.append(" " * 6 + f"- {sc.name}: {sc.detail}"[:110])
    return "\n".join(lines)


def render_trace(result: RunResult) -> str:
    """Step-by-step dump of one trajectory."""
    t, traj = result.task, result.trajectory
    lines = [
        f"TASK:   {t.id}   [{t.suite}]",
        f"INPUT:  {t.input}",
        f"EXPECT: {t.expected}",
        f"RESULT: {'pass' if result.passed else 'FAIL'}"
        + (f"   INFRA ERROR: {result.infra_error}" if result.infra_error else ""),
        f"END:    {traj.terminated_reason}   {len(traj.steps)} steps   "
        f"${traj.total_cost_usd:.4f}   {traj.wall_time_s:.2f}s",
        "-" * 70,
    ]
    if not traj.steps:
        lines.append("(no steps recorded)")
    for st in traj.steps:
        head = (
            f"[{st.index:>2}] {st.kind:<9} {st.name:<14}"
            f" {st.latency_s:>6.2f}s  ${st.cost_usd:.4f}"
            f"  {st.tokens_in}+{st.tokens_out} tok"
        )
        lines.append(head)
        if st.input is not None:
            lines.append(f"      in   {str(st.input)[:160]}")
        if st.output is not None:
            lines.append(f"      out  {str(st.output)[:160]}")
        if st.error:
            lines.append(f"      ERROR {str(st.error)[:160]}")
    lines += ["-" * 70, f"FINAL:  {str(traj.final_output)[:300]}", "", "SCORES"]
    if not result.scores:
        lines.append("  (none)")
    name_w = max((len(s.name) for s in result.scores), default=4) + 2
    for sc in result.scores:
        lines.append(
            "  "
            + ("pass" if sc.passed else "FAIL").ljust(6)
            + sc.name.ljust(name_w)
            + f"{sc.value:.2f}  {sc.detail}"[:120]
        )
    return "\n".join(lines)
