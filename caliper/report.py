"""Standalone HTML report for a single run.

One file, no assets, no scripts, no icons, no external requests. It opens from
``file://`` and can be attached to a CI artifact or emailed. The design language
matches the project website: warm off-white, one ink-blue accent, hairline
rules, serif prose, monospace numbers, and terminal output in a dark panel.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone

from caliper.compare import Comparison
from caliper.render import render_comparison
from caliper.types import SuiteRun

CSS = """
:root {
  --bg: #faf8f3; --fg: #16150f; --accent: #1f3a5f; --rule: #d8d3c7;
  --dim: #6b665a; --panel: #f3f0e8; --term-bg: #16150f; --term-fg: #e8e4da;
  --term-dim: #8b8576;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #12110e; --fg: #e8e4da; --accent: #8fb0d6; --rule: #35322b;
    --dim: #97917f; --panel: #1a1815; --term-bg: #0c0b09; --term-fg: #e8e4da;
    --term-dim: #7d7768;
  }
}
* { box-sizing: border-box; }
body {
  background: var(--bg); color: var(--fg); margin: 0;
  font-family: ui-serif, Georgia, 'Times New Roman', serif;
  font-size: 17px; line-height: 1.65;
}
main { max-width: 1080px; margin: 0 auto; padding: 3.5rem 1.5rem 5rem; }
code, pre, table, .mono, .metric, .eyebrow {
  font-family: ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
  font-variant-numeric: tabular-nums;
}
.eyebrow {
  font-size: .68rem; letter-spacing: .18em; text-transform: uppercase;
  color: var(--dim); margin: 0 0 1rem;
}
h1 {
  font-size: clamp(1.8rem, 4vw, 2.4rem); font-weight: 600;
  letter-spacing: -.02em; line-height: 1.15; margin: 0 0 .4rem;
}
h2 {
  font-size: .72rem; font-weight: 600; letter-spacing: .14em; text-transform: uppercase;
  font-family: ui-monospace, 'SF Mono', Menlo, Consolas, monospace; color: var(--dim);
  border-top: 1px solid var(--rule); padding-top: 1.2rem; margin: 3.2rem 0 1.2rem;
}
.sub { color: var(--dim); font-size: .85rem; margin: 0 0 2.4rem; }
.metrics {
  display: grid; grid-template-columns: repeat(4, 1fr);
  border-top: 1px solid var(--rule); border-bottom: 1px solid var(--rule);
}
.metric { padding: 1.4rem 1.2rem; border-left: 1px solid var(--rule); }
.metric:first-child { border-left: 0; padding-left: 0; }
.metric .fig { font-size: 1.5rem; font-weight: 600; letter-spacing: -.03em; display: block; }
.metric .lbl {
  font-size: .62rem; letter-spacing: .14em; text-transform: uppercase;
  color: var(--dim); display: block; margin-top: .4rem;
}
@media (max-width: 760px) {
  .metrics { grid-template-columns: repeat(2, 1fr); }
  .metric:nth-child(3) { border-left: 0; padding-left: 0; }
  .metric:nth-child(-n+2) { border-bottom: 1px solid var(--rule); }
}
.term {
  background: var(--term-bg); color: var(--term-fg);
  border: 1px solid var(--rule); border-radius: 4px; overflow: hidden; margin: 1.4rem 0;
}
.term .bar {
  font-size: .66rem; letter-spacing: .12em; text-transform: uppercase;
  color: var(--term-dim); padding: .45rem 1rem;
  border-bottom: 1px solid rgba(139, 133, 118, .3);
}
.term pre {
  margin: 0; padding: 1.1rem 1rem; overflow-x: auto;
  font-size: .8rem; line-height: 1.6; color: var(--term-fg);
}
table { width: 100%; border-collapse: collapse; font-size: .82rem; }
th {
  text-align: left; font-weight: 600; color: var(--dim); text-transform: uppercase;
  letter-spacing: .1em; font-size: .64rem; padding: .45rem .8rem .45rem 0;
  border-bottom: 1px solid var(--fg);
}
td { padding: .4rem .8rem .4rem 0; border-bottom: 1px solid var(--rule); }
td.num, th.num { text-align: right; padding-right: 1.4rem; }
.pass { color: var(--accent); }
.fail { font-weight: 600; }
.kv { display: grid; grid-template-columns: 14rem 1fr; gap: .1rem 1rem; font-size: .84rem; }
.kv dt { color: var(--dim); }
.kv dd { margin: 0; }
footer {
  border-top: 1px solid var(--rule); margin-top: 3.5rem; padding-top: 1.2rem;
  font-size: .76rem; color: var(--dim);
  font-family: ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
}
footer a { color: var(--dim); text-decoration: none; margin-left: 1.2rem; }
footer a:hover { color: var(--fg); }
"""


def _esc(value) -> str:
    return html.escape(str(value), quote=False)


def _ts(value: float) -> str:
    if not value:
        return "-"
    return datetime.fromtimestamp(value, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def render_html(run: SuiteRun, cmp: Comparison | None = None) -> str:
    s = run.summary or {}
    title = f"Caliper report: {run.suite} / {run.agent_name} {run.agent_version}"

    headline = [
        (f"{s.get('task_success', 0.0) * 100:.1f}%", "task success"),
        (f"{s.get('steps_median', 0.0):.1f}", "steps (median)"),
        (f"${s.get('cost_per_task', 0.0):.4f}", "cost / task"),
        (f"{s.get('tool_error_rate', 0.0) * 100:.1f}%", "tool-error rate"),
    ]
    detail = [
        ("passed", f"{s.get('passed', 0)} / {s.get('graded', 0)} graded"),
        ("failed", str(s.get("failed", 0))),
        ("infra errors", str(s.get("infra_errors", 0))),
        ("timeouts", str(s.get("timeouts", 0))),
        ("total steps", str(s.get("total_steps", 0))),
        ("tool calls", str(s.get("tool_calls", 0))),
        ("total cost", f"${s.get('total_cost_usd', 0.0):.4f}"),
        ("total tokens", f"{s.get('total_tokens', 0):,}"),
        ("mean latency", f"{s.get('latency_mean', 0.0):.2f}s"),
        ("p95 latency", f"{s.get('latency_p95', 0.0):.2f}s"),
        ("wall time", f"{s.get('wall_time_s', 0.0):.2f}s"),
    ]

    parts: list[str] = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_esc(title)}</title><style>{CSS}</style></head><body><main>",
        '<p class="eyebrow">Caliper run report</p>',
        f"<h1>{_esc(run.suite)} / {_esc(run.agent_name)} {_esc(run.agent_version)}</h1>",
        f'<p class="sub">run {_esc(run.run_id)} &middot; {_ts(run.started_at)}'
        + (f" &middot; model {_esc(run.model)}" if run.model else "")
        + "</p>",
        '<div class="metrics">',
    ]
    for figure, label in headline:
        parts.append(
            f'<div class="metric"><span class="fig">{_esc(figure)}</span>'
            f'<span class="lbl">{_esc(label)}</span></div>'
        )
    parts.append("</div>")

    if cmp is not None:
        parts += [
            f"<h2>Comparison against {_esc(cmp.baseline.agent_version)}</h2>",
            '<div class="term"><div class="bar">caliper compare '
            f"{_esc(cmp.candidate.run_id)} {_esc(cmp.baseline.run_id)}</div>",
            f"<pre>{_esc(render_comparison(cmp))}</pre></div>",
        ]

    parts.append("<h2>Run detail</h2>")
    parts.append('<dl class="kv">')
    for label, value in detail:
        parts.append(f"<dt>{_esc(label)}</dt><dd>{_esc(value)}</dd>")
    parts.append("</dl>")

    graders = s.get("graders") or {}
    if graders:
        parts += [
            "<h2>Graders</h2>",
            "<table><thead><tr><th>grader</th>"
            '<th class="num">pass rate</th><th class="num">mean value</th>'
            '<th class="num">n</th></tr></thead><tbody>',
        ]
        for name in sorted(graders):
            g = graders[name]
            parts.append(
                f"<tr><td class='mono'>{_esc(name)}</td>"
                f"<td class='num mono'>{g.get('pass_rate', 0.0) * 100:.1f}%</td>"
                f"<td class='num mono'>{g.get('mean_value', 0.0):.2f}</td>"
                f"<td class='num mono'>{int(g.get('n', 0))}</td></tr>"
            )
        parts.append("</tbody></table>")

    parts += [
        "<h2>Tasks</h2>",
        "<table><thead><tr><th>status</th><th>task</th>"
        '<th class="num">steps</th><th class="num">cost</th>'
        '<th class="num">latency</th><th>note</th></tr></thead><tbody>',
    ]
    for r in run.results:
        if r.infra_error:
            status, cls, note = "infra", "fail", r.infra_error
        elif r.passed:
            status, cls, note = "pass", "pass", ""
        else:
            failing = [sc for sc in r.scores if not sc.passed]
            status, cls = "fail", "fail"
            note = f"{failing[0].name}: {failing[0].detail}" if failing else ""
        parts.append(
            f"<tr><td class='mono {cls}'>{status}</td>"
            f"<td class='mono'>{_esc(r.task.id)}</td>"
            f"<td class='num mono'>{len(r.trajectory.steps)}</td>"
            f"<td class='num mono'>${r.trajectory.total_cost_usd:.4f}</td>"
            f"<td class='num mono'>{r.trajectory.wall_time_s:.2f}s</td>"
            f"<td>{_esc(note[:160])}</td></tr>"
        )
    parts.append("</tbody></table>")

    parts += [
        '<footer>caliper &middot; 2026'
        '<a href="https://github.com/Manavarya09/quarry">quarry</a>'
        '<a href="https://github.com/Manavarya09/strata">strata</a>'
        '<a href="https://github.com/Manavarya09/caliper">caliper</a>'
        "</footer>",
        "</main></body></html>",
    ]
    return "\n".join(parts)
