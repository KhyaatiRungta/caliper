"""Standalone HTML report for a single run.

One file, no assets, no scripts, no icons. It opens from ``file://`` and can be
attached to a CI artifact or emailed. The design language matches the project
website: warm off-white, one ink-blue accent, hairline rules, serif prose and
monospace numbers.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone

from caliper.compare import Comparison
from caliper.render import render_comparison
from caliper.types import SuiteRun

CSS = """
:root {
  --bg: #faf8f3; --fg: #16150f; --accent: #1f3a5f; --rule: #d8d3c7; --dim: #6b665a;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #12110e; --fg: #e8e4da; --accent: #8fb0d6; --rule: #35322b; --dim: #97917f; }
}
* { box-sizing: border-box; }
body {
  background: var(--bg); color: var(--fg); margin: 0;
  font-family: ui-serif, Georgia, 'Times New Roman', serif;
  font-size: 17px; line-height: 1.65;
}
main { max-width: 1000px; margin: 0 auto; padding: 4rem 1.5rem 6rem; }
h1 { font-size: 1.7rem; font-weight: 600; margin: 0 0 .3rem; letter-spacing: -.01em; }
h2 {
  font-size: .8rem; font-weight: 600; letter-spacing: .12em; text-transform: uppercase;
  font-family: ui-monospace, 'SF Mono', Menlo, monospace; color: var(--dim);
  border-top: 1px solid var(--rule); padding-top: 1.2rem; margin: 3rem 0 1.2rem;
}
.sub { color: var(--dim); font-size: .95rem; margin: 0 0 2rem; }
code, pre, table, .mono {
  font-family: ui-monospace, 'SF Mono', Menlo, monospace;
  font-variant-numeric: tabular-nums;
}
pre {
  font-size: .85rem; line-height: 1.5; overflow-x: auto;
  border-top: 1px solid var(--rule); border-bottom: 1px solid var(--rule);
  padding: 1.2rem 0; margin: 0;
}
table { width: 100%; border-collapse: collapse; font-size: .85rem; }
th {
  text-align: left; font-weight: 600; color: var(--dim); text-transform: uppercase;
  letter-spacing: .08em; font-size: .7rem; padding: .4rem .6rem .4rem 0;
  border-bottom: 1px solid var(--rule);
}
td { padding: .38rem .6rem .38rem 0; border-bottom: 1px solid var(--rule); }
td.num, th.num { text-align: right; padding-right: 1.2rem; }
.pass { color: var(--accent); }
.fail { font-weight: 600; }
.kv { display: grid; grid-template-columns: 14rem 1fr; gap: .1rem 1rem; font-size: .85rem; }
.kv dt { color: var(--dim); }
.kv dd { margin: 0; }
footer {
  border-top: 1px solid var(--rule); margin-top: 4rem; padding-top: 1.2rem;
  font-size: .8rem; color: var(--dim);
  font-family: ui-monospace, 'SF Mono', Menlo, monospace;
}
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
        ("task success", f"{s.get('task_success', 0.0) * 100:.1f}%"),
        ("passed", f"{s.get('passed', 0)} / {s.get('graded', 0)}"),
        ("infra errors", str(s.get("infra_errors", 0))),
        ("timeouts", str(s.get("timeouts", 0))),
        ("steps (median)", f"{s.get('steps_median', 0.0):.1f}"),
        ("cost / task", f"${s.get('cost_per_task', 0.0):.4f}"),
        ("total cost", f"${s.get('total_cost_usd', 0.0):.4f}"),
        ("total tokens", f"{s.get('total_tokens', 0):,}"),
        ("p95 latency", f"{s.get('latency_p95', 0.0):.2f}s"),
        ("tool-error rate", f"{s.get('tool_error_rate', 0.0) * 100:.1f}%"),
        ("tool calls", str(s.get("tool_calls", 0))),
        ("wall time", f"{s.get('wall_time_s', 0.0):.2f}s"),
    ]

    parts: list[str] = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_esc(title)}</title><style>{CSS}</style></head><body><main>",
        f"<h1>{_esc(run.suite)} / {_esc(run.agent_name)} {_esc(run.agent_version)}</h1>",
        f'<p class="sub mono">run {_esc(run.run_id)} &middot; {_ts(run.started_at)}'
        + (f" &middot; model {_esc(run.model)}" if run.model else "")
        + "</p>",
        "<h2>Summary</h2>",
        '<dl class="kv mono">',
    ]
    for label, value in headline:
        parts.append(f"<dt>{_esc(label)}</dt><dd>{_esc(value)}</dd>")
    parts.append("</dl>")

    if cmp is not None:
        parts += [
            f"<h2>Comparison against {_esc(cmp.baseline.agent_version)}</h2>",
            f"<pre>{_esc(render_comparison(cmp))}</pre>",
        ]

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
        "<footer>caliper &middot; 2026</footer>",
        "</main></body></html>",
    ]
    return "\n".join(parts)
