"""Command line interface.

    caliper run <suite> --agent <name> [--workers N] [--limit N] [--tag t] [--offline]
    caliper list
    caliper show <run_id>
    caliper compare <run_a> <run_b> [--fail-on-regression]
    caliper trace <run_id> <task_id>
    caliper report <run_id> [--html out.html]
    caliper graders
    caliper prune [--keep N]

Exit codes are part of the interface:
    0  success, no regressions
    1  regressions detected (with --fail-on-regression), or the run had failures
       (with --fail-on-failure)
    2  usage or configuration error

Run ids accept a unique prefix, ``latest``, or ``latest~1`` for the one before.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from caliper.compare import compare_runs
from caliper.graders import registered
from caliper.registry import UnknownAgent, available, load_agent
from caliper.render import (
    render_comparison,
    render_run_list,
    render_run_summary,
    render_trace,
)
from caliper.runner import DEFAULT_RETRIES, DEFAULT_WORKERS, run_suite
from caliper.store import RunStore
from caliper.suite import SuiteError, filter_tasks, load_suite

EXIT_OK = 0
EXIT_REGRESSION = 1
EXIT_USAGE = 2


def _die(message: str, code: int = EXIT_USAGE) -> int:
    print(f"caliper: error: {message}", file=sys.stderr)
    return code


# --- commands -------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    try:
        suite_name, tasks = load_suite(args.suite)
    except SuiteError as exc:
        return _die(str(exc))

    tasks = filter_tasks(tasks, tags=args.tag, limit=args.limit)
    if not tasks:
        return _die("no tasks matched the given --tag / --limit filters")

    try:
        agent = load_agent(args.agent)
    except UnknownAgent as exc:
        return _die(str(exc))
    except Exception as exc:  # noqa: BLE001 - adapters raise their own guidance
        return _die(str(exc))

    overrides: dict[str, dict] = {}
    if args.offline:
        from caliper.llm import OfflineLLMClient

        overrides["llm_judge"] = {"client": OfflineLLMClient()}

    store = RunStore(args.home)
    try:
        run = run_suite(
            agent,
            tasks,
            suite_name=suite_name,
            workers=args.workers,
            retries=args.retries,
            progress=not args.quiet,
            store=store,
            save=not args.no_save,
            grader_overrides=overrides,
        )
    except Exception as exc:  # noqa: BLE001 - setup() failures land here
        return _die(str(exc))

    print()
    print(render_run_summary(run, verbose=args.verbose))
    if not args.no_save:
        print(f"\nsaved: {store.path_for(run.run_id)}")
    if args.fail_on_failure and run.summary.get("failed", 0):
        return EXIT_REGRESSION
    return EXIT_OK


def cmd_list(args: argparse.Namespace) -> int:
    store = RunStore(args.home)
    print(render_run_list(store.list_runs(limit=args.limit)))
    return EXIT_OK


def cmd_show(args: argparse.Namespace) -> int:
    store = RunStore(args.home)
    try:
        run = store.load(args.run_id)
    except FileNotFoundError as exc:
        return _die(str(exc))
    if args.json:
        print(json.dumps(run.to_dict(), indent=2, default=str))
    else:
        print(render_run_summary(run, verbose=not args.summary_only))
    return EXIT_OK


def cmd_compare(args: argparse.Namespace) -> int:
    store = RunStore(args.home)
    try:
        candidate = store.load(args.run_a)
        baseline = store.load(args.run_b)
    except FileNotFoundError as exc:
        return _die(str(exc))

    cmp = compare_runs(candidate, baseline)
    if args.json:
        print(json.dumps(cmp.to_dict(), indent=2, default=str))
    else:
        print(render_comparison(cmp))

    if cmp.regressions and args.fail_on_regression:
        print(
            f"\ncaliper: {len(cmp.regressions)} regression(s) detected; failing the gate",
            file=sys.stderr,
        )
        return EXIT_REGRESSION
    return EXIT_OK


def cmd_trace(args: argparse.Namespace) -> int:
    store = RunStore(args.home)
    try:
        run = store.load(args.run_id)
    except FileNotFoundError as exc:
        return _die(str(exc))
    for result in run.results:
        if result.task.id == args.task_id:
            print(render_trace(result))
            return EXIT_OK
    known = ", ".join(r.task.id for r in run.results[:12])
    return _die(f"no task {args.task_id!r} in run {run.run_id}. Tasks include: {known}")


def cmd_report(args: argparse.Namespace) -> int:
    from caliper.report import render_html

    store = RunStore(args.home)
    try:
        run = store.load(args.run_id)
        baseline = store.load(args.against) if args.against else None
    except FileNotFoundError as exc:
        return _die(str(exc))

    cmp = compare_runs(run, baseline) if baseline else None
    html = render_html(run, cmp)
    if args.html:
        Path(args.html).write_text(html, encoding="utf-8")
        print(f"wrote {args.html}")
    else:
        print(html)
    return EXIT_OK


def cmd_graders(args: argparse.Namespace) -> int:
    from caliper.graders.registry import get_grader

    for name in registered():
        doc = (get_grader(name).__doc__ or "").strip().splitlines()
        print(f"{name.ljust(24)}{doc[0] if doc else ''}")
    return EXIT_OK


def cmd_prune(args: argparse.Namespace) -> int:
    store = RunStore(args.home)
    removed = store.prune(keep=args.keep, older_than_days=args.older_than_days)
    print(f"pruned {len(removed)} run(s); {len(store.list_ids())} remaining")
    return EXIT_OK


# --- parser ---------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="caliper",
        description="Evaluation and observability harness for LLM agents.",
    )
    p.add_argument("--home", default=None, help="run store root (default .caliper)")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run a suite against an agent")
    r.add_argument("suite", help="path to a suite YAML file")
    r.add_argument(
        "--agent", required=True, help=f"agent name ({', '.join(available())}) or module:attr"
    )
    r.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    r.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    r.add_argument("--limit", type=int, default=None, help="run only the first N tasks")
    r.add_argument("--tag", action="append", default=None, help="repeatable tag filter")
    r.add_argument(
        "--offline",
        action="store_true",
        help="use recorded fixtures instead of live LLM calls (no API key needed)",
    )
    r.add_argument("--quiet", action="store_true", help="suppress per-task progress")
    r.add_argument("--verbose", action="store_true", help="per-task breakdown after the run")
    r.add_argument("--no-save", action="store_true")
    r.add_argument("--fail-on-failure", action="store_true", help="exit 1 if any task failed")
    r.set_defaults(func=cmd_run)

    l = sub.add_parser("list", help="list recent runs")
    l.add_argument("--limit", type=int, default=20)
    l.set_defaults(func=cmd_list)

    s = sub.add_parser("show", help="summary and per-task breakdown for one run")
    s.add_argument("run_id")
    s.add_argument("--summary-only", action="store_true")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_show)

    c = sub.add_parser("compare", help="diff two runs and detect regressions")
    c.add_argument("run_a", help="candidate (the new run)")
    c.add_argument("run_b", help="baseline (the run being compared against)")
    c.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="exit 1 when any task went pass -> fail (use this in CI)",
    )
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_compare)

    t = sub.add_parser("trace", help="step-by-step dump of one trajectory")
    t.add_argument("run_id")
    t.add_argument("task_id")
    t.set_defaults(func=cmd_trace)

    rep = sub.add_parser("report", help="standalone HTML report")
    rep.add_argument("run_id")
    rep.add_argument("--html", default=None, help="output path (default: stdout)")
    rep.add_argument("--against", default=None, help="baseline run id for a comparison section")
    rep.set_defaults(func=cmd_report)

    g = sub.add_parser("graders", help="list registered graders")
    g.set_defaults(func=cmd_graders)

    pr = sub.add_parser("prune", help="delete old runs")
    pr.add_argument("--keep", type=int, default=50)
    pr.add_argument("--older-than-days", type=float, default=None)
    pr.set_defaults(func=cmd_prune)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\ncaliper: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
