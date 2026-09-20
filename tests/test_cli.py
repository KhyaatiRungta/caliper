"""CLI end-to-end. No API key, no network."""

import json

import pytest

from caliper.cli import main


@pytest.fixture
def home(tmp_path):
    return str(tmp_path / ".caliper")


def run_cli(args, home):
    return main(["--home", home] + args)


def do_run(home, agent, extra=None):
    code = run_cli(["run", "suites/reference.yaml", "--agent", agent, "--offline",
                    "--quiet", "--workers", "8"] + (extra or []), home)
    assert code == 0
    return code


def test_run_reference_v1_end_to_end(home, capsys):
    do_run(home, "reference-v1")
    out = capsys.readouterr().out
    assert "task success" in out
    assert "reference-agent v1" in out


def test_run_then_list_show_trace_compare(home, capsys):
    do_run(home, "reference-v1")
    do_run(home, "reference-v2")
    capsys.readouterr()

    assert run_cli(["list"], home) == 0
    assert "RUN ID" in capsys.readouterr().out

    assert run_cli(["show", "latest"], home) == 0
    assert "GRADERS" in capsys.readouterr().out

    assert run_cli(["trace", "latest", "add_two_numbers"], home) == 0
    trace = capsys.readouterr().out
    assert "SCORES" in trace and "calculator" in trace

    assert run_cli(["compare", "latest", "latest~1"], home) == 0
    table = capsys.readouterr().out
    assert "SUITE: reference-agent v2   vs   v1" in table
    assert "task success" in table


def test_compare_exits_non_zero_on_regression(home, capsys):
    """The CI gate. v1 measured against v2 is a regression in every sense."""
    do_run(home, "reference-v2")
    do_run(home, "reference-v1")
    capsys.readouterr()
    code = run_cli(["compare", "latest", "latest~1", "--fail-on-regression"], home)
    assert code == 1
    assert "REGRESSIONS" in capsys.readouterr().out


def test_compare_exits_zero_when_there_are_no_regressions(home, capsys):
    do_run(home, "reference-v1")
    do_run(home, "reference-v2")
    capsys.readouterr()
    assert run_cli(["compare", "latest", "latest~1", "--fail-on-regression"], home) == 0


def test_compare_json_output(home, capsys):
    do_run(home, "reference-v1")
    do_run(home, "reference-v2")
    capsys.readouterr()
    run_cli(["compare", "latest", "latest~1", "--json"], home)
    data = json.loads(capsys.readouterr().out)
    assert data["regressions"] == []
    assert data["fixes"]


def test_all_cli_output_is_plain_ascii(home, capsys):
    do_run(home, "reference-v2")
    do_run(home, "reference-v1")
    capsys.readouterr()
    for args in (["list"], ["show", "latest"], ["compare", "latest", "latest~1"],
                 ["trace", "latest", "capital_of_france"], ["graders"]):
        run_cli(args, home)
        out = capsys.readouterr().out
        assert out.isascii(), f"non-ascii output from {args}"


def test_tag_and_limit_filters(home, capsys):
    do_run(home, "reference-v2", ["--tag", "arithmetic", "--limit", "3"])
    out = capsys.readouterr().out
    assert "(3/3 graded)" in out


def test_no_tasks_matched_is_a_usage_error(home):
    assert run_cli(["run", "suites/reference.yaml", "--agent", "reference-v2",
                    "--tag", "no-such-tag", "--quiet"], home) == 2


def test_unknown_agent_is_a_usage_error(home, capsys):
    assert run_cli(["run", "suites/reference.yaml", "--agent", "nope", "--quiet"], home) == 2
    assert "Built-ins" in capsys.readouterr().err


def test_missing_suite_file_is_a_usage_error(home):
    assert run_cli(["run", "suites/nope.yaml", "--agent", "reference-v2", "--quiet"], home) == 2


def test_unknown_run_id_is_a_usage_error(home, capsys):
    assert run_cli(["show", "nope"], home) == 2
    assert "caliper list" in capsys.readouterr().err


def test_unknown_task_in_trace_lists_known_tasks(home, capsys):
    do_run(home, "reference-v2")
    capsys.readouterr()
    assert run_cli(["trace", "latest", "no_such_task"], home) == 2
    assert "Tasks include" in capsys.readouterr().err


def test_list_on_empty_store_is_not_an_error(home, capsys):
    assert run_cli(["list"], home) == 0
    assert "no runs recorded" in capsys.readouterr().out


def test_graders_command_lists_every_grader(home, capsys):
    assert run_cli(["graders"], home) == 0
    out = capsys.readouterr().out
    for name in ("exact_match", "step_efficiency", "llm_judge", "no_redundant_calls"):
        assert name in out


def test_html_report_is_standalone(home, tmp_path, capsys):
    do_run(home, "reference-v1")
    do_run(home, "reference-v2")
    capsys.readouterr()
    out = tmp_path / "report.html"
    assert run_cli(["report", "latest", "--against", "latest~1", "--html", str(out)], home) == 0
    html = out.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    assert "<script" not in html.lower()
    assert "http://" not in html and "https://" not in html
    assert "SUITE: reference-agent v2" in html


def test_prune(home, capsys):
    do_run(home, "reference-v1")
    do_run(home, "reference-v2")
    capsys.readouterr()
    assert run_cli(["prune", "--keep", "1"], home) == 0
    assert "pruned 1 run" in capsys.readouterr().out


def test_fail_on_failure_flag(home):
    code = run_cli(["run", "suites/reference.yaml", "--agent", "reference-v1",
                    "--offline", "--quiet", "--fail-on-failure"], home)
    assert code == 1


def test_run_id_prefix_resolution(home, capsys):
    do_run(home, "reference-v2")
    out = capsys.readouterr().out
    run_id = [l for l in out.splitlines() if l.startswith("RUN:")][0].split()[1]
    assert run_cli(["show", run_id[:15]], home) == 0
