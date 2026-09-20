"""Regression detection and the comparison table geometry."""

import pytest
from conftest import make_run

from caliper.compare import compare_runs, has_regressions
from caliper.render import RULE_W, metric_row, render_comparison
from caliper.compare import MetricDelta


# --- transitions ----------------------------------------------------------


def test_detects_regression():
    after = make_run("r2", "v2", {"a": False})
    before = make_run("r1", "v1", {"a": True})
    cmp = compare_runs(after, before)
    assert [d.task_id for d in cmp.regressions] == ["a"]
    assert cmp.task_deltas[0].transition == "pass->fail"
    assert has_regressions(cmp)


def test_detects_fix_in_the_other_direction():
    cmp = compare_runs(make_run("r2", "v2", {"a": True}), make_run("r1", "v1", {"a": False}))
    assert [d.task_id for d in cmp.fixes] == ["a"]
    assert cmp.task_deltas[0].transition == "fail->pass"
    assert not has_regressions(cmp)


def test_a_fix_is_never_counted_as_a_regression():
    cmp = compare_runs(make_run("r2", "v2", {"a": True, "b": True}),
                       make_run("r1", "v1", {"a": False, "b": False}))
    assert cmp.regressions == []
    assert len(cmp.fixes) == 2


def test_all_four_transitions():
    after = make_run("r2", "v2", {"reg": False, "fix": True, "good": True, "bad": False})
    before = make_run("r1", "v1", {"reg": True, "fix": False, "good": True, "bad": False})
    cmp = compare_runs(after, before)
    got = {d.task_id: d.transition for d in cmp.task_deltas}
    assert got == {
        "reg": "pass->fail",
        "fix": "fail->pass",
        "good": "pass->pass",
        "bad": "fail->fail",
    }
    assert len(cmp.unchanged) == 2


def test_infra_errors_are_not_regressions():
    """A rate limit is not a regression, and must not block a merge."""
    from conftest import make_result
    from caliper.runner import summarise

    after = make_run("r2", "v2", {"a": True})
    after.results[0] = make_result("a", False, infra_error="429 rate limited")
    after.summary = summarise(after)
    cmp = compare_runs(after, make_run("r1", "v1", {"a": True}))
    assert cmp.regressions == []
    assert cmp.task_deltas[0].transition == "pass->infra"


def test_new_and_removed_tasks_are_not_regressions():
    cmp = compare_runs(make_run("r2", "v2", {"a": True, "new": False}),
                       make_run("r1", "v1", {"a": True, "gone": True}))
    got = {d.task_id: d.transition for d in cmp.task_deltas}
    assert got["new"] == "absent->fail"
    assert got["gone"] == "pass->absent"
    assert cmp.regressions == []


def test_task_delta_ordering_follows_the_candidate_run():
    after = make_run("r2", "v2", {"z": True, "a": True})
    cmp = compare_runs(after, make_run("r1", "v1", {"a": True, "z": True}))
    assert [d.task_id for d in cmp.task_deltas] == ["z", "a"]


def test_mismatched_suites_warn():
    after = make_run("r2", "v2", {"a": True})
    after.suite = "other"
    cmp = compare_runs(after, make_run("r1", "v1", {"a": True}))
    assert any("suites differ" in w for w in cmp.warnings)


def test_mismatched_agents_warn():
    after = make_run("r2", "v2", {"a": True})
    after.agent_name = "different-agent"
    cmp = compare_runs(after, make_run("r1", "v1", {"a": True}))
    assert any("agents differ" in w for w in cmp.warnings)


def test_metric_deltas_are_computed():
    after = make_run("r2", "v2", {"a": True, "b": True}, steps=2, cost=0.02, latency=1.0)
    before = make_run("r1", "v1", {"a": True, "b": False}, steps=4, cost=0.04, latency=2.0)
    cmp = compare_runs(after, before)
    by_label = {m.label: m for m in cmp.metrics}
    assert by_label["task success"].delta == pytest.approx(0.5)
    assert by_label["steps (median)"].delta == -2.0
    assert by_label["cost / task"].relative == pytest.approx(-0.5)
    assert by_label["steps (median)"].improved is True


def test_relative_delta_against_zero_baseline_is_none_not_a_crash():
    m = MetricDelta("x", 1.0, 0.0, "usd", True, "pct")
    assert m.relative is None
    assert "n/a" in metric_row(m)


def test_comparison_to_dict_is_json_safe():
    import json

    cmp = compare_runs(make_run("r2", "v2", {"a": False}), make_run("r1", "v1", {"a": True}))
    assert json.loads(json.dumps(cmp.to_dict()))["regressions"] == ["a"]


# --- table geometry (golden) ---------------------------------------------

GOLDEN = """\
SUITE: analyst-agent v3   vs   v2
-------------------------------------------
task success      78.0%      64.0%   +14.0
steps (median)      4.0        6.0    -2.0
cost / task      $0.031     $0.048    -35%
p95 latency       12.4s      19.1s    -35%
tool-error rate    9.2%      22.5%   -13.3"""


def test_metric_rows_match_the_golden_table():
    rows = [
        MetricDelta("task success", 0.78, 0.64, "pct", False, "points"),
        MetricDelta("steps (median)", 4.0, 6.0, "num", True, "abs"),
        MetricDelta("cost / task", 0.031, 0.048, "usd", True, "pct"),
        MetricDelta("p95 latency", 12.4, 19.1, "secs", True, "pct"),
        MetricDelta("tool-error rate", 0.092, 0.225, "pct", True, "points"),
    ]
    rendered = "\n".join(
        ["SUITE: analyst-agent v3   vs   v2", "-" * RULE_W] + [metric_row(r) for r in rows]
    )
    assert rendered == GOLDEN


def test_every_metric_row_has_the_same_width():
    rows = [
        MetricDelta("task success", 0.78, 0.64, "pct", False, "points"),
        MetricDelta("steps (median)", 4.0, 6.0, "num", True, "abs"),
        MetricDelta("cost / task", 0.031, 0.048, "usd", True, "pct"),
        MetricDelta("tool-error rate", 0.092, 0.225, "pct", True, "points"),
    ]
    widths = {len(metric_row(r)) for r in rows}
    assert len(widths) == 1


def test_columns_line_up_in_a_real_comparison():
    after = make_run("r2", "v2", {"a": True, "b": True})
    before = make_run("r1", "v1", {"a": True, "b": False})
    lines = render_comparison(compare_runs(after, before)).splitlines()
    metric_lines = lines[2:7]
    assert len({len(l) for l in metric_lines}) == 1
    # Values right-aligned: last character of each column is never a space.
    for line in metric_lines:
        assert line[22] != " " or line[23] != " "


def test_regression_block_alignment():
    after = make_run("r2", "v2", {"join_three_tables": False, "null_handling_edge": False})
    before = make_run("r1", "v1", {"join_three_tables": True, "null_handling_edge": True})
    text = render_comparison(compare_runs(after, before))
    assert "REGRESSIONS (2)" in text
    assert "  join_three_tables      pass -> fail" in text
    assert "  null_handling_edge     pass -> fail" in text


def test_no_regressions_renders_none():
    text = render_comparison(compare_runs(make_run("r2", "v2", {"a": True}),
                                          make_run("r1", "v1", {"a": True})))
    assert "REGRESSIONS (0)" in text and "none" in text


def test_table_is_pure_ascii():
    after = make_run("r2", "v2", {"a": False})
    text = render_comparison(compare_runs(after, make_run("r1", "v1", {"a": True})))
    assert text.isascii()
    assert not any(ch in text for ch in "─│┌")
