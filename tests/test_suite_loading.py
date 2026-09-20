"""Suite file parsing."""

import pytest

from caliper.suite import SuiteError, filter_tasks, load_suite, parse_suite


def test_defaults_are_merged_into_every_task():
    name, tasks = parse_suite({
        "suite": "s",
        "defaults": {"timeout_s": 99, "grader": "exact_match"},
        "tasks": [{"id": "a"}, {"id": "b", "timeout_s": 5}],
    })
    assert name == "s"
    assert tasks[0].timeout_s == 99 and tasks[1].timeout_s == 5
    assert tasks[0].grader == "exact_match"


def test_task_carries_its_suite_name():
    _, tasks = parse_suite({"suite": "s", "tasks": [{"id": "a"}]})
    assert tasks[0].suite == "s"


def test_duplicate_ids_are_rejected():
    """A silently dropped task is a silently missing regression."""
    with pytest.raises(SuiteError, match="duplicate"):
        parse_suite({"tasks": [{"id": "a"}, {"id": "a"}]})


def test_missing_id_is_rejected():
    with pytest.raises(SuiteError, match="no 'id'"):
        parse_suite({"tasks": [{"input": "x"}]})


def test_empty_task_list_is_rejected():
    with pytest.raises(SuiteError):
        parse_suite({"tasks": []})


def test_unknown_keys_land_in_metadata():
    _, tasks = parse_suite({"tasks": [{"id": "a", "dataset": "orders.csv"}]})
    assert tasks[0].metadata["dataset"] == "orders.csv"


def test_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(SuiteError, match="not found"):
        load_suite(tmp_path / "nope.yaml")


def test_filter_by_tag():
    _, tasks = parse_suite({"tasks": [
        {"id": "a", "tags": ["x"]}, {"id": "b", "tags": ["y"]}]})
    assert [t.id for t in filter_tasks(tasks, tags=["x"])] == ["a"]
    assert [t.id for t in filter_tasks(tasks, tags=["x", "y"])] == ["a", "b"]
    assert filter_tasks(tasks, tags=["z"]) == []


def test_filter_by_limit_preserves_order():
    _, tasks = parse_suite({"tasks": [{"id": c} for c in "abcde"]})
    assert [t.id for t in filter_tasks(tasks, limit=2)] == ["a", "b"]


@pytest.mark.parametrize("path", ["suites/reference.yaml", "suites/strata.yaml",
                                  "suites/quarry.yaml"])
def test_shipped_suites_all_parse(path):
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    name, tasks = load_suite(root / path)
    assert tasks
    assert len({t.id for t in tasks}) == len(tasks)


def test_reference_suite_exercises_every_grader():
    """A grader nothing exercises is a grader nothing tests."""
    from pathlib import Path

    from caliper.graders.registry import normalise_spec

    root = Path(__file__).resolve().parent.parent
    _, tasks = load_suite(root / "suites/reference.yaml")
    used = {name for t in tasks for name, _ in normalise_spec(t.grader)}
    required = {
        "exact_match", "numeric_tolerance", "contains_all", "regex_match",
        "json_schema_match", "step_efficiency", "tool_choice_precision",
        "no_redundant_calls", "recovered_from_error", "cost_budget", "latency_budget",
    }
    assert required <= used


def test_reference_suite_has_at_least_twenty_tasks():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    _, tasks = load_suite(root / "suites/reference.yaml")
    assert len(tasks) >= 20
