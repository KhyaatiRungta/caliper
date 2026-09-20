"""Final-answer graders, including the edge cases that bite."""

import pytest
from conftest import make_task, make_trajectory

from caliper.graders import get_grader
from caliper.graders.registry import GraderError, grade, normalise_spec, registered


def run(name, task, traj, **kw):
    return get_grader(name)(task, traj, **kw)


# --- registry -------------------------------------------------------------


def test_every_documented_grader_is_registered():
    expected = {
        "exact_match", "numeric_tolerance", "contains_all", "regex_match",
        "json_schema_match", "llm_judge", "step_efficiency",
        "tool_choice_precision", "no_redundant_calls", "recovered_from_error",
        "cost_budget", "latency_budget",
    }
    assert expected <= set(registered())


def test_unknown_grader_raises_with_a_useful_message():
    with pytest.raises(GraderError) as exc:
        get_grader("no_such_grader")
    assert "Registered graders" in str(exc.value)


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("exact_match", [("exact_match", {})]),
        (["a", "b"], [("a", {}), ("b", {})]),
        ({"numeric_tolerance": {"rel": 0.1}}, [("numeric_tolerance", {"rel": 0.1})]),
        ({"name": "x", "k": 1}, [("x", {"k": 1})]),
        (None, []),
    ],
)
def test_normalise_spec(spec, expected):
    assert normalise_spec(spec) == expected


def test_broken_grader_does_not_abort_the_others():
    """A grader that raises yields a failing score; siblings still run."""
    from caliper.graders.registry import grader

    @grader("_explodes_for_test")
    def _explodes(task, trajectory, **kw):
        raise RuntimeError("grader bug")

    task = make_task(expected="a", grader=["_explodes_for_test", "exact_match"])
    scores = grade(task, make_trajectory(final_output="a"))
    assert len(scores) == 2
    assert not scores[0].passed and "grader error" in scores[0].detail
    assert scores[1].passed


# --- exact_match ----------------------------------------------------------


@pytest.mark.parametrize(
    "expected,output,ok",
    [
        ("Paris", "Paris", True),
        ("Paris", "paris", True),
        ("Paris", "  Paris  ", True),
        ("Paris", "Paris.", True),
        ("Paris", "Lyon", False),
        ("42", 42, True),
        ("", None, True),
    ],
)
def test_exact_match(expected, output, ok):
    score = run("exact_match", make_task(expected=expected), make_trajectory(final_output=output))
    assert score.passed is ok
    assert score.value == (1.0 if ok else 0.0)


def test_exact_match_case_sensitive():
    task = make_task(expected="Paris")
    assert not run("exact_match", task, make_trajectory(final_output="paris"),
                   case_sensitive=True).passed


# --- numeric_tolerance ----------------------------------------------------


@pytest.mark.parametrize(
    "expected,output,kw,ok",
    [
        ("42", "42", {}, True),
        ("42", "The answer is 42.", {}, True),
        ("100", "101", {"rel": 0.02}, True),
        ("100", "101", {"rel": 0.001}, False),
        ("100", "100.4", {"abs": 0.5}, True),
        ("1000", "1,000", {}, True),
        ("3.142857", "3.14", {"rel": 0.001}, True),
        ("3.142857", "3.14", {"rel": 0.0001}, False),
        ("-5", "-5", {}, True),
    ],
)
def test_numeric_tolerance(expected, output, kw, ok):
    score = run("numeric_tolerance", make_task(expected=expected),
                make_trajectory(final_output=output), **kw)
    assert score.passed is ok


def test_numeric_tolerance_zero_expected_does_not_divide_by_zero():
    """Relative tolerance against zero must fall back, not explode."""
    score = run("numeric_tolerance", make_task(expected="0"),
                make_trajectory(final_output="0"), rel=0.01)
    assert score.passed
    score = run("numeric_tolerance", make_task(expected="0"),
                make_trajectory(final_output="0.5"), rel=0.01)
    assert not score.passed


def test_numeric_tolerance_unparseable_output():
    score = run("numeric_tolerance", make_task(expected="42"),
                make_trajectory(final_output="no numbers here"))
    assert not score.passed
    assert "could not parse" in score.detail


def test_numeric_tolerance_ignores_booleans():
    score = run("numeric_tolerance", make_task(expected=1),
                make_trajectory(final_output=True))
    assert not score.passed


# --- contains_all ---------------------------------------------------------


def test_contains_all_partial_credit():
    score = run("contains_all", make_task(), make_trajectory(final_output="alpha beta"),
                terms=["alpha", "beta", "gamma"])
    assert score.value == pytest.approx(2 / 3)
    assert not score.passed
    assert "gamma" in score.detail


def test_contains_all_reads_expected_list():
    score = run("contains_all", make_task(expected=["Sony", "1991"]),
                make_trajectory(final_output="Sony shipped it in 1991"))
    assert score.passed


def test_contains_all_with_no_terms_fails_loudly():
    score = run("contains_all", make_task(expected="x"), make_trajectory(final_output="x"))
    assert not score.passed
    assert "no terms" in score.detail


# --- regex_match ----------------------------------------------------------


def test_regex_match_search_and_fullmatch():
    traj = make_trajectory(final_output="value: 299792458 m/s")
    assert run("regex_match", make_task(), traj, pattern=r"\d{9}").passed
    assert not run("regex_match", make_task(), traj, pattern=r"\d{9}", fullmatch=True).passed


def test_regex_match_invalid_pattern_is_a_failing_score_not_a_crash():
    score = run("regex_match", make_task(), make_trajectory(final_output="x"), pattern="(unclosed")
    assert not score.passed
    assert "bad pattern" in score.detail


# --- json_schema_match ----------------------------------------------------


def test_json_schema_match_accepts_object():
    schema = {"type": "object", "required": ["region", "total"],
              "properties": {"region": {"type": "string"}, "total": {"type": "number"}}}
    traj = make_trajectory(final_output='{"region": "emea", "total": 12.5}')
    assert run("json_schema_match", make_task(), traj, schema=schema).passed


def test_json_schema_match_reports_missing_key():
    schema = {"type": "object", "required": ["region"]}
    score = run("json_schema_match", make_task(),
                make_trajectory(final_output='{"total": 1}'), schema=schema)
    assert not score.passed
    assert "required key missing" in score.detail


def test_json_schema_match_strips_code_fence():
    traj = make_trajectory(final_output='```json\n{"a": 1}\n```')
    assert run("json_schema_match", make_task(), traj,
               schema={"type": "object", "required": ["a"]}).passed


def test_json_schema_match_invalid_json():
    score = run("json_schema_match", make_task(), make_trajectory(final_output="not json"),
                schema={"type": "object"})
    assert not score.passed
    assert "not valid JSON" in score.detail


def test_json_schema_match_validates_array_items():
    schema = {"type": "array", "items": {"type": "number"}}
    assert run("json_schema_match", make_task(), make_trajectory(final_output="[1, 2]"),
               schema=schema).passed
    assert not run("json_schema_match", make_task(), make_trajectory(final_output='[1, "x"]'),
                   schema=schema).passed
