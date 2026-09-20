"""The reference agent, its tools, and the v1-vs-v2 improvement claim.

The website shows measured numbers from these two versions. These tests pin
down the claim: v2 must genuinely beat v1 on the reference suite, on
correctness AND on trajectory cost.
"""

import pytest
from conftest import make_task

from caliper.reference import tools as T
from caliper.reference.agent import (
    ReferenceAgent,
    classify,
    extract_expression,
    find_kv_key,
)


# --- tools ----------------------------------------------------------------


@pytest.mark.parametrize(
    "expr,expected",
    [("17+25", 42), ("1000-373", 627), ("23*19", 437), ("144/12", 12),
     ("13**2", 169), ("(15/100)*240", 36), ("-5+3", -2)],
)
def test_calculator(expr, expected):
    assert T.calculator(expr) == pytest.approx(expected)


def test_calculator_rejects_division_by_zero():
    with pytest.raises(T.ToolError, match="division by zero"):
        T.calculator("1/0")


def test_calculator_rejects_code_execution():
    """AST-walked, never eval."""
    for hostile in ("__import__('os').system('ls')", "open('/etc/passwd')", "1 if True else 2"):
        with pytest.raises(T.ToolError):
            T.calculator(hostile)


def test_calculator_rejects_empty_and_garbage():
    with pytest.raises(T.ToolError):
        T.calculator("")
    with pytest.raises(T.ToolError):
        T.calculator("+*/")


def test_lookup_hit_and_miss():
    assert T.lookup("capital of france") == "Paris"
    assert T.lookup("Capital Of France") == "Paris"
    with pytest.raises(T.ToolError, match="no entry"):
        T.lookup("capital of atlantis")


def test_string_ops():
    assert T.string_op("reverse", text="abc") == "cba"
    assert T.string_op("upper", text="abc") == "ABC"
    assert T.string_op("word_count", text="a b c") == 3
    with pytest.raises(T.ToolError, match="unknown string op"):
        T.string_op("explode", text="x")


def test_call_tool_unknown_name():
    with pytest.raises(T.ToolError, match="unknown tool"):
        T.call_tool("nope")


# --- routing --------------------------------------------------------------


@pytest.mark.parametrize(
    "question,kind",
    [("What is 17 plus 25?", "arithmetic"),
     ("What is 5 * 3?", "arithmetic"),
     ("Reverse the string 'abc'", "string"),
     ("How many words are in 'a b'?", "string"),
     ("What is the capital of France?", "lookup")],
)
def test_classify(question, kind):
    assert classify(question) == kind


def test_extract_expression_handles_words_and_percentages():
    assert T.calculator(extract_expression("What is 17 plus 25?")) == 42
    assert T.calculator(extract_expression("What is 15 percent of 240?")) == pytest.approx(36)
    assert T.calculator(extract_expression("What is 13 squared?")) == 169


def test_find_kv_key_picks_the_longest_match():
    assert find_kv_key("Reverse the capital of Japan.") == "capital of japan"
    assert find_kv_key("nothing here") is None


# --- agent behaviour ------------------------------------------------------


def test_v2_answers_a_simple_lookup_in_two_steps():
    traj = ReferenceAgent("v2").run(make_task(input="What is the capital of France?"))
    assert traj.final_output == "Paris"
    assert len(traj.steps) == 2


def test_v2_decomposes_a_composite_question():
    traj = ReferenceAgent("v2").run(make_task(input="Reverse the capital of Japan."))
    assert traj.final_output == "oykoT"
    assert traj.tool_names == ["lookup", "string_op"]


def test_v1_fails_the_same_composite_question():
    traj = ReferenceAgent("v1").run(make_task(input="Reverse the capital of Japan."))
    assert traj.final_output != "oykoT"


def test_v1_makes_a_redundant_calculator_call():
    traj = ReferenceAgent("v1").run(make_task(input="What is 17 plus 25?"))
    assert traj.final_output == "42"
    assert traj.tool_names.count("calculator") == 2


def test_both_versions_agree_on_the_answer_but_not_the_cost():
    """The intellectual core: identical final answer, very different trajectory."""
    task = make_task(input="What is 17 plus 25?")
    v1 = ReferenceAgent("v1").run(task)
    v2 = ReferenceAgent("v2").run(task)
    assert v1.final_output == v2.final_output == "42"
    assert len(v1.steps) > len(v2.steps)
    assert v1.total_cost_usd > v2.total_cost_usd


def test_agent_records_tool_errors_rather_than_raising():
    traj = ReferenceAgent("v1").run(make_task(input="What is the capital of Atlantis?"))
    assert traj.errored_steps
    assert traj.terminated_reason in ("completed", "error")


def test_invalid_version_is_rejected():
    with pytest.raises(ValueError):
        ReferenceAgent("v99")


def test_agent_is_stateless_across_tasks():
    """run() may be called concurrently, so nothing may live on self."""
    agent = ReferenceAgent("v2")
    first = agent.run(make_task(input="What is 2 plus 2?"))
    second = agent.run(make_task(input="What is 2 plus 2?"))
    assert first.final_output == second.final_output
    assert len(first.steps) == len(second.steps)


# --- the headline claim ---------------------------------------------------


def test_v2_beats_v1_on_the_reference_suite():
    from pathlib import Path

    from caliper.runner import run_suite
    from caliper.suite import load_suite

    root = Path(__file__).resolve().parent.parent
    _, tasks = load_suite(root / "suites/reference.yaml")

    v1 = run_suite(ReferenceAgent("v1"), tasks, workers=8, progress=False, save=False)
    v2 = run_suite(ReferenceAgent("v2"), tasks, workers=8, progress=False, save=False)

    assert v2.summary["task_success"] > v1.summary["task_success"]
    assert v2.summary["steps_median"] < v1.summary["steps_median"]
    assert v2.summary["cost_per_task"] < v1.summary["cost_per_task"]
    assert v2.summary["tool_error_rate"] < v1.summary["tool_error_rate"]


def test_comparing_v2_to_v1_finds_fixes_and_no_regressions():
    from pathlib import Path

    from caliper.compare import compare_runs
    from caliper.runner import run_suite
    from caliper.suite import load_suite

    root = Path(__file__).resolve().parent.parent
    _, tasks = load_suite(root / "suites/reference.yaml")
    v1 = run_suite(ReferenceAgent("v1"), tasks, workers=8, progress=False, save=False)
    v2 = run_suite(ReferenceAgent("v2"), tasks, workers=8, progress=False, save=False)

    forward = compare_runs(v2, v1)
    assert forward.regressions == []
    assert len(forward.fixes) >= 10

    backward = compare_runs(v1, v2)
    assert len(backward.regressions) >= 10
    assert backward.fixes == []
