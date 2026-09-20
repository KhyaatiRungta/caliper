"""Round-trip and derivation behaviour of the core data model."""

from conftest import make_step, make_task, make_trajectory

from caliper.types import RunResult, Score, Step, SuiteRun, Task, Trajectory


def test_task_round_trip():
    task = make_task("x", input="q", expected="a", tags=["t"], reference_steps=3)
    assert Task.from_dict(task.to_dict()) == task


def test_from_dict_ignores_unknown_keys():
    """Forward compatibility: a newer run file must still load."""
    task = Task.from_dict({"id": "x", "invented_field_from_the_future": 1})
    assert task.id == "x"


def test_step_ok_property():
    assert make_step().ok
    assert not make_step(error="boom").ok


def test_trajectory_recompute_sums_steps():
    traj = make_trajectory(
        "t",
        [
            make_step(0, tokens_in=10, tokens_out=5, cost_usd=0.01),
            make_step(1, tokens_in=20, tokens_out=1, cost_usd=0.02),
        ],
    )
    assert traj.total_tokens == 36
    assert traj.total_cost_usd == 0.03


def test_trajectory_recompute_is_idempotent():
    traj = make_trajectory("t", [make_step(0, cost_usd=0.5)])
    traj.recompute().recompute()
    assert traj.total_cost_usd == 0.5


def test_trajectory_accessors():
    traj = make_trajectory(
        "t",
        [
            make_step(0, kind="llm", name="plan"),
            make_step(1, kind="tool", name="calc"),
            make_step(2, kind="tool", name="calc", error="bad"),
        ],
    )
    assert traj.tool_names == ["calc", "calc"]
    assert len(traj.errored_steps) == 1
    assert len(traj.steps_of("llm")) == 1


def test_trajectory_round_trip_preserves_steps():
    traj = make_trajectory("t", [make_step(0, error="e"), make_step(1)], final_output="x")
    back = Trajectory.from_dict(traj.to_dict())
    assert back.steps[0].error == "e"
    assert back.final_output == "x"
    assert isinstance(back.steps[0], Step)


def test_run_result_passed_requires_all_scores():
    result = RunResult(
        task=make_task(),
        trajectory=make_trajectory(),
        scores=[Score("a", 1.0, passed=True), Score("b", 0.0, passed=False)],
    )
    assert not result.recompute_passed().passed


def test_run_result_with_no_scores_does_not_pass():
    """A task nothing graded is not a task that passed."""
    result = RunResult(task=make_task(), trajectory=make_trajectory(), scores=[])
    assert not result.recompute_passed().passed


def test_infra_error_forces_fail():
    result = RunResult(
        task=make_task(),
        trajectory=make_trajectory(),
        scores=[Score("a", 1.0, passed=True)],
        infra_error="429",
    )
    assert not result.recompute_passed().passed


def test_suite_run_round_trip():
    run = SuiteRun(
        run_id="r1",
        suite="s",
        agent_name="a",
        agent_version="v1",
        results=[
            RunResult(task=make_task("t"), trajectory=make_trajectory("t"), passed=True)
        ],
        summary={"task_success": 1.0},
    )
    back = SuiteRun.from_dict(run.to_dict())
    assert back.run_id == "r1"
    assert back.results[0].task.id == "t"
    assert back.summary["task_success"] == 1.0
