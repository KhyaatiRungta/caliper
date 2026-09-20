"""Run store round-trip, resolution and pruning."""

import json
import time

import pytest
from conftest import make_run

from caliper.store import RunStore, new_run_id


def test_round_trip_preserves_everything(store):
    run = make_run("r1", "v1", {"a": True, "b": False})
    store.save(run)
    back = store.load("r1")
    assert back.to_dict() == run.to_dict()


def test_round_trip_preserves_step_detail(store):
    from caliper.types import Step

    run = make_run("r1", "v1", {"a": True})
    run.results[0].trajectory.steps.append(
        Step(9, "tool", "calc", input={"x": 1}, output=2, error="boom")
    )
    store.save(run)
    step = store.load("r1").results[0].trajectory.steps[-1]
    assert step.input == {"x": 1} and step.error == "boom" and step.kind == "tool"


def test_save_is_atomic_leaving_no_temp_files(store):
    store.save(make_run("r1", "v1", {"a": True}))
    assert list(store.runs_dir.glob("*.tmp")) == []


def test_missing_run_raises_with_guidance(store):
    with pytest.raises(FileNotFoundError) as exc:
        store.load("nope")
    assert "caliper list" in str(exc.value)


def test_list_ids_is_newest_first(store):
    for rid in ("r1", "r2", "r3"):
        store.save(make_run(rid, "v1", {"a": True}))
        time.sleep(0.01)
    assert store.list_ids() == ["r3", "r2", "r1"]


def test_resolve_latest_and_offset(store):
    for rid in ("r1", "r2"):
        store.save(make_run(rid, "v1", {"a": True}))
        time.sleep(0.01)
    assert store.resolve("latest") == "r2"
    assert store.resolve("latest~1") == "r1"
    assert store.load("latest").run_id == "r2"


def test_resolve_unique_prefix(store):
    store.save(make_run("20260101-abcd", "v1", {"a": True}))
    assert store.resolve("20260101") == "20260101-abcd"


def test_ambiguous_prefix_resolves_to_nothing(store):
    store.save(make_run("aa1", "v1", {"a": True}))
    store.save(make_run("aa2", "v1", {"a": True}))
    assert store.resolve("aa") is None


def test_corrupt_run_file_does_not_break_listing(store):
    store.save(make_run("good", "v1", {"a": True}))
    (store.runs_dir / "corrupt.json").write_text("{not json", encoding="utf-8")
    ids = store.list_ids()
    assert "corrupt" in ids
    assert [r.run_id for r in store.list_runs()] == ["good"]


def test_prune_keeps_the_newest_n(store):
    for i in range(5):
        store.save(make_run(f"r{i}", "v1", {"a": True}))
        time.sleep(0.01)
    removed = store.prune(keep=2)
    assert len(removed) == 3
    assert store.list_ids() == ["r4", "r3"]


def test_prune_by_age(store):
    store.save(make_run("old", "v1", {"a": True}))
    path = store.path_for("old")
    import os

    old = time.time() - 10 * 86400
    os.utime(path, (old, old))
    store.save(make_run("new", "v1", {"a": True}))
    removed = store.prune(keep=100, older_than_days=5)
    assert removed == ["old"]


def test_list_on_a_fresh_store_is_empty(tmp_path):
    assert RunStore(tmp_path / "nothing").list_runs() == []


def test_run_ids_are_unique_and_sortable():
    ids = [new_run_id("suite", "agent") for _ in range(50)]
    assert len(set(ids)) == 50
    assert all(i.startswith("20") for i in ids)


def test_run_id_slug_is_filesystem_safe():
    rid = new_run_id("my suite/x", "agent name!")
    assert "/" not in rid and " " not in rid and "!" not in rid


def test_saved_file_is_readable_json(store):
    path = store.save(make_run("r1", "v1", {"a": True}))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["run_id"] == "r1"
    assert isinstance(data["results"], list)


def test_home_can_be_set_by_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("CALIPER_HOME", str(tmp_path / "custom"))
    s = RunStore()
    s.save(make_run("r1", "v1", {"a": True}))
    assert (tmp_path / "custom" / "runs" / "r1.json").exists()
