"""Sibling adapters degrade with a clear message rather than a traceback.

Strata and Quarry are separate repositories. Someone who cloned only Caliper
must get instructions, not an ImportError stack.
"""

import pytest

from caliper.adapters.strata import SiblingMissing, StrataAdapter
from caliper.adapters.quarry import QuarryAdapter
from caliper.registry import UnknownAgent, available, load_agent


@pytest.mark.parametrize("Adapter,repo", [(StrataAdapter, "Strata"), (QuarryAdapter, "Quarry")])
def test_missing_sibling_raises_actionable_error(Adapter, repo, monkeypatch, tmp_path):
    monkeypatch.setenv(f"{repo.upper()}_HOME", str(tmp_path / "nowhere"))
    monkeypatch.setattr(
        "caliper.adapters.strata._candidate_paths", lambda *a, **k: [tmp_path / "nowhere"]
    )
    import sys

    for mod in list(sys.modules):
        if mod.startswith(repo.lower()):
            monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setattr(
        "builtins.__import__",
        _blocking_import(repo.lower()),
    )
    with pytest.raises(SiblingMissing) as exc:
        Adapter().setup()
    message = str(exc.value)
    assert "reference-v2" in message, "must offer a working fallback"
    assert any(word in message for word in ("clone", "export", "pip install"))


def _blocking_import(prefix):
    import builtins

    real = builtins.__import__

    def fake(name, *args, **kwargs):
        if name.startswith(prefix):
            raise ImportError(f"No module named {name!r}")
        return real(name, *args, **kwargs)

    return fake


def test_adapters_are_importable_without_the_siblings():
    """Importing Caliper must never require a sibling repo."""
    assert StrataAdapter().name == "strata"
    assert QuarryAdapter().name == "quarry"


def test_adapter_construction_does_not_import_anything():
    """Lazy: the sibling is imported in setup(), not in __init__."""
    adapter = StrataAdapter(version="v3")
    assert adapter.version == "v3"
    assert adapter._agent is None


def test_registry_lists_the_builtins():
    assert {"reference-v1", "reference-v2", "strata", "quarry"} <= set(available())


def test_registry_loads_a_builtin():
    agent = load_agent("reference-v2")
    assert agent.name == "reference-agent" and agent.version == "v2"


def test_registry_loads_a_dotted_path():
    agent = load_agent("caliper.reference.agent:build", version="v1")
    assert agent.version == "v1"


def test_registry_rejects_an_unknown_name():
    with pytest.raises(UnknownAgent, match="Built-ins"):
        load_agent("not-an-agent")


def test_registry_rejects_a_bad_module_path():
    with pytest.raises(UnknownAgent):
        load_agent("caliper.nope:build")


def test_registry_rejects_a_missing_attribute():
    with pytest.raises(UnknownAgent, match="no attribute"):
        load_agent("caliper.reference.agent:no_such_factory")
