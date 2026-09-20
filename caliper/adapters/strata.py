"""Adapter for Strata, the deep research agent.

Strata lives in its own repository (``../strata``) and knows nothing about
Caliper. This file is the entire coupling between the two: about eighty lines
that translate a Caliper Task into a Strata call and a Strata result back into
a Trajectory. That is the argument for the adapter boundary -- the agent repo
stays clean, and Caliper stays able to measure agents it was not designed for.

The import is lazy and the failure message is actionable, because a missing
sibling repository is the normal case for someone who cloned only Caliper.

Status: provisional. Strata is being written in parallel with Caliper, so this
adapter binds to whichever conventional entry point that repo exposes and
tolerates several shapes of trace. It will be pinned to one once the sibling
API settles. The reference agent, not this adapter, is what the test suite
guarantees.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from caliper.adapter import TrajectoryRecorder
from caliper.types import Task, Trajectory


class SiblingMissing(RuntimeError):
    """Raised when the sibling agent repository is not importable."""


def _candidate_paths(env_var: str, default_dir: str) -> list[Path]:
    here = Path(__file__).resolve().parent.parent.parent
    paths = []
    override = os.environ.get(env_var)
    if override:
        paths.append(Path(override).expanduser())
    paths += [here.parent / default_dir, Path.home() / default_dir]
    return paths


def _import_sibling(module: str, env_var: str, default_dir: str, repo: str):
    """Import a sibling agent module, or raise an actionable SiblingMissing.

    A sibling that exists but fails to import -- a missing dependency, a broken
    virtualenv, a half-written module -- is reported as a distinct failure with
    the underlying cause attached, because "not installed" and "installed but
    broken" want completely different fixes.
    """
    broken: Exception | None = None
    try:
        return __import__(module, fromlist=["*"])
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001 - present but unimportable
        broken = exc
    if broken is None:
        for path in _candidate_paths(env_var, default_dir):
            if (path / default_dir).is_dir() or (path / f"{default_dir}.py").exists():
                sys.path.insert(0, str(path))
                try:
                    return __import__(module, fromlist=["*"])
                except ImportError:
                    sys.path.pop(0)
                except Exception as exc:  # noqa: BLE001
                    sys.path.pop(0)
                    broken = exc
                    break
    if broken is not None:
        raise SiblingMissing(
            f"Found {repo} but could not import {module!r}: "
            f"{type(broken).__name__}: {broken}\n"
            f"The {repo} repository is present but its environment is not usable "
            "from this one.\n"
            "Fix by either:\n"
            f"  pip install -r /path/to/{default_dir}/requirements.txt\n"
            f"  # or run Caliper from {repo}'s own virtualenv\n"
            "Or evaluate the built-in reference agent instead:\n"
            "  caliper run suites/reference.yaml --agent reference-v2 --offline"
        ) from broken
    raise SiblingMissing(
        f"Cannot import {module!r}: the {repo} repository is not on the import path.\n"
        f"Caliper does not vendor it -- {repo} is a separate project.\n"
        "Fix by either:\n"
        f"  git clone <{repo}-repo> ../{default_dir}\n"
        f"  export {env_var}=/path/to/{default_dir}\n"
        f"  pip install -e /path/to/{default_dir}\n"
        "Or evaluate the built-in reference agent instead:\n"
        "  caliper run suites/reference.yaml --agent reference-v2 --offline"
    )


def _construct(factory, kwargs: dict, where: str):
    """Build the sibling agent, turning a signature mismatch into guidance.

    The sibling repos own their own constructors and are still moving. When the
    zero-argument call does not fit, say exactly that rather than surfacing a
    bare TypeError from someone else's code.
    """
    try:
        return factory(**kwargs)
    except TypeError as exc:
        import inspect

        try:
            signature = str(inspect.signature(factory))
        except (TypeError, ValueError):
            signature = "(unavailable)"
        raise SiblingMissing(
            f"Could not construct the {where} agent: {exc}\n"
            f"Its constructor signature is {signature}.\n"
            "This adapter builds the agent with no arguments by default. Pass the "
            "wiring it needs by constructing the adapter yourself:\n"
            f"  from caliper.adapters.{where} import build\n"
            f"  agent = build(retriever=..., client=...)\n"
            "and register it with --agent yourmodule:yourfactory."
        ) from exc


def _resolve_entry_point(module: Any, names: tuple[str, ...], where: str):
    """First attribute of ``module`` named in ``names``, else a clear error."""
    for name in names:
        factory = getattr(module, name, None)
        if factory is not None:
            return factory
    available = ", ".join(n for n in dir(module) if not n.startswith("_"))[:200]
    raise SiblingMissing(
        f"{where} exposes none of {', '.join(names)}, so Caliper does not know how "
        "to construct the agent.\n"
        f"What it does expose: {available}\n"
        "Fix by adding a build() factory to that module, or point Caliper at the "
        "right callable directly:\n"
        f"  caliper run suites/... --agent {where}:YourAgentClass"
    )


def _steps_from(raw: Any) -> list[dict]:
    """Accept several plausible shapes of a sibling's trace.

    Strata is being written in parallel with Caliper, so this deliberately
    tolerates a ``trace``, ``steps`` or ``events`` attribute, and a dataclass,
    dict or object element type. The contract Caliper actually documents is
    ``TrajectoryRecorder``; this is the compatibility shim for a sibling that
    has not adopted it yet.
    """
    for attr in ("trajectory", "trace", "steps", "events"):
        value = getattr(raw, attr, None) if not isinstance(raw, dict) else raw.get(attr)
        if value:
            return [v if isinstance(v, dict) else vars(v) for v in value]
    return []


class StrataAdapter:
    """Runs a Strata research query and records its trace as a Trajectory."""

    name = "strata"

    def __init__(self, version: str = "v1", model: str | None = None, **kwargs: Any):
        self.version = version
        self.model = model or os.environ.get("STRATA_MODEL", "")
        self._kwargs = kwargs
        self._agent = None

    #: Entry points tried in order. Strata is a separate project being written in
    #: parallel, so the adapter accepts any of the conventional names rather than
    #: dictating one to a repo that does not depend on Caliper.
    ENTRY_POINTS = ("build", "StrataAgent", "ResearchAgent", "Agent")

    def setup(self) -> None:
        module = _import_sibling("strata.agent", "STRATA_HOME", "strata", "Strata")
        factory = _resolve_entry_point(module, self.ENTRY_POINTS, "strata.agent")
        self._agent = _construct(factory, self._kwargs, "strata")

    def run(self, task: Task) -> Trajectory:
        if self._agent is None:
            self.setup()
        rec = TrajectoryRecorder(task.id)
        with rec.step("internal", "dispatch", input={"query": task.input}) as s:
            result = self._agent.run(str(task.input))
            s.output = "ok"

        for i, raw in enumerate(_steps_from(result)):
            rec.add(
                raw.get("kind", "llm"),
                str(raw.get("name") or raw.get("tool") or f"step_{i}"),
                input=raw.get("input"),
                output=raw.get("output"),
                tokens_in=int(raw.get("tokens_in") or 0),
                tokens_out=int(raw.get("tokens_out") or 0),
                cost_usd=float(raw.get("cost_usd") or 0.0),
                latency_s=float(raw.get("latency_s") or 0.0),
                error=raw.get("error"),
            )

        answer = getattr(result, "answer", None)
        if answer is None:
            answer = result.get("answer") if isinstance(result, dict) else str(result)
        return rec.finish(final_output=answer)


def build(version: str = "v1", **kwargs: Any) -> StrataAdapter:
    return StrataAdapter(version=version, **kwargs)
