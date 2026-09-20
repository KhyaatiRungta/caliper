"""Adapter for Quarry, the data analyst agent.

Quarry lives in its own repository (``../quarry``) and knows nothing about
Caliper. This file is the entire coupling between the two: about eighty lines
that translate a Caliper Task into a Quarry call and a Quarry result back into
a Trajectory. That is the argument for the adapter boundary -- the agent repo
stays clean, and Caliper stays able to measure agents it was not designed for.

The import is lazy and the failure message is actionable, because a missing
sibling repository is the normal case for someone who cloned only Caliper.

Status: provisional, for the same reason as the Strata adapter. See that module.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from caliper.adapter import TrajectoryRecorder
from caliper.types import Task, Trajectory


from caliper.adapters.strata import (  # noqa: F401
    SiblingMissing,
    _construct,
    _import_sibling,
    _resolve_entry_point,
)


def _steps_from(raw: Any) -> list[dict]:
    """Accept several plausible shapes of a sibling's trace.

    Quarry is being written in parallel with Caliper, so this deliberately
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


class QuarryAdapter:
    """Runs a Quarry analysis task and records its tool calls as a Trajectory."""

    name = "quarry"

    def __init__(self, version: str = "v1", model: str | None = None, **kwargs: Any):
        self.version = version
        self.model = model or os.environ.get("QUARRY_MODEL", "")
        self._kwargs = kwargs
        self._agent = None

    #: Entry points tried in order. See the note in the Strata adapter.
    ENTRY_POINTS = ("build", "QuarryAgent", "AnalystAgent", "Agent")

    def setup(self) -> None:
        module = _import_sibling("quarry.agent", "QUARRY_HOME", "quarry", "Quarry")
        factory = _resolve_entry_point(module, self.ENTRY_POINTS, "quarry.agent")
        self._agent = _construct(factory, self._kwargs, "quarry")

    def run(self, task: Task) -> Trajectory:
        if self._agent is None:
            self.setup()
        rec = TrajectoryRecorder(task.id)
        with rec.step("internal", "dispatch", input={"question": task.input}) as s:
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


def build(version: str = "v1", **kwargs: Any) -> QuarryAdapter:
    return QuarryAdapter(version=version, **kwargs)
