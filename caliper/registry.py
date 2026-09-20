"""Named agent registry for the CLI.

``--agent reference-v2`` resolves here. Anything importable can be registered
at runtime with ``--agent path.to.module:factory``, so the registry is a
convenience for the built-ins rather than a gate.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable

AgentFactory = Callable[..., Any]

_BUILTIN: dict[str, tuple[str, str, dict]] = {
    "reference-v1": ("caliper.reference.agent", "build", {"version": "v1"}),
    "reference-v2": ("caliper.reference.agent", "build", {"version": "v2"}),
    "strata": ("caliper.adapters.strata", "build", {}),
    "quarry": ("caliper.adapters.quarry", "build", {}),
}


class UnknownAgent(ValueError):
    pass


def available() -> list[str]:
    return sorted(_BUILTIN)


def load_agent(spec: str, **kwargs: Any):
    """Resolve ``spec`` to an adapter instance.

    ``spec`` is either a built-in name or ``module:attribute``. The attribute
    may be a factory function or a class; both are called with ``**kwargs``.
    """
    if spec in _BUILTIN:
        module_name, attr, defaults = _BUILTIN[spec]
        merged = {**defaults, **kwargs}
    elif ":" in spec:
        module_name, _, attr = spec.partition(":")
        merged = kwargs
    else:
        raise UnknownAgent(
            f"unknown agent {spec!r}. Built-ins: {', '.join(available())}. "
            "Or give module:attribute for your own adapter."
        )

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise UnknownAgent(f"cannot import {module_name!r} for agent {spec!r}: {exc}") from exc
    factory = getattr(module, attr, None)
    if factory is None:
        raise UnknownAgent(f"{module_name!r} has no attribute {attr!r}")
    return factory(**merged)
