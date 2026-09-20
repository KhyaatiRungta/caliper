"""Rubric-based LLM judge.

A note on judge variance, stated up front because it is the honest framing
------------------------------------------------------------------------

An LLM judge is a measuring instrument with real noise. Re-grading the same
response can land in a different bucket, particularly near the boundary between
"correct with a minor flaw" and "partially correct". Three mitigations are
applied here and none of them make the noise zero:

1. Temperature is pinned at 0.0.
2. The rubric (``caliper/prompts/judge_rubric.md``) fixes a five-point scale
   with named anchors, so the judge chooses a bucket rather than a number.
3. ``n_samples > 1`` grades repeatedly and takes the majority bucket, with the
   spread reported in ``Score.detail``.

Read ``judge_agreement`` in the detail string before trusting a single
judge-scored regression. When a comparison hinges entirely on llm_judge deltas
of a few points, it hinges on noise. The deterministic graders do not have this
problem and should carry the weight wherever a task admits one.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from caliper.graders.registry import grader
from caliper.types import Score, Task, Trajectory

RUBRIC_PATH = Path(__file__).resolve().parent.parent / "prompts" / "judge_rubric.md"

VALID_BUCKETS = (0.0, 0.25, 0.5, 0.75, 1.0)


@lru_cache(maxsize=1)
def load_rubric() -> str:
    return RUBRIC_PATH.read_text(encoding="utf-8")


def _snap(value: float) -> float:
    return min(VALID_BUCKETS, key=lambda b: abs(b - value))


def _parse_verdict(text: str) -> tuple[float | None, str]:
    """Extract ``{"score": ..., "reasoning": ...}`` from a judge reply."""
    candidate = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", candidate, re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()
    obj: Any = None
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        brace = re.search(r"\{.*\}", candidate, re.DOTALL)
        if brace:
            try:
                obj = json.loads(brace.group(0))
            except json.JSONDecodeError:
                obj = None
    if not isinstance(obj, dict) or "score" not in obj:
        return None, f"could not parse judge reply: {text[:120]!r}"
    try:
        score = float(obj["score"])
    except (TypeError, ValueError):
        return None, f"non-numeric score in judge reply: {obj.get('score')!r}"
    return _snap(max(0.0, min(1.0, score))), str(obj.get("reasoning", "")).strip()


def _build_prompt(task: Task, trajectory: Trajectory, criteria: str) -> list[dict]:
    reference = task.expected
    if not isinstance(reference, str):
        reference = json.dumps(reference, default=str) if reference is not None else "(none)"
    response = trajectory.final_output
    if not isinstance(response, str):
        response = json.dumps(response, default=str) if response is not None else ""
    user = (
        f"TASK:\n{task.input}\n\n"
        f"REFERENCE:\n{reference}\n\n"
        f"RESPONSE:\n{response}\n\n"
        f"CRITERIA:\n{criteria or '(none beyond the rubric)'}"
    )
    return [
        {"role": "system", "content": load_rubric()},
        {"role": "user", "content": user},
    ]


@grader("llm_judge")
def llm_judge(
    task: Task,
    trajectory: Trajectory,
    criteria: str = "",
    threshold: float = 0.75,
    n_samples: int = 1,
    client: Any = None,
    model: str | None = None,
) -> Score:
    """Grade the final output against the rubric via OpenRouter.

    ``n_samples`` > 1 takes the majority bucket across independent gradings and
    reports agreement. Ties break towards the LOWER score, because an optimistic
    tiebreak on a noisy instrument is how eval suites quietly drift upward.

    ``client`` may be injected -- pass an ``OfflineLLMClient`` in tests and in
    ``--offline`` runs so no API key is needed.
    """
    if client is None:
        from caliper.llm import LLMClient

        client = LLMClient(model=model, temperature=0.0)

    messages = _build_prompt(task, trajectory, criteria)
    n = max(1, int(n_samples))
    values: list[float] = []
    reasons: list[str] = []
    for _ in range(n):
        try:
            resp = client.complete(messages, max_tokens=300)
        except Exception as exc:  # noqa: BLE001
            return Score(
                name="llm_judge",
                value=0.0,
                unit="ratio",
                passed=False,
                detail=f"judge call failed: {type(exc).__name__}: {exc}",
            )
        value, reasoning = _parse_verdict(resp.text)
        if value is None:
            reasons.append(reasoning)
            continue
        values.append(value)
        reasons.append(reasoning)

    if not values:
        return Score(
            name="llm_judge",
            value=0.0,
            unit="ratio",
            passed=False,
            detail="; ".join(reasons[:2]) or "judge returned no parseable verdict",
        )

    counts = Counter(values)
    top = max(counts.values())
    # Tie-break towards the lower bucket.
    value = min(v for v, c in counts.items() if c == top)
    agreement = top / len(values)

    detail = reasons[0] if reasons else ""
    if n > 1:
        spread = ", ".join(f"{v:.2f}x{c}" for v, c in sorted(counts.items()))
        detail = (
            f"{detail} [judge_agreement {agreement:.0%} over {len(values)} samples; {spread}]"
        )
    return Score(
        name="llm_judge",
        value=value,
        unit="ratio",
        passed=value >= threshold,
        detail=detail.strip(),
    )
