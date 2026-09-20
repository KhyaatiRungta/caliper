"""LLM judge. Runs entirely against a stub client -- no API key, no network."""

import json

from conftest import make_task, make_trajectory

from caliper.graders.llm_judge import _parse_verdict, llm_judge, load_rubric
from caliper.llm import LLMResponse, OfflineLLMClient, Usage


class StubClient:
    """Returns a scripted sequence of judge replies."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.total_usage = Usage()

    def complete(self, messages, **kwargs):
        self.calls += 1
        text = self.replies[min(self.calls - 1, len(self.replies) - 1)]
        return LLMResponse(text=text, usage=Usage(), model="stub")


class ExplodingClient:
    total_usage = Usage()

    def complete(self, messages, **kwargs):
        raise RuntimeError("provider is down")


def verdict(score, reasoning="because"):
    return json.dumps({"score": score, "reasoning": reasoning})


def test_rubric_file_loads_and_fixes_a_five_point_scale():
    rubric = load_rubric()
    assert "0.75" in rubric and "0.25" in rubric
    assert "JSON" in rubric


def test_judge_passes_above_threshold():
    score = llm_judge(make_task(expected="Paris"), make_trajectory(final_output="Paris"),
                      client=StubClient([verdict(1.0)]))
    assert score.value == 1.0 and score.passed


def test_judge_fails_below_threshold():
    score = llm_judge(make_task(), make_trajectory(), client=StubClient([verdict(0.5)]))
    assert score.value == 0.5 and not score.passed


def test_judge_snaps_to_the_nearest_bucket():
    score = llm_judge(make_task(), make_trajectory(), client=StubClient([verdict(0.71)]))
    assert score.value == 0.75


def test_judge_clamps_out_of_range_scores():
    assert llm_judge(make_task(), make_trajectory(),
                     client=StubClient([verdict(7)])).value == 1.0
    assert llm_judge(make_task(), make_trajectory(),
                     client=StubClient([verdict(-3)])).value == 0.0


def test_judge_parses_a_fenced_reply():
    score = llm_judge(make_task(), make_trajectory(),
                      client=StubClient(['```json\n{"score": 1.0, "reasoning": "ok"}\n```']))
    assert score.value == 1.0


def test_judge_parses_json_embedded_in_prose():
    score = llm_judge(make_task(), make_trajectory(),
                      client=StubClient(['Here is my verdict: {"score": 0.25} Thanks.']))
    assert score.value == 0.25


def test_judge_unparseable_reply_fails_loudly():
    score = llm_judge(make_task(), make_trajectory(), client=StubClient(["no idea"]))
    assert not score.passed
    assert "could not parse" in score.detail


def test_judge_provider_failure_is_a_failing_score_not_a_crash():
    score = llm_judge(make_task(), make_trajectory(), client=ExplodingClient())
    assert not score.passed
    assert "judge call failed" in score.detail


def test_majority_vote_over_n_samples():
    client = StubClient([verdict(1.0), verdict(0.5), verdict(1.0)])
    score = llm_judge(make_task(), make_trajectory(), client=client, n_samples=3)
    assert client.calls == 3
    assert score.value == 1.0
    assert "judge_agreement 67%" in score.detail


def test_ties_break_towards_the_lower_score():
    """An optimistic tiebreak on a noisy instrument is how suites drift up."""
    client = StubClient([verdict(1.0), verdict(0.5)])
    score = llm_judge(make_task(), make_trajectory(), client=client, n_samples=2)
    assert score.value == 0.5


def test_agreement_is_reported_for_multi_sample_runs():
    client = StubClient([verdict(0.75)] * 3)
    score = llm_judge(make_task(), make_trajectory(), client=client, n_samples=3)
    assert "judge_agreement 100%" in score.detail


def test_prompt_carries_task_reference_and_response():
    class Capturing(StubClient):
        def complete(self, messages, **kwargs):
            self.seen = messages
            return super().complete(messages, **kwargs)

    client = Capturing([verdict(1.0)])
    llm_judge(make_task(input="Q?", expected="A"), make_trajectory(final_output="R"),
              client=client, criteria="must cite")
    user = client.seen[1]["content"]
    assert "Q?" in user and "A" in user and "R" in user and "must cite" in user


def test_offline_client_returns_a_labelled_miss_rather_than_crashing():
    score = llm_judge(make_task(), make_trajectory(), client=OfflineLLMClient())
    assert not score.passed
    assert "offline" in score.detail


def test_parse_verdict_rejects_non_numeric_score():
    value, reason = _parse_verdict('{"score": "high"}')
    assert value is None and "non-numeric" in reason
