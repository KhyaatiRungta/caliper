"""LLM client configuration and offline replay. No network, no key."""

import json

import pytest

from caliper.llm import (
    LLMConfigError,
    OfflineLLMClient,
    Usage,
    _estimate_cost,
    _is_retryable,
    get_api_key,
    load_dotenv,
)


def test_missing_key_error_is_actionable(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("caliper.llm._repo_root", lambda: tmp_path)
    with pytest.raises(LLMConfigError) as exc:
        get_api_key()
    message = str(exc.value)
    assert "OPENROUTER_API_KEY" in message
    assert "--offline" in message


def test_dotenv_is_loaded_without_overriding_the_environment(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text('OPENROUTER_API_KEY="from-file"\n# comment\nOTHER=1\n')
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    load_dotenv(tmp_path / ".env")
    assert get_api_key() == "from-file"

    monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
    load_dotenv(tmp_path / ".env")
    assert get_api_key() == "from-env"


def test_importing_the_module_never_requires_a_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    import importlib

    import caliper.llm as mod

    importlib.reload(mod)  # must not raise


def test_usage_adds():
    total = Usage(1, 2, 0.5, 1.0) + Usage(3, 4, 0.25, 2.0)
    assert (total.prompt_tokens, total.completion_tokens) == (4, 6)
    assert total.cost_usd == 0.75 and total.latency_s == 3.0


def test_cost_estimation_is_positive_and_scales():
    small = _estimate_cost("anthropic/claude-3.5-sonnet", 1000, 100)
    large = _estimate_cost("anthropic/claude-3.5-sonnet", 10000, 1000)
    assert 0 < small < large


def test_unknown_model_still_gets_a_price():
    assert _estimate_cost("some/unknown-model", 1000, 100) > 0


@pytest.mark.parametrize("status,retryable", [(429, True), (500, True), (503, True),
                                              (400, False), (401, False), (404, False)])
def test_retryable_status_codes(status, retryable):
    exc = RuntimeError("x")
    exc.status_code = status
    assert _is_retryable(exc) is retryable


def test_offline_client_replays_a_fixture(tmp_path):
    (tmp_path / "f.json").write_text(json.dumps({
        "grade this": {"text": '{"score": 1.0}', "prompt_tokens": 5, "cost_usd": 0.001}
    }))
    client = OfflineLLMClient(fixtures_dir=tmp_path)
    resp = client.complete([{"role": "user", "content": "grade this"}])
    assert resp.text == '{"score": 1.0}'
    assert resp.usage.cost_usd == 0.001
    assert client.total_usage.prompt_tokens == 5


def test_offline_client_missing_fixture_is_labelled_not_fatal(tmp_path):
    resp = OfflineLLMClient(fixtures_dir=tmp_path).complete(
        [{"role": "user", "content": "unseen"}]
    )
    assert "offline" in resp.text
    assert resp.raw["offline"] is True


def test_offline_client_ignores_a_corrupt_fixture_file(tmp_path):
    (tmp_path / "bad.json").write_text("{not json")
    resp = OfflineLLMClient(fixtures_dir=tmp_path).complete(
        [{"role": "user", "content": "x"}]
    )
    assert "offline" in resp.text


def test_offline_client_keys_on_the_last_user_message():
    key = OfflineLLMClient.key_for([
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
    ])
    assert key == "second"


def test_offline_client_accumulates_usage(tmp_path):
    client = OfflineLLMClient(fixtures_dir=tmp_path)
    for _ in range(3):
        client.complete([{"role": "user", "content": "x"}])
    assert client.total_usage.cost_usd == 0.0
