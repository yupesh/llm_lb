import pytest

from llm_lb.adapters import get_adapter
from llm_lb.models import ModelCard


def _card(provider: str, **kw) -> ModelCard:
    base = dict(
        model_id=f"x@{provider}",
        display_name="x",
        provider=provider,
        endpoint_kind="openai_chat" if provider != "hf" else "hf_inference",
    )
    base.update(kw)
    return ModelCard(**base)


def test_openai_adapter_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        get_adapter(_card("openai"))


def test_openai_adapter_constructs_with_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    adapter = get_adapter(_card("openai"))
    assert adapter.served_name == "x"  # @openai suffix stripped
    assert adapter.headers["Authorization"] == "Bearer sk-test"


def test_openai_compat_requires_endpoint_url():
    with pytest.raises(RuntimeError, match="endpoint_url"):
        get_adapter(_card("openai_compat"))


def test_openai_compat_no_key_needed(monkeypatch):
    monkeypatch.delenv("LLM_LB_API_KEY", raising=False)
    adapter = get_adapter(_card("openai_compat", endpoint_url="http://localhost:8000/v1"))
    assert adapter.headers == {}
    assert adapter.base_url == "http://localhost:8000/v1"


def test_hf_requires_hf_uri():
    with pytest.raises(RuntimeError, match="hf_uri"):
        get_adapter(_card("hf"))


def test_hf_constructs(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    adapter = get_adapter(_card("hf", hf_uri="meta-llama/Llama-3-8b-instruct"))
    assert "Llama-3-8b-instruct" in adapter.url
    assert adapter.headers["Authorization"] == "Bearer hf_test"
