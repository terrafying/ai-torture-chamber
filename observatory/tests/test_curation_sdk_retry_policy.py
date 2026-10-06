"""SDK retries must not conceal unknown paid-review outcomes."""
import pytest

from observatory.research_llm import researcher_model


@pytest.mark.asyncio
async def test_openai_review_factory_disables_sdk_retry_without_request(monkeypatch):
    import openai
    observed = []
    class Client:
        def __init__(self, **kwargs):
            observed.append(kwargs)
    monkeypatch.setattr(openai, "AsyncOpenAI", Client)
    settings = {"research_provider": "openai", "research_model": "fixture-model",
                "openai_api_key": "fixture-key"}
    researcher_model(settings, sdk_max_retries=0)
    researcher_model(settings)
    assert [item["max_retries"] for item in observed] == [0, 2]


@pytest.mark.asyncio
async def test_anthropic_review_factory_disables_sdk_retry_without_request():
    settings = {"research_provider": "anthropic", "research_model": "fixture-model",
                "anthropic_api_key": "fixture-key", "research_protocol": "messages"}
    review = researcher_model(settings, sdk_max_retries=0)
    ordinary = researcher_model(settings)
    try:
        assert review._get_client_params()["max_retries"] == 0
        assert ordinary._get_client_params()["max_retries"] == 2
    finally:
        await review.http_client.aclose()
        await ordinary.http_client.aclose()
