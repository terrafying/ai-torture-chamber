"""Responses-native Browser Use adapter; imports paid SDKs only when configured."""
from __future__ import annotations

from typing import Any

from .x402_client import ResearchPermanentError, X402BrokerClient, classify_failure


def validate_research_settings(settings: dict, *, require_config: bool = False) -> dict:
    """Pure research validation; partially configured setup can still be saved."""
    import os
    from .settings import validate_research_updates
    try:
        result = validate_research_updates(settings)
    except ValueError as exc:
        raise ResearchPermanentError("INVALID_RESEARCH_SETTINGS", str(exc)) from None
    provider = result.get("research_provider", result.get("agent_provider", "x402"))
    if not isinstance(provider, str) or provider not in {"x402", "openai", "anthropic"}:
        raise ResearchPermanentError("INVALID_PROVIDER", "Choose x402, OpenAI or Claude as the research provider.")
    result["research_provider"] = provider
    for key, default, lower, upper in (("agent_count", 6, 1, 6), ("steps_per_pass", 25, 5, 50)):
        value = result.get(key, default)
        try:
            integer = int(value)
        except (ValueError, TypeError, OverflowError):
            raise ResearchPermanentError("INVALID_" + key.upper(), f"{key} must be an integer between {lower} and {upper}.") from None
        if isinstance(value, bool) or str(value) != str(integer) or not lower <= integer <= upper:
            raise ResearchPermanentError("INVALID_" + key.upper(), f"{key} must be an integer between {lower} and {upper}.")
        result[key] = integer
    protocol = result.get("research_protocol", "responses")
    if not isinstance(protocol, str) or protocol not in {"responses", "messages"}:
        raise ResearchPermanentError("INVALID_PROTOCOL", "Choose responses or messages as the native research protocol.")
    result["research_protocol"] = protocol
    if result.get("reasoning_effort", "high") not in {"low", "medium", "high", "xhigh"}:
        raise ResearchPermanentError("INVALID_REASONING_EFFORT", "Choose a supported research reasoning effort.")
    model = result.get("research_model", result.get("agent_model", ""))
    if model is not None and (not isinstance(model, str) or len(model) > 256 or any(char.isspace() for char in model)):
        raise ResearchPermanentError("INVALID_MODEL", "Research model must be one explicit model identifier.")
    if require_config and provider == "x402" and not model:
        raise ResearchPermanentError("MODEL_REQUIRED", "Select an explicit model from the payment gateway catalog.")
    browser = result.get("browser_provider", "local")
    if not isinstance(browser, str) or browser not in {"local", "browseruse", "cdp"}:
        raise ResearchPermanentError("INVALID_BROWSER_PROVIDER", "Choose local Chromium, Browser Use Cloud or custom CDP.")
    result["browser_provider"] = browser
    if require_config:
        if browser == "browseruse" and not (result.get("browser_use_api_key") or os.environ.get("BROWSER_USE_API_KEY")):
            raise ResearchPermanentError("BROWSER_KEY_REQUIRED", "Configure Browser Use Cloud credentials before starting research.")
        if browser == "cdp" and (result.get("cdp_isolated_ack") is not True or not result.get("cdp_url")):
            raise ResearchPermanentError("ISOLATED_CDP_REQUIRED", "Configure and acknowledge a dedicated unauthenticated CDP browser.")
        key_name = {"openai": "openai_api_key", "anthropic": "anthropic_api_key"}.get(provider)
        if key_name and not (result.get(key_name) or os.environ.get(key_name.upper())):
            raise ResearchPermanentError("RESEARCH_KEY_REQUIRED", "Configure the selected direct research provider's API key.")
    return result


class ResponsesResearchModel:
    _verified_api_keys = False
    provider = "openai"

    def __init__(self, model: str, api_key: str, *, client: Any = None, effort: str = "high", max_output_tokens: int = 12000,
                 use_reasoning: bool = True, max_retries: int = 2):
        self.model = model
        self.effort = effort
        self.max_output_tokens = max_output_tokens
        self.use_reasoning = use_reasoning
        if client is None:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=api_key, timeout=120, max_retries=max_retries)
        self.client = client

    @property
    def name(self):
        return self.model

    @property
    def model_name(self):
        return self.model

    @staticmethod
    def inputs(messages):
        values = []
        for message in messages:
            role = getattr(message, "role", "user")
            content = getattr(message, "content", "")
            if isinstance(content, str):
                values.append({"role": role, "content": content})
                continue
            parts = []
            for part in content:
                if getattr(part, "type", None) == "text":
                    parts.append({"type": "output_text" if role == "assistant" else "input_text", "text": part.text})
                elif getattr(part, "type", None) == "image_url":
                    parts.append({"type": "input_image", "image_url": part.image_url.url, "detail": part.image_url.detail})
            values.append({"role": role, "content": parts})
        return values

    async def ainvoke(self, messages, output_format=None, **kwargs):
        from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage
        arguments = {"model": self.model, "input": self.inputs(messages), "store": False,
                     "max_output_tokens": self.max_output_tokens}
        if self.use_reasoning:
            arguments["reasoning"] = {"effort": self.effort}
        if output_format:
            from browser_use.llm.schema import SchemaOptimizer
            schema = SchemaOptimizer.create_optimized_json_schema(output_format, remove_defaults=True)
            arguments["text"] = {"format": {"type": "json_schema", "name": output_format.__name__, "strict": True, "schema": schema}}
        response = await self.client.responses.create(**arguments)
        if response.status != "completed":
            raise RuntimeError("Research model response incomplete; retry this research pass")
        if any(getattr(item, "type", None) == "refusal" for output in response.output for item in getattr(output, "content", [])):
            raise ResearchPermanentError("MODEL_REFUSED", "Research model declined this action; inspect the mission before resuming.")
        text = response.output_text
        if not text:
            raise ResearchPermanentError("MODEL_SCHEMA_INVALID", "Research model returned no actionable output.")
        completion = output_format.model_validate_json(text) if output_format else text
        usage = response.usage
        receipt = ChatInvokeUsage(prompt_tokens=usage.input_tokens, completion_tokens=usage.output_tokens,
                                  total_tokens=usage.total_tokens,
                                  prompt_cached_tokens=getattr(usage.input_tokens_details, "cached_tokens", 0),
                                  prompt_cache_creation_tokens=None, prompt_image_tokens=None) if usage else None
        return ChatInvokeCompletion(completion=completion, usage=receipt, stop_reason="end_turn")


class X402ResponsesResearchModel(ResponsesResearchModel):
    """The exact Responses payload travels through the internal payment broker."""

    def __init__(self, model: str, broker: X402BrokerClient, *, effort: str = "high", max_output_tokens: int = 12000):
        class Transport:
            responses = None

            def __init__(self):
                self.responses = self

            async def create(self, **body):
                from openai.types.responses import Response
                result = await broker.request("responses", body)
                try:
                    return Response.model_validate(result)
                except ValueError:
                    raise ResearchPermanentError("VENDOR_SCHEMA_INVALID", "Gateway Responses output does not match the native vendor schema.") from None

            async def close(self):
                await broker.aclose()

        reasoning = model.rsplit("/", 1)[-1].startswith(("gpt-5", "gpt-6", "o1", "o3", "o4"))
        super().__init__(model, "", client=Transport(), effort=effort, max_output_tokens=max_output_tokens,
                         use_reasoning=reasoning)
        self.broker = broker

    async def preflight(self):
        selected = await self.broker.preflight(self.model, "responses")
        capabilities = selected.get("capabilities", {})
        efforts = capabilities.get("reasoning_efforts")
        if isinstance(efforts, list) and self.effort not in efforts:
            raise ResearchPermanentError("REASONING_EFFORT_UNSUPPORTED", "The selected gateway model does not advertise this reasoning effort.")
        self.use_reasoning = capabilities.get("reasoning", self.use_reasoning)
        return selected


class X402MessagesResearchModel:
    """Reuse Browser Use's Anthropic message/image/tool serializer without a vendor key."""

    _verified_api_keys = False
    provider = "anthropic"

    def __init__(self, model: str, broker: X402BrokerClient, *, max_output_tokens: int = 12000):
        from browser_use import ChatAnthropic

        class BrokerAnthropic(ChatAnthropic):
            async def _create_message(self, **params):
                from anthropic.types import Message
                from anthropic import omit
                params = {key: value for key, value in params.items() if value is not omit}
                extra = params.pop("extra_body", None)
                if extra:
                    params.update(extra)
                if params.pop("betas", None):
                    raise ResearchPermanentError("UNSUPPORTED_BETA", "Gateway research does not enable unadvertised vendor beta controls.")
                result = await broker.request("messages", params)
                try:
                    return Message.model_validate(result)
                except ValueError:
                    raise ResearchPermanentError("VENDOR_SCHEMA_INVALID", "Gateway Messages output does not match the native vendor schema.") from None

        self.model = model
        self.broker = broker
        self.client = broker
        # Older native Messages models need not accept optional adaptive thinking.
        # The adapter handles adaptive-only tool choice when the model requires it.
        self.inner = BrokerAnthropic(model=model, max_tokens=max_output_tokens, max_retries=0)

    @property
    def name(self):
        return self.model

    @property
    def model_name(self):
        return self.model

    async def preflight(self):
        return await self.broker.preflight(self.model, "messages")

    async def ainvoke(self, messages, output_format=None, **kwargs):
        return await self.inner.ainvoke(messages, output_format=output_format, **kwargs)


class MonitoredResearchModel:
    """Surface terminal failures even when Browser Use consumes a failed model step."""

    _verified_api_keys = False

    def __init__(self, model, on_failure):
        self.inner = model
        self.on_failure = on_failure
        self.failures = 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    async def ainvoke(self, messages, output_format=None, **kwargs):
        import asyncio
        from .x402_client import ResearchFundingError, ResearchTransientError
        try:
            result = await self.inner.ainvoke(messages, output_format=output_format, **kwargs)
        except Exception as exc:
            error = classify_failure(exc)
            if isinstance(error, ResearchTransientError):
                self.failures += 1
                if self.failures < 3:
                    await asyncio.sleep(2 ** (self.failures - 1))
                else:
                    error = ResearchPermanentError("TRANSIENT_RETRY_EXHAUSTED", "Research retries were exhausted; inspect the provider and explicitly resume.")
            if isinstance(error, (ResearchPermanentError, ResearchFundingError)):
                self.on_failure(error)
            raise error from None
        self.failures = 0
        return result


def researcher_model(settings, *, intent_store=None, scope: str | None = None, sdk_max_retries: int = 2):
    import os
    settings = validate_research_settings(settings, require_config=True)
    provider = settings.get("research_provider", settings.get("agent_provider", "x402"))
    model = settings.get("research_model", settings.get("agent_model"))
    if provider == "x402":
        validate_research_settings(settings, require_config=True)
        broker = X402BrokerClient(intent_store=intent_store, scope=scope)
        if settings.get("research_protocol", "responses") == "messages":
            return X402MessagesResearchModel(model, broker, max_output_tokens=settings.get("research_max_output_tokens", 12000))
        return X402ResponsesResearchModel(model, broker, effort=settings.get("reasoning_effort", "high"),
                                         max_output_tokens=settings.get("research_max_output_tokens", 12000))
    if provider == "openai":
        key = settings.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("Connect an OpenAI API key in operator setup")
        return ResponsesResearchModel(model or "gpt-6-astra", key, effort=settings.get("reasoning_effort", "high"),
                                      max_output_tokens=settings.get("research_max_output_tokens", 12000), max_retries=sdk_max_retries)
    if provider == "anthropic":
        from browser_use import ChatAnthropic
        key = settings.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("Connect a Claude API key in operator setup")
        import httpx
        return ChatAnthropic(model=model or "claude-fable-5-1", api_key=key, thinking={"type": "adaptive"},
                             max_tokens=settings.get("research_max_output_tokens", 12000), max_retries=sdk_max_retries,
                             http_client=httpx.AsyncClient(timeout=120))
    raise ValueError("Select x402, OpenAI or Claude as the researcher model provider")
