"""Internal payment-broker transport. Wallet keys and signing never enter this process."""
from __future__ import annotations

import os
import asyncio
import hashlib
import json
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import httpx

BROKER_POST_TIMEOUT_SECONDS = 450
BROKER_REQUEST_DEADLINE_SECONDS = 480
X402_AGENT_LLM_TIMEOUT_SECONDS = 600


class ResearchError(RuntimeError):
    """An actionable, deliberately safe research failure."""

    def __init__(self, code: str, message: str, *, definitively_unpaid: bool = False):
        super().__init__(message)
        self.code = code
        self.definitively_unpaid = definitively_unpaid


class ResearchPermanentError(ResearchError):
    pass


class ResearchFundingError(ResearchError):
    pass


class ResearchTransientError(ResearchError):
    pass


PERMANENT_BROKER_CODES = {
    "SETTLEMENT_UNKNOWN", "REQUEST_CONFLICT", "AUTH_FAILED", "NOT_CONFIGURED",
    "INVALID_REQUEST", "CAP_EXCEEDED", "PAID_UPSTREAM_REJECTED", "PAID_RESPONSE_UNAVAILABLE",
    "UNEXPECTED_GATEWAY_RESPONSE", "UNSUPPORTED_MODEL", "UNAPPROVED_QUOTE",
    "SIGNER_INVALID", "SIGNER_NOT_CONFIGURED",
}
TRANSIENT_UNPAID_CODES = {"UPSTREAM_UNAVAILABLE", "CATALOG_UNAVAILABLE", "SIGNING_FAILED"}
DEFINITIVELY_UNPAID_CODES = TRANSIENT_UNPAID_CODES | {
    "UNFUNDED", "FUNDING_UNKNOWN", "DAILY_LIMIT", "INVALID_REQUEST",
    "UNAPPROVED_QUOTE", "UNSUPPORTED_MODEL", "CAP_EXCEEDED",
}


def classify_failure(exc: Exception) -> ResearchError:
    """SDKs sometimes wrap errors; inspect causes without publishing their bodies."""
    chain = []
    current = exc
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    for error in chain:
        if isinstance(error, ResearchError):
            return error
    for error in chain:
        status = getattr(error, "status_code", None)
        if status is None and isinstance(error, httpx.HTTPStatusError):
            status = error.response.status_code
        if status in {400, 401, 403, 404, 409, 422}:
            return ResearchPermanentError("PROVIDER_CONFIGURATION", "Research provider authentication, access or request configuration needs operator attention.")
        if isinstance(error, (ValueError, TypeError, ImportError)):
            return ResearchPermanentError("INVALID_CONFIGURATION_OR_SCHEMA", "Research configuration or structured action schema is invalid.")
    return ResearchTransientError("RESEARCH_TEMPORARILY_UNAVAILABLE", "Research service temporarily unavailable; retrying with backoff.")


class X402BrokerClient:
    def __init__(self, *, client: httpx.AsyncClient | None = None, intent_store=None, scope: str | None = None):
        if intent_store is not None and (not isinstance(scope, str) or not scope):
            raise ResearchPermanentError("INTENT_SCOPE_REQUIRED", "Durable research payment requests require a stable agent scope.")
        base = os.environ.get("OBSERVATORY_PAYMENT_BROKER_URL", "http://127.0.0.1:8062").rstrip("/")
        parts = urlsplit(base)
        if (parts.scheme not in {"http", "https"} or not parts.hostname or parts.username
                or parts.password or parts.query or parts.fragment):
            raise ResearchPermanentError("INVALID_BROKER_URL", "Configure a trusted HTTP(S) payment broker URL in the server environment.")
        if parts.scheme == "http" and parts.hostname not in {"localhost", "127.0.0.1", "::1", "payments"}:
            raise ResearchPermanentError("BROKER_HTTPS_REQUIRED", "Remote payment brokers require HTTPS; HTTP is restricted to loopback or the payments container.")
        token = os.environ.get("OBSERVATORY_PAYMENT_BROKER_TOKEN", "")
        if not token:
            raise ResearchPermanentError("BROKER_AUTH_NOT_CONFIGURED", "Configure the internal payment broker token in the server environment.")
        self.base = base
        self.headers = {"Authorization": "Bearer " + token}
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(BROKER_POST_TIMEOUT_SECONDS, connect=10), follow_redirects=False, trust_env=False)
        self._owned_client = client is None
        self._request_ids: dict[str, str] = {}
        self.intent_store = intent_store
        self.scope = scope

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()

    async def aclose(self):
        if self._owned_client:
            await self.client.aclose()

    async def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        try:
            response = await self.client.request(method, self.base + path, headers=self.headers, json=payload,
                                                 timeout=httpx.Timeout(30 if method == "GET" else BROKER_POST_TIMEOUT_SECONDS, connect=10))
        except httpx.HTTPError:
            if method == "POST":
                # The request may already have been paid. Do not create a new ID.
                raise ResearchPermanentError("BROKER_REQUEST_UNKNOWN", "Payment request outcome is unknown; reconcile the broker receipt before resuming.") from None
            raise ResearchTransientError("BROKER_UNREACHABLE", "Payment broker is temporarily unreachable.") from None
        try:
            data = response.json()
        except ValueError:
            raise ResearchPermanentError("BROKER_INVALID_RESPONSE", "Payment broker returned an invalid protocol response.") from None
        if not isinstance(data, dict):
            raise ResearchPermanentError("BROKER_INVALID_RESPONSE", "Payment broker returned an invalid protocol response.")
        if response.is_success:
            return data
        error = data.get("error", {})
        code = error.get("code") if isinstance(error, dict) else None
        code = code or data.get("code")
        code = code if isinstance(code, str) else None
        if code in PERMANENT_BROKER_CODES or response.status_code in {400, 401, 403, 404, 409, 422}:
            safe_code = code if code in PERMANENT_BROKER_CODES else "BROKER_CONFIGURATION"
            raise ResearchPermanentError(safe_code, "Payment broker configuration or an unresolved payment needs operator attention.",
                                         definitively_unpaid=code in DEFINITIVELY_UNPAID_CODES)
        if code in {"UNFUNDED", "FUNDING_UNKNOWN", "DAILY_LIMIT"} or response.status_code == 402:
            raise ResearchFundingError(str(code) if code in {"UNFUNDED", "FUNDING_UNKNOWN", "DAILY_LIMIT"} else "UNFUNDED",
                                       "Research is waiting for available, authorized broker funds.",
                                       definitively_unpaid=code in DEFINITIVELY_UNPAID_CODES)
        if method == "POST" and code not in TRANSIENT_UNPAID_CODES:
            raise ResearchPermanentError("BROKER_REQUEST_UNKNOWN", "Payment request outcome is unknown; inspect the broker before retrying.")
        raise ResearchTransientError(code if code in TRANSIENT_UNPAID_CODES else "BROKER_TEMPORARILY_UNAVAILABLE",
                                     "Payment broker request temporarily unavailable.",
                                     definitively_unpaid=code in DEFINITIVELY_UNPAID_CODES)

    async def get_health(self) -> dict:
        return await self._request("GET", "/health")

    async def get_catalog(self) -> dict:
        return await self._request("GET", "/catalog")

    async def get_funding(self) -> dict:
        return await self._request("GET", "/funding")

    async def get_request(self, request_id: str) -> dict:
        try:
            if str(UUID(request_id)) != request_id:
                raise ValueError
        except (ValueError, TypeError, AttributeError):
            raise ResearchPermanentError("INVALID_REQUEST_ID", "Select a canonical broker request ID.") from None
        return await self._request("GET", "/v1/requests/" + request_id)

    async def abandon_request(self, request_id: str, protocol: str, body: dict) -> dict:
        """Tombstone an idle intent in the broker ledger; never request inference/payment."""
        try:
            if str(UUID(request_id)) != request_id:
                raise ValueError
        except (ValueError, TypeError, AttributeError):
            raise ResearchPermanentError("INVALID_REQUEST_ID", "Select a canonical broker request ID.") from None
        if not isinstance(protocol, str) or protocol not in {"responses", "messages"} or not isinstance(body, dict):
            raise ResearchPermanentError("INVALID_REQUEST", "An exact native protocol and private body are required for intent recovery.")
        return await self._request("POST", "/v1/requests/" + request_id + "/abandon",
                                   {"protocol": protocol, "body": body})

    def _complete_pending(self, request_id: str, digest: str):
        if self.intent_store is not None:
            self.intent_store.complete_research_request(request_id)
        self._request_ids.pop(digest, None)

    async def preflight(self, model: str, protocol: str) -> dict:
        """Validate advertised capabilities and funds before allocating a browser."""
        if self.intent_store is not None and self.intent_store.pending_research_requests(self.scope):
            raise ResearchPermanentError("CALLER_RECOVERY_REQUIRED", "Review the agent's unresolved broker request before allocating a new research browser.")
        if not isinstance(model, str) or not model.strip():
            raise ResearchPermanentError("MODEL_REQUIRED", "Select an explicit model from the payment gateway catalog.")
        if not isinstance(protocol, str) or protocol not in {"responses", "messages"}:
            raise ResearchPermanentError("INVALID_PROTOCOL", "Choose the catalog model's responses or messages protocol.")
        health = await self.get_health()
        if not isinstance(health.get("status"), str) or health.get("status") not in {"ok", "ready", "healthy"}:
            raise ResearchPermanentError("BROKER_NOT_READY", "Payment broker is not configured and healthy.")
        catalog = await self.get_catalog()
        models = catalog.get("models")
        if not isinstance(models, list):
            raise ResearchPermanentError("CATALOG_INVALID", "Payment gateway catalog is invalid.")
        selected = next((item for item in models if isinstance(item, dict) and item.get("id") == model), None)
        if not selected:
            raise ResearchPermanentError("MODEL_NOT_IN_CATALOG", "The selected model is not advertised by the payment gateway; no substitute will be chosen.")
        protocols = selected.get("supportedprotocols", selected.get("supported_protocols", selected.get("protocols", [])))
        if not isinstance(protocols, list) or protocol not in protocols:
            raise ResearchPermanentError("MODEL_PROTOCOL_UNSUPPORTED", "The selected model does not advertise the requested native protocol.")
        capabilities = selected.get("capabilities", {})
        if not isinstance(capabilities, dict) or capabilities.get("vision") is not True or capabilities.get("structured_actions") is not True:
            raise ResearchPermanentError("MODEL_CAPABILITIES_UNSUPPORTED", "Research requires advertised vision and structured actions.")
        required = "structured_outputs" if protocol == "responses" else "tool_calling"
        if capabilities.get(required) is not True:
            raise ResearchPermanentError("MODEL_SCHEMA_UNSUPPORTED", "The selected model does not advertise the required action-schema controls.")
        funding = await self.get_funding()
        reason = funding.get("reason")
        if isinstance(reason, str) and reason in {"SETTLEMENT_UNKNOWN", "SIGNER_INVALID", "SIGNER_NOT_CONFIGURED", "NOT_CONFIGURED"}:
            raise ResearchPermanentError(reason, "Payment broker configuration or an unresolved settlement needs operator attention before research resumes.")
        if funding.get("status") != "ready":
            raise ResearchFundingError("FUNDING_NOT_READY", "Research is waiting for available, authorized broker funds.")
        if funding.get("configured") is not True or funding.get("enabled") is not True or reason is not None:
            raise ResearchPermanentError("FUNDING_SCHEMA_INVALID", "Broker readiness requires explicit configuration, spending authorization and no unresolved funding reason.")
        return selected

    async def request(self, protocol: str, body: dict) -> dict:
        if not isinstance(protocol, str) or protocol not in {"responses", "messages"} or not isinstance(body, dict) or not isinstance(body.get("model"), str):
            raise ResearchPermanentError("INVALID_REQUEST", "A pinned model and native research protocol are required.")
        digest = hashlib.sha256(json.dumps({"protocol": protocol, "body": body}, sort_keys=True).encode()).hexdigest()
        if self.intent_store is not None:
            pending = self.intent_store.pending_research_requests(self.scope)
            if any(item["body_hash"] != digest for item in pending):
                raise ResearchPermanentError("CALLER_RECOVERY_REQUIRED", "Review the agent's unresolved broker request before changing its research context.")
            try:
                request_id = self.intent_store.reserve_research_request(self.scope, digest, protocol, body)
            except ValueError:
                raise ResearchPermanentError("CALLER_RECOVERY_REQUIRED", "Review the agent's unresolved broker request before changing its research context.") from None
            self._request_ids[digest] = request_id
        if self._request_ids and digest not in self._request_ids:
            raise ResearchPermanentError("REQUEST_PENDING", "Resolve the previous broker request before changing research context.")
        request_id = self._request_ids.setdefault(digest, str(uuid4()))
        try:
            async with asyncio.timeout(BROKER_REQUEST_DEADLINE_SECONDS):
                for attempt in range(3):
                    try:
                        result = await self._request("POST", "/v1/request", {
                            "request_id": request_id, "protocol": protocol, "body": body,
                        })
                        break
                    except ResearchTransientError as exc:
                        if attempt == 2:
                            raise ResearchPermanentError("BROKER_RETRY_EXHAUSTED", "Unpaid broker retries were exhausted; inspect the gateway before resuming.",
                                                         definitively_unpaid=exc.definitively_unpaid) from None
                        await asyncio.sleep(2 ** attempt)
        except TimeoutError:
            raise ResearchPermanentError("BROKER_REQUEST_UNKNOWN", "The broker request deadline expired; reconcile its receipt before resuming.") from None
        except ResearchError as exc:
            if exc.definitively_unpaid:
                self._complete_pending(request_id, digest)
            raise
        payment = result.get("payment", {})
        if not isinstance(payment, dict) or payment.get("status") != "settled":
            raise ResearchPermanentError("SETTLEMENT_UNKNOWN", "Payment settlement is unresolved; reconcile the broker receipt before resuming.")
        if not isinstance(result.get("response"), dict):
            raise ResearchPermanentError("BROKER_INVALID_RESPONSE", "Payment broker did not return the requested vendor response.")
        self._complete_pending(request_id, digest)
        return result["response"]
