"""LangChain TypeSafeClassifier routed through OpenRouter's Jev decisions endpoint.

OpenRouter serves Jev at a different path/model-id shape than TypeSafe's own API
(POST https://openrouter.ai/api/alpha/decisions, model "~typesafe/jev-latest"), so we
subclass the LangChain TypeSafeClassifier to point at OpenRouter instead of typesafe.ai.
Everything else -- typed Choice/Score/Noul questions, ClassifierResponse parsing,
tracing, error handling -- comes straight from langchain_typesafe.
"""

import time

import httpx2
from langchain_core.utils import secret_from_env
from langchain_typesafe import TypeSafeClassifier
from langchain_typesafe.client import (
    TypeSafeAPIConnectionError,
    TypeSafeAPITimeoutError,
    parse_response,
)
from langchain_typesafe.types import ClassifierRequest, ClassifierResponse
from pydantic import Field, SecretStr


class OpenRouterClassifierResponse(ClassifierResponse):
    """ClassifierResponse plus the per-request USD cost OpenRouter reports and the request latency."""

    cost: float | None = None
    latency_s: float | None = None


class OpenRouterJevClassifier(TypeSafeClassifier):
    """TypeSafeClassifier routed through OpenRouter's Jev decisions endpoint."""

    model: str = "~typesafe/jev-latest"
    base_url: str = "https://openrouter.ai/api"
    api_key: SecretStr | str = Field(
        default_factory=secret_from_env("OPENROUTER_API_KEY", default=""),
        exclude=True,
        repr=False,
    )

    @property
    def _endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/alpha/decisions"

    def _payload(self, request: ClassifierRequest) -> dict:
        # OpenRouter's usage-accounting flag: ask it to include cost in the response.
        return {**super()._payload(request), "usage": {"include": True}}

    def _classify(self, request: ClassifierRequest) -> OpenRouterClassifierResponse:
        payload = self._payload(request)
        start = time.perf_counter()
        try:
            response = self.client.post(
                self._endpoint, json=payload, headers=self._request_headers
            )
        except httpx2.TimeoutException as error:
            raise TypeSafeAPITimeoutError(self.client.timeout) from error
        except httpx2.HTTPError as error:
            message = "Unable to connect to the TypeSafe API."
            raise TypeSafeAPIConnectionError(message) from error
        latency_s = time.perf_counter() - start
        base = parse_response(response)  # raises on non-2xx / malformed bodies
        usage = response.json().get("usage")
        cost = usage.get("cost") if isinstance(usage, dict) else None
        full = OpenRouterClassifierResponse(**base.model_dump(), cost=cost, latency_s=latency_s)
        return self._record_usage(full)

    # Note: the inherited validator's error message still says "TYPESAFE_API_KEY" if
    # this is unset -- that text is hardcoded upstream. The env var actually read here
    # is OPENROUTER_API_KEY.
