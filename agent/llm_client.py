from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LLMClientError(RuntimeError):
    """Base exception for local LLM client failures."""


class LLMConnectionError(LLMClientError):
    """The inference server could not be reached."""


class LLMTimeoutError(LLMClientError):
    """The inference request exceeded its timeout."""


class LLMHTTPError(LLMClientError):
    """The inference server returned a non-success HTTP status."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMResponseError(LLMClientError):
    """The server response was invalid or incomplete."""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    latency_seconds: float


class LocalLLMClient:
    """Minimal synchronous client for a local OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        self.base_url = (
            base_url or os.environ.get(
                "LLM_BASE_URL", "http://127.0.0.1:8080"
            )
        ).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL", "")
        self.timeout_seconds = timeout_seconds

    def _resolve_model(self) -> str:
        if self.model:
            return self.model

        # Ask the server for its actual model identifier instead of
        # assuming that an alias is configured.
        try:
            request = Request(
                f"{self.base_url}/v1/models",
                headers={"Accept": "application/json"},
                method="GET",
            )
            with urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise LLMHTTPError(
                exc.code, f"Model discovery returned HTTP {exc.code}"
            ) from exc
        except TimeoutError as exc:
            raise LLMTimeoutError("Model discovery timed out") from exc
        except (URLError, OSError) as exc:
            if isinstance(exc, TimeoutError):
                raise LLMTimeoutError("Model discovery timed out") from exc
            raise LLMConnectionError(
                "Could not connect to the inference server"
            ) from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LLMResponseError(
                "Model discovery returned invalid JSON"
            ) from exc

        models = data.get("data") if isinstance(data, dict) else None
        if not isinstance(models, list) or not models:
            raise LLMResponseError(
                "Model discovery returned no models in the data field"
            )

        model_id = models[0].get("id") if isinstance(models[0], dict) else None
        if not isinstance(model_id, str) or not model_id:
            raise LLMResponseError("Model discovery returned an invalid model ID")

        self.model = model_id
        return model_id

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> LLMResponse:
        if not messages:
            raise ValueError("messages must not be empty")
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")

        for message in messages:
            if (
                not isinstance(message, dict)
                or not isinstance(message.get("role"), str)
                or not isinstance(message.get("content"), str)
            ):
                raise ValueError(
                    "Each message must contain string 'role' and 'content' fields"
                )

        model = self._resolve_model()
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        request = Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        started = time.perf_counter()
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
        except HTTPError as exc:
            # Do not propagate a potentially verbose server response that
            # might include submitted prompt content.
            raise LLMHTTPError(
                exc.code, f"Chat completion returned HTTP {exc.code}"
            ) from exc
        except TimeoutError as exc:
            raise LLMTimeoutError("Chat completion timed out") from exc
        except (URLError, OSError) as exc:
            if isinstance(exc, TimeoutError):
                raise LLMTimeoutError("Chat completion timed out") from exc
            raise LLMConnectionError(
                "Could not connect to the inference server"
            ) from exc

        latency = time.perf_counter() - started

        try:
            data: Any = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LLMResponseError(
                "Chat completion returned invalid JSON"
            ) from exc

        try:
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            response_model = data.get("model", model)

            if not isinstance(text, str):
                raise TypeError("content is not a string")
            if not isinstance(response_model, str):
                raise TypeError("model is not a string")
            if not isinstance(usage, dict):
                raise TypeError("usage is not an object")

            def token_count(key: str) -> int | None:
                value = usage.get(key)
                if value is None:
                    return None
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise TypeError(f"{key} is not a non-negative integer")
                return value

            return LLMResponse(
                text=text,
                model=response_model,
                prompt_tokens=token_count("prompt_tokens"),
                completion_tokens=token_count("completion_tokens"),
                total_tokens=token_count("total_tokens"),
                latency_seconds=latency,
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError(
                "Chat completion response has an unexpected structure"
            ) from exc
