from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
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


def _validate_messages(messages: list[dict[str, str]]) -> None:
    if not messages:
        raise ValueError("messages must not be empty")

    for message in messages:
        if (
            not isinstance(message, dict)
            or not isinstance(message.get("role"), str)
            or not isinstance(message.get("content"), str)
        ):
            raise ValueError(
                "Each message must contain string 'role' and 'content' fields"
            )


def _validate_generation(temperature: float, max_tokens: int) -> None:
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("temperature must be between 0 and 2")


def _token_count(usage: dict[str, Any], key: str) -> int | None:
    value = usage.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{key} is not a non-negative integer")
    return value


@contextmanager
def _http_request(
    request: Request, *, timeout: float, label: str
) -> Iterator[Any]:
    """Open a request and map transport failures onto the client exceptions.

    Server error bodies are never propagated: they can echo prompt content.
    """
    try:
        with urlopen(request, timeout=timeout) as response:
            yield response
    except HTTPError as exc:
        exc.close()
        raise LLMHTTPError(
            exc.code, f"{label} returned HTTP {exc.code}"
        ) from exc
    except TimeoutError as exc:
        raise LLMTimeoutError(f"{label} timed out") from exc
    except (URLError, OSError) as exc:
        raise LLMConnectionError(
            "Could not connect to the inference server"
        ) from exc


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
        self._token_cache: dict[str, int] = {}
        self._token_cache_limit = 512

    def _resolve_model(self) -> str:
        if self.model:
            return self.model

        # Ask the server for its actual model identifier instead of
        # assuming that an alias is configured.
        request = Request(
            f"{self.base_url}/v1/models",
            headers={"Accept": "application/json"},
            method="GET",
        )
        try:
            with _http_request(
                request,
                timeout=self.timeout_seconds,
                label="Model discovery",
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
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
        _validate_messages(messages)
        _validate_generation(temperature, max_tokens)

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
        with _http_request(
            request, timeout=self.timeout_seconds, label="Chat completion"
        ) as response:
            raw = response.read()
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

            return LLMResponse(
                text=text,
                model=response_model,
                prompt_tokens=_token_count(usage, "prompt_tokens"),
                completion_tokens=_token_count(usage, "completion_tokens"),
                total_tokens=_token_count(usage, "total_tokens"),
                latency_seconds=latency,
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError(
                "Chat completion response has an unexpected structure"
            ) from exc

    def _post_json(self, path: str, payload: dict[str, Any], label: str) -> Any:
        request = Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _http_request(
            request, timeout=self.timeout_seconds, label=label
        ) as response:
            raw = response.read()

        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LLMResponseError(f"{label} returned invalid JSON") from exc

    def _tokenize(self, content: str) -> list[int]:
        data = self._post_json("/tokenize", {"content": content}, "Tokenize")
        tokens = data.get("tokens") if isinstance(data, dict) else None
        if not isinstance(tokens, list) or not all(
            isinstance(token, int) and not isinstance(token, bool)
            for token in tokens
        ):
            raise LLMResponseError("Tokenize returned an unexpected structure")
        return tokens

    def count_tokens(self, text: str) -> int:
        """Exact token count for text, using the server's own tokenizer."""
        if not isinstance(text, str):
            raise ValueError("text must be a string")

        cached = self._token_cache.get(text)
        if cached is not None:
            return cached

        count = len(self._tokenize(text))
        if len(self._token_cache) >= self._token_cache_limit:
            self._token_cache.pop(next(iter(self._token_cache)))
        self._token_cache[text] = count
        return count

    def count_prompt_tokens(self, messages: list[dict[str, str]]) -> int:
        """Exact prompt length after the server applies its chat template."""
        _validate_messages(messages)
        data = self._post_json(
            "/apply-template", {"messages": messages}, "Chat template"
        )
        prompt = data.get("prompt") if isinstance(data, dict) else None
        if not isinstance(prompt, str):
            raise LLMResponseError(
                "Chat template returned an unexpected structure"
            )
        return len(self._tokenize(prompt))

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> ChatStream:
        """Start a streaming completion; iterate it to receive text deltas."""
        _validate_messages(messages)
        _validate_generation(temperature, max_tokens)
        return ChatStream(
            client=self,
            messages=list(messages),
            temperature=temperature,
            max_tokens=max_tokens,
        )


class ChatStream:
    """One streaming completion, iterable once for its text deltas.

    After the iteration ends, ``text`` holds the full response and the usage
    and timing attributes are filled in. Pass ``stream_options``-style usage
    is optional on the server side, so token counts may be ``None``.
    """

    def __init__(
        self,
        client: LocalLLMClient,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> None:
        self._client = client
        self._messages = messages
        self._temperature = temperature
        self._max_tokens = max_tokens

        self.text = ""
        self.model: str | None = None
        self.prompt_tokens: int | None = None
        self.completion_tokens: int | None = None
        self.total_tokens: int | None = None
        self.first_token_seconds: float | None = None
        self.latency_seconds: float | None = None

        self._started = False

    def _record_chunk(self, chunk: dict[str, Any]) -> str | None:
        chunk_model = chunk.get("model")
        if isinstance(chunk_model, str) and chunk_model:
            self.model = chunk_model

        usage = chunk.get("usage")
        if isinstance(usage, dict):
            self.prompt_tokens = _token_count(usage, "prompt_tokens")
            self.completion_tokens = _token_count(usage, "completion_tokens")
            self.total_tokens = _token_count(usage, "total_tokens")

        choices = chunk.get("choices")
        if choices is None:
            return None
        if not isinstance(choices, list):
            raise LLMResponseError("Streaming chunk has an invalid choices field")
        if not choices:
            return None

        choice = choices[0]
        if not isinstance(choice, dict):
            raise LLMResponseError("Streaming chunk has an invalid choice")
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            return None
        content = delta.get("content")
        if content is None:
            return None
        if not isinstance(content, str):
            raise LLMResponseError("Streaming chunk content is not a string")
        return content

    def __iter__(self) -> Iterator[str]:
        if self._started:
            raise RuntimeError("A ChatStream can only be iterated once")
        self._started = True

        client = self._client
        model = client._resolve_model()
        self.model = model
        payload = {
            "model": model,
            "messages": self._messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": True,
        }
        request = Request(
            f"{client.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            method="POST",
        )

        started = time.perf_counter()
        try:
            with _http_request(
                request,
                timeout=client.timeout_seconds,
                label="Chat completion",
            ) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", "replace").strip()
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue

                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise LLMResponseError(
                            "Streaming chunk was not valid JSON"
                        ) from exc
                    if not isinstance(chunk, dict):
                        raise LLMResponseError(
                            "Streaming chunk was not a JSON object"
                        )

                    delta = self._record_chunk(chunk)
                    if delta:
                        if self.first_token_seconds is None:
                            self.first_token_seconds = (
                                time.perf_counter() - started
                            )
                        self.text += delta
                        yield delta
        finally:
            self.latency_seconds = time.perf_counter() - started
