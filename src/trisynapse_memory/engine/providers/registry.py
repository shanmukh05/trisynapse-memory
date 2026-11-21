"""Completion, embedding, model discovery, and provider credentials.

Owns provider descriptors, catalog fetch, completion JSON helpers, and embedder
construction. Engine configuration reads this registry; it must not log secrets.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from trisynapse_memory.engine.providers.embedding import (
    GeminiEmbedder,
    OpenAICompatibleEmbedder,
    SentenceTransformerEmbedder,
)
from trisynapse_memory.engine.models import (
    ModelDescriptor,
    ProviderDescriptor,
    ProviderRole,
    ProviderSelection,
)


class ProviderError(RuntimeError):
    pass


class EmbeddingRebuildRequired(ProviderError):
    """Raised when an embedding change needs an explicitly approved rebuild."""


@dataclass(frozen=True)
class ProviderSettings:
    provider: str = "none"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    timeout_seconds: float = 60


class CompletionProvider(Protocol):
    def complete_json(self, system: str, user: str) -> dict[str, Any]: ...

    def __call__(self, system: str, user: str) -> dict[str, Any]: ...

    def complete_multimodal(
        self, system: str, user: str, image: bytes, media_type: str
    ) -> dict[str, Any]: ...


DEFAULT_COMPLETION_PARALLEL_WORKERS = 8
_JSON_PARSE_ATTEMPTS = 3


def parse_json_response(text: str) -> dict[str, Any]:
    """Parse a provider's JSON-object response.

    Accepts a Markdown fence, leading commentary, and trailing extra tokens that
    some OpenAI-compatible models append after a valid object.
    """

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, count=1)
        fence = cleaned.find("```")
        if fence != -1:
            cleaned = cleaned[:fence]
        cleaned = cleaned.strip()
    try:
        value = _load_json_value(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"completion returned invalid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError("completion returned JSON that is not an object")
    return value


def _load_json_value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise
        return json.JSONDecoder().raw_decode(text, start)[0]


def complete_json_parallel(
    complete_json: Callable[[str, str], dict[str, Any]],
    requests: Sequence[tuple[str, str]],
    *,
    max_workers: int = DEFAULT_COMPLETION_PARALLEL_WORKERS,
    on_progress: Callable[[int, int], None] | None = None,
    on_item: Callable[[int, dict[str, Any]], None] | None = None,
    on_error: Callable[[int, BaseException], dict[str, Any] | None] | None = None,
) -> list[dict[str, Any]]:
    """Run independent JSON completions concurrently, preserving request order."""

    if not requests:
        return []
    if len(requests) == 1:
        return _complete_json_one(
            complete_json, requests[0], on_progress=on_progress, on_item=on_item, on_error=on_error,
        )
    return _complete_json_pool(
        complete_json, requests, max_workers=max_workers,
        on_progress=on_progress, on_item=on_item, on_error=on_error,
    )


def _resolve_complete_error(
    on_error: Callable[[int, BaseException], dict[str, Any] | None] | None,
    index: int,
    exc: BaseException,
) -> dict[str, Any]:
    replacement = on_error(index, exc) if on_error is not None else None
    if replacement is None:
        raise exc
    return replacement


def _complete_json_one(
    complete_json: Callable[[str, str], dict[str, Any]],
    request: tuple[str, str],
    *,
    on_progress: Callable[[int, int], None] | None,
    on_item: Callable[[int, dict[str, Any]], None] | None,
    on_error: Callable[[int, BaseException], dict[str, Any] | None] | None,
) -> list[dict[str, Any]]:
    try:
        results = [_invoke_complete_json(complete_json, request[0], request[1])]
    except Exception as exc:
        results = [_resolve_complete_error(on_error, 0, exc)]
    if on_item is not None:
        on_item(0, results[0])
    if on_progress is not None:
        on_progress(1, 1)
    return results


def _complete_json_future_result(
    future: Any,
    index: int,
    on_error: Callable[[int, BaseException], dict[str, Any] | None] | None,
) -> tuple[dict[str, Any] | None, BaseException | None]:
    try:
        return future.result(), None
    except Exception as exc:
        try:
            return _resolve_complete_error(on_error, index, exc), None
        except Exception as resolved:
            return None, resolved


def _record_complete_json_item(
    ordered: list[dict[str, Any] | None],
    index: int,
    payload: dict[str, Any],
    completed: int,
    total: int,
    on_item: Callable[[int, dict[str, Any]], None] | None,
    on_progress: Callable[[int, int], None] | None,
) -> int:
    ordered[index] = payload
    if on_item is not None:
        on_item(index, payload)
    completed += 1
    if on_progress is not None:
        on_progress(completed, total)
    return completed


def _complete_json_pool(
    complete_json: Callable[[str, str], dict[str, Any]],
    requests: Sequence[tuple[str, str]],
    *,
    max_workers: int,
    on_progress: Callable[[int, int], None] | None,
    on_item: Callable[[int, dict[str, Any]], None] | None,
    on_error: Callable[[int, BaseException], dict[str, Any] | None] | None,
) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any] | None] = [None] * len(requests)
    workers = max(1, min(max_workers, len(requests)))
    errors: list[BaseException] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_invoke_complete_json, complete_json, system, user): index
            for index, (system, user) in enumerate(requests)
        }
        completed = 0
        for future in as_completed(futures):
            index = futures[future]
            payload, error = _complete_json_future_result(future, index, on_error)
            if error is not None:
                errors.append(error)
                continue
            assert payload is not None
            completed = _record_complete_json_item(
                ordered, index, payload, completed, len(requests), on_item, on_progress,
            )
    if errors:
        raise errors[0]
    return [item for item in ordered if item is not None]


def _invoke_complete_json(
    complete_json: Callable[[str, str], dict[str, Any]],
    system: str,
    user: str,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(_JSON_PARSE_ATTEMPTS):
        try:
            return complete_json(system, user)
        except ValueError as exc:
            last_error = exc
            if not _invalid_json_error(exc) or attempt + 1 >= _JSON_PARSE_ATTEMPTS:
                raise
            time.sleep(min(0.05 * (2 ** attempt), 1.0))
    raise last_error or ValueError("completion returned invalid JSON")


def _invalid_json_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "invalid json" in text or "not an object" in text


def invalid_json_completion_error(exc: Exception) -> bool:
    """Return True when a completion failed because the model returned unusable JSON."""

    return _invalid_json_error(exc)


def timeout_completion_error(exc: Exception) -> bool:
    """Return True when a completion failed because the provider timed out."""

    if isinstance(exc, TimeoutError):
        return True
    text = str(exc).lower()
    return "timed out" in text or "deadline exceeded" in text


def skippable_completion_error(exc: Exception) -> bool:
    """Return True when ingest can skip a session instead of aborting the case."""

    return invalid_json_completion_error(exc) or timeout_completion_error(exc)


def gemini_json_thinking_config(model: str, types: Any) -> Any | None:
    """Prefer cheap thinking for structured JSON on Gemini Flash models."""

    name = (model or "").lower()
    if "flash" in name and "gemini-3" in name:
        return types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL)
    if "gemini-2.5-flash" in name and "lite" not in name:
        return types.ThinkingConfig(thinking_budget=0)
    return None


def openai_json_chat_payload(model: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a Chat Completions JSON payload.

    GPT-5 and o-series models only accept the default temperature. Sending 0 is a 400.
    """

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    if _openai_supports_temperature(model):
        payload["temperature"] = 0
    return payload


def _openai_supports_temperature(model: str) -> bool:
    name = (model or "").rsplit("/", 1)[-1].lower()
    return not name.startswith(("gpt-5", "o1", "o3", "o4"))


def _retryable_provider_error(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    text = str(exc).upper()
    return any(
        token in text
        for token in (
            "429",
            "RESOURCE_EXHAUSTED",
            "UNAVAILABLE",
            "503",
            "DEADLINE EXCEEDED",
            "TIMED OUT",
        )
    )


def _unsupported_thinking_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "thinking" in text and any(token in text for token in ("400", "invalid", "unsupported"))


@dataclass(frozen=True)
class _ProviderSpec:
    display_name: str
    roles: tuple[ProviderRole, ...]
    credential_env: str | None
    default_base_url: str | None
    native: bool = False
    notes: str | None = None


_SPECS: dict[str, _ProviderSpec] = {
    "none": _ProviderSpec("None", (ProviderRole.COMPLETION,), None, None),
    "sentence-transformers": _ProviderSpec(
        "SentenceTransformers", (ProviderRole.EMBEDDING,), None, None,
        notes="Runs locally; model files may be downloaded by SentenceTransformers.",
    ),
    "openai": _ProviderSpec(
        "OpenAI", (ProviderRole.COMPLETION, ProviderRole.EMBEDDING),
        "OPENAI_API_KEY", "https://api.openai.com/v1",
    ),
    "openrouter": _ProviderSpec(
        "OpenRouter", (ProviderRole.COMPLETION, ProviderRole.EMBEDDING),
        "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1",
    ),
    "gemini": _ProviderSpec(
        "Google Gemini", (ProviderRole.COMPLETION, ProviderRole.EMBEDDING),
        "GEMINI_API_KEY", None, native=True,
    ),
    "anthropic": _ProviderSpec(
        "Anthropic", (ProviderRole.COMPLETION,), "ANTHROPIC_API_KEY",
        "https://api.anthropic.com", native=True,
    ),
    "deepinfra": _ProviderSpec(
        "DeepInfra", (ProviderRole.COMPLETION, ProviderRole.EMBEDDING),
        "DEEPINFRA_API_TOKEN", "https://api.deepinfra.com/v1/openai",
    ),
    "deepseek": _ProviderSpec(
        "DeepSeek", (ProviderRole.COMPLETION,), "DEEPSEEK_API_KEY",
        "https://api.deepseek.com",
    ),
    "kimi": _ProviderSpec(
        "Kimi", (ProviderRole.COMPLETION,), "MOONSHOT_API_KEY",
        "https://api.moonshot.ai/v1",
    ),
    "openai-compatible": _ProviderSpec(
        "OpenAI-compatible", (ProviderRole.COMPLETION, ProviderRole.EMBEDDING),
        "OPENAI_COMPATIBLE_API_KEY", None,
        notes="A custom /v1 endpoint. Its API key may be omitted for an unauthenticated service.",
    ),
}


_DEFAULT_MODELS = {
    ("openai", ProviderRole.COMPLETION): "gpt-4o-mini",
    ("openai", ProviderRole.EMBEDDING): "text-embedding-3-small",
    ("openrouter", ProviderRole.COMPLETION): "openai/gpt-4o-mini",
    ("openrouter", ProviderRole.EMBEDDING): "openai/text-embedding-3-small",
    ("gemini", ProviderRole.COMPLETION): "gemini-2.5-flash-lite",
    ("gemini", ProviderRole.EMBEDDING): "gemini-embedding-001",
    ("anthropic", ProviderRole.COMPLETION): "claude-sonnet-4-5",
    ("deepinfra", ProviderRole.COMPLETION): "meta-llama/Llama-3.3-70B-Instruct",
    ("deepinfra", ProviderRole.EMBEDDING): "BAAI/bge-m3",
    ("deepseek", ProviderRole.COMPLETION): "deepseek-chat",
    ("kimi", ProviderRole.COMPLETION): "kimi-k2.5",
    ("sentence-transformers", ProviderRole.EMBEDDING): "all-MiniLM-L6-v2",
}


def list_provider_descriptors() -> list[ProviderDescriptor]:
    return [
        ProviderDescriptor(
            id=provider_id,
            display_name=spec.display_name,
            roles=list(spec.roles),
            credential_env=spec.credential_env,
            credential_configured=credential_for(provider_id) is not None,
            default_base_url=spec.default_base_url,
            native_protocol=spec.native,
            notes=spec.notes,
        )
        for provider_id, spec in _SPECS.items()
    ]


def provider_descriptor(provider: str) -> ProviderDescriptor:
    provider_id = provider.strip().lower()
    for descriptor in list_provider_descriptors():
        if descriptor.id == provider_id:
            return descriptor
    raise ProviderError(f"unsupported provider: {provider}")


def credential_for(provider: str, explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    provider_id = provider.strip().lower()
    spec = _SPECS.get(provider_id)
    if spec is None or spec.credential_env is None:
        return None
    value = os.getenv(spec.credential_env)
    if not value and provider_id == "deepinfra":
        value = os.getenv("DEEPINFRA_TOKEN")
    return value or None


def validate_selection(
    selection: ProviderSelection,
    role: ProviderRole,
    *,
    api_key: str | None = None,
) -> None:
    descriptor = provider_descriptor(selection.provider)
    if role not in descriptor.roles:
        raise ProviderError(f"provider {selection.provider} does not support {role.value}")
    if selection.provider == "none":
        return
    if selection.provider == "openai-compatible" and not selection.base_url:
        raise ProviderError("openai-compatible requires an explicit /v1 base URL")
    if (
        descriptor.credential_env
        and selection.provider != "openai-compatible"
        and not credential_for(selection.provider, api_key)
    ):
        raise ProviderError(
            f"{descriptor.credential_env} is required for provider {selection.provider}"
        )


def selection_from_settings(settings: ProviderSettings, role: ProviderRole) -> ProviderSelection:
    provider = settings.provider.strip().lower()
    model = settings.model or _DEFAULT_MODELS.get((provider, role))
    return ProviderSelection(provider=provider, model=model, base_url=settings.base_url)


def settings_from_selection(selection: ProviderSelection) -> ProviderSettings:
    return ProviderSettings(
        provider=selection.provider,
        model=selection.model,
        base_url=selection.base_url,
    )


def provider_provenance(provider: Any | None, *, kind: str) -> dict[str, Any]:
    """Describe a configured provider without serializing credentials or endpoints."""

    if provider is None:
        return {"kind": kind, "provider": "none", "model": None, "implementation": None}
    settings = getattr(provider, "settings", None)
    provider_name = getattr(settings, "provider", None) or getattr(provider, "provider_name", None)
    model = getattr(provider, "model", None) or getattr(provider, "model_name", None)
    if provider_name is None:
        known_embeddings = {
            "SentenceTransformerEmbedder": "sentence-transformers",
            "GeminiEmbedder": "gemini",
            "OpenAICompatibleEmbedder": "openai-compatible",
        }
        provider_name = "custom" if kind == "completion" else known_embeddings.get(
            provider.__class__.__name__, "custom"
        )
    return {
        "kind": kind,
        "provider": str(provider_name),
        "model": str(model) if model else None,
        "implementation": f"{provider.__class__.__module__}.{provider.__class__.__qualname__}",
    }


def embedding_cache_key(provider: str, base_url: str | None, model: str) -> str:
    endpoint = (base_url or _SPECS[provider].default_base_url or "local").strip().rstrip("/").lower()
    endpoint_hash = hashlib.sha256(endpoint.encode("utf-8")).hexdigest()[:12]
    return f"{provider}:{endpoint_hash}:{model}"


class OpenAICompatibleCompletion:
    def __init__(self, settings: ProviderSettings) -> None:
        self.settings = settings
        self.model = settings.model or "gpt-4o-mini"
        spec = _SPECS.get(settings.provider, _SPECS["openai-compatible"])
        self.base_url = (settings.base_url or spec.default_base_url or "").rstrip("/")
        if not self.base_url:
            raise ProviderError("an OpenAI-compatible provider requires a base URL")
        self.api_key = settings.api_key

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        payload = openai_json_chat_payload(
            self.model,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        value = _post_json(
            f"{self.base_url}/chat/completions", payload, self.api_key,
            self.settings.timeout_seconds,
        )
        try:
            content = value["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("OpenAI-compatible completion returned an invalid response") from exc
        return parse_json_response(str(content))

    __call__ = complete_json

    def complete_multimodal(self, system: str, user: str, image: bytes, media_type: str) -> dict[str, Any]:
        encoded = base64.b64encode(image).decode("ascii")
        payload = openai_json_chat_payload(
            self.model,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": user},
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{encoded}"}},
                ]},
            ],
        )
        value = _post_json(
            f"{self.base_url}/chat/completions", payload, self.api_key,
            self.settings.timeout_seconds,
        )
        try:
            content = value["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("vision completion returned an invalid response") from exc
        return parse_json_response(str(content))


class AnthropicCompletion:
    def __init__(self, settings: ProviderSettings) -> None:
        self.settings = settings
        self.model = settings.model or _DEFAULT_MODELS[("anthropic", ProviderRole.COMPLETION)]
        self.base_url = (settings.base_url or _SPECS["anthropic"].default_base_url or "").rstrip("/")
        self.api_key = settings.api_key

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key or "",
            "anthropic-version": "2023-06-01",
        }

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "temperature": 0,
            "system": system + "\nReturn only a valid JSON object.",
            "messages": [{"role": "user", "content": user}],
        }
        value = _request_json(
            f"{self.base_url}/v1/messages", method="POST", payload=payload,
            headers=self._headers, timeout=self.settings.timeout_seconds,
        )
        return parse_json_response(_anthropic_text(value))

    __call__ = complete_json

    def complete_multimodal(self, system: str, user: str, image: bytes, media_type: str) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "temperature": 0,
            "system": system + "\nReturn only a valid JSON object.",
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": media_type,
                    "data": base64.b64encode(image).decode("ascii"),
                }},
                {"type": "text", "text": user},
            ]}],
        }
        value = _request_json(
            f"{self.base_url}/v1/messages", method="POST", payload=payload,
            headers=self._headers, timeout=self.settings.timeout_seconds,
        )
        return parse_json_response(_anthropic_text(value))


class GeminiCompletion:
    def __init__(self, settings: ProviderSettings) -> None:
        self.settings = settings
        self.model = settings.model or "gemini-2.5-flash-lite"
        self._local = threading.local()

    def _client(self) -> Any:
        client = getattr(self._local, "client", None)
        if client is None:
            from google import genai
            client = genai.Client(api_key=self.settings.api_key)
            self._local.client = client
        return client

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        from google.genai import types
        thinking = gemini_json_thinking_config(self.model, types)
        delay = 1.0
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                return self._generate_json(system, user, types, thinking)
            except Exception as exc:
                last_error = exc
                if thinking is not None and _unsupported_thinking_error(exc):
                    thinking = None
                    continue
                retryable = _retryable_provider_error(exc) or _invalid_json_error(exc)
                if attempt == 3 or not retryable:
                    raise ProviderError(f"Gemini completion failed: {exc}") from exc
                time.sleep(delay)
                delay = min(delay * 2, 16)
        raise ProviderError(f"Gemini completion failed: {last_error}") from last_error

    __call__ = complete_json

    def _generate_json(
        self,
        system: str,
        user: str,
        types: Any,
        thinking: Any | None,
        *,
        image: bytes | None = None,
        media_type: str | None = None,
    ) -> dict[str, Any]:
        contents: Any = user
        if image is not None and media_type is not None:
            contents = [user, types.Part.from_bytes(data=image, mime_type=media_type)]
        config_kwargs: dict[str, Any] = {
            "system_instruction": system,
            "temperature": 0,
            "response_mime_type": "application/json",
        }
        if thinking is not None:
            config_kwargs["thinking_config"] = thinking
        response = self._client().models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        return parse_json_response(response.text or "")

    def complete_multimodal(self, system: str, user: str, image: bytes, media_type: str) -> dict[str, Any]:
        try:
            from google.genai import types
            return self._generate_json(
                system, user, types, gemini_json_thinking_config(self.model, types),
                image=image, media_type=media_type,
            )
        except Exception as exc:
            raise ProviderError(f"Gemini vision completion failed: {exc}") from exc


def completion_from_settings(settings: ProviderSettings | None) -> Any | None:
    if settings is None or settings.provider in {"", "none"}:
        return None
    provider = settings.provider.strip().lower()
    selection = selection_from_settings(settings, ProviderRole.COMPLETION)
    validate_selection(selection, ProviderRole.COMPLETION, api_key=settings.api_key)
    key = credential_for(provider, settings.api_key)
    normalized = ProviderSettings(
        provider=provider, model=selection.model, api_key=key,
        base_url=selection.base_url, timeout_seconds=settings.timeout_seconds,
    )
    if provider == "gemini":
        return GeminiCompletion(normalized)
    if provider == "anthropic":
        return AnthropicCompletion(normalized)
    return OpenAICompatibleCompletion(normalized)


def embedder_from_settings(settings: ProviderSettings | None) -> Any:
    if settings is None or settings.provider in {"local", "sentence-transformers", "sbert"}:
        return _local_embedder(settings)
    provider = settings.provider.strip().lower()
    selection = selection_from_settings(settings, ProviderRole.EMBEDDING)
    validate_selection(selection, ProviderRole.EMBEDDING, api_key=settings.api_key)
    key = credential_for(provider, settings.api_key)
    if provider == "gemini":
        return _gemini_embedder(selection, key)
    return _compatible_embedder(provider, settings, selection, key)


def _stamp_embedder(value: Any, provider: str, base_url: str | None = None) -> Any:
    value.provider_name = provider
    value.cache_key = embedding_cache_key(provider, base_url, value.model_name)
    return value


def _local_embedder(settings: ProviderSettings | None) -> Any:
    model = settings.model if settings and settings.model else "all-MiniLM-L6-v2"
    return _stamp_embedder(SentenceTransformerEmbedder(model), "sentence-transformers")


def _gemini_embedder(selection: ProviderSelection, key: str | None) -> Any:
    value = GeminiEmbedder(model_name=selection.model or "gemini-embedding-001", api_key=key)
    return _stamp_embedder(value, "gemini")


def _compatible_embedder(
    provider: str,
    settings: ProviderSettings,
    selection: ProviderSelection,
    key: str | None,
) -> Any:
    spec = _SPECS[provider]
    base_url = selection.base_url or spec.default_base_url
    if not base_url:
        raise ProviderError(f"provider {provider} requires a base URL")
    value = OpenAICompatibleEmbedder(
        model_name=selection.model or "text-embedding-3-small",
        base_url=base_url,
        api_key=key or "",
        timeout_seconds=settings.timeout_seconds,
    )
    return _stamp_embedder(value, provider, base_url)


def fetch_model_catalog(
    provider: str,
    role: ProviderRole,
    *,
    base_url: str | None = None,
    timeout_seconds: float = 30,
) -> list[ModelDescriptor]:
    """Fetch a non-billable model catalog from the provider's official API."""

    provider_id = provider.strip().lower()
    descriptor = provider_descriptor(provider_id)
    if role not in descriptor.roles:
        raise ProviderError(f"provider {provider_id} does not support {role.value}")
    if provider_id == "none":
        return []
    if provider_id == "sentence-transformers":
        return _sentence_transformers_catalog()
    key = credential_for(provider_id)
    _require_catalog_credential(descriptor, provider_id, key)
    models = _catalog_for_provider(provider_id, role, descriptor, key, base_url, timeout_seconds)
    return [item for item in models if role in item.roles]


def _sentence_transformers_catalog() -> list[ModelDescriptor]:
    return [ModelDescriptor(
        provider="sentence-transformers", id="all-MiniLM-L6-v2", display_name="all-MiniLM-L6-v2",
        roles=[ProviderRole.EMBEDDING], source="curated", capability_status="verified",
    )]


def _require_catalog_credential(
    descriptor: ProviderDescriptor, provider_id: str, key: str | None,
) -> None:
    if descriptor.credential_env and provider_id != "openai-compatible" and not key:
        raise ProviderError(f"{descriptor.credential_env} is required to list {provider_id} models")


def _catalog_for_provider(
    provider_id: str,
    role: ProviderRole,
    descriptor: ProviderDescriptor,
    key: str | None,
    base_url: str | None,
    timeout_seconds: float,
) -> list[ModelDescriptor]:
    catalogs = {
        "gemini": lambda: _gemini_models(role, key),
        "anthropic": lambda: _anthropic_catalog(descriptor, key, base_url, timeout_seconds),
        "deepinfra": lambda: _deepinfra_catalog(key, timeout_seconds),
    }
    fetch = catalogs.get(
        provider_id,
        lambda: _openai_compatible_catalog(provider_id, descriptor, key, base_url, timeout_seconds),
    )
    return fetch()


def _anthropic_catalog(
    descriptor: ProviderDescriptor,
    key: str | None,
    base_url: str | None,
    timeout_seconds: float,
) -> list[ModelDescriptor]:
    url = f"{(base_url or descriptor.default_base_url).rstrip('/')}/v1/models?{urlencode({'limit': 1000})}"
    payload = _request_json(url, headers={
        "x-api-key": key or "", "anthropic-version": "2023-06-01",
    }, timeout=timeout_seconds)
    return [_anthropic_model(item) for item in payload.get("data", [])]


def _deepinfra_raw_models(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    return payload.get("data") or payload.get("models") or []


def _deepinfra_catalog(key: str | None, timeout_seconds: float) -> list[ModelDescriptor]:
    payload = _request_json(
        "https://api.deepinfra.com/models/list",
        headers={"Authorization": f"Bearer {key}"}, timeout=timeout_seconds,
    )
    return [_compatible_model("deepinfra", item) for item in _deepinfra_raw_models(payload)]


def _map_catalog_models(provider_id: str, raw_models: list[Any]) -> list[ModelDescriptor]:
    if provider_id == "openai":
        return [model for item in raw_models if (model := _openai_model(item)) is not None]
    return [_compatible_model(provider_id, item) for item in raw_models]


def _openai_compatible_catalog(
    provider_id: str,
    descriptor: ProviderDescriptor,
    key: str | None,
    base_url: str | None,
    timeout_seconds: float,
) -> list[ModelDescriptor]:
    endpoint = base_url or descriptor.default_base_url
    if not endpoint:
        raise ProviderError("openai-compatible model discovery requires a base URL")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    payload = _request_json(
        f"{endpoint.rstrip('/')}/models", headers=headers, timeout=timeout_seconds,
    )
    raw_models = payload.get("data") or payload.get("models") or []
    return _map_catalog_models(provider_id, raw_models)


def _openai_model(item: Any) -> ModelDescriptor | None:
    data = item if isinstance(item, dict) else {"id": str(item)}
    model_id = str(data.get("id") or "").strip()
    lowered = model_id.lower()
    if lowered.startswith("text-embedding-"):
        roles = [ProviderRole.EMBEDDING]
        vision = None
        structured = None
    elif lowered.startswith(("gpt-", "chatgpt-", "o1", "o3", "o4")):
        roles = [ProviderRole.COMPLETION]
        vision = True if lowered.startswith(("gpt-4o", "gpt-4.1", "gpt-5")) else None
        structured = True
    else:
        return None
    return ModelDescriptor(
        provider="openai", id=model_id, display_name=model_id, roles=roles,
        vision=vision, structured_output=structured, source="live",
        capability_status="verified", metadata={"catalog": "openai+trisynapse-capabilities"},
    )


def _catalog_item_data(item: Any) -> dict[str, Any]:
    return item if isinstance(item, dict) else {"id": str(item)}


def _catalog_model_id(data: dict[str, Any]) -> str:
    return str(data.get("id") or data.get("model_name") or data.get("name") or "").strip()


def _compatible_roles(provider: str, data: dict[str, Any], text: str) -> list[ProviderRole]:
    embedding = any(token in text for token in ("embedding", "embeddings", "bge-", "e5-", "gte-"))
    if embedding:
        return [ProviderRole.EMBEDDING]
    if (
        provider in {"openai", "openrouter", "deepinfra", "openai-compatible"}
        and data.get("type") in {"embeddings", "embedding"}
    ):
        return [ProviderRole.EMBEDDING]
    return [ProviderRole.COMPLETION]


def _compatible_capability_fields(
    data: dict[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    architecture = data.get("architecture") if isinstance(data.get("architecture"), dict) else {}
    modality = str(architecture.get("modality") or data.get("modality") or "").lower()
    capabilities = data.get("capabilities") if isinstance(data.get("capabilities"), dict) else {}
    return architecture, modality, capabilities


def _name_suggests_vision(modality: str, model_name: str) -> bool:
    return any(token in modality or token in model_name for token in ("image", "vision", "vl-"))


def _compatible_vision(
    data: dict[str, Any],
    architecture: dict[str, Any],
    capabilities: dict[str, Any],
    model_id: str,
    modality: str,
) -> bool | None:
    _ = architecture
    if "supports_image_in" in data:
        return bool(data["supports_image_in"])
    if "image_input" in capabilities:
        return bool(capabilities["image_input"])
    if _name_suggests_vision(modality, model_id.lower()):
        return True
    return None


def _catalog_display_name(data: dict[str, Any], model_id: str) -> str:
    return str(data.get("display_name") or data.get("name") or model_id)


def _catalog_context_length(data: dict[str, Any]) -> int | None:
    context = data.get("context_length") or data.get("context_window")
    if isinstance(context, (int, float)):
        return int(context)
    return None


def _compatible_capability_status(
    data: dict[str, Any], architecture: dict[str, Any], modality: str,
) -> str:
    if data.get("type") or architecture or modality:
        return "verified"
    return "unknown"


def _completion_structured_output(roles: list[ProviderRole]) -> bool | None:
    if ProviderRole.COMPLETION in roles:
        return True
    return None


def _catalog_metadata(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if key not in {"pricing"}}


def _compatible_model(provider: str, item: Any) -> ModelDescriptor:
    data = _catalog_item_data(item)
    model_id = _catalog_model_id(data)
    architecture, modality, capabilities = _compatible_capability_fields(data)
    roles = _compatible_roles(provider, data, json.dumps(data, default=str).lower())
    return ModelDescriptor(
        provider=provider,
        id=model_id,
        display_name=_catalog_display_name(data, model_id),
        roles=roles,
        vision=_compatible_vision(data, architecture, capabilities, model_id, modality),
        structured_output=_completion_structured_output(roles),
        context_length=_catalog_context_length(data),
        source="live",
        capability_status=_compatible_capability_status(data, architecture, modality),
        metadata=_catalog_metadata(data),
    )


def _anthropic_model(item: dict[str, Any]) -> ModelDescriptor:
    capabilities = item.get("capabilities") if isinstance(item.get("capabilities"), dict) else {}
    return ModelDescriptor(
        provider="anthropic", id=str(item["id"]),
        display_name=str(item.get("display_name") or item["id"]),
        roles=[ProviderRole.COMPLETION],
        vision=capabilities.get("image_input"),
        structured_output=capabilities.get("structured_outputs", True),
        context_length=item.get("context_window"), source="live",
        capability_status="verified", metadata=capabilities,
    )


def _gemini_model_roles(item: Any) -> tuple[list[ProviderRole], list[str]]:
    actions = list(getattr(item, "supported_actions", None) or [])
    roles: list[ProviderRole] = []
    if "generateContent" in actions:
        roles.append(ProviderRole.COMPLETION)
    if "embedContent" in actions:
        roles.append(ProviderRole.EMBEDDING)
    return roles, actions


def _gemini_model_descriptor(item: Any) -> ModelDescriptor | None:
    roles, actions = _gemini_model_roles(item)
    if not roles:
        return None
    model_id = str(getattr(item, "name", "")).removeprefix("models/")
    completion = ProviderRole.COMPLETION in roles
    return ModelDescriptor(
        provider="gemini", id=model_id,
        display_name=str(getattr(item, "display_name", None) or model_id),
        roles=roles,
        vision=True if completion else None,
        structured_output=True if completion else None,
        context_length=getattr(item, "input_token_limit", None),
        source="live", capability_status="verified",
        metadata={"supported_actions": actions},
    )


def _gemini_models(role: ProviderRole, api_key: str | None) -> list[ModelDescriptor]:
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        result: list[ModelDescriptor] = []
        for item in client.models.list():
            descriptor = _gemini_model_descriptor(item)
            if descriptor is None or role not in descriptor.roles:
                continue
            result.append(descriptor)
        return result
    except Exception as exc:
        raise ProviderError(f"Gemini model discovery failed: {exc}") from exc


def _anthropic_text(payload: dict[str, Any]) -> str:
    for block in payload.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            return str(block.get("text") or "")
    raise ProviderError("Anthropic completion returned no text content")


def _post_json(url: str, payload: dict[str, Any], api_key: str | None, timeout: float) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    return _request_json(url, method="POST", payload=payload, headers=headers, timeout=timeout)


def _request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 60,
) -> dict[str, Any] | list[Any]:
    request_headers = _json_request_headers(headers)
    data = _request_body(payload, request_headers)
    return _request_json_with_retries(url, data, request_headers, method, timeout)


def _json_request_headers(headers: dict[str, str] | None) -> dict[str, str]:
    return {"Accept": "application/json", **(headers or {})}


def _should_retry_request(retryable: bool, attempt: int, limit: int) -> bool:
    return retryable and attempt + 1 < limit


def _request_json_with_retries(
    url: str,
    data: bytes | None,
    headers: dict[str, str],
    method: str,
    timeout: float,
) -> dict[str, Any] | list[Any]:
    delay = 2.0
    attempts = 6
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return _read_json_url(url, data, headers, method, timeout)
        except HTTPError as exc:
            last_error, retryable, wait = _http_error_retry(exc, delay)
            if not _should_retry_request(retryable, attempt, attempts):
                raise last_error from exc
            time.sleep(min(wait, 60.0))
            delay = min(delay * 2, 32.0)
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error, retryable, limit = _transport_error_retry(exc, attempts)
            if not _should_retry_request(retryable, attempt, limit):
                raise last_error from exc
            time.sleep(delay)
            delay = min(delay * 2, 32.0)
    raise last_error or ProviderError("provider request failed")


def _request_body(payload: dict[str, Any] | None, headers: dict[str, str]) -> bytes | None:
    if payload is None:
        return None
    headers["Content-Type"] = "application/json"
    return json.dumps(payload).encode("utf-8")


def _read_json_url(
    url: str, data: bytes | None, headers: dict[str, str], method: str, timeout: float,
) -> dict[str, Any] | list[Any]:
    request = Request(url, data=data, headers=headers, method=method)
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - configured provider URL.
        return json.loads(response.read().decode("utf-8"))


def _http_error_retry(exc: HTTPError, delay: float) -> tuple[ProviderError, bool, float]:
    detail = exc.read().decode("utf-8", errors="replace")[:1000]
    error = ProviderError(f"provider HTTP {exc.code}: {detail}")
    retryable = exc.code in {429, 502, 503} or _retryable_provider_error(error)
    retry_after = exc.headers.get("Retry-After") if exc.headers else None
    if retry_after and str(retry_after).isdigit():
        return error, retryable, float(retry_after)
    return error, retryable, delay


def _transport_error_retry(exc: Exception, attempts: int) -> tuple[ProviderError, bool, int]:
    error = ProviderError(f"provider request failed: {exc}")
    is_timeout = isinstance(exc, TimeoutError) or "timed out" in str(exc).lower()
    retryable = is_timeout or _retryable_provider_error(exc)
    limit = 3 if is_timeout else attempts
    return error, retryable, limit
