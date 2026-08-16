"""Completion, embedding, model-discovery, and provider registry support."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from trisynapse_memory.engine.compilation import parse_json_response
from trisynapse_memory.engine.embedding import (
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
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
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
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": user},
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{encoded}"}},
                ]},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
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

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=self.settings.api_key)
            response = client.models.generate_content(
                model=self.model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0,
                    response_mime_type="application/json",
                ),
            )
            return parse_json_response(response.text or "")
        except Exception as exc:
            raise ProviderError(f"Gemini completion failed: {exc}") from exc

    __call__ = complete_json

    def complete_multimodal(self, system: str, user: str, image: bytes, media_type: str) -> dict[str, Any]:
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=self.settings.api_key)
            response = client.models.generate_content(
                model=self.model,
                contents=[user, types.Part.from_bytes(data=image, mime_type=media_type)],
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0,
                    response_mime_type="application/json",
                ),
            )
            return parse_json_response(response.text or "")
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
        model = settings.model if settings and settings.model else "all-MiniLM-L6-v2"
        value = SentenceTransformerEmbedder(model)
        value.provider_name = "sentence-transformers"
        value.cache_key = embedding_cache_key("sentence-transformers", None, model)
        return value
    provider = settings.provider.strip().lower()
    selection = selection_from_settings(settings, ProviderRole.EMBEDDING)
    validate_selection(selection, ProviderRole.EMBEDDING, api_key=settings.api_key)
    key = credential_for(provider, settings.api_key)
    if provider == "gemini":
        value = GeminiEmbedder(model_name=selection.model or "gemini-embedding-001", api_key=key)
        value.provider_name = "gemini"
        value.cache_key = embedding_cache_key("gemini", None, value.model_name)
        return value
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
    value.provider_name = provider
    value.cache_key = embedding_cache_key(provider, base_url, value.model_name)
    return value


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
        return [ModelDescriptor(
            provider=provider_id, id="all-MiniLM-L6-v2", display_name="all-MiniLM-L6-v2",
            roles=[ProviderRole.EMBEDDING], source="curated", capability_status="verified",
        )]
    key = credential_for(provider_id)
    if descriptor.credential_env and provider_id != "openai-compatible" and not key:
        raise ProviderError(f"{descriptor.credential_env} is required to list {provider_id} models")
    if provider_id == "gemini":
        return _gemini_models(role, key)
    if provider_id == "anthropic":
        url = f"{(base_url or descriptor.default_base_url).rstrip('/')}/v1/models?{urlencode({'limit': 1000})}"
        payload = _request_json(url, headers={
            "x-api-key": key or "", "anthropic-version": "2023-06-01",
        }, timeout=timeout_seconds)
        return [_anthropic_model(item) for item in payload.get("data", [])]
    if provider_id == "deepinfra":
        payload = _request_json(
            "https://api.deepinfra.com/models/list",
            headers={"Authorization": f"Bearer {key}"}, timeout=timeout_seconds,
        )
        raw_models = payload if isinstance(payload, list) else payload.get("data") or payload.get("models") or []
    else:
        endpoint = base_url or descriptor.default_base_url
        if not endpoint:
            raise ProviderError("openai-compatible model discovery requires a base URL")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        payload = _request_json(f"{endpoint.rstrip('/')}/models", headers=headers, timeout=timeout_seconds)
        raw_models = payload.get("data") or payload.get("models") or []
    if provider_id == "openai":
        models = [model for item in raw_models if (model := _openai_model(item)) is not None]
    else:
        models = [_compatible_model(provider_id, item) for item in raw_models]
    return [item for item in models if role in item.roles]


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


def _compatible_model(provider: str, item: Any) -> ModelDescriptor:
    data = item if isinstance(item, dict) else {"id": str(item)}
    model_id = str(data.get("id") or data.get("model_name") or data.get("name") or "").strip()
    text = json.dumps(data, default=str).lower()
    embedding = any(token in text for token in ("embedding", "embeddings", "bge-", "e5-", "gte-"))
    roles = [ProviderRole.EMBEDDING] if embedding else [ProviderRole.COMPLETION]
    if provider in {"openai", "openrouter", "deepinfra", "openai-compatible"} and not embedding:
        if data.get("type") in {"embeddings", "embedding"}:
            roles = [ProviderRole.EMBEDDING]
    architecture = data.get("architecture") if isinstance(data.get("architecture"), dict) else {}
    modality = str(architecture.get("modality") or data.get("modality") or "").lower()
    capabilities = data.get("capabilities") if isinstance(data.get("capabilities"), dict) else {}
    if "supports_image_in" in data:
        vision = bool(data["supports_image_in"])
    elif "image_input" in capabilities:
        vision = bool(capabilities["image_input"])
    else:
        model_name = model_id.lower()
        vision = True if any(
            token in modality or token in model_name for token in ("image", "vision", "vl-")
        ) else None
    context = data.get("context_length") or data.get("context_window")
    return ModelDescriptor(
        provider=provider,
        id=model_id,
        display_name=str(data.get("display_name") or data.get("name") or model_id),
        roles=roles,
        vision=vision,
        structured_output=True if ProviderRole.COMPLETION in roles else None,
        context_length=int(context) if isinstance(context, (int, float)) else None,
        source="live",
        capability_status="verified" if data.get("type") or architecture or modality else "unknown",
        metadata={key: value for key, value in data.items() if key not in {"pricing"}},
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


def _gemini_models(role: ProviderRole, api_key: str | None) -> list[ModelDescriptor]:
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        result: list[ModelDescriptor] = []
        for item in client.models.list():
            actions = list(getattr(item, "supported_actions", None) or [])
            roles: list[ProviderRole] = []
            if "generateContent" in actions:
                roles.append(ProviderRole.COMPLETION)
            if "embedContent" in actions:
                roles.append(ProviderRole.EMBEDDING)
            if role not in roles:
                continue
            model_id = str(getattr(item, "name", "")).removeprefix("models/")
            result.append(ModelDescriptor(
                provider="gemini", id=model_id,
                display_name=str(getattr(item, "display_name", None) or model_id),
                roles=roles,
                vision=True if ProviderRole.COMPLETION in roles else None,
                structured_output=True if ProviderRole.COMPLETION in roles else None,
                context_length=getattr(item, "input_token_limit", None),
                source="live", capability_status="verified",
                metadata={"supported_actions": actions},
            ))
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
    request_headers = {"Accept": "application/json", **(headers or {})}
    data = None
    if payload is not None:
        request_headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers=request_headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - configured provider URL.
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise ProviderError(f"provider HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ProviderError(f"provider request failed: {exc}") from exc
