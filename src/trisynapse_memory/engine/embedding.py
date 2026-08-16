"""Strict local embedding provider used by hybrid retrieval."""

from __future__ import annotations

import json
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class Embedder(Protocol):
    model_name: str

    def encode(self, texts: list[str]) -> list[list[float]]: ...


class EmbeddingProviderError(RuntimeError):
    """Raised when the configured real embedding model cannot be used."""


class UnavailableEmbedder:
    """Keeps a store open while its selected provider credential is unavailable."""

    def __init__(self, model_name: str, cache_key: str, error: str) -> None:
        self.model_name = model_name
        self.cache_key = cache_key
        self.error = error
        self.provider_name = "unavailable"

    def encode(self, texts: list[str]) -> list[list[float]]:
        del texts
        raise EmbeddingProviderError(self.error)


class SentenceTransformerEmbedder:
    """Lazy SentenceTransformers wrapper with no substitute-vector fallback."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", *, local_files_only: bool = False) -> None:
        self.model_name = model_name
        self.local_files_only = local_files_only
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer

            # Cache-first avoids an unnecessary network metadata request when
            # the complete model is already present. If it is not cached and
            # downloads are allowed, retry the same real model online.
            try:
                self._model = SentenceTransformer(self.model_name, local_files_only=True)
            except Exception:
                if self.local_files_only:
                    raise
                self._model = SentenceTransformer(self.model_name, local_files_only=False)
        except Exception as exc:
            raise EmbeddingProviderError(
                f"Could not load SentenceTransformers model '{self.model_name}'. "
                "Install sentence-transformers and make the model available; "
                "trisynapse-memory never substitutes fake embeddings. "
                f"Original error: {exc}"
            ) from exc
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            values = self._load().encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return [list(map(float, vector)) for vector in values]
        except EmbeddingProviderError:
            raise
        except Exception as exc:
            raise EmbeddingProviderError(
                f"SentenceTransformers embedding failed for model '{self.model_name}': {exc}"
            ) from exc


class OpenAICompatibleEmbedder:
    """Embedding client for OpenAI and compatible HTTP APIs."""

    def __init__(self, *, model_name: str, base_url: str, api_key: str, timeout_seconds: float = 60) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        request = Request(
            f"{self.base_url}/embeddings",
            data=json.dumps({"model": self.model_name, "input": texts}).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310 - configured endpoint.
                payload = json.loads(response.read().decode("utf-8"))
            ordered = sorted(payload["data"], key=lambda item: item.get("index", 0))
            vectors = [list(map(float, item["embedding"])) for item in ordered]
        except (HTTPError, URLError, TimeoutError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise EmbeddingProviderError(f"OpenAI-compatible embedding failed for '{self.model_name}': {exc}") from exc
        if len(vectors) != len(texts):
            raise EmbeddingProviderError("embedding provider returned the wrong number of vectors")
        return vectors


class GeminiEmbedder:
    def __init__(self, model_name: str = "gemini-embedding-001", *, api_key: str | None = None) -> None:
        self.model_name = model_name
        self.api_key = api_key

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.api_key:
            raise EmbeddingProviderError("GEMINI_API_KEY is required for Gemini embeddings")
        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)
            response = client.models.embed_content(model=self.model_name, contents=texts)
            vectors = [list(map(float, item.values)) for item in response.embeddings]
        except Exception as exc:
            raise EmbeddingProviderError(f"Gemini embedding failed for '{self.model_name}': {exc}") from exc
        if len(vectors) != len(texts):
            raise EmbeddingProviderError("Gemini returned the wrong number of embeddings")
        return vectors
