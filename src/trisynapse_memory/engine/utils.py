"""Shared numerical and package utilities used across the engine."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import tomllib

import numpy as np


def _source_tree_version() -> str | None:
    """Read the canonical version when running directly from a source checkout."""

    pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"
    if not pyproject.is_file():
        return None
    with pyproject.open("rb") as stream:
        value = tomllib.load(stream).get("project", {}).get("version")
    return str(value) if value else None


def get_version() -> str:
    """Return the source version or installed distribution metadata."""

    source_version = _source_tree_version()
    if source_version is not None:
        return source_version
    try:
        return version("trisynapse-memory")
    except PackageNotFoundError:
        return "0+unknown"


__version__ = get_version()


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity for one pair, or zero for invalid vectors."""

    if len(left) == 0 or len(left) != len(right):
        return 0.0
    matrix = np.asarray([left, right], dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        return 0.0
    normalized = normalize_rows(matrix)
    return float(np.dot(normalized[0], normalized[1]))


def cosine_scores(query: Sequence[float], candidates: Sequence[Sequence[float]]) -> np.ndarray:
    """Vectorized cosine scores for one query against many candidates."""

    if len(query) == 0 or len(candidates) == 0:
        return np.empty(0, dtype=np.float32)
    query_array = np.asarray(query, dtype=np.float32)
    matrix = np.asarray(candidates, dtype=np.float32)
    if query_array.ndim != 1 or matrix.ndim != 2 or matrix.shape[1] != query_array.shape[0]:
        raise ValueError("candidate embeddings must share the query vector dimension")
    normalized_query = normalize_vector(query_array)
    normalized_candidates = normalize_rows(matrix)
    return normalized_candidates @ normalized_query


def normalize_vector(vector: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return one float32 unit vector; invalid/zero vectors become zeros."""

    value = np.asarray(vector, dtype=np.float32)
    if value.ndim != 1:
        raise ValueError("expected one vector")
    norm = np.linalg.norm(value)
    if not np.isfinite(norm) or norm <= 0:
        return np.zeros_like(value)
    return value / norm


def normalize_rows(vectors: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Return a float32 matrix with every valid row normalized once."""

    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("expected a vector matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(
        matrix,
        norms,
        out=np.zeros_like(matrix),
        where=np.isfinite(norms) & (norms > 0),
    )


def bm25_term_score(
    *,
    term_frequency: int,
    document_length: int,
    average_document_length: float,
    document_count: int,
    document_frequency: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Return the standard Okapi BM25 contribution for one query term."""

    if term_frequency <= 0 or document_count <= 0:
        return 0.0
    inverse_document_frequency = math.log(
        1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
    )
    denominator = term_frequency + k1 * (
        1 - b + b * document_length / max(average_document_length, 1.0)
    )
    return inverse_document_frequency * (term_frequency * (k1 + 1)) / denominator


def bm25_document_score(
    query_terms: Iterable[str],
    document_terms: Sequence[str],
    *,
    average_document_length: float,
    document_frequencies: Mapping[str, int],
    document_count: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Score one tokenized document with the same formula as SQLite BM25."""

    frequencies = Counter(document_terms)
    return sum(
        bm25_term_score(
            term_frequency=frequencies.get(term, 0),
            document_length=len(document_terms),
            average_document_length=average_document_length,
            document_count=document_count,
            document_frequency=document_frequencies.get(term, 0),
            k1=k1,
            b=b,
        )
        for term in set(query_terms)
    )


def reciprocal_rank_fusion(
    rankings: Iterable[tuple[str, Sequence[str]]],
    *,
    weights: Mapping[str, float] | None = None,
    rank_constant: int = 60,
) -> dict[str, float]:
    """Fuse rankings with the standard weighted RRF formula.

    score(document) = sum(weight(route) / (rank_constant + rank + 1))
    """

    route_weights = weights or {}
    scores: dict[str, float] = defaultdict(float)
    for route, ranking in rankings:
        weight = float(route_weights.get(route, 1.0))
        if weight <= 0:
            continue
        for rank, item_id in enumerate(ranking):
            scores[item_id] += weight / (rank_constant + rank + 1)
    return dict(scores)
