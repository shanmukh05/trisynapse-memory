"""Read-only Recall inspection used by Studio Memory Viewer."""

from __future__ import annotations

from typing import Any

import numpy as np

from trisynapse_memory.engine.models import (
    CompiledClaim,
    MemoryCatalog,
    MemoryCatalogHelper,
    MemoryCatalogRoute,
    MemoryDocumentPage,
    MemoryDocumentRow,
    MemoryGraphEdge,
    MemoryGraphNode,
    MemoryGraphPage,
    MemoryHelperItem,
    MemoryHelperPage,
    MemoryNamespace,
    MemoryTerm,
    MemoryTermPage,
    MemoryTermPosting,
    VectorNeighbor,
    VectorNeighbors,
    VectorProjection,
    VectorProjectionPoint,
)
from trisynapse_memory.engine.providers.registry import embedding_cache_key
from trisynapse_memory.engine.recall.catalog import BUILTIN_RECALL_HELPERS, builtin_helper_ids, helper_spec
from trisynapse_memory.engine.recall.compilation import compile_claims
from trisynapse_memory.engine.retrieval.contracts import DEFAULT_ROUTE_WEIGHTS, RouteRegistry, route_weights


def memory_catalog(engine: Any, namespace: MemoryNamespace) -> MemoryCatalog:
    counts = engine.store.retrieval_index_counts(namespace)
    deltas = engine.store.list_deltas(namespace=namespace, include_retracted=True)
    observations = {item.id: item for item in deltas if item.kind == "observation"}
    claims = compile_claims([item for item in deltas if item.kind == "extraction"], observations)
    episodes = engine.store.episode_recall_views(namespace=namespace)
    cache_key = str(getattr(engine.embedder, "cache_key", getattr(engine.embedder, "model_name", "")))
    active_docs, _total_docs = _document_hashes(engine, namespace, limit=2_000)
    embedded = 0
    if active_docs:
        embedded = len(engine.vector_cache.get(list(active_docs.values()), cache_key))
    contested = sum(1 for item in claims if item.status == "CONTESTED")
    stale_episodes = sum(1 for item in episodes if item.stale)
    helpers: list[MemoryCatalogHelper] = []
    health_by_id = {
        "trace": {"count": len(deltas), "observations": sum(1 for item in deltas if item.kind == "observation")},
        "documents": {
            "count": counts["documents_active"],
            "inactive": counts["documents"] - counts["documents_active"],
        },
        "bm25": {"count": counts["distinct_terms"], "postings": counts["postings"]},
        "vectors": {
            "count": embedded,
            "searchable": len(active_docs),
            "coverage": round(embedded / len(active_docs), 3) if active_docs else 0.0,
            "fingerprint": cache_key,
        },
        "episodes": {"count": len(episodes), "stale": stale_episodes, "fresh": len(episodes) - stale_episodes},
        "claims": {"count": len(claims), "contested": contested},
        "graph": {"count": sum(counts["graph_edges"].values()), "edges": counts["graph_edges"]},
    }
    count_by_id = {
        "trace": len(deltas),
        "documents": counts["documents_active"],
        "bm25": counts["distinct_terms"],
        "vectors": embedded,
        "episodes": len(episodes),
        "claims": len(claims),
        "graph": sum(counts["graph_edges"].values()),
    }
    for spec in BUILTIN_RECALL_HELPERS:
        helpers.append(MemoryCatalogHelper(
            id=spec.id,
            title=spec.title,
            kind=spec.kind,
            inspect_path=spec.inspect_path,
            playground_seed=spec.playground_seed,
            count=count_by_id.get(spec.id, 0),
            health=health_by_id.get(spec.id, {}),
        ))
    known = builtin_helper_ids()
    for view_type, extra_count in engine.store.recall_view_type_counts(namespace).items():
        if view_type in known or view_type == "episode_recall":
            continue
        helpers.append(MemoryCatalogHelper(
            id=view_type,
            title=view_type.replace("_", " ").title(),
            kind="cards",
            inspect_path=f"/api/v1/memory/helpers/{view_type}",
            count=extra_count,
            health={"count": extra_count},
        ))
    registry = engine.retrieval_routes or RouteRegistry()
    configuration = engine.get_retrieval_configuration()
    weights = route_weights(configuration.retrieval_profile, configuration.route_weights)
    enabled = set(configuration.enabled_routes)
    routes = [
        MemoryCatalogRoute(
            name=name,
            title=name.replace("_", " "),
            enabled=name in enabled,
            weight=float(weights.get(name, DEFAULT_ROUTE_WEIGHTS.get(name, 1.0))),
        )
        for name in registry.names
    ]
    return MemoryCatalog(helpers=helpers, retrieval_routes=routes)


def helper_items(
    engine: Any,
    helper_id: str,
    namespace: MemoryNamespace,
    *,
    search: str | None = None,
    cursor: str | None = None,
    limit: int = 50,
) -> MemoryHelperPage:
    spec = helper_spec(helper_id)
    offset = int(cursor or 0)
    if helper_id == "trace":
        page = engine.list(namespace=namespace, cursor=offset or None, limit=limit, include_retracted=True)
        items = [
            MemoryHelperItem(
                id=item.id,
                helper_id=helper_id,
                kind=item.kind,
                title=(item.text or item.kind)[:120],
                subtitle=item.episode_id,
                excerpt=item.text[:400],
                status="retracted" if item.id in engine.store.retracted_ids() else "active",
                data={"seq": item.seq, "locator": item.locator, "source_ref": item.source_ref},
            )
            for item in page.items
        ]
        return MemoryHelperPage(
            helper_id=helper_id, kind=spec.kind if spec else "timeline", items=items,
            next_cursor=str(page.next_cursor) if page.next_cursor is not None else None,
            truncated=page.next_cursor is not None,
        )
    if helper_id == "documents":
        documents, total = engine.store.list_retrieval_documents_page(
            namespace, search=search, cursor=offset, limit=limit
        )
        items = [
            MemoryHelperItem(
                id=item["id"], helper_id=helper_id, kind="document",
                title=item["text"][:120], subtitle=item["modality"], excerpt=item["text"][:400],
                status="active" if item["active"] else "inactive",
                data=item,
            )
            for item in documents
        ]
        next_cursor = offset + limit if offset + limit < total else None
        return MemoryHelperPage(
            helper_id=helper_id, kind="table", items=items,
            next_cursor=str(next_cursor) if next_cursor is not None else None,
            truncated=next_cursor is not None,
        )
    if helper_id == "bm25":
        terms, total = engine.store.list_retrieval_terms_page(
            namespace, search=search, cursor=offset, limit=limit
        )
        items = [
            MemoryHelperItem(
                id=item["term"], helper_id=helper_id, kind="term",
                title=item["term"], subtitle=f"{item['document_frequency']} documents",
                status="indexed", data=item, score=float(item["document_frequency"]),
            )
            for item in terms
        ]
        next_cursor = offset + limit if offset + limit < total else None
        return MemoryHelperPage(
            helper_id=helper_id, kind="postings", items=items,
            next_cursor=str(next_cursor) if next_cursor is not None else None,
            truncated=next_cursor is not None,
        )
    if helper_id == "claims":
        claims = _claims(engine, namespace)
        if search:
            term = search.lower()
            claims = [item for item in claims if term in f"{item.text} {item.subject} {item.object}".lower()]
        page = claims[offset:offset + limit]
        items = [
            MemoryHelperItem(
                id=item.id, helper_id=helper_id, kind="claim", title=item.text[:160],
                subtitle=item.relation, status=item.status, score=item.confidence,
                data=item.model_dump(mode="json"),
            )
            for item in page
        ]
        next_cursor = offset + limit if offset + limit < len(claims) else None
        return MemoryHelperPage(
            helper_id=helper_id, kind="table", items=items,
            next_cursor=str(next_cursor) if next_cursor is not None else None,
            truncated=next_cursor is not None,
        )
    if helper_id == "episodes":
        views = engine.store.episode_recall_views(namespace=namespace)
        if search:
            term = search.lower()
            views = [item for item in views if term in f"{item.concept_or_topic} {item.summary}".lower()]
        page = views[offset:offset + limit]
        items = [
            MemoryHelperItem(
                id=item.id, helper_id=helper_id, kind="recall",
                title=item.concept_or_topic, subtitle=item.episode_id, excerpt=item.summary,
                status="stale" if item.stale else "fresh",
                data=item.model_dump(mode="json"),
            )
            for item in page
        ]
        next_cursor = offset + limit if offset + limit < len(views) else None
        return MemoryHelperPage(
            helper_id=helper_id, kind="cards", items=items,
            next_cursor=str(next_cursor) if next_cursor is not None else None,
            truncated=next_cursor is not None,
        )
    if helper_id == "vectors":
        projection = vector_projection(engine, namespace, sample=min(limit, 200))
        items = [
            MemoryHelperItem(
                id=point.id, helper_id=helper_id, kind="embedding",
                title=point.excerpt[:120] or point.id, subtitle=point.modality,
                excerpt=point.excerpt, data=point.model_dump(mode="json"),
            )
            for point in projection.points
        ]
        return MemoryHelperPage(helper_id=helper_id, kind="embedding", items=items, truncated=projection.searchable > len(items))
    if helper_id == "graph":
        graph = retrieval_graph_page(engine, namespace, limit=limit)
        items = [
            MemoryHelperItem(
                id=node.id, helper_id=helper_id, kind=node.type, title=node.label,
                subtitle=node.subtitle, status=node.status, data=node.data,
            )
            for node in graph.nodes
        ]
        return MemoryHelperPage(
            helper_id=helper_id, kind="graph", items=items, truncated=graph.truncated,
            next_cursor=graph.next_cursor,
        )
    extra, total = engine.store.recall_cache_items(helper_id, namespace=namespace, cursor=offset, limit=limit)
    items = [
        MemoryHelperItem(
            id=str(item.get("id") or item.get("cache_key") or offset + index),
            helper_id=helper_id,
            kind=helper_id,
            title=str(item.get("concept_or_topic") or item.get("title") or helper_id),
            excerpt=str(item.get("summary") or item.get("text") or "")[:400],
            status="stale" if item.get("stale") else "fresh",
            data=item,
        )
        for index, item in enumerate(extra)
    ]
    next_cursor = offset + limit if offset + limit < total else None
    return MemoryHelperPage(
        helper_id=helper_id, kind=spec.kind if spec else "cards", items=items,
        next_cursor=str(next_cursor) if next_cursor is not None else None,
        truncated=next_cursor is not None,
    )


def document_page(
    engine: Any,
    namespace: MemoryNamespace,
    *,
    search: str | None = None,
    modality: str | None = None,
    cursor: int = 0,
    limit: int = 50,
) -> MemoryDocumentPage:
    rows, total = engine.store.list_retrieval_documents_page(
        namespace, search=search, modality=modality, cursor=cursor, limit=limit
    )
    next_cursor = cursor + limit if cursor + limit < total else None
    return MemoryDocumentPage(
        documents=[MemoryDocumentRow.model_validate(item) for item in rows],
        next_cursor=next_cursor,
        total=total,
    )


def term_page(
    engine: Any,
    namespace: MemoryNamespace,
    *,
    search: str | None = None,
    cursor: int = 0,
    limit: int = 40,
) -> MemoryTermPage:
    rows, total = engine.store.list_retrieval_terms_page(
        namespace, search=search, cursor=cursor, limit=limit
    )
    next_cursor = cursor + limit if cursor + limit < total else None
    return MemoryTermPage(
        terms=[
            MemoryTerm(
                term=item["term"],
                document_frequency=item["document_frequency"],
                postings=[MemoryTermPosting.model_validate(posting) for posting in item["postings"]],
            )
            for item in rows
        ],
        next_cursor=next_cursor,
        total=total,
    )


def compiled_claims(engine: Any, namespace: MemoryNamespace) -> list[CompiledClaim]:
    return _claims(engine, namespace)


def vector_projection(
    engine: Any,
    namespace: MemoryNamespace,
    *,
    sample: int = 500,
) -> VectorProjection:
    hashes, documents = _document_hashes(engine, namespace, limit=max(sample, 1))
    cache_key = str(getattr(engine.embedder, "cache_key", getattr(engine.embedder, "model_name", "")))
    configuration = engine.get_model_configuration()
    vectors = engine.vector_cache.get(list(hashes.values()), cache_key) if hashes else {}
    paired: list[tuple[str, str, list[float]]] = []
    for delta_id, text_hash in hashes.items():
        vector = vectors.get(text_hash)
        if vector:
            paired.append((delta_id, text_hash, vector))
    sampled = paired[:sample]
    points: list[VectorProjectionPoint] = []
    if sampled:
        matrix = np.array([item[2] for item in sampled], dtype=float)
        coords = _pca_2d(matrix)
        for index, (delta_id, text_hash, _vector) in enumerate(sampled):
            document = documents.get(delta_id) or {}
            points.append(VectorProjectionPoint(
                id=delta_id,
                x=float(coords[index][0]),
                y=float(coords[index][1]),
                modality=str(document.get("modality") or "text"),
                excerpt=str(document.get("text") or "")[:240],
                text_hash=text_hash,
            ))
    return VectorProjection(
        points=points,
        model=configuration.embedding.model,
        fingerprint=cache_key or embedding_cache_key(
            configuration.embedding.provider, configuration.embedding.base_url,
            configuration.embedding.model or "unknown",
        ),
        embedded=len(vectors),
        searchable=len(hashes),
        sampled=len(points),
    )


def vector_neighbors(
    engine: Any,
    namespace: MemoryNamespace,
    delta_id: str,
    *,
    limit: int = 12,
) -> VectorNeighbors:
    documents = engine.store.retrieval_documents_by_ids(namespace, [delta_id])
    if not documents:
        return VectorNeighbors(delta_id=delta_id, neighbors=[])
    cache_key = str(getattr(engine.embedder, "cache_key", getattr(engine.embedder, "model_name", "")))
    seed_hash = documents[0]["text_hash"]
    seed_vectors = engine.vector_cache.get([seed_hash], cache_key)
    seed = seed_vectors.get(seed_hash)
    configuration = engine.get_model_configuration()
    if not seed:
        return VectorNeighbors(delta_id=delta_id, neighbors=[], model=configuration.embedding.model)
    nearest = engine.vector_cache.nearest(seed, cache_key, limit=limit + 8)
    hash_to_doc: dict[str, dict[str, Any]] = {}
    hashes, docs = _document_hashes(engine, namespace, limit=2_000)
    inverse = {text_hash: identifier for identifier, text_hash in hashes.items()}
    hash_to_doc.update({docs[identifier]["text_hash"]: docs[identifier] for identifier in docs})
    neighbors: list[VectorNeighbor] = []
    for text_hash, score in nearest:
        other_id = inverse.get(text_hash)
        if other_id is None or other_id == delta_id:
            continue
        document = hash_to_doc.get(text_hash) or {}
        neighbors.append(VectorNeighbor(
            id=other_id,
            score=float(score),
            excerpt=str(document.get("text") or "")[:240],
            modality=str(document.get("modality") or "text"),
        ))
        if len(neighbors) >= limit:
            break
    return VectorNeighbors(delta_id=delta_id, neighbors=neighbors, model=configuration.embedding.model)


def retrieval_graph_page(
    engine: Any,
    namespace: MemoryNamespace,
    *,
    seed_id: str | None = None,
    edge_kind: str | None = None,
    limit: int = 400,
) -> MemoryGraphPage:
    if seed_id:
        edges = engine.store.retrieval_graph_neighborhood(
            namespace, seed_id, edge_kind=edge_kind, limit=limit
        )
    else:
        edges = engine.store.list_retrieval_graph_edges(namespace, edge_kind=edge_kind, limit=limit)
    ids: list[str] = []
    if seed_id:
        ids.append(seed_id)
    for source, target, _kind, _weight in edges:
        ids.extend((source, target))
    documents = {item["id"]: item for item in engine.store.retrieval_documents_by_ids(namespace, ids)}
    nodes = [
        MemoryGraphNode(
            id=item["id"], type="trace", label=item["text"][:90],
            subtitle=item["modality"], status="active" if item["active"] else "inactive",
            data={"seq": item["seq"], "source_type": item["source_type"]},
        )
        for item in documents.values()
    ]
    graph_edges = [
        MemoryGraphEdge(
            id=f"{source}:{target}:{kind}", source=source, target=target,
            type=kind, label=kind.replace("_", " "), weight=weight,
        )
        for source, target, kind, weight in edges
        if source in documents and target in documents
    ]
    counts: dict[str, int] = {}
    for node in nodes:
        counts[node.type] = counts.get(node.type, 0) + 1
    return MemoryGraphPage(
        view="retrieval",
        nodes=nodes,
        edges=graph_edges,
        counts=counts,
        truncated=len(edges) >= limit,
    )


def _claims(engine: Any, namespace: MemoryNamespace) -> list[CompiledClaim]:
    deltas = engine.store.list_deltas(
        kinds=["observation", "extraction"], namespace=namespace, include_retracted=False
    )
    observations = {item.id: item for item in deltas if item.kind == "observation"}
    return compile_claims([item for item in deltas if item.kind == "extraction"], observations)


def _document_hashes(
    engine: Any,
    namespace: MemoryNamespace,
    *,
    limit: int,
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    rows, _total = engine.store.list_retrieval_documents_page(namespace, cursor=0, limit=limit)
    hashes = {item["id"]: item["text_hash"] for item in rows}
    documents = {item["id"]: item for item in rows}
    return hashes, documents


def _pca_2d(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((0, 2))
    if matrix.shape[0] == 1:
        return np.zeros((1, 2))
    centered = matrix - matrix.mean(axis=0)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    if components.shape[1] == 1:
        components = np.hstack([components, np.zeros((components.shape[0], 1))])
    projected = centered @ components
    scale = np.max(np.abs(projected)) or 1.0
    return projected / scale
