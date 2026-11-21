"""Read-only Recall inspection used by Studio Memory Viewer.

Owns catalog assembly, helper pages, document and term listings, vector
projection, and the retrieval graph. Call these from MemoryEngine inspect
methods; go to ingest or the store to mutate state. This module must not
write the memory log.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from trisynapse_memory.engine.models import (
    Claim,
    MemoryCatalog,
    MemoryCatalogExtension,
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
from trisynapse_memory.engine.extensions import InspectRequest, IndexReader
from trisynapse_memory.engine.indexes.catalog import BUILTIN_INDEX_HELPERS, builtin_helper_ids, helper_spec
from trisynapse_memory.engine.indexes.claims import compile_claims
from trisynapse_memory.engine.retrieval.contracts import DEFAULT_ROUTE_WEIGHTS, RouteRegistry, route_weights


def memory_catalog(engine: Any, namespace: MemoryNamespace) -> MemoryCatalog:
    return MemoryCatalog(
        helpers=_catalog_helpers(engine, namespace),
        retrieval_routes=_catalog_routes(engine),
        extensions=_catalog_extensions(engine),
    )


def helper_items(
    engine: Any,
    helper_id: str,
    namespace: MemoryNamespace,
    *,
    search: str | None = None,
    cursor: str | None = None,
    limit: int = 50,
) -> MemoryHelperPage:
    page = _channel_helper_page(engine, helper_id, namespace, search, cursor, limit)
    if page is not None:
        return page
    builder = _HELPER_BUILDERS.get(helper_id, _extra_helper_items)
    return builder(engine, helper_id, namespace, search, cursor, limit)


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


def compiled_claims(engine: Any, namespace: MemoryNamespace) -> list[Claim]:
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
    vectors = engine.embedding_index.get(list(hashes.values()), cache_key) if hashes else {}
    points = _projection_points(_paired_vectors(hashes, vectors)[:sample], documents)
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
    record_id: str,
    *,
    limit: int = 12,
) -> VectorNeighbors:
    documents = engine.store.retrieval_documents_by_ids(namespace, [record_id])
    if not documents:
        return VectorNeighbors(record_id=record_id, neighbors=[])
    cache_key = str(getattr(engine.embedder, "cache_key", getattr(engine.embedder, "model_name", "")))
    seed_hash = documents[0]["text_hash"]
    seed_vectors = engine.embedding_index.get([seed_hash], cache_key)
    seed = seed_vectors.get(seed_hash)
    configuration = engine.get_model_configuration()
    if not seed:
        return VectorNeighbors(record_id=record_id, neighbors=[], model=configuration.embedding.model)
    nearest = engine.embedding_index.nearest(seed, cache_key, limit=limit + 8)
    return VectorNeighbors(
        record_id=record_id,
        neighbors=_neighbor_hits(engine, namespace, record_id, nearest, limit),
        model=configuration.embedding.model,
    )


def retrieval_graph_page(
    engine: Any,
    namespace: MemoryNamespace,
    *,
    seed_id: str | None = None,
    edge_kind: str | None = None,
    limit: int = 400,
) -> MemoryGraphPage:
    edges = _retrieval_edges(engine, namespace, seed_id, edge_kind, limit)
    documents = {
        item["id"]: item
        for item in engine.store.retrieval_documents_by_ids(namespace, _graph_ids(seed_id, edges))
    }
    nodes = [_graph_node(item) for item in documents.values()]
    return MemoryGraphPage(
        view="retrieval",
        nodes=nodes,
        edges=_graph_edges(edges, documents),
        counts=_node_type_counts(nodes),
        truncated=len(edges) >= limit,
    )


def _catalog_helpers(engine: Any, namespace: MemoryNamespace) -> list[MemoryCatalogHelper]:
    health_by_id, count_by_id = _builtin_helper_stats(engine, namespace)
    helpers = [
        MemoryCatalogHelper(
            id=spec.id,
            title=spec.title,
            kind=spec.kind,
            inspect_path=spec.inspect_path,
            playground_seed=spec.playground_seed,
            count=count_by_id.get(spec.id, 0),
            health=health_by_id.get(spec.id, {}),
        )
        for spec in BUILTIN_INDEX_HELPERS
    ]
    channels = getattr(getattr(engine, "extensions", None), "index_channels", None)
    helpers.extend(_channel_helpers(engine, namespace, channels))
    known = builtin_helper_ids() | ({*channels.names} if channels is not None else set())
    helpers.extend(_orphan_view_helpers(engine, namespace, known))
    return helpers


def _builtin_helper_stats(
    engine: Any, namespace: MemoryNamespace
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    counts = engine.store.retrieval_index_counts(namespace)
    deltas = engine.store.list_records(namespace=namespace, include_retracted=True)
    claims = _catalog_compiled_claims(deltas)
    groups = engine.store.group_summary_views(namespace=namespace)
    cache_key = str(getattr(engine.embedder, "cache_key", getattr(engine.embedder, "model_name", "")))
    active_docs, _total_docs = _document_hashes(engine, namespace, limit=2_000)
    embedded = _embedded_count(engine, active_docs, cache_key)
    contested = sum(1 for item in claims if item.status == "CONTESTED")
    stale_groups = sum(1 for item in groups if item.stale)
    health_by_id = {
        "log": {"count": len(deltas), "raw_sources": sum(1 for item in deltas if item.kind == "raw_source")},
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
        "groups": {"count": len(groups), "stale": stale_groups, "fresh": len(groups) - stale_groups},
        "claims": {"count": len(claims), "contested": contested},
        "graph": {"count": sum(counts["graph_edges"].values()), "edges": counts["graph_edges"]},
    }
    count_by_id = {
        "log": len(deltas),
        "documents": counts["documents_active"],
        "bm25": counts["distinct_terms"],
        "vectors": embedded,
        "groups": len(groups),
        "claims": len(claims),
        "graph": sum(counts["graph_edges"].values()),
    }
    return health_by_id, count_by_id


def _catalog_compiled_claims(deltas: list[Any]) -> list[Claim]:
    observations = {item.id: item for item in deltas if item.kind == "raw_source"}
    return compile_claims([item for item in deltas if item.kind == "fact"], observations)


def _embedded_count(engine: Any, active_docs: dict[str, str], cache_key: str) -> int:
    if not active_docs:
        return 0
    return len(engine.embedding_index.get(list(active_docs.values()), cache_key))


def _channel_helpers(engine: Any, namespace: MemoryNamespace, channels: Any) -> list[MemoryCatalogHelper]:
    if channels is None:
        return []
    reader = IndexReader(engine.store, namespace)
    channel_counts = engine.store.recall_channel_counts(namespace)
    return [
        _channel_catalog_helper(channel, reader, channel_counts)
        for channel in channels.channels
    ]


def _channel_catalog_helper(channel: Any, reader: IndexReader, channel_counts: dict[str, Any]) -> MemoryCatalogHelper:
    health = channel_counts.get(channel.spec.id, {"count": 0, "total": 0, "stale": 0})
    if callable(getattr(channel, "health", None)):
        try:
            health = {**health, **dict(channel.health(reader))}
        except Exception as exc:
            health = {**health, "status": "degraded", "error": str(exc)[:500]}
    return MemoryCatalogHelper(
        id=channel.spec.id,
        title=channel.spec.title,
        kind=channel.spec.kind,
        inspect_path=f"/api/v1/memory/helpers/{channel.spec.id}",
        playground_seed=channel.spec.playground_seed,
        count=int(health.get("count", 0)),
        health=health,
    )


def _orphan_view_helpers(
    engine: Any, namespace: MemoryNamespace, known: set[str]
) -> list[MemoryCatalogHelper]:
    helpers: list[MemoryCatalogHelper] = []
    for view_type, extra_count in engine.store.recall_view_type_counts(namespace).items():
        if view_type in known or view_type == "group_summary":
            continue
        helpers.append(MemoryCatalogHelper(
            id=view_type,
            title=view_type.replace("_", " ").title(),
            kind="cards",
            inspect_path=f"/api/v1/memory/helpers/{view_type}",
            count=extra_count,
            health={"count": extra_count},
        ))
    return helpers


def _catalog_routes(engine: Any) -> list[MemoryCatalogRoute]:
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
    routes.extend(_extension_routes(engine, configuration, enabled))
    known = {route.name for route in routes}
    routes.extend(_unknown_configured_routes(configuration, enabled, known))
    return routes


def _extension_routes(engine: Any, configuration: Any, enabled: set[str]) -> list[MemoryCatalogRoute]:
    branches = getattr(getattr(engine, "extensions", None), "retrieval_branches", None)
    if branches is None:
        return []
    extension_states = {
        state.extension_id: state for state in engine.store.extension_states()
    }
    return [
        _extension_route(engine, branch, configuration, enabled, extension_states)
        for branch in branches.branches
    ]


def _extension_route(
    engine: Any,
    branch: Any,
    configuration: Any,
    enabled: set[str],
    extension_states: dict[str, Any],
) -> MemoryCatalogRoute:
    profile_weight = branch.spec.profile_weights.get(configuration.retrieval_profile)
    owner = engine.extensions.owner("retrieval", branch.spec.name)
    state = extension_states.get(owner) if owner else None
    available = state is None or state.status == "available"
    return MemoryCatalogRoute(
        name=branch.spec.name,
        title=branch.spec.title or branch.spec.name.replace("_", " "),
        enabled=branch.spec.name in enabled,
        weight=float(configuration.route_weights.get(
            branch.spec.name,
            profile_weight if profile_weight is not None else branch.spec.default_weight,
        )),
        dependencies=list(branch.spec.depends_on),
        cost_tier=branch.spec.cost_tier,
        available=available,
    )


def _unknown_configured_routes(
    configuration: Any, enabled: set[str], known_route_names: set[str]
) -> list[MemoryCatalogRoute]:
    configured_route_names = list(configuration.enabled_routes) + list(configuration.route_weights)
    routes: list[MemoryCatalogRoute] = []
    for name in dict.fromkeys(configured_route_names):
        if name in known_route_names:
            continue
        routes.append(MemoryCatalogRoute(
            name=name,
            title=name.replace("_", " "),
            enabled=name in enabled,
            weight=float(configuration.route_weights.get(name, 1.0)),
            available=False,
        ))
    return routes


def _catalog_extensions(engine: Any) -> list[MemoryCatalogExtension]:
    specs = getattr(getattr(engine, "extensions", None), "extensions", {})
    states = {state.extension_id: state for state in engine.store.extension_states()}
    return [
        _catalog_extension(extension_id, specs.get(extension_id), states.get(extension_id))
        for extension_id in sorted(set(specs) | set(states))
    ]


def _catalog_extension(extension_id: str, spec: Any, state: Any) -> MemoryCatalogExtension:
    return MemoryCatalogExtension(
        id=extension_id,
        version=spec.version if spec else state.installed_version,
        engine_api=spec.engine_api if spec else state.engine_api,
        storage_revision=spec.storage_revision if spec else state.storage_revision,
        status=_extension_status(spec, state),
        last_projected_seq=state.last_projected_seq if state else 0,
        last_error=state.last_error if state else None,
    )


def _extension_status(spec: Any, state: Any) -> str:
    if spec is None:
        return "unavailable"
    if state is None:
        return "available"
    return state.status


def _paged_helper_page(
    helper_id: str,
    kind: str,
    items: list[MemoryHelperItem],
    offset: int,
    limit: int,
    total: int,
) -> MemoryHelperPage:
    next_cursor = offset + limit if offset + limit < total else None
    return MemoryHelperPage(
        helper_id=helper_id,
        kind=kind,
        items=items,
        next_cursor=str(next_cursor) if next_cursor is not None else None,
        truncated=next_cursor is not None,
    )


def _channel_helper_page(
    engine: Any,
    helper_id: str,
    namespace: MemoryNamespace,
    search: str | None,
    cursor: str | None,
    limit: int,
) -> MemoryHelperPage | None:
    channels = getattr(getattr(engine, "extensions", None), "index_channels", None)
    channel = channels.get(helper_id) if channels is not None else None
    if channel is None:
        return None
    reader = IndexReader(engine.store, namespace)
    if callable(getattr(channel, "inspect", None)):
        return channel.inspect(
            InspectRequest(search=search, cursor=cursor, limit=limit),
            reader,
        )
    offset = int(cursor or 0)
    records, total = reader.records(helper_id, search=search, cursor=offset, limit=limit)
    items = [_channel_record_item(helper_id, record) for record in records]
    return _paged_helper_page(helper_id, channel.spec.kind, items, offset, limit, total)


def _channel_record_item(helper_id: str, record: Any) -> MemoryHelperItem:
    return MemoryHelperItem(
        id=record.record_id,
        helper_id=helper_id,
        kind="index",
        title=record.text[:160] or record.record_id,
        excerpt=record.text[:400],
        status="stale" if record.stale else "active",
        data={
            "fields": dict(record.fields),
            "metadata": dict(record.metadata),
            "evidence_refs": list(record.evidence_refs),
            "evidence_version": record.evidence_version,
            "producer_version": record.producer_version,
        },
    )


def _log_helper_items(
    engine: Any,
    helper_id: str,
    namespace: MemoryNamespace,
    search: str | None,
    cursor: str | None,
    limit: int,
) -> MemoryHelperPage:
    spec = helper_spec(helper_id)
    offset = int(cursor or 0)
    page = engine.list(namespace=namespace, cursor=offset or None, limit=limit, include_retracted=True)
    items = [
        MemoryHelperItem(
            id=item.id,
            helper_id=helper_id,
            kind=item.kind,
            title=(item.text or item.kind)[:120],
            subtitle=item.group_id,
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


def _documents_helper_items(
    engine: Any,
    helper_id: str,
    namespace: MemoryNamespace,
    search: str | None,
    cursor: str | None,
    limit: int,
) -> MemoryHelperPage:
    offset = int(cursor or 0)
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
    return _paged_helper_page(helper_id, "table", items, offset, limit, total)


def _bm25_helper_items(
    engine: Any,
    helper_id: str,
    namespace: MemoryNamespace,
    search: str | None,
    cursor: str | None,
    limit: int,
) -> MemoryHelperPage:
    offset = int(cursor or 0)
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
    return _paged_helper_page(helper_id, "postings", items, offset, limit, total)


def _claims_helper_items(
    engine: Any,
    helper_id: str,
    namespace: MemoryNamespace,
    search: str | None,
    cursor: str | None,
    limit: int,
) -> MemoryHelperPage:
    offset = int(cursor or 0)
    claims = _filter_claims(_claims(engine, namespace), search)
    page = claims[offset:offset + limit]
    items = [
        MemoryHelperItem(
            id=item.id, helper_id=helper_id, kind="claim", title=item.text[:160],
            subtitle=item.relation, status=item.status, score=item.confidence,
            data=item.model_dump(mode="json"),
        )
        for item in page
    ]
    return _paged_helper_page(helper_id, "table", items, offset, limit, len(claims))


def _filter_claims(claims: list[Claim], search: str | None) -> list[Claim]:
    if not search:
        return claims
    term = search.lower()
    return [item for item in claims if term in f"{item.text} {item.subject} {item.object}".lower()]


def _groups_helper_items(
    engine: Any,
    helper_id: str,
    namespace: MemoryNamespace,
    search: str | None,
    cursor: str | None,
    limit: int,
) -> MemoryHelperPage:
    offset = int(cursor or 0)
    views = _filter_groups(engine.store.group_summary_views(namespace=namespace), search)
    page = views[offset:offset + limit]
    items = [
        MemoryHelperItem(
            id=item.id, helper_id=helper_id, kind="index",
            title=item.concept_or_topic, subtitle=item.group_id, excerpt=item.summary,
            status="stale" if item.stale else "fresh",
            data=item.model_dump(mode="json"),
        )
        for item in page
    ]
    return _paged_helper_page(helper_id, "cards", items, offset, limit, len(views))


def _filter_groups(views: list[Any], search: str | None) -> list[Any]:
    if not search:
        return views
    term = search.lower()
    return [item for item in views if term in f"{item.concept_or_topic} {item.summary}".lower()]


def _vectors_helper_items(
    engine: Any,
    helper_id: str,
    namespace: MemoryNamespace,
    search: str | None,
    cursor: str | None,
    limit: int,
) -> MemoryHelperPage:
    projection = vector_projection(engine, namespace, sample=min(limit, 200))
    items = [
        MemoryHelperItem(
            id=point.id, helper_id=helper_id, kind="embedding",
            title=point.excerpt[:120] or point.id, subtitle=point.modality,
            excerpt=point.excerpt, data=point.model_dump(mode="json"),
        )
        for point in projection.points
    ]
    return MemoryHelperPage(
        helper_id=helper_id, kind="embedding", items=items, truncated=projection.searchable > len(items)
    )


def _graph_helper_items(
    engine: Any,
    helper_id: str,
    namespace: MemoryNamespace,
    search: str | None,
    cursor: str | None,
    limit: int,
) -> MemoryHelperPage:
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


def _extra_helper_items(
    engine: Any,
    helper_id: str,
    namespace: MemoryNamespace,
    search: str | None,
    cursor: str | None,
    limit: int,
) -> MemoryHelperPage:
    spec = helper_spec(helper_id)
    offset = int(cursor or 0)
    extra, total = engine.store.recall_cache_items(helper_id, namespace=namespace, cursor=offset, limit=limit)
    items = [_extra_helper_item(helper_id, item, offset, index) for index, item in enumerate(extra)]
    return _paged_helper_page(helper_id, spec.kind if spec else "cards", items, offset, limit, total)


def _extra_helper_item(helper_id: str, item: dict[str, Any], offset: int, index: int) -> MemoryHelperItem:
    return MemoryHelperItem(
        id=str(item.get("id") or item.get("cache_key") or offset + index),
        helper_id=helper_id,
        kind=helper_id,
        title=str(item.get("concept_or_topic") or item.get("title") or helper_id),
        excerpt=str(item.get("summary") or item.get("text") or "")[:400],
        status="stale" if item.get("stale") else "fresh",
        data=item,
    )


_HELPER_BUILDERS = {
    "log": _log_helper_items,
    "documents": _documents_helper_items,
    "bm25": _bm25_helper_items,
    "claims": _claims_helper_items,
    "groups": _groups_helper_items,
    "vectors": _vectors_helper_items,
    "graph": _graph_helper_items,
}


def _paired_vectors(
    hashes: dict[str, str],
    vectors: dict[str, list[float]],
) -> list[tuple[str, str, list[float]]]:
    paired: list[tuple[str, str, list[float]]] = []
    for record_id, text_hash in hashes.items():
        vector = vectors.get(text_hash)
        if not vector:
            continue
        paired.append((record_id, text_hash, vector))
    return paired


def _projection_points(
    sampled: list[tuple[str, str, list[float]]],
    documents: dict[str, dict[str, Any]],
) -> list[VectorProjectionPoint]:
    if not sampled:
        return []
    matrix = np.array([item[2] for item in sampled], dtype=float)
    coords = _pca_2d(matrix)
    points: list[VectorProjectionPoint] = []
    for index, (record_id, text_hash, _vector) in enumerate(sampled):
        document = documents.get(record_id) or {}
        points.append(VectorProjectionPoint(
            id=record_id,
            x=float(coords[index][0]),
            y=float(coords[index][1]),
            modality=str(document.get("modality") or "text"),
            excerpt=str(document.get("text") or "")[:240],
            text_hash=text_hash,
        ))
    return points


def _neighbor_hits(
    engine: Any,
    namespace: MemoryNamespace,
    record_id: str,
    nearest: Any,
    limit: int,
) -> list[VectorNeighbor]:
    hashes, docs = _document_hashes(engine, namespace, limit=2_000)
    inverse = {text_hash: identifier for identifier, text_hash in hashes.items()}
    hash_to_doc = {docs[identifier]["text_hash"]: docs[identifier] for identifier in docs}
    neighbors: list[VectorNeighbor] = []
    for text_hash, score in nearest:
        other_id = inverse.get(text_hash)
        if other_id is None or other_id == record_id:
            continue
        document = hash_to_doc.get(text_hash) or {}
        neighbors.append(_vector_neighbor(other_id, score, document))
        if len(neighbors) >= limit:
            break
    return neighbors


def _vector_neighbor(other_id: str, score: float, document: dict[str, Any]) -> VectorNeighbor:
    return VectorNeighbor(
        id=other_id,
        score=float(score),
        excerpt=str(document.get("text") or "")[:240],
        modality=str(document.get("modality") or "text"),
    )


def _retrieval_edges(
    engine: Any,
    namespace: MemoryNamespace,
    seed_id: str | None,
    edge_kind: str | None,
    limit: int,
) -> list[Any]:
    if seed_id:
        return engine.store.retrieval_graph_neighborhood(
            namespace, seed_id, edge_kind=edge_kind, limit=limit
        )
    return engine.store.list_retrieval_graph_edges(namespace, edge_kind=edge_kind, limit=limit)


def _graph_ids(seed_id: str | None, edges: list[Any]) -> list[str]:
    ids: list[str] = []
    if seed_id:
        ids.append(seed_id)
    for source, target, _kind, _weight in edges:
        ids.extend((source, target))
    return ids


def _graph_node(item: dict[str, Any]) -> MemoryGraphNode:
    return MemoryGraphNode(
        id=item["id"], type="log", label=item["text"][:90],
        subtitle=item["modality"], status="active" if item["active"] else "inactive",
        data={"seq": item["seq"], "source_type": item["source_type"]},
    )


def _graph_edges(edges: list[Any], documents: dict[str, dict[str, Any]]) -> list[MemoryGraphEdge]:
    return [
        MemoryGraphEdge(
            id=f"{source}:{target}:{kind}", source=source, target=target,
            type=kind, label=kind.replace("_", " "), weight=weight,
        )
        for source, target, kind, weight in edges
        if source in documents and target in documents
    ]


def _node_type_counts(nodes: list[MemoryGraphNode]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in nodes:
        counts[node.type] = counts.get(node.type, 0) + 1
    return counts


def _claims(engine: Any, namespace: MemoryNamespace) -> list[Claim]:
    deltas = engine.store.list_records(
        kinds=["raw_source", "fact"], namespace=namespace, include_retracted=False
    )
    observations = {item.id: item for item in deltas if item.kind == "raw_source"}
    return compile_claims([item for item in deltas if item.kind == "fact"], observations)


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
