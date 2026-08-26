import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useApi } from "../../context";
import { Empty, Loading } from "../../ui";
import { classNames } from "../../util";
import type { GraphPage, InspectorRecord } from "../../types";
import { registerHelper, type HelperViewProps } from "./index";
import { CardList, GenericItemsView, ItemTable, MemoryFlowGraph, PostingsList, TimelineList, VectorCanvas } from "./families";

function TraceView(props: HelperViewProps) {
  const api = useApi();
  const page = useQuery({ queryKey: ["memories", api.connection], queryFn: () => api.memories() });
  if (page.isLoading) return <Loading label="Loading Trace" />;
  if (page.isError) return <Empty title="Trace unavailable" detail={(page.error as Error).message} />;
  const items = (page.data?.items || []).filter(item => {
    if (!props.search) return true;
    return `${item.text} ${item.kind} ${item.episode_id || ""}`.toLowerCase().includes(props.search.toLowerCase());
  }).map(item => ({
    id: item.id, helper_id: "trace", kind: item.kind, title: (item.text || item.kind).slice(0, 120),
    subtitle: item.episode_id, excerpt: item.text.slice(0, 400), status: item.kind,
    data: item as unknown as Record<string, unknown>,
  }));
  return items.length ? <TimelineList items={items} selectedId={props.selectedId} onSelect={props.onSelect} /> : <Empty title="Trace is empty" detail="Accepted sources and observations appear here as ordered evidence." />;
}

function DocumentsView(props: HelperViewProps) {
  const api = useApi();
  const page = useQuery({ queryKey: ["documents", props.search, api.connection], queryFn: () => api.documents(props.search) });
  if (page.isLoading) return <Loading label="Loading documents" />;
  if (page.isError) return <Empty title="Documents unavailable" detail={(page.error as Error).message} />;
  const items = (page.data?.documents || []).map(item => ({
    id: item.id, helper_id: "documents", kind: item.modality, title: item.text.slice(0, 120),
    subtitle: `${item.source_type} · ${item.token_count} tokens`, excerpt: item.text.slice(0, 400),
    status: item.active ? "active" : "inactive", data: item as unknown as Record<string, unknown>,
  }));
  return items.length ? <ItemTable items={items} selectedId={props.selectedId} onSelect={props.onSelect} /> : <Empty title="No retrieval documents" detail="Accepted Trace records become searchable documents during indexing." />;
}

function Bm25View(props: HelperViewProps) {
  const api = useApi();
  const [term, setTerm] = useState(props.search);
  useEffect(() => { setTerm(props.search); }, [props.search]);
  const page = useQuery({ queryKey: ["terms", term, api.connection], queryFn: () => api.terms(term) });
  const items = (page.data?.terms || []).map(item => ({
    id: item.term, helper_id: "bm25", kind: "term", title: item.term,
    subtitle: `${item.document_frequency} documents`, status: "indexed",
    score: item.document_frequency, data: item as unknown as Record<string, unknown>,
  }));
  const maxDf = Math.max(1, ...items.map(item => Number(item.score || 1)));
  return <div className="bm25-view">
    <label className="search-field"><input value={term} onChange={event => setTerm(event.target.value)} placeholder="Look up a term" aria-label="Look up a term" /></label>
    {page.isLoading ? <Loading label="Loading BM25 postings" /> : page.isError ? <Empty title="BM25 unavailable" detail={(page.error as Error).message} /> : <>
      <div className="df-bars" aria-hidden="true">
        {items.slice(0, 18).map(item => <span key={item.id} style={{ height: `${Math.max(8, (Number(item.score) / maxDf) * 64)}px` }} title={`${item.title} ${item.subtitle}`} />)}
      </div>
      {items.length ? <PostingsList items={items} selectedId={props.selectedId} onSelect={props.onSelect} /> : <Empty title="No postings" detail="Lexical terms appear here after Trace is indexed." />}
    </>}
  </div>;
}

function VectorsView(props: HelperViewProps) {
  const api = useApi();
  const projection = useQuery({ queryKey: ["vectors", api.connection], queryFn: () => api.vectorProjection() });
  if (projection.isLoading) return <Loading label="Projecting embeddings" />;
  if (projection.isError) return <Empty title="Vectors unavailable" detail={(projection.error as Error).message} />;
  const data = projection.data;
  if (!data?.points.length) return <Empty title="No embeddings yet" detail="Semantic vectors are stored after searchable Trace is embedded." />;
  return <div className="vector-view">
    <p className="vector-caption">{data.embedded} embedded / {data.searchable} searchable · {data.model || "active model"} · {data.fingerprint}</p>
    <VectorCanvas points={data.points} selectedId={props.selectedId} onSelect={id => {
      const point = data.points.find(item => item.id === id);
      if (!point) return;
      props.onSelect({ id: point.id, helperId: "vectors", kind: "embedding", title: point.excerpt.slice(0, 120) || point.id, subtitle: point.modality, excerpt: point.excerpt, data: point as unknown as Record<string, unknown> });
    }} />
  </div>;
}

function EpisodesView(props: HelperViewProps) {
  const api = useApi();
  const recall = useQuery({ queryKey: ["helper-items", "episodes", props.search, api.connection], queryFn: () => api.helperItems("episodes", props.search) });
  const episodes = useQuery({ queryKey: ["episodes", api.connection], queryFn: () => api.episodes() });
  if (recall.isLoading) return <Loading label="Loading Episode Recall" />;
  if (recall.isError) return <Empty title="Episode Recall unavailable" detail={(recall.error as Error).message} />;
  const items = [...(recall.data?.items || [])];
  const seen = new Set(items.map(item => String(item.data.episode_id || item.subtitle || "")));
  for (const episode of episodes.data?.episodes || []) {
    if (seen.has(episode.episode_id)) continue;
    items.push({
      id: episode.episode_id, helper_id: "episodes", kind: "episode", title: episode.episode_id,
      subtitle: `${episode.delta_count} deltas`, excerpt: null, status: episode.stale ? "stale" : "fresh",
      data: episode as unknown as Record<string, unknown>,
    });
  }
  const filtered = props.search
    ? items.filter(item => `${item.title} ${item.excerpt || ""} ${item.subtitle || ""}`.toLowerCase().includes(props.search.toLowerCase()))
    : items;
  return filtered.length ? <CardList items={filtered} selectedId={props.selectedId} onSelect={props.onSelect} /> : <Empty title="No Episode Recall views" detail="Episode summaries appear after compilation." />;
}

function ClaimsView(props: HelperViewProps) {
  const api = useApi();
  const claims = useQuery({ queryKey: ["claims", api.connection], queryFn: () => api.claims() });
  if (claims.isLoading) return <Loading label="Loading claims" />;
  if (claims.isError) return <Empty title="Claims unavailable" detail={(claims.error as Error).message} />;
  const filtered = (claims.data?.claims || []).filter(item => {
    if (!props.search) return true;
    return `${item.text} ${item.subject} ${item.object} ${item.relation} ${(item.objects || []).join(" ")}`.toLowerCase().includes(props.search.toLowerCase());
  });
  if (!filtered.length) return <Empty title="No compiled claims" detail="Claims appear after evidence-linked extractions are compiled." />;
  return <div className="claims-table" role="table">
    <div role="row" className="graph-table-head"><span>Claim</span><span>Relation</span><span>Status</span></div>
    {filtered.map(item => {
      const record: InspectorRecord = {
        id: item.id, helperId: "claims", kind: "claim", title: item.text.slice(0, 160),
        subtitle: `${item.subject || "—"} ${item.relation || ""} ${item.object || ""}`,
        status: item.status, data: item as unknown as Record<string, unknown>,
      };
      const objects = item.objects?.length ? item.objects : [item.object].filter(Boolean) as string[];
      return <div key={item.id} className="claim-block">
        <button role="row" className={classNames(props.selectedId === item.id && "active")} onClick={() => props.onSelect(record)}>
          <span><strong>{item.text.slice(0, 160)}</strong><small>{item.subject || "—"}</small></span>
          <span>{item.relation || "—"}</span>
          <span className={`status-chip ${item.status}`}>{item.status}</span>
        </button>
        {item.status === "CONTESTED" && props.selectedId === item.id && (
          <div className="contested-objects">
            <strong>Competing objects</strong>
            {objects.map(value => <span key={value}>{value}</span>)}
            {item.source_delta_ids.map(id => <code key={id}>{id}</code>)}
          </div>
        )}
      </div>;
    })}
  </div>;
}

function GraphView(props: HelperViewProps) {
  const api = useApi();
  const [view, setView] = useState<"knowledge" | "lineage" | "retrieval">("knowledge");
  const [nodeType, setNodeType] = useState("");
  const [sourceId, setSourceId] = useState("");
  const [episodeId, setEpisodeId] = useState("");
  const [extra, setExtra] = useState<GraphPage | null>(null);
  const graph = useQuery({
    queryKey: ["memory-graph", view, props.search, nodeType, sourceId, episodeId, api.connection],
    queryFn: () => view === "retrieval"
      ? api.retrievalGraph(sourceId || props.selectedId || "", "")
      : api.graph(view, props.search, { nodeType, sourceId, episodeId }),
  });
  useEffect(() => { setExtra(null); }, [view, props.search, nodeType, sourceId, episodeId]);
  useEffect(() => {
    if (!props.selectedId || view === "retrieval") return;
    void api.graphNeighbors(props.selectedId, view).then(setExtra).catch(() => setExtra(null));
  }, [api, props.selectedId, view]);
  const merged = useMemo(() => mergeGraphs(graph.data, extra), [graph.data, extra]);
  if (graph.isLoading) return <Loading label="Building graph" />;
  if (graph.isError) return <Empty title="Graph unavailable" detail={(graph.error as Error).message} />;
  if (!merged) return null;
  return <div className="graph-helper">
    <div className="graph-filters">
      <div className="segmented">
        {(["knowledge", "lineage", "retrieval"] as const).map(value => (
          <button key={value} className={view === value ? "active" : ""} onClick={() => setView(value)}>{value}</button>
        ))}
      </div>
      <select aria-label="Node type" value={nodeType} onChange={event => setNodeType(event.target.value)}>
        <option value="">All node types</option>
        {["source", "trace", "episode", "claim", "concept", "recall"].map(value => <option key={value} value={value}>{value}</option>)}
      </select>
      <input aria-label="Source ID" value={sourceId} onChange={event => setSourceId(event.target.value)} placeholder="source id" />
      <input aria-label="Episode ID" value={episodeId} onChange={event => setEpisodeId(event.target.value)} placeholder="episode id" />
    </div>
    {merged.nodes.length ? <MemoryFlowGraph graph={merged} selectedId={props.selectedId} onSelect={node => {
      props.onSelect({ id: node.id, helperId: "graph", kind: node.type, title: node.label, subtitle: node.subtitle, status: node.status, data: node.data });
    }} /> : <Empty title="Graph is empty" detail="Concepts, claims, and retrieval edges appear after Recall is built." />}
  </div>;
}

function mergeGraphs(base?: GraphPage, extra?: GraphPage | null): GraphPage | undefined {
  if (!base) return extra || undefined;
  if (!extra) return base;
  const nodes = [...base.nodes];
  const seen = new Set(nodes.map(item => item.id));
  for (const node of extra.nodes) {
    if (seen.has(node.id)) continue;
    nodes.push(node);
    seen.add(node.id);
  }
  const edges = [...base.edges];
  const edgeIds = new Set(edges.map(item => item.id));
  for (const edge of extra.edges) {
    if (edgeIds.has(edge.id)) continue;
    edges.push(edge);
    edgeIds.add(edge.id);
  }
  return { ...base, nodes, edges };
}

export function registerBuiltinHelpers() {
  registerHelper({ id: "trace", kind: "timeline", render: TraceView, seedPlayground: record => record.excerpt || record.title });
  registerHelper({ id: "documents", kind: "table", render: DocumentsView, seedPlayground: record => record.excerpt || record.title });
  registerHelper({ id: "bm25", kind: "postings", render: Bm25View, seedPlayground: record => record.title });
  registerHelper({ id: "vectors", kind: "embedding", render: VectorsView, seedPlayground: record => record.excerpt || record.title });
  registerHelper({ id: "episodes", kind: "cards", render: EpisodesView, seedPlayground: record => record.excerpt || record.title });
  registerHelper({ id: "claims", kind: "table", render: ClaimsView, seedPlayground: record => record.title });
  registerHelper({ id: "graph", kind: "graph", render: GraphView, seedPlayground: record => record.title });
}

export function FallbackHelper(props: HelperViewProps) {
  return <GenericItemsView {...props} />;
}
