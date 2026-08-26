import type {
  Connection, GraphPage, IngestionRun, MemoryCatalog, HelperPage, MemoryDocument,
  MemoryPage, MemoryTerm, ModelConfigurationState, Namespace, Provider, QueryRun, Source,
  QueryRunPage, RetrievalConfiguration, SearchResult, SourceInput, SourcePage,
  SourcePreview, VectorNeighbors, VectorProjection, CompiledClaim, EpisodeInfo,
} from "./types";

export class StudioApi {
  constructor(readonly connection: Connection) {}

  private namespaceQuery(): URLSearchParams {
    const q = new URLSearchParams({ project_id: this.connection.namespace.project_id || "default" });
    for (const key of ["user_id", "agent_id", "session_id"] as const) {
      const value = this.connection.namespace[key];
      if (value) q.set(key, value);
    }
    return q;
  }

  async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const headers = new Headers(init.headers);
    headers.set("Accept", "application/json");
    if (init.body && !(init.body instanceof FormData)) headers.set("Content-Type", "application/json");
    if (this.connection.token) headers.set("Authorization", `Bearer ${this.connection.token}`);
    const response = await fetch(`${this.connection.baseUrl}${path}`, { ...init, headers });
    if (!response.ok) {
      const data = await response.json().catch(() => ({})) as { detail?: string };
      throw new Error(data.detail || `Request failed with HTTP ${response.status}`);
    }
    return response.json() as Promise<T>;
  }

  health() { return this.request<{ status: string; version: string; storage_ready: boolean; pending_jobs: number }>("/api/v1/health"); }
  session() { return this.request<{ role: string; effective_namespace: Namespace; capabilities: Record<string, boolean> }>("/api/v1/session"); }

  listSources(filters: { q?: string; kind?: string; status?: string; sort?: string; cursor?: number } = {}) {
    const q = this.namespaceQuery();
    q.set("limit", "48");
    if (filters.q) q.set("q", filters.q);
    if (filters.kind) q.append("kind", filters.kind);
    if (filters.status) q.set("status", filters.status);
    if (filters.sort) q.set("sort", filters.sort);
    if (filters.cursor) q.set("cursor", String(filters.cursor));
    return this.request<SourcePage>(`/api/v1/sources?${q}`);
  }
  getSource(id: string) { return this.request<Source>(`/api/v1/sources/${encodeURIComponent(id)}?${this.namespaceQuery()}`); }
  sourcePreview(id: string) { return this.request<SourcePreview>(`/api/v1/sources/${encodeURIComponent(id)}/preview?${this.namespaceQuery()}`); }
  async sourceBlob(id: string, disposition: "inline" | "attachment" = "inline") {
    const q = this.namespaceQuery(); q.set("disposition", disposition);
    const headers: Record<string, string> = {};
    if (this.connection.token) headers.Authorization = `Bearer ${this.connection.token}`;
    const response = await fetch(`${this.connection.baseUrl}/api/v1/sources/${encodeURIComponent(id)}/content?${q}`, { headers });
    if (!response.ok) throw new Error(`Source content failed with HTTP ${response.status}`);
    return response.blob();
  }
  ingest(sources: SourceInput[]) { return this.request<IngestionRun>("/api/v1/sources/ingest", { method: "POST", body: JSON.stringify({ sources, namespace: this.connection.namespace }) }); }
  ingestionRun(id: string) { return this.request<IngestionRun>(`/api/v1/ingestion-runs/${encodeURIComponent(id)}`); }
  listIngestionRuns() { return this.request<{ runs: IngestionRun[] }>(`/api/v1/ingestion-runs?${this.namespaceQuery()}`); }
  retryIngestion(id: string) { return this.request<IngestionRun>(`/api/v1/ingestion-runs/${encodeURIComponent(id)}/retry`, { method: "POST", body: "{}" }); }
  removeSource(id: string) { return this.request(`/api/v1/sources/${encodeURIComponent(id)}/remove`, { method: "POST", body: JSON.stringify({ reason: "Removed from Studio", namespace: this.connection.namespace }) }); }

  createQuery(query: string, mode: "query" | "search" = "query") { return this.request<QueryRun>("/api/v1/query-runs", { method: "POST", body: JSON.stringify({ query, mode, namespace: this.connection.namespace }) }); }
  getQuery(id: string) { return this.request<QueryRun>(`/api/v1/query-runs/${encodeURIComponent(id)}?${this.namespaceQuery()}`); }
  listQueries(search = "") { const q = this.namespaceQuery(); q.set("limit", "100"); if (search) q.set("q", search); return this.request<QueryRunPage>(`/api/v1/query-runs?${q}`); }
  removeQuery(id: string) { return this.request(`/api/v1/query-runs/${encodeURIComponent(id)}/remove`, { method: "POST", body: JSON.stringify({ query_ids: [id], confirm: true, namespace: this.connection.namespace }) }); }
  clearQueries() { return this.request("/api/v1/query-runs/remove", { method: "POST", body: JSON.stringify({ all_in_namespace: true, confirm: true, namespace: this.connection.namespace }) }); }

  async streamQuery(id: string, onRun: (run: QueryRun) => void, signal?: AbortSignal) {
    const q = this.namespaceQuery();
    const headers: Record<string, string> = { Accept: "text/event-stream" };
    if (this.connection.token) headers.Authorization = `Bearer ${this.connection.token}`;
    const response = await fetch(`${this.connection.baseUrl}/api/v1/query-runs/${encodeURIComponent(id)}/events?${q}`, { headers, signal });
    if (!response.ok || !response.body) throw new Error(`Live query stream failed with HTTP ${response.status}`);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop() || "";
      for (const event of events) {
        const data = event.split("\n").filter(line => line.startsWith("data: ")).map(line => line.slice(6)).join("\n");
        if (!data || event.includes("event: complete")) continue;
        onRun(JSON.parse(data) as QueryRun);
      }
    }
  }

  graph(view: GraphPage["view"], search = "", filters: { nodeType?: string; sourceId?: string; episodeId?: string } = {}) {
    const q = this.namespaceQuery();
    q.set("view", view);
    q.set("limit", "800");
    if (search) q.set("q", search);
    if (filters.nodeType) q.append("node_type", filters.nodeType);
    if (filters.sourceId) q.set("source_id", filters.sourceId);
    if (filters.episodeId) q.set("episode_id", filters.episodeId);
    return this.request<GraphPage>(`/api/v1/memory-graph?${q}`);
  }
  graphNeighbors(id: string, view: GraphPage["view"]) { const q = this.namespaceQuery(); q.set("view", view); return this.request<GraphPage>(`/api/v1/memory-graph/nodes/${encodeURIComponent(id)}/neighbors?${q}`); }

  catalog() { return this.request<MemoryCatalog>(`/api/v1/memory/catalog?${this.namespaceQuery()}`); }
  helperItems(helperId: string, search = "", cursor = "") {
    const q = this.namespaceQuery();
    q.set("limit", "80");
    if (search) q.set("q", search);
    if (cursor) q.set("cursor", cursor);
    return this.request<HelperPage>(`/api/v1/memory/helpers/${encodeURIComponent(helperId)}?${q}`);
  }
  memories(cursor?: number) {
    const q = this.namespaceQuery();
    q.set("limit", "80");
    q.set("include_retracted", "true");
    if (cursor) q.set("cursor", String(cursor));
    return this.request<MemoryPage>(`/api/v1/memories?${q}`);
  }
  documents(search = "", modality = "") {
    const q = this.namespaceQuery();
    q.set("limit", "80");
    if (search) q.set("q", search);
    if (modality) q.set("modality", modality);
    return this.request<{ documents: MemoryDocument[]; next_cursor?: number | null; total: number }>(`/api/v1/memory/documents?${q}`);
  }
  terms(search = "") {
    const q = this.namespaceQuery();
    q.set("limit", "60");
    if (search) q.set("q", search);
    return this.request<{ terms: MemoryTerm[]; next_cursor?: number | null; total: number }>(`/api/v1/memory/terms?${q}`);
  }
  claims() { return this.request<{ claims: CompiledClaim[] }>(`/api/v1/memory/claims?${this.namespaceQuery()}`); }
  episodes() { return this.request<{ episodes: EpisodeInfo[] }>(`/api/v1/episodes?${this.namespaceQuery()}`); }
  vectorProjection() { return this.request<VectorProjection>(`/api/v1/memory/vectors/projection?${this.namespaceQuery()}`); }
  vectorNeighbors(deltaId: string) {
    const q = this.namespaceQuery();
    q.set("delta_id", deltaId);
    return this.request<VectorNeighbors>(`/api/v1/memory/vectors/neighbors?${q}`);
  }
  retrievalGraph(seedId = "", edgeKind = "") {
    const q = this.namespaceQuery();
    if (seedId) q.set("seed_id", seedId);
    if (edgeKind) q.set("edge_kind", edgeKind);
    return this.request<GraphPage>(`/api/v1/memory/retrieval-graph?${q}`);
  }
  search(query: string, persist = false) {
    return this.request<SearchResult>("/api/v1/search", {
      method: "POST",
      body: JSON.stringify({ query, persist, include_diagnostics: true, namespace: this.connection.namespace }),
    });
  }

  providers() { return this.request<{ providers: Provider[] }>("/api/v1/providers"); }
  modelConfiguration() { return this.request<ModelConfigurationState>("/api/v1/model-configuration"); }
  saveModels(state: ModelConfigurationState, confirmEmbeddingRebuild: boolean) { return this.request<ModelConfigurationState>("/api/v1/model-configuration", { method: "PUT", body: JSON.stringify({ ...state.configuration, confirm_embedding_rebuild: confirmEmbeddingRebuild }) }); }
  testModel(role: "completion" | "embedding", selection: unknown) { return this.request<{ ok: boolean; message: string }>("/api/v1/model-configuration/test", { method: "POST", body: JSON.stringify({ role, selection }) }); }
  retrievalConfiguration() { return this.request<RetrievalConfiguration>("/api/v1/retrieval-configuration"); }
  saveRetrieval(configuration: RetrievalConfiguration) { return this.request<RetrievalConfiguration>("/api/v1/retrieval-configuration", { method: "PUT", body: JSON.stringify(configuration) }); }
}

export function initialConnection(): Connection {
  const storedNamespace = localStorage.getItem("trisynapse.namespace");
  return {
    baseUrl: sessionStorage.getItem("trisynapse.baseUrl") || "",
    token: sessionStorage.getItem("trisynapse.token") || "",
    namespace: storedNamespace ? JSON.parse(storedNamespace) as Namespace : { project_id: "default" },
  };
}
