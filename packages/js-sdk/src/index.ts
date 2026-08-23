export interface MemoryNamespace {
  user_id?: string;
  agent_id?: string;
  project_id?: string;
  session_id?: string;
}

export type DeltaKind = "observation" | "extraction" | "annotation" | "access" | "retraction";

export interface MemoryDelta {
  id: string;
  seq: number;
  written_at: string;
  observed_at?: string | null;
  kind: DeltaKind;
  namespace: Required<Pick<MemoryNamespace, "project_id">> & MemoryNamespace;
  episode_id?: string | null;
  evidence_refs: string[];
  confidence: number;
  scope: Record<string, unknown>;
  payload: Record<string, unknown>;
  text: string;
  source_ref?: Record<string, unknown> | string | null;
  locator?: Record<string, unknown> | string | null;
}

export interface SearchHit {
  item_id: string;
  kind: "observation" | "extraction" | "compiled" | "episode_recall";
  text: string;
  score: number;
  route: string;
  episode_id?: string | null;
  observed_at?: string | null;
  temporal_anchor?: string | null;
  source_delta_ids: string[];
  source_ref?: Record<string, unknown> | string | null;
  locator?: Record<string, unknown> | string | null;
  confidence: number;
  metadata: Record<string, unknown>;
}

export interface RetrievalTrace {
  query_id: string;
  query: string;
  namespace: Required<Pick<MemoryNamespace, "project_id">> & MemoryNamespace;
  query_kind: "fact" | "temporal" | "list" | "inference" | "multi_hop";
  stage: "fast" | "refine_1" | "refine_2" | "deep_recall" | "cold";
  confident: boolean;
  escalated: boolean;
  routing_seeds: string[];
  drilled_trace_count: number;
  episode_recall_in_answer_context: number;
}

export interface SearchResult {
  query_id: string;
  hits: SearchHit[];
  stage: string;
  confident: boolean;
  retrieval_trace: RetrievalTrace;
}

export interface QueryResult {
  query_id: string;
  question: string;
  answer: string;
  abstain: boolean;
  citations: Array<{ delta_id: string; excerpt: string; source_ref?: unknown; locator?: unknown }>;
  retrieval_hits: SearchHit[];
  retrieval_trace: RetrievalTrace;
}

export type SourceKind = "text" | "file" | "directory" | "archive" | "git" | "url" | "image";

export interface SourceInput {
  kind?: SourceKind;
  source_key?: string;
  url?: string;
  text?: string;
  content_base64?: string;
  filename?: string;
  title?: string;
  ref?: string;
  metadata?: Record<string, unknown>;
  scope?: Record<string, unknown>;
}

export interface SourceRecord {
  id: string;
  source_key: string;
  kind: SourceKind;
  title: string;
  uri?: string | null;
  content_hash: string;
  blob_path: string;
  media_type: string;
  filename?: string | null;
  byte_size: number;
  chunk_count: number;
  ingestion_run_id?: string | null;
  preview_type?: string | null;
  skipped_count: number;
  version: number;
  status: "active" | "superseded" | "removed";
  namespace: MemoryNamespace;
  metadata: Record<string, unknown>;
  delta_ids: string[];
  previous_source_id?: string | null;
  created_at: string;
  removed_at?: string | null;
}

export interface SourceIngestionResult {
  index: number;
  source_id?: string | null;
  source_key?: string | null;
  kind: SourceKind;
  status: "success" | "skipped" | "failed";
  episode_id?: string | null;
  delta_ids: string[];
  skipped_paths: string[];
  error?: string | null;
}

export interface IngestionRun {
  id: string;
  status: "pending" | "running" | "completed" | "partial" | "failed";
  namespace: MemoryNamespace;
  inputs: SourceInput[];
  results: SourceIngestionResult[];
  created_at: string;
  updated_at: string;
}

export interface RemoveResult {
  remove_id: string;
  removed_delta_ids: string[];
  requested_by: string;
  created_at: string;
}

export interface StoreValidation {
  ok: boolean;
  database_ok: boolean;
  delta_count: number;
  sequence_contiguous: boolean;
  source_blobs_checked: number;
  missing_source_ids: string[];
  corrupted_source_ids: string[];
  malformed_delta_ids: string[];
  broken_evidence_refs: string[];
  issues: string[];
}

export type ProviderRole = "completion" | "embedding";

export interface ProviderSelection {
  provider: string;
  model?: string | null;
  base_url?: string | null;
}

export interface ModelConfiguration {
  completion: ProviderSelection;
  embedding: ProviderSelection;
  revision: number;
  updated_at: string;
}

export interface ProviderDescriptor {
  id: string;
  display_name: string;
  roles: ProviderRole[];
  credential_env?: string | null;
  credential_configured: boolean;
  default_base_url?: string | null;
  native_protocol: boolean;
  notes?: string | null;
}

export interface ModelDescriptor {
  provider: string;
  id: string;
  display_name: string;
  roles: ProviderRole[];
  vision?: boolean | null;
  structured_output?: boolean | null;
  context_length?: number | null;
  source: "live" | "curated" | "custom";
  capability_status: "verified" | "unknown";
  metadata: Record<string, unknown>;
}

export interface ModelConfigurationChange {
  status: "applied" | "rebuild_pending" | "rebuild_failed";
  configuration: ModelConfiguration;
  pending_configuration?: ModelConfiguration | null;
  job_id?: string | null;
  rebuild_required: boolean;
  message?: string | null;
}

export interface ConnectionTestResult {
  ok: boolean;
  role: ProviderRole;
  provider: string;
  model?: string | null;
  message: string;
  billed_request: boolean;
  vision_supported?: boolean | null;
}

export interface RetrievalConfiguration {
  default_top_k: number;
  max_context_items: number;
  max_context_tokens: number;
  per_source_context_tokens: number;
  max_refinement_rounds: number;
  graph_hops: number;
  confidence_margin: number;
  deep_recall_enabled: boolean;
  answer_abstain_threshold: number;
  retrieval_profile: "auto" | "balanced" | "precise" | "broad" | "mixed" | "code" | "table" | "image" | "document" | "conversation";
  enabled_routes: string[];
  route_weights: Record<string, number>;
  revision: number;
  updated_at: string;
}

export interface QueryCandidateSnapshot {
  item_id: string;
  kind: string;
  route: string;
  rank: number;
  score: number;
  excerpt: string;
  source_delta_ids: string[];
  source_ref?: unknown;
  locator?: unknown;
}

export interface QueryStep {
  id: string;
  phase: string;
  label: string;
  sequence: number;
  status: "pending" | "running" | "succeeded" | "failed" | "skipped";
  parent_ids: string[];
  input: Record<string, unknown>;
  output: Record<string, unknown>;
  metrics: Record<string, string | number | boolean | null>;
  candidates: QueryCandidateSnapshot[];
  duration_ms?: number | null;
  created_at: string;
}

export interface QueryRun {
  id: string;
  mode: "query" | "search";
  status: "pending" | "running" | "completed" | "failed" | "interrupted";
  namespace: MemoryNamespace;
  query: string;
  answer?: string | null;
  abstain?: boolean | null;
  citations: QueryResult["citations"];
  steps: QueryStep[];
  retrieval_trace?: RetrievalTrace | null;
  retrieval_configuration: RetrievalConfiguration;
  generation_provenance: Record<string, unknown>;
  error?: string | null;
  attempt: number;
  partial: boolean;
  created_at: string;
  updated_at: string;
  completed_at?: string | null;
  duration_ms?: number | null;
}

export interface SourcePreviewItem {
  delta_id: string;
  kind: string;
  text: string;
  locator?: unknown;
  metadata: Record<string, unknown>;
}
export interface SourcePreview {
  source_id: string;
  preview_type: string;
  media_type: string;
  items: SourcePreviewItem[];
  next_cursor?: number | null;
  manifest: string[];
}

export interface MemoryGraphNode { id: string; type: "source" | "trace" | "episode" | "recall" | "claim" | "concept"; label: string; subtitle?: string | null; status?: string | null; data: Record<string, unknown>; }
export interface MemoryGraphEdge { id: string; source: string; target: string; type: string; label?: string | null; weight: number; data: Record<string, unknown>; }
export interface MemoryGraphPage { view: "knowledge" | "lineage" | "trace"; nodes: MemoryGraphNode[]; edges: MemoryGraphEdge[]; counts: Record<string, number>; truncated: boolean; next_cursor?: string | null; }

export interface ClientOptions {
  baseUrl?: string;
  apiKey?: string;
  namespace?: MemoryNamespace;
  fetch?: typeof globalThis.fetch;
}

export class TrisynapseMemory {
  readonly baseUrl: string;
  readonly namespace: MemoryNamespace;
  private readonly apiKey?: string;
  private readonly fetcher: typeof globalThis.fetch;

  constructor(options: ClientOptions = {}) {
    this.baseUrl = (options.baseUrl ?? "http://127.0.0.1:8765").replace(/\/$/, "");
    this.apiKey = options.apiKey;
    this.namespace = { project_id: "default", ...(options.namespace ?? {}) };
    this.fetcher = options.fetch ?? globalThis.fetch;
    if (!this.fetcher) throw new Error("A Fetch API implementation is required");
  }

  async health(): Promise<{ status: string; version: string; storage_ready: boolean; pending_jobs: number }> {
    return this.request("GET", "/api/v1/health", undefined, false);
  }

  async check(): Promise<Record<string, unknown> & { storage?: StoreValidation }> {
    return this.request("GET", "/api/v1/check");
  }

  async session(): Promise<{ role: "open" | "admin" | "scoped"; effective_namespace: MemoryNamespace; capabilities: Record<string, boolean> }> {
    return this.request("GET", "/api/v1/session");
  }

  async getRetrievalConfiguration(): Promise<RetrievalConfiguration> {
    return this.request("GET", "/api/v1/retrieval-configuration");
  }

  async setRetrievalConfiguration(configuration: RetrievalConfiguration): Promise<RetrievalConfiguration> {
    return this.request("PUT", "/api/v1/retrieval-configuration", configuration);
  }

  async listProviders(): Promise<{ providers: ProviderDescriptor[] }> {
    return this.request("GET", "/api/v1/providers");
  }

  async listModels(provider: string, role: ProviderRole, options: { refresh?: boolean; baseUrl?: string } = {}): Promise<{ models: ModelDescriptor[] }> {
    const query = new URLSearchParams({ role, refresh: String(options.refresh ?? false) });
    if (options.baseUrl) query.set("base_url", options.baseUrl);
    return this.request("GET", `/api/v1/providers/${encodeURIComponent(provider)}/models?${query}`);
  }

  async getModelConfiguration(): Promise<ModelConfigurationChange> {
    return this.request("GET", "/api/v1/model-configuration");
  }

  async setModelConfiguration(configuration: ModelConfiguration, options: { confirmEmbeddingRebuild?: boolean } = {}): Promise<ModelConfigurationChange> {
    return this.request("PUT", "/api/v1/model-configuration", {
      completion: configuration.completion,
      embedding: configuration.embedding,
      revision: configuration.revision,
      confirm_embedding_rebuild: options.confirmEmbeddingRebuild ?? false,
    });
  }

  async testModelConnection(role: ProviderRole, selection?: ProviderSelection): Promise<ConnectionTestResult> {
    return this.request("POST", "/api/v1/model-configuration/test", { role, selection });
  }

  async add(text: string, options: { episodeId?: string; scope?: Record<string, unknown>; externalKey?: string; modality?: string; sourceType?: string; retrievalFields?: Record<string, unknown> } = {}) {
    return this.request<{ delta_id: string }>("POST", "/api/v1/memory/observations", {
      text,
      episode_id: options.episodeId,
      scope: options.scope,
      external_key: options.externalKey,
      modality: options.modality,
      source_type: options.sourceType,
      retrieval_fields: options.retrievalFields,
      namespace: this.namespace,
    });
  }

  async addBatch(items: Array<{ text: string; episode_id?: string; external_key?: string; modality?: string; source_type?: string; retrieval_fields?: Record<string, unknown> }>) {
    return this.request<{ delta_ids: string[] }>("POST", "/api/v1/memories/batch", {
      items: items.map(item => ({ ...item, namespace: this.namespace })),
    });
  }

  async addFile(filename: string, contentBase64: string, options: { documentId?: string; title?: string; chunkChars?: number } = {}) {
    return this.request<{ document_id: string; chunk_count: number; delta_ids: string[] }>("POST", "/api/v1/memory/files", {
      filename,
      content_base64: contentBase64,
      document_id: options.documentId,
      title: options.title,
      chunk_chars: options.chunkChars,
      namespace: this.namespace,
    });
  }

  async search(query: string, options: { topK?: number; episodePrefix?: string; scope?: Record<string, unknown>; diagnostics?: boolean } = {}): Promise<SearchResult> {
    return this.request("POST", "/api/v1/search", {
      query,
      top_k: options.topK,
      episode_prefix: options.episodePrefix,
      scope: options.scope,
      include_diagnostics: options.diagnostics,
      namespace: this.namespace,
    });
  }

  async query(question: string, options: { topK?: number; episodePrefix?: string; scope?: Record<string, unknown> } = {}): Promise<QueryResult> {
    return this.request("POST", "/api/v1/query", {
      question,
      top_k: options.topK,
      episode_prefix: options.episodePrefix,
      scope: options.scope,
      namespace: this.namespace,
    });
  }

  async list(options: { cursor?: number; limit?: number; kinds?: DeltaKind[]; includeRetracted?: boolean } = {}) {
    const query = new URLSearchParams({ project_id: this.namespace.project_id ?? "default", limit: String(options.limit ?? 50) });
    if (this.namespace.user_id) query.set("user_id", this.namespace.user_id);
    if (this.namespace.agent_id) query.set("agent_id", this.namespace.agent_id);
    if (this.namespace.session_id) query.set("session_id", this.namespace.session_id);
    if (options.cursor) query.set("cursor", String(options.cursor));
    if (options.includeRetracted) query.set("include_retracted", "true");
    for (const kind of options.kinds ?? []) query.append("kind", kind);
    return this.request<{ items: MemoryDelta[]; next_cursor?: number | null }>("GET", `/api/v1/memories?${query}`);
  }

  async get(id: string, includeRetracted = false): Promise<MemoryDelta> {
    const query = new URLSearchParams({ project_id: this.namespace.project_id ?? "default", include_retracted: String(includeRetracted) });
    return this.request("GET", `/api/v1/memories/${encodeURIComponent(id)}?${query}`);
  }

  async history(id: string) {
    const query = new URLSearchParams({ project_id: this.namespace.project_id ?? "default" });
    return this.request<{ memory_id: string; events: MemoryDelta[] }>("GET", `/api/v1/memories/${encodeURIComponent(id)}/history?${query}`);
  }

  async correct(id: string, text: string, reason = "user correction") {
    return this.request<{ correction_id: string }>("POST", `/api/v1/memories/${encodeURIComponent(id)}/corrections`, { text, reason, namespace: this.namespace });
  }

  async forget(id: string, reason: string) {
    return this.request<{ retraction_id: string }>("POST", `/api/v1/memories/${encodeURIComponent(id)}/forget`, { reason, namespace: this.namespace });
  }

  async remove(ids: string[], reason: string): Promise<RemoveResult> {
    return this.request("POST", "/api/v1/memory/remove", {
      delta_ids: ids,
      reason,
      confirm: true,
      namespace: this.namespace,
    });
  }

  async ingest(sources: SourceInput[]): Promise<IngestionRun> {
    return this.request("POST", "/api/v1/sources/ingest", {
      sources,
      namespace: this.namespace,
    });
  }

  async listSources(includeRemoved = false, options: { search?: string; kind?: SourceKind; status?: string; sort?: "newest" | "oldest" | "title"; cursor?: number; limit?: number } = {}): Promise<{ sources: SourceRecord[]; next_cursor?: number | null; total: number; facets: Record<string, number> }> {
    const query = this.namespaceQuery();
    query.set("include_removed", String(includeRemoved));
    if (options.search) query.set("q", options.search);
    if (options.kind) query.append("kind", options.kind);
    if (options.status) query.set("status", options.status);
    if (options.sort) query.set("sort", options.sort);
    if (options.cursor) query.set("cursor", String(options.cursor));
    if (options.limit) query.set("limit", String(options.limit));
    return this.request("GET", `/api/v1/sources?${query}`);
  }

  async getSource(sourceId: string): Promise<SourceRecord> {
    return this.request("GET", `/api/v1/sources/${encodeURIComponent(sourceId)}?${this.namespaceQuery()}`);
  }

  async getSourcePreview(sourceId: string, options: { cursor?: number; limit?: number } = {}): Promise<SourcePreview> {
    const query = this.namespaceQuery();
    if (options.cursor) query.set("cursor", String(options.cursor));
    if (options.limit) query.set("limit", String(options.limit));
    return this.request("GET", `/api/v1/sources/${encodeURIComponent(sourceId)}/preview?${query}`);
  }

  async getSourceContent(sourceId: string, disposition: "inline" | "attachment" = "attachment"): Promise<Blob> {
    const query = this.namespaceQuery(); query.set("disposition", disposition);
    const headers: Record<string, string> = {};
    if (this.apiKey) headers.Authorization = `Bearer ${this.apiKey}`;
    const response = await this.fetcher(`${this.baseUrl}/api/v1/sources/${encodeURIComponent(sourceId)}/content?${query}`, { headers });
    if (!response.ok) throw new Error(`Trisynapse source download failed with HTTP ${response.status}`);
    return response.blob();
  }

  async removeSource(sourceId: string, reason = "source removed"): Promise<RemoveResult> {
    return this.request("POST", `/api/v1/sources/${encodeURIComponent(sourceId)}/remove`, {
      reason,
      namespace: this.namespace,
    });
  }

  async getIngestionRun(runId: string): Promise<IngestionRun> {
    return this.request("GET", `/api/v1/ingestion-runs/${encodeURIComponent(runId)}`);
  }

  async listIngestionRuns(limit = 100): Promise<{ runs: IngestionRun[] }> {
    const query = this.namespaceQuery(); query.set("limit", String(limit));
    return this.request("GET", `/api/v1/ingestion-runs?${query}`);
  }

  async createQueryRun(queryText: string, options: { mode?: "query" | "search"; topK?: number; abstainThreshold?: number } = {}): Promise<QueryRun> {
    return this.request("POST", "/api/v1/query-runs", { query: queryText, mode: options.mode ?? "query", top_k: options.topK, abstain_threshold: options.abstainThreshold, namespace: this.namespace });
  }

  async listQueryRuns(options: { search?: string; cursor?: string; limit?: number; mode?: "query" | "search"; status?: string; stage?: string } = {}): Promise<{ runs: QueryRun[]; next_cursor?: string | null }> {
    const query = this.namespaceQuery();
    if (options.search) query.set("q", options.search);
    if (options.cursor) query.set("cursor", options.cursor);
    if (options.limit) query.set("limit", String(options.limit));
    if (options.mode) query.set("mode", options.mode);
    if (options.status) query.set("status", options.status);
    if (options.stage) query.set("stage", options.stage);
    return this.request("GET", `/api/v1/query-runs?${query}`);
  }

  async getQueryRun(queryId: string): Promise<QueryRun> {
    return this.request("GET", `/api/v1/query-runs/${encodeURIComponent(queryId)}?${this.namespaceQuery()}`);
  }

  async removeQueryRuns(queryIds: string[] = [], allInNamespace = false): Promise<{ removed_query_ids: string[] }> {
    return this.request("POST", "/api/v1/query-runs/remove", { query_ids: queryIds, all_in_namespace: allInNamespace, confirm: true, namespace: this.namespace });
  }

  async getMemoryGraph(view: MemoryGraphPage["view"] = "knowledge", options: { search?: string; sourceId?: string; episodeId?: string; cursor?: string; limit?: number } = {}): Promise<MemoryGraphPage> {
    const query = this.namespaceQuery(); query.set("view", view);
    if (options.search) query.set("q", options.search);
    if (options.sourceId) query.set("source_id", options.sourceId);
    if (options.episodeId) query.set("episode_id", options.episodeId);
    if (options.cursor) query.set("cursor", options.cursor);
    if (options.limit) query.set("limit", String(options.limit));
    return this.request("GET", `/api/v1/memory-graph?${query}`);
  }

  async getMemoryGraphNeighbors(nodeId: string, view: MemoryGraphPage["view"] = "lineage"): Promise<MemoryGraphPage> {
    const query = this.namespaceQuery(); query.set("view", view);
    return this.request("GET", `/api/v1/memory-graph/nodes/${encodeURIComponent(nodeId)}/neighbors?${query}`);
  }

  async retryIngestion(runId: string): Promise<IngestionRun> {
    return this.request("POST", `/api/v1/ingestion-runs/${encodeURIComponent(runId)}/retry`, {});
  }

  async profile(query?: string) {
    return this.request<{ static: string[]; dynamic: string[]; search_results?: SearchHit[] }>("POST", "/api/v1/profile", { query, namespace: this.namespace });
  }

  async feedback(queryId: string, helpful: boolean, comment?: string) {
    return this.request<{ feedback_id: string }>("POST", "/api/v1/feedback", { query_id: queryId, helpful, comment, namespace: this.namespace });
  }

  private namespaceQuery(): URLSearchParams {
    const query = new URLSearchParams({ project_id: this.namespace.project_id ?? "default" });
    if (this.namespace.user_id) query.set("user_id", this.namespace.user_id);
    if (this.namespace.agent_id) query.set("agent_id", this.namespace.agent_id);
    if (this.namespace.session_id) query.set("session_id", this.namespace.session_id);
    return query;
  }

  private async request<T>(method: string, path: string, body?: unknown, authenticated = true): Promise<T> {
    const headers: Record<string, string> = { Accept: "application/json" };
    if (body !== undefined) headers["Content-Type"] = "application/json";
    if (authenticated && this.apiKey) headers.Authorization = `Bearer ${this.apiKey}`;
    const response = await this.fetcher(`${this.baseUrl}${path}`, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({})) as { detail?: string };
      throw new Error(payload.detail ?? `Trisynapse API request failed with HTTP ${response.status}`);
    }
    return response.json() as Promise<T>;
  }
}

export default TrisynapseMemory;
