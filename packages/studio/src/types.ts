export type Namespace = {
  project_id: string;
  user_id?: string | null;
  agent_id?: string | null;
  session_id?: string | null;
};

export type SourceKind = "text" | "file" | "directory" | "archive" | "git" | "url" | "image";

export type Source = {
  id: string;
  source_key: string;
  kind: SourceKind;
  title: string;
  uri?: string | null;
  content_hash: string;
  media_type: string;
  filename?: string | null;
  byte_size: number;
  chunk_count: number;
  ingestion_run_id?: string | null;
  preview_type?: string | null;
  skipped_count: number;
  version: number;
  status: "active" | "superseded" | "removed";
  namespace: Namespace;
  metadata: Record<string, unknown>;
  delta_ids: string[];
  previous_source_id?: string | null;
  created_at: string;
  removed_at?: string | null;
};

export type SourcePage = {
  sources: Source[];
  next_cursor?: number | null;
  total: number;
  facets: Record<string, number>;
};

export type PreviewItem = {
  delta_id: string;
  kind: string;
  text: string;
  locator?: Record<string, unknown> | string | null;
  metadata: Record<string, unknown>;
};

export type SourcePreview = {
  source_id: string;
  preview_type: string;
  media_type: string;
  items: PreviewItem[];
  next_cursor?: number | null;
  manifest: string[];
};

export type SourceInput = {
  kind: SourceKind;
  source_key?: string;
  text?: string;
  url?: string;
  content_base64?: string;
  filename?: string;
  title?: string;
  ref?: string;
  metadata?: Record<string, unknown>;
};

export type IngestionResult = {
  index: number;
  source_id?: string | null;
  source_key?: string | null;
  kind: SourceKind;
  status: "success" | "skipped" | "failed";
  delta_ids: string[];
  skipped_paths: string[];
  error?: string | null;
};

export type IngestionRun = {
  id: string;
  status: "pending" | "running" | "completed" | "partial" | "failed";
  namespace: Namespace;
  inputs: SourceInput[];
  results: IngestionResult[];
  created_at: string;
  updated_at: string;
};

export type Candidate = {
  item_id: string;
  kind: string;
  route: string;
  rank: number;
  score: number;
  excerpt: string;
  source_delta_ids: string[];
  source_ref?: unknown;
  locator?: unknown;
};

export type QueryStep = {
  id: string;
  phase: string;
  label: string;
  sequence: number;
  status: "pending" | "running" | "succeeded" | "failed" | "skipped";
  parent_ids: string[];
  input: Record<string, unknown>;
  output: Record<string, unknown>;
  metrics: Record<string, string | number | boolean | null>;
  candidates: Candidate[];
  duration_ms?: number | null;
  created_at: string;
};

export type Citation = {
  delta_id: string;
  excerpt: string;
  source_ref?: { id?: string; title?: string } | string | null;
  locator?: unknown;
};

export type RetrievalConfiguration = {
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
};

export type QueryRun = {
  id: string;
  mode: "query" | "search";
  status: "pending" | "running" | "completed" | "failed" | "interrupted";
  namespace: Namespace;
  query: string;
  answer?: string | null;
  abstain?: boolean | null;
  citations: Citation[];
  steps: QueryStep[];
  retrieval_trace?: {
    stage: string;
    confident: boolean;
    escalated: boolean;
    query_kind: string;
    top_score: number;
    margin?: number | null;
  } | null;
  retrieval_configuration: RetrievalConfiguration;
  generation_provenance: Record<string, unknown>;
  error?: string | null;
  attempt: number;
  partial: boolean;
  created_at: string;
  updated_at: string;
  completed_at?: string | null;
  duration_ms?: number | null;
};

export type QueryRunPage = { runs: QueryRun[]; next_cursor?: string | null };

export type GraphNode = {
  id: string;
  type: "source" | "trace" | "episode" | "recall" | "claim" | "concept";
  label: string;
  subtitle?: string | null;
  status?: string | null;
  data: Record<string, unknown>;
};

export type GraphEdge = {
  id: string;
  source: string;
  target: string;
  type: string;
  label?: string | null;
  weight: number;
  data: Record<string, unknown>;
};

export type GraphPage = {
  view: "knowledge" | "lineage" | "trace" | "retrieval";
  nodes: GraphNode[];
  edges: GraphEdge[];
  counts: Record<string, number>;
  truncated: boolean;
  next_cursor?: string | null;
};

export type HelperKind = "timeline" | "table" | "postings" | "embedding" | "cards" | "graph";

export type CatalogHelper = {
  id: string;
  title: string;
  kind: HelperKind;
  inspect_path: string;
  playground_seed?: string | null;
  count: number;
  health: Record<string, unknown>;
};

export type CatalogRoute = {
  name: string;
  title: string;
  enabled: boolean;
  weight: number;
};

export type MemoryCatalog = {
  helpers: CatalogHelper[];
  retrieval_routes: CatalogRoute[];
};

export type HelperItem = {
  id: string;
  helper_id: string;
  kind: string;
  title: string;
  subtitle?: string | null;
  excerpt?: string | null;
  status?: string | null;
  score?: number | null;
  data: Record<string, unknown>;
};

export type HelperPage = {
  helper_id: string;
  kind: HelperKind | string;
  items: HelperItem[];
  next_cursor?: string | null;
  truncated: boolean;
};

export type MemoryDocument = {
  id: string;
  seq: number;
  kind: string;
  modality: string;
  source_type: string;
  text: string;
  token_count: number;
  active: boolean;
  episode_id?: string | null;
  text_hash: string;
  fields: Record<string, string>;
  locator?: Record<string, unknown> | string | null;
};

export type MemoryTermPosting = {
  delta_id: string;
  term_frequency: number;
  excerpt: string;
  token_count: number;
};

export type MemoryTerm = {
  term: string;
  document_frequency: number;
  postings: MemoryTermPosting[];
};

export type VectorPoint = {
  id: string;
  x: number;
  y: number;
  modality: string;
  excerpt: string;
  text_hash: string;
};

export type VectorProjection = {
  points: VectorPoint[];
  model?: string | null;
  fingerprint?: string | null;
  embedded: number;
  searchable: number;
  sampled: number;
};

export type VectorNeighbor = { id: string; score: number; excerpt: string; modality: string };
export type VectorNeighbors = { delta_id: string; neighbors: VectorNeighbor[]; model?: string | null };

export type CompiledClaim = {
  id: string;
  claim_key: string;
  text: string;
  status: "ACTIVE" | "SUPERSEDED" | "CONTESTED";
  source_delta_ids: string[];
  observation_delta_ids: string[];
  temporal_anchor?: string | null;
  confidence: number;
  subject?: string | null;
  relation?: string | null;
  object?: string | null;
  objects?: string[];
};

export type EpisodeInfo = {
  episode_id: string;
  delta_count: number;
  observation_count: number;
  extraction_count: number;
  first_observed_at?: string | null;
  last_observed_at?: string | null;
  stale: boolean;
};

export type SearchHit = {
  item_id: string;
  kind: string;
  text: string;
  score: number;
  route: string;
  episode_id?: string | null;
  source_delta_ids: string[];
  locator?: unknown;
};

export type SearchResult = {
  query_id: string;
  hits: SearchHit[];
  stage: string;
  confident: boolean;
  retrieval_trace?: {
    query_kind?: string;
    stage?: string;
    routes?: Record<string, string[]>;
    top_score?: number;
    margin?: number | null;
  };
};

export type MemoryDelta = {
  id: string;
  seq: number;
  kind: string;
  text: string;
  episode_id?: string | null;
  evidence_refs: string[];
  locator?: Record<string, unknown> | string | null;
  source_ref?: unknown;
  observed_at?: string | null;
  subject?: string | null;
  relation?: string | null;
  object?: string | null;
};

export type MemoryPage = { items: MemoryDelta[]; next_cursor?: number | null };

export type InspectorRecord = {
  id: string;
  helperId: string;
  kind: string;
  title: string;
  subtitle?: string | null;
  excerpt?: string | null;
  status?: string | null;
  data: Record<string, unknown>;
};

export type Provider = {
  id: string;
  display_name: string;
  roles: Array<"completion" | "embedding">;
  credential_env?: string | null;
  credential_configured: boolean;
};

export type ModelSelection = { provider: string; model?: string | null; base_url?: string | null };
export type ModelConfigurationState = {
  status: string;
  configuration: {
    completion: ModelSelection;
    embedding: ModelSelection;
    revision: number;
  };
  pending_configuration?: unknown;
  job_id?: string | null;
  message?: string | null;
};

export type Connection = { baseUrl: string; token: string; namespace: Namespace };
