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
  max_refinement_rounds: number;
  graph_hops: number;
  confidence_margin: number;
  deep_recall_enabled: boolean;
  answer_abstain_threshold: number;
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
  view: "knowledge" | "lineage" | "trace";
  nodes: GraphNode[];
  edges: GraphEdge[];
  counts: Record<string, number>;
  truncated: boolean;
  next_cursor?: string | null;
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
