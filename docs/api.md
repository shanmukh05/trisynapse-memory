# API and Interfaces

Trisynapse has one engine and four ways to call it. The namespace, source, lifecycle, model, and citation rules are the same everywhere.

## Python

```python
from trisynapse_memory import MemoryEngine, MemoryNamespace, SourceInput

ns = MemoryNamespace(user_id="maya", project_id="personal")
memory = MemoryEngine.from_env("~/.trisynapse-memory/store", namespace=ns)
```

### Observations and conversations

```python
item = memory.add(
    "Maya prefers three-bullet updates.",
    episode_id="chat:8",
    external_key="personal:maya:message:8:4",
)

memory.ingest_messages(
    [
        {"id": "m1", "role": "user", "content": "Send the update Friday."},
        {"id": "m2", "role": "assistant", "content": "Understood."},
    ],
    episode_id="chat:9",
)
```

Use `external_key` when a caller may repeat an observation write. Use `add_batch()` only for text that is already normalized:

```python
memory.add_batch([
    {"text": "First observation", "episode_id": "chat:10"},
    {"text": "Second observation", "episode_id": "chat:10"},
])
```

### Unified source ingestion

```python
one = memory.ingest(SourceInput(kind="file", path="report.pdf"))

run = memory.ingest_many([
    SourceInput(kind="text", text="Release notes must name the owner.", source_key="release-rule"),
    SourceInput(kind="directory", path="./src", source_key="service-repo"),
    SourceInput(kind="url", url="https://example.com/handbook"),
    SourceInput(kind="git", url="https://github.com/org/public-repo.git", ref="main"),
], on_progress=lambda stage: print(stage))

for result in run.results:
    print(result.index, result.status, result.source_id, result.error)
```

`ingest_many()` preprocesses at most four sources concurrently and commits valid sources in input order. The optional `on_progress` callback reports durable-run creation, preprocessing, Trace writes, and finalization. The run status is `completed`, `partial`, or `failed`.

```python
memory.list_sources()
source = memory.get_source("src_...")
memory.remove_source(source.id, reason="approved source deletion")

run = memory.get_ingestion_run("run_...")
retried = memory.retry_ingestion(run.id)  # only failed inputs; resumes interrupted runs
```

`SourceInput` accepts exactly one of `path`, `url`, `text`, or `content_base64`. REST does not accept `path`; it accepts uploaded base64 bytes and public URLs.

### Search and grounded answers

```python
search = memory.search("How should the update look?", top_k=8)
for hit in search.hits:
    print(hit.score, hit.text, hit.locator)

answer = memory.query(
    "How should I send the update?",
    on_step=lambda step: print(step.phase, step.label, step.duration_ms),
)
print(answer.answer, answer.abstain)
for hit in answer.retrieval_hits:  # populated even if answer.abstain is true
    print(hit.score, hit.locator, hit.metadata["modality"], hit.metadata["retrieval_fields"])
for citation in answer.citations:
    print(citation.delta_id, citation.source_ref, citation.locator)
```

`search()` returns ranked evidence for application logic. `query()` returns an answer or abstention with citations and `retrieval_hits`. Retrieval hits are captured before answer generation and remain present after an abstention; citations are only the evidence attached to a completed answer. Both methods accept `on_step`, which receives each durable input, classification, retrieval, confidence, refinement, Deep Recall, grounding, answer, and audit step as it completes. `retrieval_trace` explains routing and grounding.

Content, query text, metadata, locators, and retrieval fields are accepted as supplied. Sanitize sensitive values in the application before calling Python, REST, or TypeScript ingestion and query methods.

For an already-normalized observation, Python and REST also accept `modality`, `source_type`, and `retrieval_fields`. The TypeScript `add()` options are `modality`, `sourceType`, and `retrievalFields`. Prefer `ingest()` for files because it derives these values and locators safely.

The production retriever has a replaceable planner and route registry:

```python
from trisynapse_memory import MemoryEngine, QueryPlan, RouteRegistry

class ProductPlanner:
    def plan(self, query, **options):
        return QueryPlan(
            query=query,
            query_kind="fact",
            terms=tuple(query.casefold().split()),
            modalities=("code",),
            routes=("bm25", "semantic", "code"),
            profile="code",
        )

memory = MemoryEngine.open("./memory", query_planner=ProductPlanner())
```

Implement `RetrievalRoute.name` and `rank(plan, context)`, then add the instance to a `RouteRegistry`. Pass the registry as `retrieval_routes=`. Custom routes return `(item_id, score)` pairs and still go through fusion, confidence checks, Trace grounding, and context limits.

Applications with a model-specific local tokenizer can also supply context accounting:

```python
class ModelTokenCounter:
    name = "my-model-tokenizer-v1"
    exact = True

    def count(self, text: str) -> int:
        return len(my_tokenizer.encode(text))

memory = MemoryEngine.open("./memory", token_counter=ModelTokenCounter())
```

The built-in `token_counter_for()` prefers a completion provider's local `count_tokens()` method, then Tiktoken for recognized OpenAI models, then `ApproximateTokenCounter`. Grounding steps and search-hit metadata expose the counter name; grounding also records its `exact` flag and selected context token count.

### Benchmark runs

`POST /api/v1/benchmarks/runs` accepts `benchmark`, `mode`, `limit`, `sampling`, and an optional non-secret `judge` provider selection. For a partial LoCoMo run, `sampling: "auto"` uses deterministic stratification across conversations and question categories. Use `sequential` only to reproduce an older prefix-based run.

Schema-v4 artifacts retain bounded pre-answer hits, query steps, and the final store-validation result. The important metrics are:

- `evidence_hit_at_k`: fraction of questions retrieving at least one gold evidence ID;
- `evidence_recall_at_k`: mean fraction of all gold evidence IDs retrieved;
- `all_evidence_retrieved_rate`: questions for which every gold evidence ID was retrieved;
- `citation_evidence_recall` and `citation_precision`: answer citation quality, measured separately;
- `abstention_rate`: answer-generation behavior, not a retrieval miss.

API keys remain environment-only. An independent judge can be selected without changing the store completion model:

```json
{
  "benchmark": "locomo",
  "mode": "end-to-end",
  "limit": 100,
  "sampling": "stratified",
  "judge": {"provider": "anthropic", "model": "claude-sonnet-4-5"}
}
```

### Lifecycle

```python
memory.correct(
    delta_id=item.id,
    text="Maya prefers at most three bullets.",
    reason="clarified maximum",
)

memory.forget(delta_id=item.id, reason="preference expired")

memory.remove(
    delta_ids=[item.id],
    reason="approved privacy deletion",
    requested_by="privacy-service",
)
```

`correct()` and `forget()` append events. `remove()` physically redacts selected deltas and returns `RemoveResult` with `remove_id` and `removed_delta_ids`. There is no `purge()` alias.

Use `memory.validate_store()` for operational validation. It returns `StoreValidation` with SQLite status, delta count, sequence continuity, malformed or broken evidence references, and missing or corrupted retained-source IDs. This is a consistency and corruption check, not proof against an administrator who can rewrite the database.

### Model configuration

Provider and model choices are stored once per memory store. They never contain API keys.

```python
from trisynapse_memory import ModelConfiguration, ProviderSelection

current = memory.get_model_configuration()
current.completion = ProviderSelection(
    provider="anthropic",
    model="claude-sonnet-4-5",
)
change = memory.set_model_configuration(current)

models = memory.list_models("completion", "anthropic")
print(models[0].id, models[0].vision)

# This sends one small, potentially billable request.
result = memory.test_model_connection("completion")
```

The Python methods are:

- `list_providers()` and `list_models(role, provider, refresh=False)`;
- `get_model_configuration()` and `get_model_configuration_status()`;
- `set_model_configuration(configuration, confirm_embedding_rebuild=False, wait=False)`;
- `test_model_connection(role, selection=None)` and `check()`.

Completion changes apply to future operations immediately. Existing generated records keep their provider, model, and prompt provenance.

Changing the embedding provider, endpoint, or model on a non-empty store raises `EmbeddingRebuildRequired` unless `confirm_embedding_rebuild=True`. The confirmed change returns a durable job ID. `wait=True` processes the job before returning. The old index and configuration remain active until the new index is complete; failure leaves them unchanged.

## REST

Start the service:

```bash
trisynapse-memory init
trisynapse-memory serve --studio
```

Use `Authorization: Bearer TOKEN` except for health. The OpenAPI document is at `/openapi.json`.

### Core routes

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/v1/health` | Version, store readiness, pending jobs |
| `POST` | `/api/v1/memory/observations` | Add one observation |
| `POST` | `/api/v1/memory/messages` | Add a conversation episode |
| `POST` | `/api/v1/memories/batch` | Add normalized observations |
| `GET` | `/api/v1/memories` | Cursor-paginated Trace records |
| `GET` | `/api/v1/memories/{id}` | Get one record |
| `GET` | `/api/v1/memories/{id}/history` | Linked lifecycle history |
| `POST` | `/api/v1/search` | Ranked evidence and diagnostics |
| `POST` | `/api/v1/query` | Grounded answer and citations |
| `POST` | `/api/v1/query-runs` | Start a durable live query or search and return `202` |
| `GET` | `/api/v1/query-runs` | Filtered, cursor-paginated query history |
| `GET` | `/api/v1/query-runs/{id}` | Saved answer, citations, retrieval configuration, and ordered steps |
| `GET` | `/api/v1/query-runs/{id}/events` | Reconnectable server-sent run updates |
| `POST` | `/api/v1/query-runs/{id}/remove` | Confirmed removal of one operational query record |
| `POST` | `/api/v1/query-runs/remove` | Confirmed bulk history removal |
| `POST` | `/api/v1/memories/{id}/corrections` | Append correction |
| `POST` | `/api/v1/memories/{id}/forget` | Append logical retraction |
| `POST` | `/api/v1/memory/remove` | Confirmed physical redaction |

### Model and operation routes

These store-wide routes require the administrator bearer token when authentication is enabled.

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/v1/providers` | Provider roles, credential variable names, and readiness |
| `GET` | `/api/v1/providers/{provider}/models?role=...` | Cached or live model catalog |
| `GET` | `/api/v1/model-configuration` | Active and pending selections with revision |
| `PUT` | `/api/v1/model-configuration` | Save choices; include rebuild confirmation when required |
| `POST` | `/api/v1/model-configuration/test` | Explicit, potentially billable connection test |
| `GET/PUT` | `/api/v1/retrieval-configuration` | Revisioned retrieval and abstention defaults |
| `GET` | `/api/v1/session` | Authenticated role, effective namespace, and safe capabilities |
| `GET` | `/api/v1/check` | Installation, store, provider, and pending-work checks |
| `GET` | `/api/v1/health` | Lightweight liveness and store readiness |

`PUT` returns `200` for an immediate change, `202` plus `job_id` for an embedding rebuild, and `409` when confirmation or a fresh revision is required. API keys are never accepted or returned.

Retrieval configuration includes `retrieval_profile`, `enabled_routes`, `route_weights`, `max_context_tokens`, and `per_source_context_tokens`, in addition to result count, graph depth, refinement, confidence, Deep Recall, and abstention settings. For example:

```json
{
  "default_top_k": 12,
  "max_context_items": 24,
  "max_context_tokens": 6000,
  "per_source_context_tokens": 2000,
  "retrieval_profile": "auto",
  "enabled_routes": ["bm25", "semantic", "temporal", "graph", "code", "table", "image", "document", "conversation"],
  "route_weights": {"code": 1.4},
  "max_refinement_rounds": 2,
  "graph_hops": 2,
  "confidence_margin": 0.018,
  "deep_recall_enabled": true,
  "answer_abstain_threshold": 0.1,
  "revision": 0
}
```

Model selection uses a separate request body (the REST wrapper also accepts the rebuild confirmation flag):

```json
{
  "completion": {"provider": "deepseek", "model": "deepseek-chat"},
  "embedding": {"provider": "deepinfra", "model": "BAAI/bge-m3"},
  "revision": 3,
  "confirm_embedding_rebuild": true
}
```

### Source routes

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/api/v1/sources/ingest` | Accept up to 100 descriptors; return `202` and durable run |
| `GET` | `/api/v1/sources` | Search, filter, sort, and paginate source cards with facets |
| `GET` | `/api/v1/sources/{source_id}` | Inspect versions, metadata, and delta IDs |
| `GET` | `/api/v1/sources/{source_id}/preview` | Safe structured, locator-aware preview items |
| `GET` | `/api/v1/sources/{source_id}/content` | Authorized inline-safe content or original download |
| `POST` | `/api/v1/sources/{source_id}/remove` | Delete source and derived memory |
| `GET` | `/api/v1/ingestion-runs` | Recent durable runs in the namespace |
| `GET` | `/api/v1/ingestion-runs/{run_id}` | Follow run status and ordered results |
| `POST` | `/api/v1/ingestion-runs/{run_id}/retry` | Resume interrupted work or retry failed items |

File and document routes remain compatibility wrappers over the source pipeline.

Example mixed request:

```bash
curl -X POST http://127.0.0.1:8765/api/v1/sources/ingest \
  -H "Authorization: Bearer $TRISYNAPSE_MEMORY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": {"project_id": "demo"},
    "sources": [
      {"kind": "url", "url": "https://example.com/guide"},
      {"kind": "text", "text": "Keep status concise", "source_key": "status-rule"}
    ]
  }'
```

REST safeguards are 25 MiB per upload, 250 MiB retained input per run, 100 descriptors, bounded archive expansion/count, public-network URL validation on every redirect, fetch timeouts, and redirect limits.

### Query workflow events and graph views

`POST /api/v1/query-runs` accepts `query`, `mode` (`query` or `search`), optional retrieval overrides, and a namespace. Its SSE endpoint emits complete current `QueryRun` snapshots as `run` events and ends with `complete`. Reconnect with `Last-Event-ID`; clients can always recover with the ordinary `GET` route.

`QueryStep` contains the stage name, parent IDs, safe input/output metadata, metrics, duration, and bounded candidate snapshots. A failed or interrupted run keeps its completed steps.

The graph routes are:

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/v1/memory-graph?view=knowledge` | Concepts, claims, and their relationships |
| `GET` | `/api/v1/memory-graph?view=lineage` | Source-to-Trace-to-Recall provenance |
| `GET` | `/api/v1/memory-graph?view=trace` | Ordered evidence and lifecycle connections |
| `GET` | `/api/v1/memory-graph/nodes/{id}/neighbors` | Expand one node without loading the full store |

Python exposes `get/list/remove_query_runs()`, `create/execute_query_run()`, `get/set_retrieval_configuration()`, `source_preview()`, `source_content_path()`, and `memory_graph()`. The TypeScript SDK mirrors these with typed methods.

## TypeScript

`@trisynapse/trisynapse-memory` is a typed Fetch client for the REST API.

```ts
import { TrisynapseMemory } from "@trisynapse/trisynapse-memory";

const memory = new TrisynapseMemory({
  baseUrl: "http://127.0.0.1:8765",
  apiKey: process.env.TRISYNAPSE_MEMORY_API_KEY,
  namespace: { user_id: "maya", project_id: "personal" },
});

const run = await memory.ingest([
  { kind: "url", url: "https://example.com/guide" },
  { kind: "file", filename: "notes.md", content_base64: notesBase64 },
]);

const finished = await memory.getIngestionRun(run.id);
const sources = await memory.listSources();
await memory.removeSource(sources.sources[0].id, "user request");

const modelState = await memory.getModelConfiguration();
modelState.configuration.completion = {
  provider: "kimi",
  model: "kimi-k2.5",
};
await memory.setModelConfiguration(modelState.configuration);
```

The client also includes `listProviders()`, `listModels()`, `getModelConfiguration()`, `setModelConfiguration()`, `testModelConnection()`, and `check()`. Pass `{confirmEmbeddingRebuild: true}` only after the user has approved an embedding rebuild.

## CLI and interactive terminal

Running `trisynapse-memory` with no arguments opens the Textual terminal only when stdin/stdout are TTYs. Piped execution prints help and exits, so it never waits for input unexpectedly.

```bash
trisynapse-memory ingest ./src ./guide.pdf https://example.com/page
trisynapse-memory ingest --manifest sources.json
trisynapse-memory sources list
trisynapse-memory sources show SOURCE_ID
trisynapse-memory --yes remove source SOURCE_ID --reason "user request"
trisynapse-memory runs list
trisynapse-memory runs retry RUN_ID
trisynapse-memory --json check
trisynapse-memory models providers
trisynapse-memory models list --role completion --provider anthropic
trisynapse-memory models set completion anthropic claude-sonnet-4-5
trisynapse-memory --yes models set embedding deepinfra BAAI/bge-m3
```

Interactive plain text runs `query()`. Slash commands are `/ingest`, `/sources`, `/search`, `/timeline`, `/history`, `/correct`, `/forget`, `/remove`, `/jobs`, `/namespace`, `/model`, `/check`, `/config`, `/help`, `/clear`, and `/exit`.

`/model` opens a searchable, scrollable selector for completion and embeddings. It shows missing credential variables, catalog source, context length, and vision capability. Refreshing the catalog does not run inference. Testing a connection does. An embedding change requires the visible rebuild confirmation before Apply succeeds.

## Namespaces

All surfaces accept `user_id`, `agent_id`, `project_id`, and `session_id`. `project_id` defaults to `default`. A namespace-scoped API key cannot override its assigned namespace. Keep values stable across writes and reads.

See [Completion and embedding providers](operations.md#completion-and-embedding-providers) for API keys, supported providers, model selection, custom endpoints, and vision behavior. The rest of [Operations](operations.md) covers installation, backups, security, and releases.
