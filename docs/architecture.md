# How Trisynapse Memory Works

This is the main introduction to the Trisynapse memory architecture. You do not need to know the codebase first.

## Vocabulary

These are the names used in code, JSON, and Studio.

| Term | Meaning |
|---|---|
| **Ingest** | The write path: accept a source and append records. |
| **Memory log** | Durable ordered evidence. This is the source of truth. |
| **Indexes** | Rebuildable search helpers over the memory log (BM25, embeddings, claims, group summaries). |
| **Retrieval** | The read path: find candidates in indexes, then ground the answer in the memory log. |
| **Query Run** | Diagnostic history of one search or answer. Not evidence. |
| **Record** | One ordered piece of evidence in the memory log. |
| **Raw source** | Original ingested chunk (page, message, code span, table row). Kind: `raw_source`. |
| **Fact** | Extracted statement that cites raw sources. Kind: `fact`. |
| **Note** | User correction or instruction. Kind: `note`. |
| **Record group** | A group of records from one conversation, document, or event (`group_id`). |
| **Group summary** | A short index card used to route a question toward a record group. |
| **Deep summary** | A wider retrieval stage when fast routes are not confident. |
| **Claim** | A compiled, disposable assertion built from facts. |
| **Embedding index** | Vector store used for semantic search. |

## One document, end to end

The easiest way to understand the architecture is to follow one document through it. This example uses a two-page PDF named `atlas-brief.pdf`:

```text
Page 1: Project Atlas launched on 14 May 2026.
Page 2: Maya Chen is the owner of Project Atlas.
```

We will ingest it, inspect what the memory log and indexes keep, ask a question, and follow every retrieval stage. IDs, timings, and scores below are shortened illustrative values; their shapes and relationships match the real records.

```mermaid
flowchart TD
    subgraph WRITE["1. Ingest · write path"]
        S["atlas-brief.pdf"] --> P["PDF pages + locators"]
        P --> AC["Accepted chunks + source metadata"]
    end

    subgraph LOG["2. Memory log · durable evidence"]
        TR["Ordered page raw sources"]
    end

    AC --> TR

    subgraph IDX["3. Indexes · rebuildable helpers"]
        TR --> BM["BM25 postings"]
        TR --> VE["Vector embeddings"]
        TR --> ER["Batch summary"]
        TR --> CL["Claims + graph links"]
    end

    subgraph READ["4. Retrieval and grounding · read path"]
        Q["When did Atlas launch, and who owns it?"] --> PL["Plan + run routes"]
        BM --> PL
        VE --> PL
        ER --> PL
        CL --> PL
        PL --> FU["Fuse + confidence check"]
        FU --> GR["Drill back to the memory log"]
        GR --> AN["Grounded answer + page citations"]
    end

    subgraph EXPLAIN["5. Query Run · explanation path"]
        Q --> QR["Durable Query Run"]
        PL -. save route candidates .-> QR
        FU -. save scores and decisions .-> QR
        GR -. save grounded evidence .-> QR
        AN -. save answer and citations .-> QR
    end
```

The five boxes are deliberately separate:

- **Ingest** decides what may be written.
- **Memory log** is the durable evidence produced by Ingest.
- **Indexes** contains disposable structures that make the memory log easier to search.
- **Retrieval and grounding** use indexes to find candidates, then return to the memory log before answering.
- **Query Run** records how a read happened; it is diagnostic history, not memory evidence.

### 1. Source processing and Ingest

The application submits one typed source:

```python
from trisynapse_memory import MemoryEngine, SourceInput

memory = MemoryEngine.open("./memory")
ingestion = memory.ingest(
    SourceInput(kind="file", path="atlas-brief.pdf", source_key="atlas-brief")
)
```

The PDF loader emits one chunk per non-empty page. The source pipeline retains the original by content hash, adds its page locator, and serially appends the accepted chunks.

| Chunk | Text | Locator |
|---|---|---|
| Page 1 | Project Atlas launched on 14 May 2026. | `{"kind": "page", "page": 1}` |
| Page 2 | Maya Chen is the owner of Project Atlas. | `{"kind": "page", "page": 2}` |

### 2. Memory log: what becomes evidence

The two chunks become separate `raw_source` records in one record group. A simplified view is:

```json
[
  {
    "id": "d_page_1",
    "seq": 41,
    "kind": "raw_source",
    "group_id": "source:src_atlas:v1",
    "text": "Project Atlas launched on 14 May 2026.",
    "locator": {"kind": "page", "page": 1}
  },
  {
    "id": "d_page_2",
    "seq": 42,
    "kind": "raw_source",
    "group_id": "source:src_atlas:v1",
    "text": "Maya Chen is the owner of Project Atlas.",
    "locator": {"kind": "page", "page": 2}
  }
]
```

If a completion model is configured, Ingest can append evidence-linked fact records:

```json
[
  {
    "id": "d_fact_launch",
    "kind": "fact",
    "text": "Project Atlas launched on 14 May 2026.",
    "subject": "Project Atlas",
    "relation": "launched_on",
    "object": "14 May 2026",
    "evidence_refs": ["d_page_1"]
  },
  {
    "id": "d_fact_owner",
    "kind": "fact",
    "text": "Maya Chen owns Project Atlas.",
    "subject": "Project Atlas",
    "relation": "owned_by",
    "object": "Maya Chen",
    "evidence_refs": ["d_page_2"]
  }
]
```

The raw source remains the original document evidence. The fact is another memory-log record that says exactly which raw source supports it. If no completion model is configured, the raw sources are still stored and searchable; only model-backed fact extraction is skipped. For the remaining example, assume that a completion model is configured and the facts and group summary jobs have completed.

### 3. Indexes: what gets built for fast lookup

Indexes build several helpers over those memory-log IDs:

| Index component | Example output | Why it exists |
|---|---|---|
| Retrieval document | Text, modality `document`, page locator, source fields | Gives every source a common searchable shape without losing structure |
| BM25 postings | `atlas → d_page_1, d_page_2`; `owner → d_page_2` | Finds exact names and terms |
| Embedding index | Embeddings keyed by text hash and model fingerprint | Finds similar meaning when words differ |
| group summary | “Atlas launch and ownership: launched 14 May 2026; owned by Maya Chen.” | Routes broad questions toward this record group |
| Claims | `Project Atlas —owned_by→ Maya Chen` | Groups compatible extracted facts and exposes conflicts |
| Graph edges | source → raw source → fact → claim | Supports neighborhood and multi-hop retrieval |

Every helper points back to the memory log. group summary and claims may help choose an area to search, but they are not accepted as final evidence by themselves.

### 4. Retrieval: what happens for a question

Now the user asks:

```text
When did Project Atlas launch, and who owns it?
```

The engine creates a durable Query Run before retrieval starts. The intermediate steps for this example look like this:

| Step | Example intermediate output |
|---|---|
| Input | Original query recorded for the durable Query Run |
| Classification | `query_kind=temporal`, `profile=balanced`; all enabled routes run, with useful results here from BM25, semantic, temporal, graph, and document routes |
| Index snapshot | Trace cutoff `44`; two raw sources, two facts, one group summary view |
| BM25 route | `d_page_2` for “owner”; `d_page_1` for “launch” and “date” |
| Semantic route | Both page raw sources plus the two linked facts |
| Temporal route | `d_page_1` and `d_fact_launch` because they carry the launch date |
| Graph route | Owner and launch fact nodes connected through their shared `Project Atlas` entity |
| Fusion | One ranked candidate list produced from the enabled route weights |
| Confidence | Top score, margin, and evidence agreement pass the configured gate |
| Grounding | Helper nodes are drilled down to `d_page_1` and `d_page_2`; token and per-source limits are applied |

The route outputs are candidate lists, not five answers. Weighted rank fusion combines their ranks, and the grounding stage resolves claims or group summary views back to their supporting raw-source or fact records.

If confidence had remained low, the saved workflow could contain up to two refinement rounds. With deep summary enabled, a low-confidence or multi-hop question can also run a wider search and one extra graph hop. Those stages are conditional, so they do not appear in every Query Run.

```mermaid
flowchart TD
    Q["Question"] --> C["Classification + index snapshot"]
    C --> B["BM25 candidates"]
    C --> S["Semantic candidates"]
    C --> T["Temporal candidates"]
    C --> G["Graph candidates"]
    C --> D["Document-field candidates"]
    B --> F["Weighted rank fusion"]
    S --> F
    T --> F
    G --> F
    D --> F
    F --> CF{"Confidence passes?"}
    CF -->|"yes in this example"| TR["Ground to page 1 + page 2 memory-log records"]
    CF -->|"no"| RF["Refine; optionally run deep summary"]
    RF --> TR
```

### 5. Grounded answer and citations

Only the grounded memory-log records enter the answer context. The result can be:

```json
{
  "answer": "Project Atlas is owned by Maya Chen and launched on 14 May 2026.",
  "abstain": false,
  "citations": [
    {"record_id": "d_page_2", "locator": {"kind": "page", "page": 2}},
    {"record_id": "d_page_1", "locator": {"kind": "page", "page": 1}}
  ]
}
```

If the owner page were missing, the engine should not fill the gap from general model knowledge. It should answer only the supported part or abstain, depending on the evidence and configured threshold.

### 6. Query Run: how the process remains inspectable

The Query Run stores the question, effective retrieval configuration, executed steps, bounded candidate snapshots, timings, branch decisions, final answer, citations, and provider/prompt provenance. Studio reads this record to reopen the same workflow later.

It does **not** store embeddings, API keys, authorization headers, full system prompts, or hidden model reasoning. Removing query history deletes these diagnostics; it does not remove the document from the memory log.

The central rule illustrated by the whole example is: **Indexes may point to an answer, but the memory log must support it.**

## What a memory-log record contains

A record is one ordered piece of evidence. Useful fields include:

- a stable ID and sequence number;
- the text;
- event kind, time, actor, and namespace;
- source and exact locator, such as a PDF page or code line range;
- evidence links and structured payload.

```mermaid
flowchart TD
    W["Serialized SQLite writes"] --> D1["Record · seq 41"]
    D1 --> D2["Record · seq 42"]
    D2 --> D3["Record · seq 43"]
    D3 --> V["validate_store(): database, order, references, source blobs"]
```

### Ordering and validation

Memory-log rows are ordered by a unique sequence number assigned inside the same SQLite transaction that writes the row. Stable IDs connect raw sources, facts, retractions, citations, and sources. This is enough for normal memory lifecycle operations without making every write depend on the previous row.

`validate_store()` performs practical store checks. The CLI exposes the same operation as `trisynapse-memory validate`:

- SQLite's own consistency check succeeds;
- sequence numbers are contiguous;
- evidence references point to existing records;
- retained source files exist and still match their recorded SHA-256 values.

This validation catches corruption, incomplete data relationships, and missing or changed source blobs. It is not a cryptographic audit ledger and does not prove that a database administrator never changed a row. If that threat model matters, protect backups and audit exports with an external signed or append-only system.

SHA-256 remains useful for content identity, so Trisynapse uses it for source deduplication and blob checking, source version detection, file provenance, cache keys, endpoint fingerprints, and prompt provenance. These hashes do not link memory-log records together and do not hide sensitive content.

Physical removal redacts only the requested rows in one transaction, invalidates affected index data, and records the operation in `removal_audit`. Older stores are migrated in place: record IDs, order, timestamps, links, snapshots, and historical `[PURGED]` payloads stay unchanged while obsolete chain-only columns are dropped.

## Ingest: turning input into the memory log

Ingest is the write path.

```mermaid
flowchart TD
    S["One or many SourceInput items"] --> RUN["Durable ingestion run"]
    RUN --> K{"Source kind"}
    K -->|"text / Office / structured"| T["Structure-aware chunks + locators"]
    K -->|"code / notebook / repository"| C["Symbols or bounded line chunks"]
    K -->|"image"| I["Vision facts"]
    K -->|"web page / public Git"| N["Safe fetch or shallow clone"]
    K -->|"archive"| A["Traversal-safe expansion"]
    T --> PS["PreparedSource"]
    C --> PS
    I --> PS
    N --> PS
    A --> PS
    PS --> B["Retain accepted original by SHA-256"]
    PS --> O["Sequential append of chunks and searchable fields to the memory log"]
    O --> IDX["Update retrieval documents, BM25 terms, and graph edges"]
    B --> ACTIVE["Activate the committed source version"]
    O --> ACTIVE
    ACTIVE --> RESULT["Record the ordered source result"]
    ACTIVE --> J["Schedule the source record group once"]
    J --> CP{"Completion model configured?"}
    CP -->|"yes"| X["Extract evidence-linked facts"]
    X --> E["Build group summary"]
    CP -->|"no"| E
    E --> JOB["Save durable job status"]
```

### Unified sources

`SourceInput` describes one item. `ingest()` handles one; `ingest_many()` handles a mixed list. Each result keeps its input index, so successes and failures remain ordered.

Preprocessing uses at most four workers. Memory-log writes stay sequential so every committed record receives one unambiguous sequence number. A broken PDF therefore does not roll back a valid code repository in the same run.

Runs and item results live in SQLite. REST can return a run ID immediately, Studio can follow it, and interrupted or failed work can be retried after restart.

### Retained originals and versions

Accepted originals are stored by SHA-256 under `sources/sha256/`. Directories and repositories become safe packages containing only accepted files.

For a stable `source_key`:

```mermaid
flowchart TD
    A["First content hash"] --> V1["Source version 1 becomes active"]
    B["Same content hash again"] --> NO["No-op: return existing source"]
    C["Changed content hash"] --> V2["Prepare and append every v2 chunk"]
    V2 --> OK{"Succeeded?"}
    OK -->|"yes"| SW["Activate v2, retract v1 chunks"]
    OK -->|"no"| KEEP["Keep v1 active"]
```

This order avoids losing a working source because its replacement failed.

### Code is not ordinary prose

Fixed character chunks often split a function in half. Trisynapse uses Tree-sitter where a parser is available and keeps symbols together. Python also has a built-in AST path. Unsupported languages fall back to bounded line chunks.

A code citation can carry:

```json
{
  "path": "src/payments.py",
  "language": "py",
  "symbol": "capture_payment",
  "symbol_kind": "function",
  "start_line": 41,
  "end_line": 79,
  "metadata": {
    "imports": ["from decimal import Decimal"],
    "file_hash": "…"
  }
}
```

Git sources also carry the resolved commit SHA and remote URL. One manifest raw source lists every accepted repository path. `.gitignore` and `.trisynapseignore` are honored; dependencies, builds, caches, secrets, binaries, and symlinks are skipped and reported.

### Images

An image is sent to the configured completion model with a versioned facts prompt. The result describes visible text, the scene, tables/charts, and important relationships. That extracted text is written to the memory log as source content.

There is no hidden OCR fallback. If the provider or model lacks vision support, that source fails clearly while other items in the run may still succeed.

### One searchable shape, without flattening everything

Every memory-log item also gets a rebuildable `RetrievalDocument`. It keeps the original text plus named fields and a broad modality:

| Source | Modality | Example searchable fields |
|---|---|---|
| Code | `code` | path, language, symbol, imports, code |
| CSV/XLSX | `table` | sheet, row, headers, record |
| PDF/DOCX/PPTX/HTML | `document` | title, page, slide, section, path |
| Image | `image` | filename, visible text, description |
| Messages | `conversation` | speaker, message |

This is different from converting every input to anonymous prose. A semantic route can search all text, while a code route can give extra weight to an exact symbol or path. Search hits return their modality, source type, and retrieval fields in `metadata`. Applications should sanitize sensitive values before submitting text, metadata, locators, or retrieval fields.

## Indexes: fast help that can be rebuilt

The memory log is the authority, but scanning and rebuilding every row for every question is slow. Indexes maintain smaller views as data is written:

- keyword and vector indexes;
- group summary summaries;
- evidence-linked claims;
- graph relations and profiles.

These products store links back to source record IDs. They can be stale, deleted, or rebuilt without losing original evidence.

```mermaid
flowchart TD
    T["memory-log records"] --> D["RetrievalDocument\ntext + modality + named fields"]
    T --> E["group summary"]
    T --> C["Evidence-linked claims"]
    D --> B["SQLite BM25 postings"]
    D --> V["Fingerprint-partitioned embedding index"]
    D --> G["Persisted graph edges"]
    B --> R["Query planner + RouteRegistry"]
    V --> R
    G --> R
    E --> R
    C --> R
    R --> IDs["Ranked candidate memory-log IDs"]
    IDs --> T
```

SQLite keeps searchable documents, BM25 postings, and graph edges incrementally. The vector backend adds each new embedding under the active provider/endpoint/model fingerprint and uses its nearest-neighbor index when available. A retraction deactivates its target postings and edges; physical removal deletes them. Opening an older store backfills these disposable tables without editing historical deltas.

Durable jobs run fact extraction and claims. Their state and errors survive restarts. This makes provider failures visible without losing the raw source that triggered the job.

## Two model roles, one store-wide setting

Trisynapse uses models for two different jobs:

- The **completion model** reads or writes text. It can extract facts, summarize an episode, answer a grounded question, or describe an image.
- The **embedding model** turns text into vectors. Retrieval compares these vectors to find related evidence.

They do not need to come from the same provider.

```mermaid
flowchart TD
    CFG["Model configuration saved in this store"]
    CFG --> C["Completion provider + model"]
    CFG --> E["Embedding provider + model"]
    C --> X["Future extraction and group summary"]
    C --> A["Future grounded answers and images"]
    E --> V["Embedding index for all searchable text"]
    V --> R["Semantic retrieval"]
```

Only the provider name, model ID, optional endpoint, revision, and update time are saved. API keys stay in environment variables. This lets the terminal, Python, REST, TypeScript, and Studio read the same choice without copying secrets into SQLite.

### Why completion changes are immediate

Changing completion does not change old evidence or old generated records. It simply changes the model used by the next provider-backed operation. Each generated artifact records its provider, model, prompt version, and prompt hash, so old and new results remain explainable.

### Why embedding changes rebuild first

Vectors from different models do not share one coordinate system. Comparing a query vector from model B with stored vectors from model A would produce meaningless scores. Trisynapse therefore never activates an embedding change until every searchable item has a compatible replacement vector.

```mermaid
flowchart TD
    S["User chooses a new embedding model"] --> C{"Store has searchable memory?"}
    C -->|"no"| I["Activate immediately"]
    C -->|"yes"| W["Show rebuild and possible cost warning"]
    W --> Y{"User confirms?"}
    Y -->|"no"| K["Keep current model and index"]
    Y -->|"yes"| J["Create durable rebuild job"]
    J --> B["Build new fingerprinted embedding index"]
    B --> O{"Complete without error?"}
    O -->|"yes"| SW["Atomically activate new embedding choice"]
    O -->|"no"| K
```

The active index continues serving queries during the rebuild. The cache identity includes provider, normalized endpoint hash, and model, preventing two services with the same model name from accidentally sharing vectors. Another process notices the new configuration revision before its next provider-backed operation, so a server restart is not required.

## Retrieval: plan, route, drill down, answer

Retrieval is not a benchmark script. `MemoryEngine.search()` and `query()` both run this production path.

The default query planner identifies the question shape and useful source modalities. It returns a `QueryPlan`, then an ordered `RouteRegistry` runs independent routes:

- BM25 postings for exact terms;
- vector nearest neighbors for meaning;
- temporal anchors and event time;
- persisted graph neighbors;
- field-aware code, table, image, document, and conversation routes.

Applications may provide another `QueryPlanner`, register a new `RetrievalRoute`, or replace an existing route. The benchmark adapters only translate benchmark records into ordinary sources; they do not control production ranking.

```mermaid
flowchart TD
    Q["Question"] --> P["Query planner"]
    P --> I["Intent + modalities + profile"]
    I --> B["BM25 route"]
    I --> S["Semantic route"]
    I --> T["Temporal route"]
    I --> GR["Graph route"]
    I --> M["Source-aware routes"]
    B --> F["Weighted rank fusion"]
    S --> F
    T --> F
    GR --> F
    M --> F
    F --> CF{"Confident?"}
    CF -->|"no"| R1["Refine query for up to the configured rounds"]
    CF -->|"yes"| ESC{"deep summary enabled and\nlow confidence or multi-hop?"}
    R1 --> ESC
    ESC -->|"yes"| DR["deep summary with a wider context and graph walk"]
    ESC -->|"no"| D["Drill to the memory log"]
    DR --> D
    D --> EG{"Enough grounded evidence?"}
    EG -->|"yes"| A["Answer with citations"]
    EG -->|"no"| X["Abstain"]
```

`retrieval_profile` selects a balanced, precise, broad, mixed, or modality-focused set of route weights. `auto` lets the planner select a profile from the question. `route_weights` can override individual routes, and `enabled_routes` can disable routes for a store. The saved configuration and the effective query plan are captured in the durable query run.

Episode summaries are routing hints. They are not placed into final answer context as unsupported truth. `retrieval_trace` exposes the chosen stage, routes, scores, seeds, and number of drilled memory-log records.

For multi-hop questions, extraction records connect through shared subjects and objects. A query can therefore walk `person → event → outcome` even when no single observation contains every word in the question. Claims and group summary locate a useful region; the engine then reranks the supporting raw sources against the original question. It does not simply return an entire record group.

Context selection is bounded twice: by record count and by tokens. The default answer window allows at most 24 records and 6,000 tokens, with 2,000 tokens per source. Trisynapse uses a provider-supplied local counter when available, Tiktoken for recognized OpenAI models when the `tokens` extra is installed, and an explicitly labelled provider-aware Unicode estimate otherwise. The query workflow records the counter name and whether it was exact. Applications can supply their own `TokenCounter`. This keeps one large PDF, repository, or conversation from crowding every other source out of the answer. Ordinary `search()` still defaults to 12 results.

Retrieval math has one implementation per formula. SQLite and in-memory BM25 call the same Okapi BM25 primitive. Pairwise and fallback nearest-neighbor cosine scoring use shared NumPy functions. LanceDB queries explicitly request cosine distance and convert cosine distance to similarity with `1 - distance`. Weighted rank fusion follows `weight / (60 + rank + 1)` and is independently tested.

Each extracted fact records the smallest supporting set of stable raw-source IDs. Those references flow into claims, graph edges, grounding, and citations. This is why a derived fact can route efficiently without becoming a replacement for its original evidence.

### Extensible pipelines

The engine supports trusted, explicitly registered extension packs. A pack can contribute one or more independent capabilities instead of inheriting from one large plugin base class:

```mermaid
flowchart LR
    E["Extension pack"] --> S["Source handler"]
    E --> F["Ingest processor"]
    E --> C["Index channel"]
    E --> B["Retrieval branch"]
    E --> U["Fusion / reranker"]
    S --> T["Core memory-log append"]
    F --> T
    T --> C
    T --> B
    C --> B
    B --> G["Core validation + grounding"]
    G --> T
```

Registration is frozen before the engine starts. Components receive scoped readers, writers, and model services rather than a raw SQLite connection. The core continues to own ordering, namespaces, evidence validation, context limits, retraction, physical removal, and citations.

The write-side distinction remains important:

- a **source handler** decodes and chunks a new `SourceInput.kind`;
- a **Ingest processor** proposes citable fact or note records and must identify supporting evidence;
- a **Index channel** creates disposable records that can be rebuilt from the memory log.

Index channels use a generic core-owned record store by default. Records carry a channel ID, namespace, producer/evidence version, searchable text and fields, and supporting memory-log IDs. Retraction makes dependent records inactive; physical removal deletes them. Optional projections run through durable jobs and cannot roll back already committed memory-log records when they fail.

A **retrieval branch** returns explicit `RetrievalCandidate` values rather than IDs assumed to exist in a private index. Log candidates identify one active record. Index candidates include routing text and active, same-namespace evidence IDs. The core rejects invalid candidates, fuses valid branch rankings, and resolves index candidates back to the memory log before answer context is assembled.

Branch metadata declares dependencies, default/profile weights, cost tier, and a candidate limit. Dependencies form a validated acyclic graph, so Graph-like expansion no longer requires a name-specific execution switch. Weighted RRF remains the default `FusionStrategy`, and the semantic/reliability/recency ordering is an explicit replaceable `CandidateReranker`.

The existing `QueryPlanner`, `RetrievalRoute`, and `RouteRegistry` APIs remain as compatibility surfaces for algorithms that rank the built-in candidate corpus. New end-to-end channels should use extension packs. The memory catalog reports extension versions/status, Index channel health, retrieval dependencies, and active weights so Studio and external clients remain catalog-driven.

The current engine loads extensions only from Python objects passed to `MemoryEngine.open(extensions=[...])`. Automatic package discovery and untrusted plugin execution are intentionally not part of this boundary.

### Query runs: the inspectable retrieval record

A query does not become evidence. Instead, Trisynapse stores it in a separate, removable **Query Run** record. This keeps the memory log focused on memory while still letting Studio reopen exactly how an answer was found.

```mermaid
flowchart TD
    Q["Question"] --> QR["Create durable Query Run\nquery + configuration snapshot"]
    QR --> P["Input, classification, and index snapshot"]
    P --> R["Run selected retrieval routes"]
    R --> F["Fuse, rerank, and test confidence"]
    F --> D{"Needs refinement or a multi-hop bridge?"}
    D -->|"yes"| RF["Refinement rounds and optional deep summary"]
    D -->|"no"| T["Ground candidates in the memory log"]
    RF --> T
    T --> A["Generate cited answer or abstain"]
    A --> AUDIT["Write citations and access audit"]
    AUDIT --> DONE["Mark Query Run completed or failed"]

    P -. append QueryStep .-> QR
    R -. append QueryStep .-> QR
    F -. append QueryStep .-> QR
    RF -. append QueryStep .-> QR
    T -. append QueryStep .-> QR
    A -. append QueryStep .-> QR
    QR --> SSE["SSE streams saved updates"]
    QR --> HIST["Studio reopens the same workflow later"]
```

Every executed box is stored as a `QueryStep`. Each route keeps at most 20 safe candidate snapshots, not the whole index. Embedding vectors, credentials, authorization headers, full prompts, and private model reasoning are never stored. Query history remains until explicitly removed. The memory log access event keeps the query ID and citations, but not the question text.

`query()` also returns `retrieval_hits`. They are the pre-answer grounded records and remain available when the completion model abstains. `citations` describe what the final answer used; `retrieval_hits` describe what retrieval found. Keeping them separate makes both Studio diagnostics and benchmarks honest.

### Memory Viewer

Studio's home is Memory Viewer. `GET /api/v1/memory/catalog` lists the memory log, every index helper, and every retrieval route the running engine actually has. Tabs, health chips, playground route filters, and Configuration's route editors are generated from that catalog — not a hardcoded page list.

Each helper declares a visual family (`timeline`, `table`, `postings`, `embedding`, `cards`, `graph`). Studio registers specialized renderers locally; an unknown helper id still renders the family view from generic `items[]`. Deep links stay stable: `/studio/memory?helper=bm25&id=atlas`.

The Graph tab keeps three architecture views over the same memory log and index data, plus the retrieval graph the graph route actually walks:

| View | Starts from | Makes clear |
|---|---|---|
| Knowledge | Concepts and claims | What Trisynapse currently knows and which relationship joins two concepts |
| Lineage | Sources, log chunks, facts, claims, and group summary | Where a derived memory came from and which original evidence grounds it |
| Retrieval | `followed_by` and `about_same_entity` edges | How the graph route expands candidates |
| Memory log (timeline tab) | Ordered records grouped by source or record group | What was written, corrected, retracted, or accessed over time |

```mermaid
flowchart TD
    subgraph K["Knowledge view"]
        KC1["Concept: Maya"] -->|"subject"| KCL["Claim: prefers three-bullet updates"]
        KCL -->|"object"| KC2["Concept: short updates"]
    end

    subgraph L["Lineage view"]
        LS["Source + version"] -->|"produced"| LO["Observation"]
        LO -->|"supports"| LE["Extraction"]
        LE -->|"grounds"| LC["Compiled claim"]
        LO -->|"summarized by"| LR["group summary"]
        LE -->|"summarized by"| LR
    end

    subgraph R["Retrieval graph"]
        RT1["Record A"] -->|"followed_by"| RT2["Record B"]
        RT1 -->|"about_same_entity"| RT3["Record C"]
    end
```

The retrieval playground runs production `search(persist=false)` from the current selection so inspection does not fill Query history. "Open as Query" is the only path that writes a Query Run.

Large stores open as a bounded overview. Search and neighbor expansion make every visible-namespace node reachable without drawing an unreadable graph all at once.

## Namespaces

Every public operation uses a namespace:

```text
user_id / agent_id / project_id / session_id
```

`project_id` defaults to `default`. Other fields are optional. A write under one namespace is not visible from another. Stable namespace values matter more than filling every field.

## Correct, forget, and remove

These operations solve different problems:

| Operation | What happens | Old content remains? | Use when |
|---|---|---:|---|
| `correct()` | Append linked corrected evidence | Yes | The old statement was wrong or incomplete |
| `forget()` | Append a logical retraction | Yes, for authorized history | Memory should stop using it |
| `remove()` | Redact selected delta fields and clear affected index data | No | Physical deletion is required |
| `remove_source()` | Remove an unshared retained blob and all derived deltas | No | Delete one imported source completely |

Delta-only removal does not silently delete a larger source package. Conversely, source removal is intentionally larger: it redacts all chunks derived from that source.

Historical stores may contain `[PURGED]` rows. Migration never rewrites them; new operations use `[REMOVED]` and removal terminology.

## Storage map

```mermaid
flowchart TD
    ROOT["Memory store"] --> DB["trace.sqlite3"]
    ROOT --> SRC["sources/sha256/"]
    ROOT --> VC["vectors.lance/"]
    ROOT --> KEY[".api-key"]
    DB --> D["memory-log records"]
    DB --> M["Sources + ingestion runs"]
    DB --> Q["Query runs with steps + retrieval traces"]
    DB --> J["Durable jobs + snapshots + removal audit"]
    DB --> R["Retrieval documents + BM25 terms + graph edges"]
    DB --> RC["group summary + SQLite vector fallback"]
    DB --> EX["Generic index records + extension state"]
    DB --> CFG["Model + retrieval configuration + provider catalog"]
    SRC --> O["Content-addressed originals and safe source packages"]
    VC --> V["Embedding tables partitioned by configuration fingerprint"]
```

SQLite uses WAL, full synchronous writes, and secure deletion. Memory-log writes are serialized. Vector storage is only a cache. Backups include SQLite, source blobs, and rebuildable products.

Extension storage revisions use the same memory-log-first rule. A revision change disables the pack's branches, schedules a durable extension rebuild across stored namespaces, and returns the pack to `available` only after its persistent channels rebuild successfully through scoped readers and atomic writers.

Source files are permission-restricted but not encrypted. Use an encrypted disk or storage layer when required.

## Code map

The engine folders use the same names as the architecture. `MemoryEngine` is the
public orchestrator; shared contracts and utilities remain at the engine root.

```text
engine/
├── memory.py                 public orchestration
├── models.py                 shared contracts
├── extensions.py             extension packs and capability registries
├── utils.py                  shared numerical and version helpers
├── ingest/                   source preparation and evidence creation
├── memory_log/               durable ordered evidence
├── indexes/                  rebuildable views and the embedding index
├── retrieval/                planning, ranking, grounding, and context
└── providers/                completion and embedding integrations
```

| Area | Path | Responsibility |
|---|---|---|
| Public engine | `src/trisynapse_memory/engine/memory.py` | Lifecycle, ingestion runs, retrieval, backups |
| Shared contracts | `engine/models.py` | Public models used across every architecture component |
| Extension contracts | `engine/extensions.py` | Packs, source/ingest/index/branch registries, fusion and reranking |
| Shared utilities | `engine/utils.py` | Runtime version lookup, NumPy cosine, shared BM25, weighted reciprocal-rank fusion |
| Ingest pipeline | `engine/ingest/pipeline.py` | Raw-source and evidence-linked fact writes |
| Ingest sources | `engine/ingest/sources.py` | Loaders, safety rules, retained blobs, code/image/archive handling |
| Memory log | `engine/memory_log/store.py` | SQLite, ordered records, jobs, sources, query runs, validation, removal audit |
| Claims and group summaries | `engine/indexes/claims.py` | Claims and group summary |
| Embedding index | `engine/indexes/embedding_index.py` | Rebuildable SQLite and LanceDB embedding indexes |
| Retrieval contracts | `engine/retrieval/contracts.py` | Canonical documents, query planner, route interface, registry, profiles |
| Retrieval execution | `engine/retrieval/engine.py` | Fusion, confidence, memory-log drill-down, context budgeting |
| Retrieval tokenization | `engine/retrieval/tokenization.py` | Unicode/code lexical tokens and replaceable context token counters |
| Provider registry | `engine/providers/registry.py` | Completion, vision, discovery, and model configuration |
| Provider embeddings | `engine/providers/embedding.py` | Local and remote embedding integrations |
| Prompt package | `prompts/` | Versioned production prompts |
| REST/Studio | `api.py`, `studio/` | Service and browser UI |
| Terminal/CLI | `cli.py`, `terminal.py` | Scriptable and interactive operations |
| Adapters | `adapters/` | Agent events, Trisynapse, benchmark formats |

## Invariants worth remembering

1. The memory log is the evidence; indexes are rebuildable helpers.
2. Accepted content is stored as supplied; applications sanitize sensitive input before ingestion.
3. Store validation checks SQLite, sequence continuity, evidence links, and retained source blobs; it is not a tamper-proof ledger.
4. Retrieval drills down to the memory log before answering.
5. Mixed sources fail independently, but memory-log writes remain sequential.
6. A replacement source becomes active only after successful processing.
7. Forget is logical; remove is physical.
8. The system should abstain when evidence is not strong enough.
9. Completion changes affect future generation; embedding changes activate only after a confirmed successful rebuild.
10. Source modalities and fields survive ingestion; final answers still cite the memory log.

Continue with [Algorithms](algorithms.md) for formulas and procedures, [API and Interfaces](api.md) for code examples, or [Operations](operations.md) for installation, security, and recovery.
