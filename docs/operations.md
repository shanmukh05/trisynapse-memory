# Operations

This guide covers installation, the terminal, providers, source safety, service operation, backups, and releases. Read [Architecture](architecture.md) for the memory model and [API](api.md) for code contracts.

## Install or upgrade

macOS/Linux:

```bash
curl -LsSf https://github.com/shanmukh05/trisynapse-memory/releases/latest/download/install.sh | sh
```

Windows PowerShell:

```powershell
irm https://github.com/shanmukh05/trisynapse-memory/releases/latest/download/install.ps1 | iex
```

The scripts install or reuse `uv`, run `uv tool install --upgrade 'trisynapse-memory[all]'`, write local install metadata, identify the tool bin directory, and run `trisynapse-memory --json check`. Re-running upgrades the tool.

```bash
uv tool uninstall trisynapse-memory
```

Direct PyPI alternatives:

```bash
pip install trisynapse-memory               # engine + terminal
pip install 'trisynapse-memory[sources]'    # Office/PDF/Tree-sitter loaders
pip install 'trisynapse-memory[server]'     # REST service
pip install 'trisynapse-memory[all]'        # user-facing features
```

Python 3.11+ is required. Native binaries and Homebrew are deferred to avoid bundling Python, PyTorch, and local model assets into very large artifacts.

## Initialize and inspect

```bash
trisynapse-memory init
trisynapse-memory --json check
trisynapse-memory verify
```

The default store is `~/.trisynapse-memory/store`. Change it with the global `--path` option or `TRISYNAPSE_MEMORY_PATH`.

`check` checks the installed command, source extras, write permissions, Trace integrity, provider credentials, the multimodal interface, port 8765, and pending/failed jobs and ingestion runs. It also warns that retained originals are not encrypted.

## Interactive terminal

```bash
trisynapse-memory
```

The terminal shows a monochrome rendering of the project logo, switching to a compact mark on narrow screens. It also shows the version, active store/namespace, completion provider, Trace integrity, and pending jobs/runs. Plain text asks a grounded question. Use `/help` for slash commands.

Typing `/` opens live recommendations. Suggestions narrow as characters are entered; **Tab** or **Right Arrow** accepts the ghost completion. `/ingest` recommends matching files and directories, while lifecycle commands recommend recent memory IDs.

Ctrl+C clears the current prompt; Ctrl+C on an empty prompt exits. Ctrl+L clears the activity view and Ctrl+D exits.

For automation:

```bash
trisynapse-memory --json --project-id demo sources list
trisynapse-memory --quiet jobs run
trisynapse-memory --no-color verify
```

Machine JSON stdout contains no logo, spinner, table, or progress text.

## Store layout and permissions

```text
store/
├── trace.sqlite3
├── trace.sqlite3-wal / trace.sqlite3-shm
├── sources/sha256/        retained originals and safe source packages
├── vectors.lance/         rebuildable cache
└── .api-key               local administrator key
```

Blob directories use mode `0700` and files use `0600` where supported. SQLite enables `secure_delete`, WAL, and full synchronous writes. The store is **not encrypted**. Use an encrypted filesystem/volume if stored sources require encryption at rest.

## Source ingestion safety

Defaults:

| Limit or rule | Default |
|---|---:|
| Uploaded file | 25 MiB |
| Retained input per run | 250 MiB |
| Source descriptors | 100 |
| Expanded archive files | 10,000 |
| Concurrent preprocessors | 4 |
| Redirects | 5 |
| Fetch timeout | 30 seconds |

URL ingestion blocks embedded credentials and private, loopback, link-local, and reserved addresses. Redirect targets are checked again. Git accepts public HTTPS remotes, shallow-clones the requested branch/ref, records the resolved commit, and does not retain credentials.

Archives reject absolute paths, `..` traversal, symlinks, oversized expansion, and excessive file counts. Directories skip symlinks, secret file names/extensions, binaries, dependency/build/cache directories, and ignored paths. Every skipped path is returned in the source result and record metadata.

Retained blobs are content-addressed, so identical content uses one file. A source is removed physically only when no other non-removed source references that blob.

## Completion and embedding providers

Trisynapse uses two independent model settings:

- The **embedding provider** turns text into vectors for semantic search. It defaults to local SentenceTransformers and needs no API key.
- The optional **completion provider** extracts facts, builds richer Recall views, generates grounded answers, and reads images. Text ingestion and retrieval still work when it is `none`; image ingestion does not.

### Fastest setup

Model choices are saved in the memory store. Environment variables contain credentials only. For Anthropic completion with the default local embeddings:

```bash
export ANTHROPIC_API_KEY="your-key"
trisynapse-memory models set completion anthropic claude-sonnet-4-5
trisynapse-memory --json check
trisynapse-memory
```

Or open the interactive terminal, enter `/model completion`, choose Anthropic, search the live model catalog, and Apply.

PowerShell uses the same credential name:

```powershell
$env:ANTHROPIC_API_KEY = "your-key"
trisynapse-memory models set completion anthropic claude-sonnet-4-5
trisynapse-memory --json check
```

Environment variables set this way last for the current terminal session. Trisynapse deliberately does **not** load `.env` files automatically. If you use the repository template on macOS/Linux, keep the local file private and explicitly load it:

```bash
cp .env.example .env.local
chmod 600 .env.local
# Edit .env.local, then load it into this shell:
set -a
source .env.local
set +a
```

`.env.local` is ignored by Git. Never commit a real provider key or pass one as an ordinary CLI argument where it may remain in shell history.

### Supported completion providers

| Provider | Credential | Default endpoint | Notes |
|---|---|---|---|
| `none` | None | None | Default. Retrieval and extractive grounded answers work. |
| `openai` | `OPENAI_API_KEY` | `https://api.openai.com/v1` | Completion, structured JSON, and model-dependent vision. |
| `openrouter` | `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1` | Live multi-provider catalog, including Qwen families when available. |
| `gemini` | `GEMINI_API_KEY` | Gemini SDK | Native Gemini generation, model listing, and vision. |
| `anthropic` | `ANTHROPIC_API_KEY` | `https://api.anthropic.com` | Native Messages API, live catalog, structured JSON, and model-dependent vision. |
| `deepinfra` | `DEEPINFRA_API_TOKEN` | `https://api.deepinfra.com/v1/openai` | Completion and vision through the OpenAI-compatible API. `DEEPINFRA_TOKEN` is accepted as an environment alias. |
| `deepseek` | `DEEPSEEK_API_KEY` | `https://api.deepseek.com` | Direct text completion and live model catalog. |
| `kimi` | `MOONSHOT_API_KEY` | `https://api.moonshot.ai/v1` | Direct completion; vision is shown only for catalog models that report it. |
| `openai-compatible` | `OPENAI_COMPATIBLE_API_KEY` when required | User-supplied `/v1` URL | Custom model IDs and best-effort `/models` discovery. |

Examples:

```bash
export DEEPSEEK_API_KEY="your-key"
trisynapse-memory models set completion deepseek deepseek-chat

export MOONSHOT_API_KEY="your-key"
trisynapse-memory models list --role completion --provider kimi
trisynapse-memory models set completion kimi kimi-k2.5

export OPENAI_COMPATIBLE_API_KEY="your-key"
trisynapse-memory models set completion openai-compatible your-model \
  --base-url https://llm.example.com/v1
```

### Supported embedding providers

Embedding configuration is separate, so OpenAI completion can be combined with local embeddings, for example.

| Provider | Credential | Endpoint or default |
|---|---|---|
| `sentence-transformers` | None | `all-MiniLM-L6-v2`, running locally |
| `openai` | `OPENAI_API_KEY` | `text-embedding-3-small` at the OpenAI `/v1` endpoint |
| `openrouter` | `OPENROUTER_API_KEY` | OpenRouter `/v1`; choose an embedding-capable model |
| `gemini` | `GEMINI_API_KEY` | `gemini-embedding-001` |
| `deepinfra` | `DEEPINFRA_API_TOKEN` | DeepInfra `/v1/openai`; Qwen and other embedding families may appear in its live catalog |
| `openai-compatible` | `OPENAI_COMPATIBLE_API_KEY` when required | User-supplied `/v1` URL and model ID |

Changing embeddings on a non-empty store rebuilds every searchable vector. The change is never silent:

```bash
export DEEPINFRA_API_TOKEN="your-key"
trisynapse-memory models list --role embedding --provider deepinfra
trisynapse-memory models set embedding deepinfra BAAI/bge-m3
# Confirm Yes when prompted. In automation, put --yes before `models`.
```

Trisynapse builds the replacement cache under a provider, endpoint, and model fingerprint. The old index stays active. Only a successful rebuild activates the new configuration; failure keeps the old one and records the job error.

### Model catalogs and connection tests

`models list` and `/model` use official model-list APIs where providers offer them. Results are cached for 15 minutes. `--refresh` bypasses a fresh cache; a stale cache remains usable during a provider-list outage. Sparse catalogs may show unknown capabilities, and an exact custom model ID can still be selected.

Catalog lookup and saving do not run inference. This command does and may use provider credits:

```bash
trisynapse-memory models test --role completion
```

### Configure providers in Python

`MemoryEngine.from_env()` reads credentials from the fixed provider variables above and reads model choices from the store. Python can update the same store-wide configuration:

```python
from trisynapse_memory import MemoryEngine, ProviderSelection

memory = MemoryEngine.from_env("~/.trisynapse-memory/store")
configuration = memory.get_model_configuration()
configuration.completion = ProviderSelection(
    provider="anthropic", model="claude-sonnet-4-5"
)
memory.set_model_configuration(configuration)
```

An explicit `ProviderSettings` passed to `MemoryEngine.open()` remains an in-memory override for advanced embedding or secret-manager integrations. It has priority over the persisted selection for that engine instance and is not written to the store.

### Vision and verification

Image ingestion reuses the selected completion model. `check` can confirm that the adapter offers the multimodal method, while model catalogs mark known vision capability. Only a real request proves model access. Unsupported vision requests fail that source; there is no OCR fallback.

Run `trisynapse-memory --json check` after changing credentials. It reports selections, missing credential variables, and provider initialization errors without printing secret values. Provider failures are reported; Trisynapse does not silently replace a failed production provider with fake vectors or another model.

## REST service and Studio

```bash
trisynapse-memory serve --studio
```

The service binds to `127.0.0.1:8765`, runs durable memory jobs, redirects `/` to Studio at `/studio/`, and serves OpenAPI at `/openapi.json`. `init` creates `<store>/.api-key`; enter it in Studio's Connection view. `--no-auth` is for trusted local development only.

Studio's Sources view supports mixed file/link batches and tailored single-source forms. Retained originals can be previewed or downloaded only after namespace authorization; HTML is never executed inline. Queries stream into a clickable retrieval workflow and remain in removable history. Memory Viewer provides knowledge, lineage, and Trace graphs. Configuration manages both model roles and revisioned retrieval defaults.

The bearer token stays in browser session storage. Namespace preferences use local storage. When Studio connects to a different origin, that server must allow Studio's origin through CORS; same-origin operation needs no extra configuration.

For a shared deployment:

- place TLS and network controls in front;
- keep authentication enabled and use namespace-scoped tokens;
- store credentials in a secret manager;
- persist and protect the entire store directory;
- monitor health, `check`, failed jobs/runs, and Trace verification;
- back up and restore-test regularly.

## Jobs and ingestion runs

Durable jobs are `extract_episode`, `compile_episode`, `rebuild_embeddings`, and single-attempt `execute_query`. States are pending, running, completed, or failed. Expired extraction/compilation leases return to the retry flow. Live queries use one attempt so a restart cannot silently repeat a billable completion; their saved partial workflow is marked interrupted. Embedding rebuild jobs also own a pending configuration; final failure clears it without changing the active model.

```bash
trisynapse-memory jobs list --status failed
trisynapse-memory jobs run --limit 100
```

Source runs contain every original input descriptor and ordered result. If the process stops while a run is active, reopening the store marks it resumable. Retry resumes interrupted runs or creates a new run containing only failed items.

```bash
trisynapse-memory runs list
trisynapse-memory runs show RUN_ID
trisynapse-memory runs retry RUN_ID
```

A source preparation failure never erases an already-active older source version.

## Backup and restore

```bash
trisynapse-memory backup ./backups/memory.zip
trisynapse-memory restore ./backups/memory.zip ./restore-check
trisynapse-memory --path ./restore-check verify
```

Backups checkpoint SQLite and include retained originals. Restore rejects paths that escape the destination, requires an empty target, and verifies Trace before reporting success.

A snapshot is not a backup. It records a recall sequence cutoff:

```bash
trisynapse-memory snapshot create --label before-import
trisynapse-memory snapshot diff SNAPSHOT_A SNAPSHOT_B
trisynapse-memory snapshot rollback SNAPSHOT_ID
```

## Removing data

Logical forgetting:

```bash
trisynapse-memory forget MEMORY_ID --reason "expired"
```

Physical removal requires confirmation (or global `--yes`):

```bash
trisynapse-memory --yes remove memory ID1 ID2 --reason "approved deletion"
trisynapse-memory --yes remove source SOURCE_ID --reason "approved deletion"
```

Removal redacts content, rebuilds Trace hashes, clears derived caches, checkpoints SQLite, and records old/new aggregate roots. Historical `[PURGED]` data is never rewritten during migration.

## Release process

The complete versioning, GitHub/PyPI setup, test matrix, installer verification, deployment, rollback, and production sign-off process lives in the [Production release runbook](release-and-production.md). The summary below describes what automation runs after a tag is pushed.

Pushing a `v*` tag triggers `.github/workflows/release.yml`:

1. test Python 3.11/3.12 and TypeScript on Linux, macOS, and Windows;
2. build wheel and source distribution;
3. publish to PyPI with trusted publishing;
4. attach installers, distributions, `SHA256SUMS`, and `release.json` to GitHub Releases;
5. smoke-test the installer and command on all three operating systems.

Before tagging:

```bash
uv run python scripts/version.py set 0.2.0
uv run python scripts/version.py check --tag v0.2.0
pytest -q
ruff check src tests scripts
pnpm --filter @trisynapse/memory check
trisynapse-memory verify
trisynapse-memory bench gate --mode retrieval --data-root data
```

`pyproject.toml` is the canonical version source. The version script updates the
JavaScript packages, installers, and `uv.lock` together; the tag check prevents a
release when any surface disagrees. Use the version you are actually releasing in
place of `0.2.0`.

Retrieval and end-to-end benchmark artifacts must remain separate. CLI and REST benchmark runs copy the active store model choices into their isolated benchmark store. End-to-end runs require a completion selection plus its credential and record provider/model and prompt version/hash provenance.

## Troubleshooting

| Symptom | Check |
|---|---|
| Known memory missing | Namespace mismatch or logical retraction |
| Source failed | `runs show`, skipped paths, file limit, loader extra |
| Image failed | Completion provider/model vision support |
| URL rejected | HTTPS/public DNS, private-network protection, redirects |
| Query abstained | Evidence score and `retrieval_trace` |
| REST 401/403 | Bearer token and scoped namespace |
| Trace invalid | Preserve store, stop writers, restore verified backup; do not hand-edit rows |
| Command not found | Add `uv tool dir --bin` to PATH and open a new shell |
